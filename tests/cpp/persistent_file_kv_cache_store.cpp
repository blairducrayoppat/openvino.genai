// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/stat.h>
#endif

#include "content_hash.hpp"
#include "continuous_batching/cache/cache_offload_coordinator.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_offload_cache.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_offload_cache.hpp"
#include "continuous_batching/cache/model_identity.hpp"
#include "continuous_batching/cache/store_registry.hpp"
#include "helper.hpp"
#include "model_identity_test_models.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/runtime/core.hpp"
#include "prefix_digest_registry.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

using namespace ov::genai;

// Every test here goes through open_kv_cache_store(), the way the pipeline opens a store, so that the same file also
// builds and runs against a tree in which persist_file_store is accepted but the store is still run-scoped. A test
// that only says something about a persistent store therefore fails there on what a reopened store holds (presence,
// bytes, free count), not on a property flag.

namespace {

constexpr size_t BLOCK_SIZE = 32;
constexpr size_t NUM_BLOCKS = 4;
// On-disk format version 1 (persistent_file_kv_cache_store.hpp): a 4096-byte file header, then slots of
// round_up(64-byte slot header + payload, 4096) bytes, the payload right after the slot header.
constexpr size_t FILE_HEADER_BYTES = 4096;
constexpr size_t SLOT_HEADER_BYTES = 64;
constexpr size_t SLOT_STRIDE = 4096;
// Offsets inside the file header and inside a slot header.
constexpr size_t FILE_FORMAT_VERSION_AT = 8;
constexpr size_t FILE_HASH_VERSION_AT = 12;
constexpr size_t SLOT_MAGIC_AT = 0;
constexpr size_t SLOT_SEQUENCE_AT = 48;

KVCacheDiskLayout make_layout(size_t block_size = BLOCK_SIZE) {
    return KVCacheDiskLayout({block_size}, {0});
}

std::vector<uint8_t> make_block(uint8_t seed, size_t size = BLOCK_SIZE) {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + i);
    }
    return data;
}

/// A fresh, empty directory removed again when the test ends.
struct ScopedDirectory {
    std::filesystem::path path;
    explicit ScopedDirectory(const std::string& name) : path(std::filesystem::temp_directory_path() / name) {
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~ScopedDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

CacheOffloadConfig make_persistent_config(const std::filesystem::path& directory,
                                          size_t num_blocks = NUM_BLOCKS,
                                          size_t block_size = BLOCK_SIZE) {
    CacheOffloadConfig config;
    config.capacity_bytes = block_size * num_blocks;
    config.kv_cache_dir = directory.string();
    config.persist_file_store = true;
    return config;
}

std::filesystem::path slot_file(const std::filesystem::path& directory, const std::string& group) {
    return directory / "ov_genai_kv_store" / group / "slots.bin";
}

size_t slot_at(size_t slot) {
    return FILE_HEADER_BYTES + slot * SLOT_STRIDE;
}

size_t count_regular_files(const std::filesystem::path& directory) {
    size_t files = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        files += entry.is_regular_file() ? 1 : 0;
    }
    return files;
}

std::vector<uint8_t> read_file_bytes(const std::filesystem::path& path, size_t offset, size_t size) {
    std::ifstream file(path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(offset));
    std::vector<uint8_t> bytes(size);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    EXPECT_EQ(static_cast<size_t>(file.gcount()), size) << "short read of " << path.string();
    return bytes;
}

void write_file_bytes(const std::filesystem::path& path, size_t offset, const std::vector<uint8_t>& bytes) {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.is_open()) << path.string();
    file.seekp(static_cast<std::streamoff>(offset));
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<uint8_t> little_endian(uint64_t value, size_t width) {
    std::vector<uint8_t> bytes(width);
    for (size_t i = 0; i < width; ++i) {
        bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    }
    return bytes;
}

uint64_t read_little_endian(const std::filesystem::path& path, size_t offset, size_t width) {
    const std::vector<uint8_t> bytes = read_file_bytes(path, offset, width);
    uint64_t value = 0;
    for (size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

/// Bytes the file system has allocated for @p path, which for a sparse file is less than its size.
uint64_t allocated_bytes(const std::filesystem::path& path) {
#ifdef _WIN32
    DWORD high = 0;
    const DWORD low = GetCompressedFileSizeW(path.wstring().c_str(), &high);
    EXPECT_FALSE(low == INVALID_FILE_SIZE && GetLastError() != NO_ERROR) << path.string();
    return (static_cast<uint64_t>(high) << 32) | low;
#else
    struct stat status {};
    EXPECT_EQ(::stat(path.c_str(), &status), 0) << path.string();
    return static_cast<uint64_t>(status.st_blocks) * 512;
#endif
}

PrefixKey key_of(size_t hash, uint64_t digest_lo) {
    return PrefixKey{hash, ContentHash{digest_lo, ~digest_lo}};
}

void write_keyed(IKVCacheStore& store, const PrefixKey& key, const std::vector<uint8_t>& data) {
    const std::vector<bool> written = store.write_blocks_by_key({key}, {&data});
    ASSERT_EQ(written.size(), 1u);
    ASSERT_TRUE(written[0]);
}

bool has_keyed(const IKVCacheStore& store, const PrefixKey& key) {
    return store.has_blocks_by_key({key}).at(0);
}

std::optional<std::vector<uint8_t>> read_keyed(const IKVCacheStore& store, const PrefixKey& key) {
    std::vector<std::vector<uint8_t>> data;
    if (!store.read_blocks_by_key({key}, data).at(0)) {
        return std::nullopt;
    }
    return data.at(0);
}

}  // namespace

TEST(TestPersistentFileKVCacheStore, RestartFindsBlocksWrittenByAnEarlierStore) {
    ScopedDirectory directory("openvino_genai_gap2_restart");
    const auto config = make_persistent_config(directory.path);
    const std::vector<PrefixKey> keys = {key_of(11, 0x1111), key_of(22, 0x2222), key_of(33, 0x3333)};
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        for (size_t i = 0; i < keys.size(); ++i) {
            write_keyed(*store, keys[i], make_block(static_cast<uint8_t>(10 * i)));
        }
        store->flush();
    }
    auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS - keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(has_keyed(*reopened, keys[i])) << "block " << i;
        EXPECT_EQ(read_keyed(*reopened, keys[i]), make_block(static_cast<uint8_t>(10 * i))) << "block " << i;
    }
    EXPECT_TRUE(reopened->outlives_process());
}

TEST(TestPersistentFileKVCacheStore, RestartWithReuseSsdCacheFalsePurges) {
    ScopedDirectory directory("openvino_genai_gap2_purge");
    auto config = make_persistent_config(directory.path);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        write_keyed(*store, key_of(1, 0xA), make_block(1));
    }
    {
        // Control: with reuse_ssd_cache left on, the block is there after the restart.
        auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        ASSERT_TRUE(has_keyed(*reopened, key_of(1, 0xA))) << "the control restart did not find the block";
    }
    config.reuse_ssd_cache = false;
    {
        auto purged = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        EXPECT_FALSE(has_keyed(*purged, key_of(1, 0xA)));
        EXPECT_EQ(purged->get_num_free_blocks(), NUM_BLOCKS);
    }
    config.reuse_ssd_cache = true;
    auto after_purge = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    EXPECT_FALSE(has_keyed(*after_purge, key_of(1, 0xA))) << "the purge must reach the disk, not only the index";
}

// Passes with and without the persistent store: it pins the default (persist_file_store = false), which must stay
// run-scoped.
TEST(TestPersistentFileKVCacheStore, DefaultConfigStillDeletesItsFileOnDestruction) {
    ScopedDirectory directory("openvino_genai_gap2_default");
    auto config = make_persistent_config(directory.path);
    config.persist_file_store = false;
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        EXPECT_FALSE(store->outlives_process());
        write_keyed(*store, key_of(1, 0xA), make_block(1));
        EXPECT_EQ(count_regular_files(directory.path), 1u);
    }
    EXPECT_EQ(count_regular_files(directory.path), 0u);
    auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    EXPECT_FALSE(has_keyed(*reopened, key_of(1, 0xA)));
}

// The miss under another group passes on a run-scoped store as well; the same-group hit before it is what fails there.
TEST(TestPersistentFileKVCacheStore, DifferentGroupDoesNotSeeBlocks) {
    ScopedDirectory directory("openvino_genai_gap2_group");
    const auto config = make_persistent_config(directory.path);
    const std::string model_a = "kv:00000000000000000000000000000001";
    const std::string model_b = "kv:00000000000000000000000000000002";
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", model_a);
        write_keyed(*store, key_of(1, 0xA), make_block(1));
    }
    {
        auto same_model = open_kv_cache_store(make_layout(), config, "CPU", model_a);
        EXPECT_TRUE(has_keyed(*same_model, key_of(1, 0xA)));
    }
    auto other_model = open_kv_cache_store(make_layout(), config, "CPU", model_b);
    EXPECT_FALSE(has_keyed(*other_model, key_of(1, 0xA)));
    EXPECT_EQ(other_model->get_num_free_blocks(), NUM_BLOCKS);
}

// The misses after a header change pass on a run-scoped store as well; the control restart is what fails there.
TEST(TestPersistentFileKVCacheStore, HeaderMismatchStartsEmpty) {
    ScopedDirectory directory("openvino_genai_gap2_header");
    {
        auto store = open_kv_cache_store(make_layout(), make_persistent_config(directory.path), "CPU", "kv");
        write_keyed(*store, key_of(1, 0xA), make_block(1));
    }
    {
        auto control = open_kv_cache_store(make_layout(), make_persistent_config(directory.path), "CPU", "kv");
        ASSERT_TRUE(has_keyed(*control, key_of(1, 0xA))) << "the control restart did not find the block";
    }
    {
        // Another capacity changes the slot count recorded in the file header.
        auto resized = open_kv_cache_store(make_layout(), make_persistent_config(directory.path, NUM_BLOCKS + 2), "CPU", "kv");
        EXPECT_FALSE(has_keyed(*resized, key_of(1, 0xA)));
        EXPECT_EQ(resized->get_num_free_blocks(), NUM_BLOCKS + 2);
        write_keyed(*resized, key_of(2, 0xB), make_block(2));
    }
    // Another block size under the same group name (which the coordinator never produces) changes the payload size.
    auto rewidened = open_kv_cache_store(make_layout(2 * BLOCK_SIZE),
                                         make_persistent_config(directory.path, NUM_BLOCKS, 2 * BLOCK_SIZE),
                                         "CPU",
                                         "kv");
    EXPECT_FALSE(has_keyed(*rewidened, key_of(2, 0xB)));
    write_keyed(*rewidened, key_of(3, 0xC), make_block(3, 2 * BLOCK_SIZE));
    EXPECT_EQ(read_keyed(*rewidened, key_of(3, 0xC)), make_block(3, 2 * BLOCK_SIZE));
}

TEST(TestPersistentFileKVCacheStore, FormatOrHashVersionMismatchStartsEmpty) {
    for (const size_t field_at : {FILE_FORMAT_VERSION_AT, FILE_HASH_VERSION_AT}) {
        SCOPED_TRACE("file header field at byte " + std::to_string(field_at));
        ScopedDirectory directory("openvino_genai_gap2_version");
        const auto config = make_persistent_config(directory.path);
        {
            auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
            write_keyed(*store, key_of(1, 0xA), make_block(1));
        }
        const auto path = slot_file(directory.path, "kv");
        ASSERT_TRUE(std::filesystem::exists(path)) << "the persistent store did not keep its file";
        const uint64_t written = read_little_endian(path, field_at, 4);
        EXPECT_EQ(written, field_at == FILE_FORMAT_VERSION_AT ? 1u : CONTENT_HASH_ALGORITHM_VERSION);
        write_file_bytes(path, field_at, little_endian(written + 1, 4));

        auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        EXPECT_FALSE(has_keyed(*reopened, key_of(1, 0xA)));
        EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS);
        write_keyed(*reopened, key_of(2, 0xB), make_block(2));
        reopened.reset();
        EXPECT_EQ(read_little_endian(path, field_at, 4), written) << "the header is rewritten for this build";
        auto again = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        EXPECT_EQ(read_keyed(*again, key_of(2, 0xB)), make_block(2));
    }
}

TEST(TestPersistentFileKVCacheStore, ShortFileStartsEmpty) {
    // Cut inside the slots, and inside the file header.
    for (const size_t cut_at : {slot_at(1) + 100, size_t{100}}) {
        SCOPED_TRACE("file cut to " + std::to_string(cut_at) + " bytes");
        ScopedDirectory directory("openvino_genai_gap2_short");
        const auto config = make_persistent_config(directory.path);
        {
            auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
            write_keyed(*store, key_of(1, 0xA), make_block(1));  // slot 0, before the cut
        }
        const auto path = slot_file(directory.path, "kv");
        ASSERT_TRUE(std::filesystem::exists(path)) << "the persistent store did not keep its file";
        std::filesystem::resize_file(path, cut_at);

        auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        EXPECT_FALSE(has_keyed(*reopened, key_of(1, 0xA)));
        EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS);
        write_keyed(*reopened, key_of(2, 0xB), make_block(2));
        reopened.reset();
        EXPECT_EQ(std::filesystem::file_size(path), slot_at(NUM_BLOCKS));
        auto again = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        EXPECT_EQ(read_keyed(*again, key_of(2, 0xB)), make_block(2));
    }
}

TEST(TestPersistentFileKVCacheStore, TornSlotReadsAsMissAndIsFreed) {
    ScopedDirectory directory("openvino_genai_gap2_torn");
    const auto config = make_persistent_config(directory.path);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        write_keyed(*store, key_of(1, 0xA), make_block(1));  // first slot handed out: slot 0
        write_keyed(*store, key_of(2, 0xB), make_block(2));  // slot 1
    }
    const auto path = slot_file(directory.path, "kv");
    ASSERT_TRUE(std::filesystem::exists(path)) << "the persistent store did not keep its file";
    write_file_bytes(path, slot_at(0) + SLOT_HEADER_BYTES + 5, {0xFF});
    {
        auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS - 2) << "both slot headers are intact, so both are indexed";
        EXPECT_EQ(read_keyed(*reopened, key_of(1, 0xA)), std::nullopt) << "a payload that fails its checksum must read as a miss";
        EXPECT_EQ(read_keyed(*reopened, key_of(2, 0xB)), make_block(2));
        EXPECT_FALSE(has_keyed(*reopened, key_of(1, 0xA)));
        EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS - 1);
    }
    EXPECT_EQ(read_little_endian(path, slot_at(0) + SLOT_MAGIC_AT, 4), 0u) << "the torn slot is freed on disk too";
}

TEST(TestPersistentFileKVCacheStore, ZeroedSlotMagicReadsAsFree) {
    ScopedDirectory directory("openvino_genai_gap2_magic");
    const auto config = make_persistent_config(directory.path);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        write_keyed(*store, key_of(1, 0xA), make_block(1));  // slot 0
        write_keyed(*store, key_of(2, 0xB), make_block(2));  // slot 1
    }
    const auto path = slot_file(directory.path, "kv");
    ASSERT_TRUE(std::filesystem::exists(path)) << "the persistent store did not keep its file";
    // What a crash between "magic zeroed" and "header written" leaves behind.
    write_file_bytes(path, slot_at(0) + SLOT_MAGIC_AT, {0, 0, 0, 0});

    auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    EXPECT_EQ(read_keyed(*reopened, key_of(2, 0xB)), make_block(2));
    EXPECT_FALSE(has_keyed(*reopened, key_of(1, 0xA)));
    EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS - 1);
}

TEST(TestPersistentFileKVCacheStore, TwoSlotsWithOneKeyKeepTheNewerWrite) {
    // The newer copy in a later slot, then in an earlier one, since the open scan meets them in slot order.
    for (const bool newer_first : {false, true}) {
        SCOPED_TRACE(newer_first ? "newer copy in the earlier slot" : "newer copy in the later slot");
        ScopedDirectory directory("openvino_genai_gap2_twoslots");
        const auto config = make_persistent_config(directory.path);
        const auto path = slot_file(directory.path, "kv");
        {
            auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
            write_keyed(*store, key_of(1, 0xA), make_block(1));  // slot 0, sequence 1
            write_keyed(*store, key_of(2, 0xB), make_block(2));  // slot 1, sequence 2
        }
        ASSERT_TRUE(std::filesystem::exists(path)) << "the persistent store did not keep its file";
        const std::vector<uint8_t> older = read_file_bytes(path, slot_at(0), SLOT_HEADER_BYTES + BLOCK_SIZE);
        {
            auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
            write_keyed(*store, key_of(1, 0xA), make_block(9));  // same key, same slot 0, sequence 3
        }
        const std::vector<uint8_t> newer = read_file_bytes(path, slot_at(0), SLOT_HEADER_BYTES + BLOCK_SIZE);
        ASSERT_NE(older, newer);
        // Slot 0 and the free slot 2 now both claim the key, with different write sequences.
        write_file_bytes(path, slot_at(0), newer_first ? newer : older);
        write_file_bytes(path, slot_at(2), newer_first ? older : newer);
        const size_t stale_slot = newer_first ? 2 : 0;
        {
            auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
            EXPECT_EQ(read_keyed(*reopened, key_of(1, 0xA)), make_block(9));
            EXPECT_EQ(read_keyed(*reopened, key_of(2, 0xB)), make_block(2));
            EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS - 2) << "the stale copy's slot is free again";
        }
        EXPECT_EQ(read_little_endian(path, slot_at(stale_slot) + SLOT_MAGIC_AT, 4), 0u) << "stale copy freed on disk";
        auto again = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        EXPECT_EQ(read_keyed(*again, key_of(1, 0xA)), make_block(9));
    }
}

TEST(TestPersistentFileKVCacheStore, ZeroOrDuplicateWriteSequenceIsNotIndexed) {
    // Sequence 0 in slot 1, or slot 1 repeating slot 0's sequence 1: no write of this store produces either.
    for (const uint64_t forged_sequence : {uint64_t{0}, uint64_t{1}}) {
        SCOPED_TRACE("slot 1 sequence forged to " + std::to_string(forged_sequence));
        ScopedDirectory directory("openvino_genai_gap2_sequence");
        const auto config = make_persistent_config(directory.path);
        const auto path = slot_file(directory.path, "kv");
        {
            auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
            write_keyed(*store, key_of(1, 0xA), make_block(1));  // slot 0, sequence 1
            write_keyed(*store, key_of(2, 0xB), make_block(2));  // slot 1, sequence 2
        }
        ASSERT_TRUE(std::filesystem::exists(path)) << "the persistent store did not keep its file";
        ASSERT_EQ(read_little_endian(path, slot_at(1) + SLOT_SEQUENCE_AT, 8), 2u);
        write_file_bytes(path, slot_at(1) + SLOT_SEQUENCE_AT, little_endian(forged_sequence, 8));
        {
            auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
            EXPECT_EQ(read_keyed(*reopened, key_of(1, 0xA)), make_block(1));
            EXPECT_FALSE(has_keyed(*reopened, key_of(2, 0xB)));
            EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS - 1);
            // A new write after the open must get a sequence above every one on disk, so it is not the next to go.
            write_keyed(*reopened, key_of(3, 0xC), make_block(3));
        }
        // The forged slot was freed and is the first free slot, so the new write took it.
        auto again = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        EXPECT_EQ(read_keyed(*again, key_of(3, 0xC)), make_block(3));
        EXPECT_EQ(read_keyed(*again, key_of(1, 0xA)), make_block(1));
        again.reset();
        EXPECT_GT(read_little_endian(path, slot_at(1) + SLOT_SEQUENCE_AT, 8),
                  read_little_endian(path, slot_at(0) + SLOT_SEQUENCE_AT, 8));
    }
}

TEST(TestPersistentFileKVCacheStore, DroppedBlockDoesNotReturnAfterRestart) {
    ScopedDirectory directory("openvino_genai_gap2_drop");
    const auto config = make_persistent_config(directory.path);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        write_keyed(*store, key_of(1, 0xA), make_block(1));
        write_keyed(*store, key_of(2, 0xB), make_block(2));
        store->drop_block_by_key(key_of(1, 0xA));
    }
    auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    EXPECT_TRUE(has_keyed(*reopened, key_of(2, 0xB)));
    EXPECT_FALSE(has_keyed(*reopened, key_of(1, 0xA)));
    EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS - 1);
}

TEST(TestPersistentFileKVCacheStore, OverwriteKeepsOneSlotAcrossRestart) {
    ScopedDirectory directory("openvino_genai_gap2_overwrite");
    const auto config = make_persistent_config(directory.path);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        write_keyed(*store, key_of(1, 0xA), make_block(1));
        write_keyed(*store, key_of(1, 0xA), make_block(9));
        EXPECT_EQ(store->get_num_free_blocks(), NUM_BLOCKS - 1);
    }
    auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS - 1);
    EXPECT_EQ(read_keyed(*reopened, key_of(1, 0xA)), make_block(9));
}

TEST(TestPersistentFileKVCacheStore, FullStoreEvictsTheOldestWriteAcrossRestart) {
    ScopedDirectory directory("openvino_genai_gap2_evict");
    const auto config = make_persistent_config(directory.path);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        for (uint64_t i = 1; i <= NUM_BLOCKS; ++i) {
            write_keyed(*store, key_of(i, i), make_block(static_cast<uint8_t>(i)));
        }
        EXPECT_EQ(store->get_num_free_blocks(), 0u);
    }
    {
        // A store an earlier process filled must still take new blocks: the offload tier cannot drop blocks it did not
        // write, so the store makes room itself, oldest write first.
        auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        for (uint64_t i = 1; i <= NUM_BLOCKS; ++i) {
            EXPECT_TRUE(has_keyed(*reopened, key_of(i, i))) << "block " << i << " before the restart's first write";
        }
        write_keyed(*reopened, key_of(9, 9), make_block(9));
        EXPECT_FALSE(has_keyed(*reopened, key_of(1, 1)));
        for (uint64_t i = 2; i <= NUM_BLOCKS; ++i) {
            EXPECT_TRUE(has_keyed(*reopened, key_of(i, i))) << "block " << i;
        }
        EXPECT_TRUE(has_keyed(*reopened, key_of(9, 9)));
        // Rewriting block 2 makes it the newest, so block 3 is the next to go.
        write_keyed(*reopened, key_of(2, 2), make_block(22));
        write_keyed(*reopened, key_of(10, 10), make_block(10));
        EXPECT_FALSE(has_keyed(*reopened, key_of(3, 3)));
        EXPECT_TRUE(has_keyed(*reopened, key_of(2, 2)));
        EXPECT_TRUE(reopened->evicts_itself());
    }
    auto again = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    std::vector<std::vector<uint8_t>> data;
    const std::vector<bool> found =
        again->read_blocks_by_key({key_of(1, 1), key_of(2, 2), key_of(3, 3), key_of(4, 4), key_of(9, 9), key_of(10, 10)}, data);
    EXPECT_EQ(found, (std::vector<bool>{false, true, false, true, true, true}));
    EXPECT_EQ(data[1], make_block(22));
}

// Both opens are in this process: a proxy for two processes on both lock paths (Windows opens LOCK with _SH_DENYRW,
// which refuses a second open from any process; POSIX flock(LOCK_EX) conflicts between two open() calls in one process
// as between two processes). SecondProcessCannotOpenALockedStore below uses a real second process.
TEST(TestPersistentFileKVCacheStore, SecondOpenWhileLockedFallsBackToRunScopedStore) {
    ScopedDirectory directory("openvino_genai_gap2_lock");
    const auto config = make_persistent_config(directory.path);
    auto first = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    write_keyed(*first, key_of(1, 0xA), make_block(1));

    auto second = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    ASSERT_NE(second, nullptr);
    EXPECT_FALSE(has_keyed(*second, key_of(1, 0xA)));
    write_keyed(*second, key_of(2, 0xB), make_block(2));
    EXPECT_FALSE(second->outlives_process()) << "a locked store directory must not be opened twice";
    second.reset();

    // The run-scoped fallback left nothing behind, and the first store is unaffected.
    EXPECT_TRUE(has_keyed(*first, key_of(1, 0xA)));
    first.reset();
    auto third = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    EXPECT_TRUE(has_keyed(*third, key_of(1, 0xA))) << "the lock must be released with the store";
    EXPECT_FALSE(has_keyed(*third, key_of(2, 0xB)));
}

namespace {

constexpr const char* LOCK_CHILD_DIRECTORY = "OV_GENAI_TEST_PERSISTENT_STORE_DIR";

void set_environment(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    if (value.empty()) {
        unsetenv(name);
    } else {
        setenv(name, value.c_str(), 1);
    }
#endif
}

/// Run in a child process: 0 when the store directory was locked and its block hidden, else what went wrong.
[[noreturn]] void open_in_child_and_exit(const std::filesystem::path& directory, bool expect_locked) {
    auto store = open_kv_cache_store(make_layout(), make_persistent_config(directory), "CPU", "kv");
    const bool persistent = store->outlives_process();
    const bool sees_block = has_keyed(*store, key_of(1, 0xA));
    store.reset();
    if (expect_locked) {
        std::exit(persistent ? 1 : sees_block ? 2 : 0);
    }
    std::exit(!sees_block ? 3 : !persistent ? 4 : 0);
}

}  // namespace

// The parent holds the store while a child process opens the same directory, then releases it and a second child
// opens it again. "threadsafe" death tests start the child as a new process that runs this test from the top, so the
// child finds the directory through an environment variable and skips the parent's part.
TEST(TestPersistentFileKVCacheStore, SecondProcessCannotOpenALockedStore) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const char* inherited = std::getenv(LOCK_CHILD_DIRECTORY);
    const bool is_child = inherited != nullptr && *inherited != '\0';
    std::optional<ScopedDirectory> owned;
    std::filesystem::path directory;
    std::unique_ptr<IKVCacheStore> first;
    if (is_child) {
        directory = inherited;
    } else {
        owned.emplace("openvino_genai_gap2_lock_process");
        directory = owned->path;
        first = open_kv_cache_store(make_layout(), make_persistent_config(directory), "CPU", "kv");
        write_keyed(*first, key_of(1, 0xA), make_block(1));
        first->flush();
        set_environment(LOCK_CHILD_DIRECTORY, directory.string());
    }

    EXPECT_EXIT(open_in_child_and_exit(directory, /*expect_locked=*/true), ::testing::ExitedWithCode(0), "")
        << "exit 1: the child opened the locked directory as a persistent store; 2: it saw the parent's block";

    if (!is_child) {
        EXPECT_TRUE(has_keyed(*first, key_of(1, 0xA)));
        first.reset();
    }
    EXPECT_EXIT(open_in_child_and_exit(directory, /*expect_locked=*/false), ::testing::ExitedWithCode(0), "")
        << "exit 3: the child did not find the block the parent stored; 4: it could not take the released lock";

    if (!is_child) {
        set_environment(LOCK_CHILD_DIRECTORY, "");
    }
}

TEST(TestPersistentFileKVCacheStore, BlocksWithoutADigestAreNotKept) {
    // The size_t hash folded to 128 bits is not an identity a store that outlives the process may be keyed by: two
    // prefixes whose 64-bit hashes collide would share a key and pass each other's checksum.
    ScopedDirectory directory("openvino_genai_gap2_nodigest");
    const auto config = make_persistent_config(directory.path);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        const std::vector<uint8_t> unkeyed = make_block(7);
        EXPECT_EQ(store->write_blocks_by_key({PrefixKey{7, std::nullopt}}, {&unkeyed}), std::vector<bool>{false});
        EXPECT_FALSE(store->write_block(8, make_block(8)));
        EXPECT_FALSE(store->has_block(7));
        EXPECT_FALSE(store->has_block(8));
        EXPECT_FALSE(has_keyed(*store, PrefixKey{7, std::nullopt}));
        EXPECT_EQ(store->get_num_free_blocks(), NUM_BLOCKS);
        // A key with a digest is kept, also when its digest happens to equal the folded form of another hash.
        write_keyed(*store, PrefixKey{9, ContentHash{7, 7}}, make_block(9));
    }
    auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    EXPECT_EQ(read_keyed(*reopened, PrefixKey{9, ContentHash{7, 7}}), make_block(9));
    EXPECT_FALSE(reopened->has_block(7));
    std::vector<uint8_t> data;
    EXPECT_FALSE(reopened->read_block(7, data)) << "the size_t API must not reach a digest-keyed block";
    EXPECT_EQ(reopened->get_num_free_blocks(), NUM_BLOCKS - 1);
}

// Passes with and without the persistent store: the field and its check are part of the plumbing commit.
TEST(TestPersistentFileKVCacheStore, ConfigToStringEqualityAndValidationCarryTheField) {
    CacheOffloadConfig a;
    CacheOffloadConfig b;
    b.persist_file_store = true;
    EXPECT_FALSE(a == b);
    EXPECT_NE(b.to_string().find("persist_file_store: true"), std::string::npos);

    SchedulerConfig scheduler;
    scheduler.enable_prefix_caching = true;
    scheduler.use_cache_offload = true;
    scheduler.cache_offload_config = b;
    // A capacity, so that the only setting validate() can object to is the missing kv_cache_dir.
    scheduler.cache_offload_config.capacity_bytes = BLOCK_SIZE * NUM_BLOCKS;
    try {
        scheduler.validate();
        ADD_FAILURE() << "persist_file_store without kv_cache_dir must be rejected";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("persist_file_store requires kv_cache_dir"), std::string::npos)
            << error.what();
    }
    scheduler.cache_offload_config.kv_cache_dir = std::filesystem::temp_directory_path().string();
    EXPECT_NO_THROW(scheduler.validate());
}

TEST(TestPersistentFileKVCacheStore, OpenDoesNotReserveTheCapacityOnDisk) {
    // 16384 slots of exactly one 4096-byte stride: a 64 MiB store. Opening it, and purging it, must not write or hold
    // the whole capacity; the space is taken as blocks are written.
    constexpr size_t payload = SLOT_STRIDE - SLOT_HEADER_BYTES;
    constexpr size_t num_slots = 16384;
    ScopedDirectory directory("openvino_genai_gap2_sparse");
    auto config = make_persistent_config(directory.path, num_slots, payload);
    const auto path = slot_file(directory.path, "kv");
    // Measured with the store closed, since an open store does not share its file.
    open_kv_cache_store(make_layout(payload), config, "CPU", "kv").reset();
    ASSERT_TRUE(std::filesystem::exists(path)) << "the persistent store did not keep its file";
    EXPECT_EQ(std::filesystem::file_size(path), slot_at(num_slots));
    EXPECT_LT(allocated_bytes(path), uint64_t{1} << 20) << "opening an empty store allocated its capacity";
    {
        auto store = open_kv_cache_store(make_layout(payload), config, "CPU", "kv");
        write_keyed(*store, key_of(1, 0xA), make_block(1, payload));
    }
    EXPECT_LT(allocated_bytes(path), uint64_t{1} << 20);
    config.reuse_ssd_cache = false;
    {
        auto purged = open_kv_cache_store(make_layout(payload), config, "CPU", "kv");
        EXPECT_FALSE(has_keyed(*purged, key_of(1, 0xA)));
    }
    EXPECT_LT(allocated_bytes(path), uint64_t{1} << 20) << "purging at open allocated the capacity";
}

TEST(TestPersistentFileKVCacheStore, LinearAttentionGroupIsKeptApartFromKvAndRestored) {
    ScopedDirectory directory("openvino_genai_gap2_la_group");
    const auto kv_config = make_persistent_config(directory.path);
    const auto la_config = make_persistent_config(directory.path, NUM_BLOCKS, 2 * BLOCK_SIZE);
    const std::string kv_group = "kv:000000000000000000000000000000a1";
    const std::string la_group = "la:000000000000000000000000000000a1";
    {
        // Both groups open at once, as the coordinator opens them for a hybrid model, under one kv_cache_dir.
        auto kv = open_kv_cache_store(make_layout(), kv_config, "CPU", kv_group);
        auto la = open_kv_cache_store(make_layout(2 * BLOCK_SIZE), la_config, "CPU", la_group);
        write_keyed(*kv, key_of(1, 0xA), make_block(1));
        write_keyed(*la, key_of(1, 0xA), make_block(5, 2 * BLOCK_SIZE));
        write_keyed(*la, key_of(2, 0xB), make_block(6, 2 * BLOCK_SIZE));
    }
    auto la = open_kv_cache_store(make_layout(2 * BLOCK_SIZE), la_config, "CPU", la_group);
    auto kv = open_kv_cache_store(make_layout(), kv_config, "CPU", kv_group);
    EXPECT_EQ(read_keyed(*la, key_of(1, 0xA)), make_block(5, 2 * BLOCK_SIZE));
    EXPECT_EQ(read_keyed(*la, key_of(2, 0xB)), make_block(6, 2 * BLOCK_SIZE));
    EXPECT_EQ(read_keyed(*kv, key_of(1, 0xA)), make_block(1));
    EXPECT_FALSE(has_keyed(*kv, key_of(2, 0xB)));
    EXPECT_TRUE(std::filesystem::exists(directory.path / "ov_genai_kv_store" / "la_000000000000000000000000000000a1"));
}

namespace {

constexpr size_t NUM_DECODER_LAYERS = 2;
constexpr size_t NUM_LA_LAYERS = 2;

/// The offload tier over a persistent store, with a real CPU cache of the given manager type to copy blocks from.
template <typename Manager, typename Tier>
struct TierFixture {
    ov::Core core;
    ov::InferRequest request;
    std::shared_ptr<Manager> cache_manager;
    std::string group;

    TierFixture(const std::shared_ptr<ov::Model>& model, std::string store_group) : group(std::move(store_group)) {
        request = core.compile_model(model).create_infer_request();
        cache_manager = std::make_shared<Manager>(request);
        cache_manager->allocate_cache_if_needed(4);
    }

    std::vector<uint8_t> fill_block(size_t block_id, uint8_t seed) {
        std::vector<uint8_t> data(cache_manager->get_block_layout().get_slot_size());
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<uint8_t>(seed + i);
        }
        cache_manager->write_block(block_id, data);
        return data;
    }

    CacheOffloadConfig make_config(const std::filesystem::path& directory) const {
        CacheOffloadConfig config;
        config.capacity_bytes = cache_manager->get_block_layout().get_slot_size() * 4;
        config.kv_cache_dir = directory.string();
        config.persist_file_store = true;
        return config;
    }

    std::unique_ptr<IKVCacheStore> open_store(const std::filesystem::path& directory) const {
        return open_kv_cache_store(cache_manager->get_block_layout(), make_config(directory), "CPU", group);
    }

    std::unique_ptr<Tier> make_tier(const std::filesystem::path& directory) {
        return std::make_unique<Tier>(*cache_manager, open_store(directory), /*max_queued_stores=*/8);
    }
};

using KvTierFixture = TierFixture<KVCacheManager, KVCacheOffloadCache>;
using LaTierFixture = TierFixture<LinearAttentionCacheManager, LinearAttentionOffloadCache>;

/// Stores block 1 of @p fixture under @p hash through one tier, then restores it into block 2 through a second tier
/// over the reopened store, the way a restarted pipeline would. Returns whether the second tier found it.
template <typename Fixture>
void store_then_restore_in_a_later_tier(Fixture& fixture,
                                        const std::filesystem::path& directory,
                                        size_t hash,
                                        const ContentHash& digest) {
    std::vector<uint8_t> expected;
    {
        auto tier = fixture.make_tier(directory);
        // What Sequence records while a digest-keyed tier holds the lease.
        PrefixDigestRegistry::instance().record(hash, digest);
        expected = fixture.fill_block(1, /*seed=*/7);
        tier->probe({hash});
        tier->on_blocks_overwritten(hash, BlocksPerLayer{std::make_shared<CacheBlock>(1)});
        tier->flush();
        EXPECT_EQ(tier->get_statistics().num_stored, 1u);
    }
    {
        // The block is in the store under its 128-bit digest, not under the hash folded to 128 bits.
        auto store = fixture.open_store(directory);
        EXPECT_TRUE(has_keyed(*store, PrefixKey{hash, digest})) << "the store did not keep the block under its digest";
        EXPECT_FALSE(has_keyed(*store, PrefixKey{hash, std::nullopt}));
    }
    fixture.fill_block(1, /*seed=*/0);
    fixture.fill_block(2, /*seed=*/0);

    auto restarted = fixture.make_tier(directory);
    // A new process computes the same digest for the same prefix.
    PrefixDigestRegistry::instance().record(hash, digest);
    EXPECT_FALSE(restarted->contains(hash)) << "a new tier only knows a store block once a probe found it";
    const std::vector<bool> present = restarted->probe({hash});
    ASSERT_EQ(present.size(), 1u);
    ASSERT_TRUE(present[0]) << "the restarted tier did not find the block the earlier instance stored";
    EXPECT_TRUE(restarted->contains(hash));
    restarted->prefetch({hash});
    ASSERT_TRUE(restarted->load_into(hash, /*block_index=*/2));
    restarted->release_prefetched();
    EXPECT_EQ(fixture.cache_manager->read_block(2), expected);
}

}  // namespace

TEST(TestPersistentFileKVCacheStore, OffloadTierOfALaterInstanceRestoresByTheRecordedDigest) {
    ScopedDirectory directory("openvino_genai_gap2_tier");
    ov::Core core;
    KvTierFixture fixture(get_dummy_model(core, NUM_DECODER_LAYERS), "kv");
    // Hashes that no other test uses, so that the process-wide digest registry holds nothing else for them.
    store_then_restore_in_a_later_tier(fixture, directory.path, /*hash=*/920011, ContentHash{0x0123456789ABCDEFull, 0x1});
}

TEST(TestPersistentFileKVCacheStore, LinearAttentionTierOfALaterInstanceRestoresByTheRecordedDigest) {
    ScopedDirectory directory("openvino_genai_gap2_la_tier");
    ov::Core core;
    LaTierFixture fixture(get_dummy_hybrid_model(core, /*kv_num_layers=*/0, NUM_LA_LAYERS), "la");
    store_then_restore_in_a_later_tier(fixture, directory.path, /*hash=*/920022, ContentHash{0x0123456789ABCDEFull, 0x2});
}

TEST(TestPersistentFileKVCacheStore, OffloadTierDoesNotPersistABlockWithoutADigest) {
    ScopedDirectory directory("openvino_genai_gap2_tier_nodigest");
    ov::Core core;
    KvTierFixture fixture(get_dummy_model(core, NUM_DECODER_LAYERS), "kv");
    constexpr size_t unrecorded = 920033;  // never recorded, so the tier has no digest for it
    {
        auto tier = fixture.make_tier(directory.path);
        fixture.fill_block(1, /*seed=*/7);
        tier->probe({unrecorded});
        tier->on_blocks_overwritten(unrecorded, BlocksPerLayer{std::make_shared<CacheBlock>(1)});
        tier->flush();
        EXPECT_EQ(tier->get_statistics().num_stored, 0u);
        EXPECT_EQ(tier->get_statistics().num_failed, 1u);
    }
    auto restarted = fixture.make_tier(directory.path);
    EXPECT_FALSE(restarted->probe({unrecorded}).at(0));
}

namespace {

SequenceGroup::Ptr make_sequence_group(const std::vector<int64_t>& tokens, uint64_t request_id) {
    return std::make_shared<SequenceGroup>(request_id,
                                           ov::Tensor(ov::element::i64, {tokens.size()}, const_cast<int64_t*>(tokens.data())),
                                           utils::get_greedy_config());
}

/// The draft identity MTP (mtp_strategy.cpp:185-188) and Eagle3 with embeddings input (eagle3_strategy.cpp:210-214)
/// give a draft sequence: the main sequence's hash of the prefix one token longer.
void hash_draft_through_main(const Sequence::Ptr& draft, const Sequence::Ptr& main, size_t publication_limit) {
    draft->set_prefix_cache_policy(publication_limit, [main](size_t length, size_t block_size) {
        return main->get_hash(length + 1, block_size);
    });
}

}  // namespace

// A draft sequence hashed through the main sequence, as speculative decoding with MTP or Eagle3 (embeddings input)
// does, gets no digest of its own; the main sequence's get_hash records one for the hash it returns. This checks that
// the draft's offload tier finds that digest, so the persistent store keeps the draft block under it, and that a
// second tier opened later in the same process restores it from the reopened store. The process-wide digest registry
// is not emptied in between, so this does not show a restore across a process restart.
TEST(TestPersistentFileKVCacheStore, DraftBlockHashedThroughTheMainSequenceIsKeptUnderTheMainDigest) {
    ScopedDirectory directory("openvino_genai_gap2_draft_identity");
    ov::Core core;
    KvTierFixture fixture(get_dummy_model(core, NUM_DECODER_LAYERS), "kv");
    constexpr size_t block_size = 4;
    // Prompt tokens no other test uses, so that the process-wide digest registry holds nothing else for their hashes.
    const std::vector<int64_t> main_tokens = {9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108, 9109};
    const std::vector<int64_t> draft_tokens(main_tokens.begin() + 1, main_tokens.end());  // shifted by one

    size_t draft_hash = 0;
    ContentHash main_digest{};
    std::vector<uint8_t> expected;
    {
        auto main_group = make_sequence_group(main_tokens, 9101);
        auto draft_group = make_sequence_group(draft_tokens, 9102);
        const Sequence::Ptr main = main_group->get_sequences().front();
        const Sequence::Ptr draft = draft_group->get_sequences().front();
        hash_draft_through_main(draft, main, draft_group->get_prompt_len());

        auto tier = fixture.make_tier(directory.path);  // holds the digest lease, as a persistent store's tier does
        ASSERT_TRUE(PrefixDigestRegistry::enabled());
        draft_hash = draft->get_hash(block_size, block_size);
        EXPECT_EQ(draft_hash, main->get_hash(block_size + 1, block_size));
        main_digest = main->get_digest(block_size + 1, block_size);
        EXPECT_EQ(PrefixDigestRegistry::instance().find(draft_hash), std::optional<ContentHash>(main_digest))
            << "the registry has no digest for the hash the draft sequence published";

        expected = fixture.fill_block(1, /*seed=*/11);
        tier->probe({draft_hash});
        tier->on_blocks_overwritten(draft_hash, BlocksPerLayer{std::make_shared<CacheBlock>(1)});
        tier->flush();
        EXPECT_EQ(tier->get_statistics().num_stored, 1u);
        EXPECT_EQ(tier->get_statistics().num_failed, 0u);
    }
    {
        auto store = fixture.open_store(directory.path);
        EXPECT_TRUE(has_keyed(*store, PrefixKey{draft_hash, main_digest})) << "not kept under the main digest";
        EXPECT_FALSE(has_keyed(*store, PrefixKey{draft_hash, std::nullopt}));
    }
    fixture.fill_block(1, /*seed=*/0);
    fixture.fill_block(2, /*seed=*/0);

    // Later in the same process: new sequences for the same prompt and a second tier over the same directory. The
    // registry still holds the digest recorded above; the main digest is a function of the tokens alone.
    auto main_group = make_sequence_group(main_tokens, 9103);
    auto draft_group = make_sequence_group(draft_tokens, 9104);
    const Sequence::Ptr main = main_group->get_sequences().front();
    const Sequence::Ptr draft = draft_group->get_sequences().front();
    hash_draft_through_main(draft, main, draft_group->get_prompt_len());
    auto restarted = fixture.make_tier(directory.path);
    EXPECT_EQ(draft->get_hash(block_size, block_size), draft_hash);
    EXPECT_EQ(main->get_digest(block_size + 1, block_size), main_digest);
    ASSERT_TRUE(restarted->probe({draft_hash}).at(0)) << "the later tier did not find the draft block";
    restarted->prefetch({draft_hash});
    ASSERT_TRUE(restarted->load_into(draft_hash, /*block_index=*/2));
    restarted->release_prefetched();
    EXPECT_EQ(fixture.cache_manager->read_block(2), expected);
}

// The namespace a pipeline's store opens under carries that pipeline's model identity (cache_offload_coordinator.hpp
// group_name), so main and draft models with different identities open different store directories. Two coordinators
// with the same identity and layout name the same directory, and the second store opened on it while the first is
// open gets a run-scoped store.
TEST(TestPersistentFileKVCacheStore, StoreGroupsOfDifferentModelIdentitiesAreDifferentDirectories) {
    ScopedDirectory directory("openvino_genai_gap2_identity_groups");
    const auto layout = make_layout();
    const CacheOffloadCoordinator main_model("identity-of-the-main-model");
    const CacheOffloadCoordinator draft_model("identity-of-the-draft-model");
    const CacheOffloadCoordinator same_as_main("identity-of-the-main-model");
    const std::string main_group = main_model.group_name("kv", layout, /*block_tokens=*/4, "CPU");
    const std::string draft_group = draft_model.group_name("kv", layout, /*block_tokens=*/4, "CPU");
    EXPECT_NE(main_group, draft_group);
    EXPECT_EQ(same_as_main.group_name("kv", layout, /*block_tokens=*/4, "CPU"), main_group);

    const auto config = make_persistent_config(directory.path);
    auto main_store = open_kv_cache_store(layout, config, "CPU", main_group);
    auto draft_store = open_kv_cache_store(layout, config, "CPU", draft_group);
    EXPECT_TRUE(main_store->outlives_process());
    EXPECT_TRUE(draft_store->outlives_process()) << "a different identity must not share the main model's directory";
    auto same_store = open_kv_cache_store(layout, config, "CPU", main_group);
    EXPECT_FALSE(same_store->outlives_process()) << "the same identity and layout name the locked directory";
}

// End to end at unit level: a block written by one model is found again by the same model after a restart, never by
// a variant whose inputs or outputs swap roles (model_identity_test_models.hpp), which computes different logits from
// the same named inputs. The namespace comes from compute_model_identity() and group_name(), as in the pipeline.
TEST(TestPersistentFileKVCacheStore, ModelsWhoseInputsOrOutputsSwapRolesDoNotRestoreEachOthersBlocks) {
    using namespace model_identity_test_models;
    ScopedDirectory directory("openvino_genai_gap2_identity_roles");
    const auto layout = make_layout();
    const auto config = make_persistent_config(directory.path);
    const auto group_of = [&](const std::shared_ptr<ov::Model>& model) {
        return CacheOffloadCoordinator(compute_model_identity(model)).group_name("kv", layout, /*block_tokens=*/4, "CPU");
    };
    const PrefixKey key = key_of(7, 0x7777);
    const auto write_as = [&](const std::string& group, uint8_t seed) {
        auto store = open_kv_cache_store(layout, config, "CPU", group);
        ASSERT_TRUE(store->outlives_process());
        write_keyed(*store, key, make_block(seed));
        store->flush();
    };
    const auto read_as = [&](const std::string& group) {
        auto store = open_kv_cache_store(layout, config, "CPU", group);
        EXPECT_TRUE(store->outlives_process());
        return read_keyed(*store, key);
    };

    write_as(group_of(make_subtract_model(Wiring::Straight)), 7);
    write_as(group_of(make_two_output_model(Wiring::Straight)), 8);
    // Controls before: the same models, rebuilt, restore their own blocks; another operator misses.
    ASSERT_EQ(read_as(group_of(make_subtract_model(Wiring::Straight))), make_block(7)) << "control before";
    ASSERT_EQ(read_as(group_of(make_two_output_model(Wiring::Straight))), make_block(8)) << "control before";
    EXPECT_EQ(read_as(group_of(make_subtract_model(Wiring::Straight, ov::PartialShape{1, 4}, /*use_add=*/true))),
              std::nullopt)
        << "control: a different operator restored a block";

    EXPECT_EQ(read_as(group_of(make_subtract_model(Wiring::Rewired))), std::nullopt)
        << "rewired inputs restored the other model's block";
    EXPECT_EQ(read_as(group_of(make_subtract_model(Wiring::NamesSwapped))), std::nullopt)
        << "swapped input names restored the other model's block";
    EXPECT_EQ(read_as(group_of(make_two_output_model(Wiring::NamesSwapped))), std::nullopt)
        << "swapped output names restored the other model's block";

    // Controls after: the writers' blocks are still there.
    EXPECT_EQ(read_as(group_of(make_subtract_model(Wiring::Straight))), make_block(7)) << "control after";
    EXPECT_EQ(read_as(group_of(make_two_output_model(Wiring::Straight))), make_block(8)) << "control after";
}
