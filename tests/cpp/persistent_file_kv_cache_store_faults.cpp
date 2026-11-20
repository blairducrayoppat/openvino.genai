// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_offload_cache.hpp"
#include "continuous_batching/cache/persistent_file_kv_cache_store.hpp"
#include "helper.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/core.hpp"
#include "prefix_digest_registry.hpp"

using namespace ov::genai;

// Failures of the file system (a full disk) while the persistent store sizes its file or writes a block. They are
// injected through PersistentFileKVCacheStore::IoFaults, which throws where the system call would have failed.

namespace {

constexpr size_t BLOCK_SIZE = 32;
constexpr size_t NUM_BLOCKS = 4;
constexpr size_t SLOT_HEADER_BYTES = PersistentFileKVCacheStore::SLOT_HEADER_BYTES;

KVCacheDiskLayout make_layout() {
    return KVCacheDiskLayout({BLOCK_SIZE}, {0});
}

std::vector<uint8_t> make_block(uint8_t seed, size_t size = BLOCK_SIZE) {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + i);
    }
    return data;
}

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

CacheOffloadConfig make_config(const std::filesystem::path& directory, size_t capacity_bytes = BLOCK_SIZE * NUM_BLOCKS) {
    CacheOffloadConfig config;
    config.capacity_bytes = capacity_bytes;
    config.kv_cache_dir = directory.string();
    config.persist_file_store = true;
    return config;
}

std::filesystem::path slot_file(const std::filesystem::path& directory) {
    return directory / "ov_genai_kv_store" / "kv" / "slots.bin";
}

PrefixKey key_of(size_t hash, uint64_t digest_lo) {
    return PrefixKey{hash, ContentHash{digest_lo, ~digest_lo}};
}

void write_keyed(IKVCacheStore& store, const PrefixKey& key, const std::vector<uint8_t>& data) {
    const std::vector<bool> written = store.write_blocks_by_key({key}, {&data});
    ASSERT_EQ(written, std::vector<bool>{true});
}

bool has_keyed(const IKVCacheStore& store, const PrefixKey& key) {
    return store.has_blocks_by_key({key}).at(0);
}

bool reads_back(const IKVCacheStore& store, const PrefixKey& key, const std::vector<uint8_t>& expected) {
    std::vector<std::vector<uint8_t>> data;
    return store.read_blocks_by_key({key}, data).at(0) && data[0] == expected;
}

void throw_disk_full() {
    OPENVINO_THROW("injected: no space left on device");
}

/// Fails every write of exactly @p size bytes while @p armed is set.
PersistentFileKVCacheStore::IoFaults fail_writes_of(size_t size, const bool& armed) {
    PersistentFileKVCacheStore::IoFaults faults;
    faults.before_write = [size, &armed](std::uint64_t, std::size_t bytes) {
        if (armed && bytes == size) {
            throw_disk_full();
        }
    };
    return faults;
}

}  // namespace

TEST(TestPersistentFileKVCacheStoreFaults, FailedSizingAtOpenRemovesTheFileAndReleasesTheLock) {
    ScopedDirectory directory("openvino_genai_gap2_fault_size");
    PersistentFileKVCacheStore::IoFaults faults;
    faults.before_resize = [](std::uint64_t) {
        throw_disk_full();
    };
    EXPECT_THROW(PersistentFileKVCacheStore(make_layout(), make_config(directory.path), "CPU", "kv", faults), ov::Exception);
    EXPECT_FALSE(std::filesystem::exists(slot_file(directory.path))) << "a store that failed to start left its file";

    PersistentFileKVCacheStore store(make_layout(), make_config(directory.path), "CPU", "kv");
    EXPECT_EQ(store.get_num_free_blocks(), NUM_BLOCKS) << "the failed open must have released the lock";
    write_keyed(store, key_of(1, 0xA), make_block(1));
}

TEST(TestPersistentFileKVCacheStoreFaults, FailedFileHeaderWriteAtOpenRemovesTheFile) {
    ScopedDirectory directory("openvino_genai_gap2_fault_header");
    PersistentFileKVCacheStore::IoFaults faults;
    faults.before_write = [](std::uint64_t offset, std::size_t) {
        if (offset == 0) {
            throw_disk_full();
        }
    };
    EXPECT_THROW(PersistentFileKVCacheStore(make_layout(), make_config(directory.path), "CPU", "kv", faults), ov::Exception);
    EXPECT_FALSE(std::filesystem::exists(slot_file(directory.path)));
    PersistentFileKVCacheStore store(make_layout(), make_config(directory.path), "CPU", "kv");
    EXPECT_EQ(store.get_num_free_blocks(), NUM_BLOCKS);
}

TEST(TestPersistentFileKVCacheStoreFaults, PurgeThatFailsToSizeTheFileRemovesIt) {
    ScopedDirectory directory("openvino_genai_gap2_fault_purge");
    {
        PersistentFileKVCacheStore store(make_layout(), make_config(directory.path), "CPU", "kv");
        write_keyed(store, key_of(1, 0xA), make_block(1));
    }
    PersistentFileKVCacheStore::IoFaults faults;
    faults.before_resize = [](std::uint64_t) {
        throw_disk_full();
    };
    {
        // An intact store is reused without sizing its file again, so the failing resize is never reached.
        PersistentFileKVCacheStore reused(make_layout(), make_config(directory.path), "CPU", "kv", faults);
        EXPECT_TRUE(reads_back(reused, key_of(1, 0xA), make_block(1)));
    }
    auto purge = make_config(directory.path);
    purge.reuse_ssd_cache = false;
    EXPECT_THROW(PersistentFileKVCacheStore(make_layout(), purge, "CPU", "kv", faults), ov::Exception);
    EXPECT_FALSE(std::filesystem::exists(slot_file(directory.path)));
    PersistentFileKVCacheStore store(make_layout(), make_config(directory.path), "CPU", "kv");
    EXPECT_FALSE(has_keyed(store, key_of(1, 0xA)));
}

TEST(TestPersistentFileKVCacheStoreFaults, FailedPayloadWriteFreesTheSlotAndKeepsTheOthers) {
    ScopedDirectory directory("openvino_genai_gap2_fault_payload");
    bool armed = false;
    {
        PersistentFileKVCacheStore store(make_layout(), make_config(directory.path), "CPU", "kv", fail_writes_of(BLOCK_SIZE, armed));
        write_keyed(store, key_of(1, 0xA), make_block(1));
        write_keyed(store, key_of(2, 0xB), make_block(2));
        armed = true;
        const std::vector<uint8_t> data = make_block(3);
        EXPECT_THROW(store.write_blocks_by_key({key_of(3, 0xC)}, {&data}), ov::Exception);
        armed = false;
        EXPECT_FALSE(has_keyed(store, key_of(3, 0xC)));
        EXPECT_EQ(store.get_num_free_blocks(), NUM_BLOCKS - 2) << "the slot of the failed write must be free again";
        EXPECT_TRUE(reads_back(store, key_of(1, 0xA), make_block(1)));
        EXPECT_TRUE(reads_back(store, key_of(2, 0xB), make_block(2)));
        write_keyed(store, key_of(4, 0xD), make_block(4));
        EXPECT_EQ(store.get_num_free_blocks(), NUM_BLOCKS - 3);
    }
    PersistentFileKVCacheStore reopened(make_layout(), make_config(directory.path), "CPU", "kv");
    EXPECT_FALSE(has_keyed(reopened, key_of(3, 0xC)));
    EXPECT_TRUE(reads_back(reopened, key_of(1, 0xA), make_block(1)));
    EXPECT_TRUE(reads_back(reopened, key_of(2, 0xB), make_block(2)));
    EXPECT_TRUE(reads_back(reopened, key_of(4, 0xD), make_block(4)));
    EXPECT_EQ(reopened.get_num_free_blocks(), NUM_BLOCKS - 3);
}

TEST(TestPersistentFileKVCacheStoreFaults, FailedSlotHeaderWriteLeavesNoBlockAfterRestart) {
    ScopedDirectory directory("openvino_genai_gap2_fault_slot_header");
    bool armed = false;
    {
        PersistentFileKVCacheStore store(make_layout(), make_config(directory.path), "CPU", "kv",
                                         fail_writes_of(SLOT_HEADER_BYTES, armed));
        write_keyed(store, key_of(1, 0xA), make_block(1));
        armed = true;
        const std::vector<uint8_t> data = make_block(3);
        EXPECT_THROW(store.write_blocks_by_key({key_of(3, 0xC)}, {&data}), ov::Exception);
        armed = false;
        EXPECT_FALSE(has_keyed(store, key_of(3, 0xC)));
        EXPECT_EQ(store.get_num_free_blocks(), NUM_BLOCKS - 1);
    }
    // The payload reached the disk but its header did not, so the slot does not claim the key after a restart.
    PersistentFileKVCacheStore reopened(make_layout(), make_config(directory.path), "CPU", "kv");
    EXPECT_FALSE(has_keyed(reopened, key_of(3, 0xC)));
    EXPECT_TRUE(reads_back(reopened, key_of(1, 0xA), make_block(1)));
    EXPECT_EQ(reopened.get_num_free_blocks(), NUM_BLOCKS - 1);
}

TEST(TestPersistentFileKVCacheStoreFaults, FailedOverwriteLosesOnlyThatKey) {
    ScopedDirectory directory("openvino_genai_gap2_fault_overwrite");
    bool armed = false;
    {
        PersistentFileKVCacheStore store(make_layout(), make_config(directory.path), "CPU", "kv", fail_writes_of(BLOCK_SIZE, armed));
        write_keyed(store, key_of(1, 0xA), make_block(1));
        write_keyed(store, key_of(2, 0xB), make_block(2));
        armed = true;
        const std::vector<uint8_t> data = make_block(9);
        EXPECT_THROW(store.write_blocks_by_key({key_of(1, 0xA)}, {&data}), ov::Exception);
        armed = false;
        // The old copy was invalidated before the new payload went in, so neither is served.
        EXPECT_FALSE(has_keyed(store, key_of(1, 0xA)));
        EXPECT_TRUE(reads_back(store, key_of(2, 0xB), make_block(2)));
    }
    PersistentFileKVCacheStore reopened(make_layout(), make_config(directory.path), "CPU", "kv");
    EXPECT_FALSE(has_keyed(reopened, key_of(1, 0xA)));
    EXPECT_TRUE(reads_back(reopened, key_of(2, 0xB), make_block(2)));
    EXPECT_EQ(reopened.get_num_free_blocks(), NUM_BLOCKS - 1);
}

TEST(TestPersistentFileKVCacheStoreFaults, OffloadTierCountsAFullDiskAsAFailedStore) {
    ScopedDirectory directory("openvino_genai_gap2_fault_tier");
    ov::Core core;
    ov::InferRequest request = core.compile_model(get_dummy_model(core, /*num_layers=*/2)).create_infer_request();
    auto cache_manager = std::make_shared<KVCacheManager>(request);
    cache_manager->allocate_cache_if_needed(4);
    const size_t slot_size = cache_manager->get_block_layout().get_slot_size();
    bool armed = false;
    // Hashes that no other test uses, so that the process-wide digest registry holds nothing else for them.
    constexpr size_t stored = 930011;
    constexpr size_t refused = 930022;
    {
        auto store = std::make_unique<PersistentFileKVCacheStore>(cache_manager->get_block_layout(),
                                                                  make_config(directory.path, slot_size * 4),
                                                                  "CPU",
                                                                  "kv",
                                                                  fail_writes_of(slot_size, armed));
        KVCacheOffloadCache tier(*cache_manager, std::move(store), /*max_queued_stores=*/8);
        PrefixDigestRegistry::instance().record(stored, ContentHash{0xFEED, 1});
        PrefixDigestRegistry::instance().record(refused, ContentHash{0xFEED, 2});
        std::vector<uint8_t> data(slot_size, 0x5A);
        cache_manager->write_block(1, data);
        tier.probe({stored});
        tier.on_blocks_overwritten(stored, BlocksPerLayer{std::make_shared<CacheBlock>(1)});
        tier.flush();
        armed = true;
        tier.probe({refused});
        tier.on_blocks_overwritten(refused, BlocksPerLayer{std::make_shared<CacheBlock>(1)});
        tier.flush();
        armed = false;
        EXPECT_EQ(tier.get_statistics().num_stored, 1u);
        EXPECT_EQ(tier.get_statistics().num_failed, 1u);
        EXPECT_EQ(tier.probe({stored, refused}), (std::vector<bool>{true, false}));
    }
    PersistentFileKVCacheStore reopened(cache_manager->get_block_layout(), make_config(directory.path, slot_size * 4), "CPU", "kv");
    EXPECT_TRUE(has_keyed(reopened, PrefixKey{stored, ContentHash{0xFEED, 1}}));
    EXPECT_FALSE(has_keyed(reopened, PrefixKey{refused, ContentHash{0xFEED, 2}}));
    EXPECT_EQ(reopened.get_num_free_blocks(), 3u);
}
