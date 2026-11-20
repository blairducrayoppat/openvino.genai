// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"
#include "openvino/genai/cache_offload.hpp"

namespace ov::genai {

/**
 * @brief In-tree IKVCacheStore that keeps its blocks under `kv_cache_dir` beyond the process (persist_file_store).
 *
 * Layout, format version 1, under `<kv_cache_dir>/ov_genai_kv_store/<group>/`:
 *   - `LOCK`: held exclusively while the store is open, so one process at a time uses the directory.
 *   - `slots.bin`: a 4096-byte file header (magic, format version, content hash algorithm version, payload bytes,
 *     slot count, slot stride, group), then fixed slots of `round_up(64 + payload bytes, 4096)` bytes. Each slot starts
 *     with a 64-byte header (magic, key, payload bytes, checksum of the payload seeded with the key, write sequence)
 *     followed by the payload.
 *
 * Blocks are keyed by the 128-bit prefix digest (PrefixKey::digest), the identity that is the same in every process. A
 * block whose digest is unknown is not stored: its size_t hash folded to 128 bits would let two prefixes whose 64-bit
 * hashes collide share a key and pass each other's checksum. The size_t API therefore never stores or finds a block.
 *
 * Opening the store rebuilds the index from the slot headers alone; a payload is checked against its checksum on every
 * read, so a slot torn by a crash reads as a miss and is freed, never returned. A file header that does not match what
 * this process would write, or `reuse_ssd_cache == false`, starts the store empty.
 *
 * `slots.bin` is sized to the capacity without writing it (sparse on Windows NTFS, ftruncate elsewhere), so disk space is
 * taken as blocks are written and a full disk shows as a failed block write, which frees that slot. If sizing or the
 * file header fails while the store starts empty, the store removes `slots.bin` and throws.
 *
 * The store evicts by itself (oldest write first) when it is full: the offload tier only reclaims blocks its own
 * instance wrote, so a store an earlier process filled would otherwise never accept another block.
 */
class PersistentFileKVCacheStore final : public IKVCacheStore {
public:
    static constexpr std::uint32_t FORMAT_VERSION = 1;
    static constexpr std::size_t FILE_HEADER_BYTES = 4096;
    static constexpr std::size_t SLOT_HEADER_BYTES = 64;
    static constexpr std::size_t SLOT_ALIGNMENT = 4096;

    /// Test seam: called before the file is sized at open and before every write; throwing from it fails that step as
    /// a full disk would. Empty outside of tests.
    struct IoFaults {
        std::function<void(std::uint64_t size)> before_resize;
        std::function<void(std::uint64_t offset, std::size_t size)> before_write;
    };

    /// Throws when the directory cannot be used or is locked by another open store; the registry then falls back.
    PersistentFileKVCacheStore(const KVCacheDiskLayout& layout,
                               const CacheOffloadConfig& config,
                               const std::string& device,
                               const std::string& group,
                               IoFaults faults = {});
    ~PersistentFileKVCacheStore() override;

    PersistentFileKVCacheStore(const PersistentFileKVCacheStore&) = delete;
    PersistentFileKVCacheStore& operator=(const PersistentFileKVCacheStore&) = delete;

    bool has_block(size_t hash) const override;
    bool write_block(size_t hash, const std::vector<uint8_t>& data) override;
    bool read_block(size_t hash, std::vector<uint8_t>& data) const override;
    void drop_block(size_t hash) override;

    std::vector<bool> has_blocks_by_key(const std::vector<PrefixKey>& chain) const override;
    std::vector<bool> write_blocks_by_key(const std::vector<PrefixKey>& keys,
                                          const std::vector<const std::vector<uint8_t>*>& data) override;
    std::vector<bool> read_blocks_by_key(const std::vector<PrefixKey>& keys,
                                         std::vector<std::vector<uint8_t>>& data) const override;
    void drop_block_by_key(const PrefixKey& key) override;

    size_t get_block_size() const override {
        return m_payload_bytes;
    }
    size_t get_capacity_blocks() const override {
        return m_num_slots;
    }
    size_t get_num_free_blocks() const override;

    void flush() override;

    bool outlives_process() const override {
        return true;
    }

    bool evicts_itself() const override {
        return true;
    }

    /// @return The directory this store keeps its files in.
    const std::filesystem::path& get_directory() const {
        return m_directory;
    }

    /// @return The directory name a group is stored under: every character other than [A-Za-z0-9_-] becomes '_'.
    static std::string directory_name(const std::string& group);

private:
    struct KeyHash {
        size_t operator()(const ContentHash& key) const noexcept {
            return static_cast<size_t>(key.lo ^ (key.hi * 0x9E3779B97F4A7C15ull));
        }
    };

    bool has_locked(const ContentHash& key) const;
    bool write_locked(const ContentHash& key, const std::vector<uint8_t>& data);
    bool read_locked(const ContentHash& key, std::vector<uint8_t>& data) const;
    void drop_locked(const ContentHash& key) const;
    /// Counts and, the first time, logs a block that has no digest; @return false, i.e. not stored.
    bool refuse_without_digest_locked() const;

    std::vector<uint8_t> make_file_header() const;
    bool open_existing_locked();
    void start_empty_locked();
    std::uint64_t slot_offset(std::size_t slot) const;
    void invalidate_slot_locked(std::size_t slot) const;
    void close_files(bool remove_slot_file = false) noexcept;
    /// Records that @p slot now holds @p key, written with @p sequence.
    void occupy_locked(std::size_t slot, const ContentHash& key, std::uint64_t sequence) const;
    /// Forgets what @p slot holds, in the index and in the eviction order.
    void vacate_locked(std::size_t slot) const;

    void write_at(std::uint64_t offset, const uint8_t* data, std::size_t size) const;
    void read_at(std::uint64_t offset, uint8_t* data, std::size_t size) const;

    std::filesystem::path m_directory;
    std::string m_group;
    std::size_t m_payload_bytes = 0;
    std::size_t m_num_slots = 0;
    std::size_t m_stride = 0;
    int m_lock_fd = -1;
    int m_fd = -1;
    // The index and free list change on a failed read too, so they are mutable like the mutex.
    mutable std::unordered_map<ContentHash, std::size_t, KeyHash> m_key_to_slot;
    mutable std::vector<std::size_t> m_free_slots;
    // Per slot: the write sequence of what it holds (0 when free) and its key; m_by_sequence orders occupied slots
    // oldest first for eviction.
    mutable std::vector<std::uint64_t> m_slot_sequence;
    mutable std::vector<ContentHash> m_slot_key;
    mutable std::map<std::uint64_t, std::size_t> m_by_sequence;
    std::uint64_t m_next_sequence = 1;
    mutable std::size_t m_num_refused_without_digest = 0;
    IoFaults m_faults;
    mutable std::mutex m_mutex;
};

}  // namespace ov::genai
