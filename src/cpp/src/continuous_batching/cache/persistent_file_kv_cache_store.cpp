// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/persistent_file_kv_cache_store.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <fcntl.h>
#    include <io.h>
#    include <share.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <windows.h>
#    include <winioctl.h>
#else
#    include <fcntl.h>
#    include <sys/file.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

#include "continuous_batching/cache/kv_cache_offload_manager.hpp"
#include "logger.hpp"
#include "openvino/core/except.hpp"

namespace ov::genai {

namespace {

constexpr char FILE_MAGIC[8] = {'O', 'V', 'G', 'K', 'V', 'S', 'T', '1'};
constexpr std::uint32_t SLOT_MAGIC = 0x31544C53u;  // "SLT1"
// Offsets inside the 64-byte slot header.
constexpr std::size_t SLOT_MAGIC_AT = 0;
constexpr std::size_t SLOT_KEY_LO_AT = 8;
constexpr std::size_t SLOT_KEY_HI_AT = 16;
constexpr std::size_t SLOT_BYTES_AT = 24;
constexpr std::size_t SLOT_SUM_LO_AT = 32;
constexpr std::size_t SLOT_SUM_HI_AT = 40;
constexpr std::size_t SLOT_SEQUENCE_AT = 48;

void put_u32(uint8_t* out, std::uint32_t value) {
    for (std::size_t i = 0; i < 4; ++i) {
        out[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

void put_u64(uint8_t* out, std::uint64_t value) {
    for (std::size_t i = 0; i < 8; ++i) {
        out[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

std::uint32_t get_u32(const uint8_t* in) {
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(in[i]) << (8 * i);
    }
    return value;
}

std::uint64_t get_u64(const uint8_t* in) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(in[i]) << (8 * i);
    }
    return value;
}

/// Binds a payload to the key it was written under, so a slot that holds another key's bytes fails the check.
ContentHash payload_checksum(const ContentHash& key, const uint8_t* data, std::size_t size) {
    return compute_content_hash(data, size, key.lo ^ (key.hi * 0x9E3779B97F4A7C15ull));
}

std::string errno_text() {
    return std::strerror(errno);
}

int open_shared_none(const std::filesystem::path& path) {
#ifdef _WIN32
    int fd = -1;
    const errno_t error = _wsopen_s(&fd,
                                    path.wstring().c_str(),
                                    _O_RDWR | _O_CREAT | _O_BINARY | _O_RANDOM | _O_NOINHERIT,
                                    _SH_DENYRW,
                                    _S_IREAD | _S_IWRITE);
    if (error != 0) {
        errno = error;
        return -1;
    }
    return fd;
#else
    return ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR);
#endif
}

void close_fd(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    ::close(fd);
#endif
}

std::uint64_t file_size(int fd) {
#ifdef _WIN32
    const __int64 size = _filelengthi64(fd);
    OPENVINO_ASSERT(size >= 0, "Failed to read the size of the KV cache store file: ", errno_text());
    return static_cast<std::uint64_t>(size);
#else
    struct stat status {};
    OPENVINO_ASSERT(::fstat(fd, &status) == 0, "Failed to read the size of the KV cache store file: ", errno_text());
    return static_cast<std::uint64_t>(status.st_size);
#endif
}

/// Lets the file system leave unwritten slots unallocated, so that opening or purging a store neither writes nor
/// holds its whole capacity (POSIX ftruncate already leaves the extension unallocated on common file systems).
void make_sparse(int fd) {
#ifdef _WIN32
    DWORD returned = 0;
    const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (handle == INVALID_HANDLE_VALUE ||
        !DeviceIoControl(handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr)) {
        // E.g. FAT32 or exFAT. Sizing the file then reserves the capacity at open, and a full disk shows there.
        GENAI_INFO("[KV_TRACE] persistent_file_store sparse=0 error=%lu", GetLastError());
    }
#else
    (void)fd;
#endif
}

/// Sets the file size without writing the extension (_chsize_s would write zeros over the whole of it).
void resize_fd(int fd, std::uint64_t size) {
#ifdef _WIN32
    const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    LARGE_INTEGER end;
    end.QuadPart = static_cast<LONGLONG>(size);
    const bool failed = handle == INVALID_HANDLE_VALUE || !SetFilePointerEx(handle, end, nullptr, FILE_BEGIN) ||
                        !SetEndOfFile(handle);
    OPENVINO_ASSERT(!failed, "Failed to size the KV cache store file to ", size, " bytes: Windows error ", GetLastError());
#else
    OPENVINO_ASSERT(::ftruncate(fd, static_cast<off_t>(size)) == 0,
                    "Failed to size the KV cache store file to ",
                    size,
                    " bytes: ",
                    errno_text());
#endif
}

}  // namespace

std::string PersistentFileKVCacheStore::directory_name(const std::string& group) {
    std::string name = group.empty() ? std::string("default") : group;
    for (char& c : name) {
        const bool keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
        if (!keep) {
            c = '_';
        }
    }
    return name;
}

PersistentFileKVCacheStore::PersistentFileKVCacheStore(const KVCacheDiskLayout& layout,
                                                       const CacheOffloadConfig& config,
                                                       const std::string& device,
                                                       const std::string& group,
                                                       IoFaults faults)
    : m_group(group),
      m_faults(std::move(faults)) {
    OPENVINO_ASSERT(KVCacheOffloadManager::is_supported_device(device),
                    "KV cache disk offload is implemented for CPU and GPU, but the inference device is '",
                    device,
                    "'");
    OPENVINO_ASSERT(config.use_page_cache,
                    "KV cache disk offload with direct I/O is not implemented yet, set use_page_cache to true");
    OPENVINO_ASSERT(!config.kv_cache_dir.empty(), "persist_file_store requires kv_cache_dir to be set");
    OPENVINO_ASSERT(std::filesystem::is_directory(config.kv_cache_dir),
                    "KV cache offload kv_cache_dir '",
                    config.kv_cache_dir,
                    "' is not an existing directory");

    m_payload_bytes = layout.get_slot_size();
    OPENVINO_ASSERT(m_payload_bytes > 0, "KV cache offload slot size must be greater than 0");
    m_num_slots = config.capacity_bytes / m_payload_bytes;
    OPENVINO_ASSERT(m_num_slots > 0,
                    "KV cache offload capacity of ",
                    config.capacity_bytes,
                    " bytes is smaller than a single cache block of ",
                    m_payload_bytes,
                    " bytes");
    OPENVINO_ASSERT(m_payload_bytes <= std::numeric_limits<std::size_t>::max() - SLOT_HEADER_BYTES - SLOT_ALIGNMENT,
                    "KV cache offload slot size overflows");
    m_stride = (SLOT_HEADER_BYTES + m_payload_bytes + SLOT_ALIGNMENT - 1) / SLOT_ALIGNMENT * SLOT_ALIGNMENT;
    OPENVINO_ASSERT(m_num_slots <= (std::numeric_limits<std::uint64_t>::max() - FILE_HEADER_BYTES) / m_stride,
                    "KV cache store file size overflows");
    OPENVINO_ASSERT(group.size() <= FILE_HEADER_BYTES - 64, "KV cache store group name is too long");

    m_directory = std::filesystem::path(config.kv_cache_dir) / "ov_genai_kv_store" / directory_name(group);
    std::error_code error_code;
    std::filesystem::create_directories(m_directory, error_code);
    OPENVINO_ASSERT(std::filesystem::is_directory(m_directory),
                    "Failed to create the KV cache store directory '",
                    m_directory.string(),
                    "': ",
                    error_code.message());

    m_lock_fd = open_shared_none(m_directory / "LOCK");
    OPENVINO_ASSERT(m_lock_fd >= 0,
                    "KV cache store directory '",
                    m_directory.string(),
                    "' is in use by another store: ",
                    errno_text());
#ifndef _WIN32
    if (::flock(m_lock_fd, LOCK_EX | LOCK_NB) != 0) {
        const std::string reason = errno_text();
        close_files();
        OPENVINO_THROW("KV cache store directory '", m_directory.string(), "' is in use by another store: ", reason);
    }
#endif

    bool starting_empty = false;
    try {
        m_fd = open_shared_none(m_directory / "slots.bin");
        OPENVINO_ASSERT(m_fd >= 0, "Failed to open the KV cache store file in '", m_directory.string(), "': ", errno_text());
        make_sparse(m_fd);
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!config.reuse_ssd_cache || !open_existing_locked()) {
            starting_empty = true;
            start_empty_locked();
        }
    } catch (...) {
        // A store that failed while starting empty holds no block; its file goes, so that a full disk is not left
        // fuller for the run-scoped store the registry falls back to. The lock is still held while it is removed.
        close_files(/*remove_slot_file=*/starting_empty);
        throw;
    }
    GENAI_INFO("[KV_TRACE] persistent_file_store dir='%s' slots=%zu stride=%zu reused_blocks=%zu",
               m_directory.string().c_str(),
               m_num_slots,
               m_stride,
               m_key_to_slot.size());
}

PersistentFileKVCacheStore::~PersistentFileKVCacheStore() {
    std::lock_guard<std::mutex> lock(m_mutex);
    close_files();
}

void PersistentFileKVCacheStore::close_files(bool remove_slot_file) noexcept {
    if (m_fd >= 0) {
        close_fd(m_fd);
        m_fd = -1;
    }
    if (remove_slot_file) {
        std::error_code error;
        std::filesystem::remove(m_directory / "slots.bin", error);
    }
    if (m_lock_fd >= 0) {
        close_fd(m_lock_fd);
        m_lock_fd = -1;
    }
}

std::vector<uint8_t> PersistentFileKVCacheStore::make_file_header() const {
    std::vector<uint8_t> header(FILE_HEADER_BYTES, 0);
    std::memcpy(header.data(), FILE_MAGIC, sizeof(FILE_MAGIC));
    put_u32(header.data() + 8, FORMAT_VERSION);
    put_u32(header.data() + 12, CONTENT_HASH_ALGORITHM_VERSION);
    put_u64(header.data() + 16, m_payload_bytes);
    put_u64(header.data() + 24, m_num_slots);
    put_u64(header.data() + 32, m_stride);
    put_u32(header.data() + 40, static_cast<std::uint32_t>(m_group.size()));
    std::memcpy(header.data() + 64, m_group.data(), m_group.size());
    return header;
}

std::uint64_t PersistentFileKVCacheStore::slot_offset(std::size_t slot) const {
    return FILE_HEADER_BYTES + static_cast<std::uint64_t>(slot) * m_stride;
}

bool PersistentFileKVCacheStore::open_existing_locked() {
    const std::uint64_t expected_size = slot_offset(m_num_slots);
    const std::uint64_t size = file_size(m_fd);
    if (size == 0) {
        return false;
    }
    const std::vector<uint8_t> expected = make_file_header();
    std::vector<uint8_t> header(FILE_HEADER_BYTES);
    if (size < expected_size) {
        GENAI_WARN("KV cache store '%s' is shorter than its layout requires; starting it empty", m_directory.string().c_str());
        return false;
    }
    read_at(0, header.data(), header.size());
    if (header != expected) {
        GENAI_WARN("KV cache store '%s' was written with another format, block size, capacity or group; starting it empty",
                   m_directory.string().c_str());
        return false;
    }

    m_slot_sequence.assign(m_num_slots, 0);
    m_slot_key.assign(m_num_slots, ContentHash{});
    m_by_sequence.clear();
    uint8_t slot_header[SLOT_HEADER_BYTES];
    for (std::size_t slot = 0; slot < m_num_slots; ++slot) {
        read_at(slot_offset(slot), slot_header, sizeof(slot_header));
        if (get_u32(slot_header + SLOT_MAGIC_AT) != SLOT_MAGIC ||
            get_u64(slot_header + SLOT_BYTES_AT) != m_payload_bytes) {
            continue;
        }
        const ContentHash key{get_u64(slot_header + SLOT_KEY_LO_AT), get_u64(slot_header + SLOT_KEY_HI_AT)};
        const std::uint64_t sequence = get_u64(slot_header + SLOT_SEQUENCE_AT);
        m_next_sequence = std::max(m_next_sequence, sequence + 1);
        if (sequence == 0 || m_by_sequence.count(sequence) != 0) {
            // A write sequence is never 0 and never reused, so such a header is not one this store wrote.
            invalidate_slot_locked(slot);
            continue;
        }
        const auto existing = m_key_to_slot.find(key);
        if (existing != m_key_to_slot.end()) {
            // Two slots claim one key, which only an interrupted write sequence leaves behind: keep the newer one.
            const std::size_t previous = existing->second;
            if (sequence < m_slot_sequence[previous]) {
                invalidate_slot_locked(slot);
                continue;
            }
            vacate_locked(previous);
            invalidate_slot_locked(previous);
        }
        occupy_locked(slot, key, sequence);
    }
    m_free_slots.clear();
    for (std::size_t slot = m_num_slots; slot > 0; --slot) {
        if (m_slot_sequence[slot - 1] == 0) {
            m_free_slots.push_back(slot - 1);
        }
    }
    return true;
}

void PersistentFileKVCacheStore::occupy_locked(std::size_t slot, const ContentHash& key, std::uint64_t sequence) const {
    m_key_to_slot[key] = slot;
    m_slot_key[slot] = key;
    m_slot_sequence[slot] = sequence;
    m_by_sequence[sequence] = slot;
}

void PersistentFileKVCacheStore::vacate_locked(std::size_t slot) const {
    if (m_slot_sequence[slot] == 0) {
        return;
    }
    m_by_sequence.erase(m_slot_sequence[slot]);
    m_key_to_slot.erase(m_slot_key[slot]);
    m_slot_sequence[slot] = 0;
    m_slot_key[slot] = ContentHash{};
}

void PersistentFileKVCacheStore::start_empty_locked() {
    m_key_to_slot.clear();
    m_slot_sequence.assign(m_num_slots, 0);
    m_slot_key.assign(m_num_slots, ContentHash{});
    m_by_sequence.clear();
    const std::uint64_t size = slot_offset(m_num_slots);
    resize_fd(m_fd, 0);
    if (m_faults.before_resize) {
        m_faults.before_resize(size);
    }
    resize_fd(m_fd, size);
    const std::vector<uint8_t> header = make_file_header();
    write_at(0, header.data(), header.size());
    m_free_slots.clear();
    m_free_slots.reserve(m_num_slots);
    for (std::size_t slot = m_num_slots; slot > 0; --slot) {
        m_free_slots.push_back(slot - 1);
    }
}

void PersistentFileKVCacheStore::invalidate_slot_locked(std::size_t slot) const {
    const uint8_t zero[4] = {0, 0, 0, 0};
    write_at(slot_offset(slot) + SLOT_MAGIC_AT, zero, sizeof(zero));
}

bool PersistentFileKVCacheStore::has_locked(const ContentHash& key) const {
    return m_key_to_slot.find(key) != m_key_to_slot.end();
}

bool PersistentFileKVCacheStore::write_locked(const ContentHash& key, const std::vector<uint8_t>& data) {
    OPENVINO_ASSERT(data.size() == m_payload_bytes,
                    "Unexpected KV cache store block size: got ",
                    data.size(),
                    ", expected ",
                    m_payload_bytes);
    std::size_t slot = 0;
    if (const auto it = m_key_to_slot.find(key); it != m_key_to_slot.end()) {
        slot = it->second;
        vacate_locked(slot);
    } else if (!m_free_slots.empty()) {
        slot = m_free_slots.back();
        m_free_slots.pop_back();
    } else {
        // Full: the oldest write makes room. The slot header is invalidated below before the new payload goes in.
        OPENVINO_ASSERT(!m_by_sequence.empty(), "KV cache store has neither a free nor an occupied slot");
        slot = m_by_sequence.begin()->second;
        vacate_locked(slot);
    }
    const std::uint64_t sequence = m_next_sequence++;
    try {
        // Header last: until it is written the slot does not claim the key, and the checksum covers a torn payload.
        invalidate_slot_locked(slot);
        write_at(slot_offset(slot) + SLOT_HEADER_BYTES, data.data(), data.size());
        uint8_t header[SLOT_HEADER_BYTES] = {};
        const ContentHash sum = payload_checksum(key, data.data(), data.size());
        put_u32(header + SLOT_MAGIC_AT, SLOT_MAGIC);
        put_u64(header + SLOT_KEY_LO_AT, key.lo);
        put_u64(header + SLOT_KEY_HI_AT, key.hi);
        put_u64(header + SLOT_BYTES_AT, data.size());
        put_u64(header + SLOT_SUM_LO_AT, sum.lo);
        put_u64(header + SLOT_SUM_HI_AT, sum.hi);
        put_u64(header + SLOT_SEQUENCE_AT, sequence);
        write_at(slot_offset(slot), header, sizeof(header));
    } catch (...) {
        // The slot no longer holds a valid block, whichever key it held before; it was vacated above.
        m_free_slots.push_back(slot);
        throw;
    }
    occupy_locked(slot, key, sequence);
    return true;
}

bool PersistentFileKVCacheStore::read_locked(const ContentHash& key, std::vector<uint8_t>& data) const {
    const auto it = m_key_to_slot.find(key);
    if (it == m_key_to_slot.end()) {
        return false;
    }
    std::vector<uint8_t> buffer(SLOT_HEADER_BYTES + m_payload_bytes);
    read_at(slot_offset(it->second), buffer.data(), buffer.size());
    const uint8_t* header = buffer.data();
    const uint8_t* payload = buffer.data() + SLOT_HEADER_BYTES;
    const ContentHash sum = payload_checksum(key, payload, m_payload_bytes);
    const bool valid = get_u32(header + SLOT_MAGIC_AT) == SLOT_MAGIC && get_u64(header + SLOT_KEY_LO_AT) == key.lo &&
                       get_u64(header + SLOT_KEY_HI_AT) == key.hi && get_u64(header + SLOT_BYTES_AT) == m_payload_bytes &&
                       get_u64(header + SLOT_SUM_LO_AT) == sum.lo && get_u64(header + SLOT_SUM_HI_AT) == sum.hi;
    if (!valid) {
        GENAI_WARN("KV cache store '%s' slot %zu failed its checksum; dropping it and treating it as a miss",
                   m_directory.string().c_str(),
                   it->second);
        try {
            drop_locked(key);
        } catch (const std::exception& error) {
            // The index already forgot the slot; a stale header on disk fails this same check after a restart.
            GENAI_WARN("KV cache store '%s' could not invalidate a slot: %s", m_directory.string().c_str(), error.what());
        }
        return false;
    }
    data.assign(payload, payload + m_payload_bytes);
    return true;
}

void PersistentFileKVCacheStore::drop_locked(const ContentHash& key) const {
    const auto it = m_key_to_slot.find(key);
    if (it == m_key_to_slot.end()) {
        return;
    }
    const std::size_t slot = it->second;
    vacate_locked(slot);
    m_free_slots.push_back(slot);
    // A dropped block must not come back after a restart.
    invalidate_slot_locked(slot);
}

// The size_t API names a block by its hash alone, which this store never keys by (refuse_without_digest_locked): such
// a block is never stored, so it is never found either.
bool PersistentFileKVCacheStore::has_block(size_t) const {
    return false;
}

bool PersistentFileKVCacheStore::write_block(size_t, const std::vector<uint8_t>&) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return refuse_without_digest_locked();
}

bool PersistentFileKVCacheStore::read_block(size_t, std::vector<uint8_t>&) const {
    return false;
}

void PersistentFileKVCacheStore::drop_block(size_t) {}

bool PersistentFileKVCacheStore::refuse_without_digest_locked() const {
    ++m_num_refused_without_digest;
    if (m_num_refused_without_digest == 1) {
        GENAI_WARN("KV cache store '%s' keeps only blocks with a 128-bit prefix digest; a block without one is not stored "
                   "(logged once per store)",
                   m_directory.string().c_str());
    }
    return false;
}

std::vector<bool> PersistentFileKVCacheStore::has_blocks_by_key(const std::vector<PrefixKey>& chain) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<bool> present;
    present.reserve(chain.size());
    for (const PrefixKey& key : chain) {
        present.push_back(key.digest.has_value() && has_locked(*key.digest));
    }
    return present;
}

std::vector<bool> PersistentFileKVCacheStore::write_blocks_by_key(const std::vector<PrefixKey>& keys,
                                                                  const std::vector<const std::vector<uint8_t>*>& data) {
    OPENVINO_ASSERT(keys.size() == data.size(), "KV cache store write needs one buffer per key");
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<bool> written;
    written.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        written.push_back(keys[i].digest ? write_locked(*keys[i].digest, *data[i]) : refuse_without_digest_locked());
    }
    return written;
}

std::vector<bool> PersistentFileKVCacheStore::read_blocks_by_key(const std::vector<PrefixKey>& keys,
                                                                 std::vector<std::vector<uint8_t>>& data) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<bool> found;
    found.reserve(keys.size());
    data.assign(keys.size(), {});
    for (std::size_t i = 0; i < keys.size(); ++i) {
        found.push_back(keys[i].digest.has_value() && read_locked(*keys[i].digest, data[i]));
    }
    return found;
}

void PersistentFileKVCacheStore::drop_block_by_key(const PrefixKey& key) {
    if (!key.digest) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    drop_locked(*key.digest);
}

size_t PersistentFileKVCacheStore::get_num_free_blocks() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_free_slots.size();
}

void PersistentFileKVCacheStore::flush() {
    std::lock_guard<std::mutex> lock(m_mutex);
#ifdef _WIN32
    OPENVINO_ASSERT(_commit(m_fd) == 0, "Failed to flush the KV cache store file: ", errno_text());
#else
    OPENVINO_ASSERT(::fsync(m_fd) == 0, "Failed to flush the KV cache store file: ", errno_text());
#endif
}

void PersistentFileKVCacheStore::write_at(std::uint64_t offset, const uint8_t* data, std::size_t size) const {
    if (m_faults.before_write) {
        m_faults.before_write(offset, size);
    }
    std::size_t written = 0;
    while (written < size) {
        const std::size_t remaining = size - written;
#ifdef _WIN32
        OPENVINO_ASSERT(_lseeki64(m_fd, static_cast<__int64>(offset + written), SEEK_SET) >= 0,
                        "Failed to seek in the KV cache store file: ",
                        errno_text());
        const auto chunk = static_cast<unsigned int>(std::min<std::size_t>(remaining, std::numeric_limits<int>::max()));
        const int result = _write(m_fd, data + written, chunk);
#else
        const ssize_t result = ::pwrite(m_fd, data + written, remaining, static_cast<off_t>(offset + written));
#endif
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            OPENVINO_THROW("Failed to write the KV cache store file: ", errno_text());
        }
        OPENVINO_ASSERT(result > 0, "Short write to the KV cache store file: wrote ", written, " of ", size, " bytes");
        written += static_cast<std::size_t>(result);
    }
}

void PersistentFileKVCacheStore::read_at(std::uint64_t offset, uint8_t* data, std::size_t size) const {
    std::size_t read_bytes = 0;
    while (read_bytes < size) {
        const std::size_t remaining = size - read_bytes;
#ifdef _WIN32
        OPENVINO_ASSERT(_lseeki64(m_fd, static_cast<__int64>(offset + read_bytes), SEEK_SET) >= 0,
                        "Failed to seek in the KV cache store file: ",
                        errno_text());
        const auto chunk = static_cast<unsigned int>(std::min<std::size_t>(remaining, std::numeric_limits<int>::max()));
        const int result = _read(m_fd, data + read_bytes, chunk);
#else
        const ssize_t result = ::pread(m_fd, data + read_bytes, remaining, static_cast<off_t>(offset + read_bytes));
#endif
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            OPENVINO_THROW("Failed to read the KV cache store file: ", errno_text());
        }
        OPENVINO_ASSERT(result > 0, "Short read from the KV cache store file: read ", read_bytes, " of ", size, " bytes");
        read_bytes += static_cast<std::size_t>(result);
    }
}

}  // namespace ov::genai
