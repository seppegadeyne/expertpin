#include "ggml-moe-cache-lru.h"

#include <cstdint>
#include <functional>

namespace {

size_t hash_combine(size_t lhs, size_t rhs) {
    return lhs ^ (rhs + 0x9e3779b9u + (lhs << 6) + (lhs >> 2));
}

} // namespace

ggml_moe_cache_lru::ggml_moe_cache_lru(size_t capacity_bytes) {
    reset(capacity_bytes);
}

size_t ggml_moe_cache_lru::key_hash::operator()(const ggml_moe_cache_key & key) const {
    const size_t data_hash = std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(key.tensor_data));
    const size_t layout_hash = std::hash<size_t>{}(key.expert_stride);
    return hash_combine(hash_combine(data_hash, layout_hash), std::hash<uint32_t>{}(key.expert));
}

ggml_moe_cache_access ggml_moe_cache_lru::access(
        const ggml_moe_cache_key & key, size_t bytes) {
    ggml_moe_cache_access result;
    ++stats_.requests;

    auto found = index_.find(key);
    if (found != index_.end() && found->second->bytes == bytes) {
        entries_.splice(entries_.begin(), entries_, found->second);
        ++stats_.hits;
        result.hit = true;
        result.cached = true;
        return result;
    }

    ++stats_.misses;
    if (found != index_.end()) {
        stats_.resident_bytes -= found->second->bytes;
        entries_.erase(found->second);
        index_.erase(found);
    }

    if (bytes == 0 || stats_.capacity_bytes == 0 || bytes > stats_.capacity_bytes) {
        ++stats_.bypasses;
        result.bypassed = true;
        return result;
    }

    while (!entries_.empty() && bytes > stats_.capacity_bytes - stats_.resident_bytes) {
        const entry & victim = entries_.back();
        result.evicted_bytes += victim.bytes;
        ++result.evicted_entries;
        stats_.resident_bytes -= victim.bytes;
        index_.erase(victim.key);
        entries_.pop_back();
    }

    entries_.push_front({key, bytes});
    index_[key] = entries_.begin();
    stats_.resident_bytes += bytes;
    stats_.evictions += result.evicted_entries;
    stats_.evicted_bytes += result.evicted_bytes;
    result.cached = true;
    return result;
}

bool ggml_moe_cache_lru::contains(const ggml_moe_cache_key & key) const {
    return index_.find(key) != index_.end();
}

const ggml_moe_cache_stats & ggml_moe_cache_lru::stats() const {
    return stats_;
}

void ggml_moe_cache_lru::reset(size_t capacity_bytes) {
    entries_.clear();
    index_.clear();
    stats_ = {};
    stats_.capacity_bytes = capacity_bytes;
}
