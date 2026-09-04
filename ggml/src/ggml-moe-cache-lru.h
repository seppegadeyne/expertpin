#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>

struct ggml_moe_cache_key {
    const void * tensor_data = nullptr;
    size_t expert_stride = 0;
    uint32_t expert = 0;

    bool operator==(const ggml_moe_cache_key & other) const {
        return tensor_data == other.tensor_data && expert_stride == other.expert_stride && expert == other.expert;
    }
};

struct ggml_moe_cache_access {
    bool hit = false;
    bool cached = false;
    bool bypassed = false;
    size_t evicted_entries = 0;
    size_t evicted_bytes = 0;
};

struct ggml_moe_cache_stats {
    size_t capacity_bytes = 0;
    size_t resident_bytes = 0;
    uint64_t requests = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t evicted_bytes = 0;
    uint64_t bypasses = 0;
};

// Byte-bounded LRU policy for mmap-backed MoE expert slices. This class only
// decides residency; callers supply the actual populate/evict operations.
class ggml_moe_cache_lru {
public:
    explicit ggml_moe_cache_lru(size_t capacity_bytes = 0);

    ggml_moe_cache_access access(const ggml_moe_cache_key & key, size_t bytes);
    bool contains(const ggml_moe_cache_key & key) const;
    const ggml_moe_cache_stats & stats() const;
    void reset(size_t capacity_bytes);

private:
    struct key_hash {
        size_t operator()(const ggml_moe_cache_key & key) const;
    };
    struct entry {
        ggml_moe_cache_key key;
        size_t bytes;
    };

    using lru_list = std::list<entry>;
    using index_map = std::unordered_map<ggml_moe_cache_key, lru_list::iterator, key_hash>;

    lru_list entries_;
    index_map index_;
    ggml_moe_cache_stats stats_;
};
