#include "ggml-moe-cache-lru.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ggml_moe_cache_key key(uintptr_t tensor, uint32_t expert) {
    return {reinterpret_cast<const void *>(tensor), 0, expert};
}

void test_hit_promotes_and_evicts_lru() {
    ggml_moe_cache_lru cache(8);

    require(!cache.access(key(1, 0), 4).hit, "A first access is a miss");
    require(!cache.access(key(1, 1), 4).hit, "B first access is a miss");
    require(cache.access(key(1, 0), 4).hit, "A second access is a hit");

    const auto insert_c = cache.access(key(1, 2), 4);
    require(!insert_c.hit, "C first access is a miss");
    require(insert_c.evicted_entries == 1 && insert_c.evicted_bytes == 4,
            "C evicts exactly one four-byte LRU entry");
    require(cache.contains(key(1, 0)), "hit-promoted A remains cached");
    require(!cache.contains(key(1, 1)), "least-recent B is evicted");
    require(cache.contains(key(1, 2)), "C is cached");

    const auto stats = cache.stats();
    require(stats.requests == 4 && stats.hits == 1 && stats.misses == 3,
            "basic access counters");
    require(stats.evictions == 1 && stats.evicted_bytes == 4,
            "basic eviction counters");
    require(stats.resident_bytes == 8 && stats.capacity_bytes == 8,
            "resident bytes stay within capacity");
}

void test_variable_sizes_can_evict_multiple_entries() {
    ggml_moe_cache_lru cache(12);
    cache.access(key(2, 0), 4);
    cache.access(key(2, 1), 4);
    cache.access(key(2, 2), 4);
    cache.access(key(2, 0), 4); // promote A; B then C are oldest

    const auto result = cache.access(key(2, 3), 8);
    require(result.evicted_entries == 2 && result.evicted_bytes == 8,
            "eight-byte insert evicts two old four-byte entries");
    require(cache.contains(key(2, 0)), "promoted A survives multi-eviction");
    require(cache.contains(key(2, 3)), "large D is inserted");
    require(!cache.contains(key(2, 1)) && !cache.contains(key(2, 2)),
            "B and C are evicted");
    require(cache.stats().resident_bytes == 12,
            "variable-size cache exactly fills its budget");
}

void test_oversized_entry_bypasses_without_destroying_hotset() {
    ggml_moe_cache_lru cache(8);
    cache.access(key(3, 0), 4);
    cache.access(key(3, 1), 4);

    const auto oversized = cache.access(key(3, 9), 9);
    require(!oversized.hit && !oversized.cached && oversized.bypassed,
            "oversized entry is a bypassed miss");
    require(oversized.evicted_entries == 0,
            "oversized bypass evicts no useful entries");
    require(cache.contains(key(3, 0)) && cache.contains(key(3, 1)),
            "hotset survives oversized bypass");
    require(cache.stats().resident_bytes == 8 && cache.stats().bypasses == 1,
            "oversized bypass keeps accounting bounded");
    require(cache.access(key(3, 0), 4).hit,
            "pre-existing entry still hits after bypass");
}

void test_capacity_check_does_not_overflow() {
    const size_t max = static_cast<size_t>(-1);
    ggml_moe_cache_lru cache(max);
    cache.access(key(5, 0), max - 4);

    const auto result = cache.access(key(5, 1), 8);
    require(result.evicted_entries == 1 && result.evicted_bytes == max - 4,
            "overflow-safe capacity check evicts the old entry");
    require(!cache.contains(key(5, 0)) && cache.contains(key(5, 1)),
            "overflow-safe insert keeps only the fitting entry");
    require(cache.stats().resident_bytes == 8,
            "overflow-safe accounting remains bounded");
}

void test_same_data_with_different_layout_does_not_alias() {
    ggml_moe_cache_lru cache(256);
    const void * data = reinterpret_cast<const void *>(static_cast<uintptr_t>(6));

    require(!cache.access({data, 64, 0}, 64).hit,
            "layout identity: first stride is a miss");
    require(!cache.access({data, 128, 0}, 128).hit,
            "layout identity: a different stride must not alias the first tensor view");
    require(cache.stats().resident_bytes == 192,
            "layout identity: both tensor-view slices are represented");
}

void test_reset_and_zero_capacity() {
    ggml_moe_cache_lru cache(4);
    cache.access(key(4, 0), 4);
    cache.reset(0);

    const auto empty = cache.stats();
    require(empty.capacity_bytes == 0 && empty.resident_bytes == 0,
            "reset applies zero capacity and clears residency");
    require(empty.requests == 0 && empty.hits == 0 && empty.misses == 0 &&
            empty.evictions == 0 && empty.bypasses == 0,
            "reset clears counters");

    const auto result = cache.access(key(4, 1), 4);
    require(!result.cached && result.bypassed,
            "zero-capacity cache bypasses every entry");
    require(cache.stats().resident_bytes == 0 && cache.stats().misses == 1,
            "zero-capacity access stays empty and records a miss");
}

} // namespace

int main() {
    try {
        test_hit_promotes_and_evicts_lru();
        test_variable_sizes_can_evict_multiple_entries();
        test_oversized_entry_bypasses_without_destroying_hotset();
        test_capacity_check_does_not_overflow();
        test_same_data_with_different_layout_does_not_alias();
        test_reset_and_zero_capacity();
        std::cout << "expert cache LRU tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-expert-cache-lru: " << error.what() << '\n';
        return 1;
    }
}
