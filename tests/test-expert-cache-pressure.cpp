#include "ggml-moe-cache-lru.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

// Synthetic logical slices only: never allocate/read an expert payload.
// This exercises the production policy, NOT model routing or NVMe reloads.
namespace {
constexpr size_t mib = size_t{1024} * 1024;
constexpr uint32_t entries = 24576; // 24 GiB logical working set
constexpr uint64_t sweeps = 3;
constexpr uint64_t requests = uint64_t{entries} * sweeps * 2;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool same(const ggml_moe_cache_stats & a, const ggml_moe_cache_stats & b) {
    return a.capacity_bytes == b.capacity_bytes && a.resident_bytes == b.resident_bytes &&
           a.requests == b.requests && a.hits == b.hits && a.misses == b.misses &&
           a.evictions == b.evictions && a.evicted_bytes == b.evicted_bytes &&
           a.bypasses == b.bypasses;
}

ggml_moe_cache_stats replay(size_t cap_mib, bool enabled, const char * order, int arm) {
    ggml_moe_cache_lru cache(cap_mib * mib);
    const int tensor = 0; // stable identity; never dereferenced by the policy
    uint64_t visited = 0;
    uint64_t rereference_misses = 0;
    for (uint64_t sweep = 0; sweep < sweeps; ++sweep) {
        for (uint32_t expert = 0; expert < entries; ++expert) {
            const ggml_moe_cache_key key{&tensor, mib, expert};
            for (int repeat = 0; repeat < 2; ++repeat) {
                ++visited;
                if (!enabled) {
                    continue; // disabled observer is not a zero-capacity observer
                }
                const auto result = cache.access(key, mib);
                require(result.cached && !result.bypassed, "fitting slice must be cached");
                const bool expected_hit = repeat == 1 || (sweep > 0 && cap_mib >= entries);
                require(result.hit == expected_hit, "closed-form cyclic trace hit mismatch");
                if (sweep > 0 && repeat == 0 && !result.hit) {
                    ++rereference_misses;
                }
                const auto & s = cache.stats();
                require(s.resident_bytes <= s.capacity_bytes, "capacity exceeded during replay");
                require(s.requests == s.hits + s.misses, "request accounting mismatch");
                require(s.misses * mib == s.resident_bytes + s.evicted_bytes,
                        "inserted bytes must equal resident plus evicted bytes");
            }
        }
    }
    require(visited == requests, "all arms must traverse the same trace");
    const auto stats = cache.stats();
    if (enabled) {
        const bool pressure = cap_mib < entries;
        const uint64_t misses = pressure ? uint64_t{entries} * sweeps : entries;
        const uint64_t resident_entries = pressure ? cap_mib : entries;
        require(stats.requests == requests && stats.misses == misses &&
                stats.hits == requests - misses, "final access counters mismatch");
        require(stats.resident_bytes == resident_entries * mib &&
                stats.evictions == misses - resident_entries &&
                stats.evicted_bytes == (misses - resident_entries) * mib &&
                stats.bypasses == 0, "final eviction counters mismatch");
        require(rereference_misses == (pressure ? uint64_t{entries} * (sweeps - 1) : 0),
                "must observe capacity-induced misses after the first sweep");
        require(pressure ? stats.evictions > 0 : stats.evictions == 0,
                "small caps must evict; the fitting control must not");
    } else {
        ggml_moe_cache_stats empty{};
        empty.capacity_bytes = cap_mib * mib;
        require(same(stats, empty), "disabled observer must leave every counter empty");
    }
    std::cout << "{\"kind\":\"synthetic_policy_replay\",\"order\":\"" << order
              << "\",\"arm\":" << arm << ",\"enabled\":" << (enabled ? "true" : "false")
              << ",\"logical_working_set_bytes\":" << uint64_t{entries} * mib
              << ",\"capacity_bytes\":" << stats.capacity_bytes
              << ",\"trace_requests\":" << visited
              << ",\"requests\":" << stats.requests << ",\"hits\":" << stats.hits
              << ",\"misses\":" << stats.misses << ",\"evictions\":" << stats.evictions
              << ",\"evicted_bytes\":" << stats.evicted_bytes
              << ",\"resident_bytes\":" << stats.resident_bytes
              << ",\"bypasses\":" << stats.bypasses
              << ",\"rereference_misses\":" << rereference_misses
              << ",\"nvme_reload_bytes\":null,\"nvme_reload_ms\":null,\"tok_s\":null}\n";
    return stats;
}
} // namespace

int main() {
    try {
        for (size_t cap_mib : {size_t{8192}, size_t{16384}, size_t{32768}}) {
            const auto on_a = replay(cap_mib, true, "on/off/on", 1);
            const auto off_a = replay(cap_mib, false, "on/off/on", 2);
            const auto on_b = replay(cap_mib, true, "on/off/on", 3);
            const auto off_b = replay(cap_mib, false, "off/on/off", 1);
            const auto on_c = replay(cap_mib, true, "off/on/off", 2);
            const auto off_c = replay(cap_mib, false, "off/on/off", 3);
            require(same(on_a, on_b) && same(on_b, on_c), "enabled replay must be repeatable");
            require(same(off_a, off_b) && same(off_b, off_c), "disabled replay must be repeatable");
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-expert-cache-pressure: " << error.what() << "\n";
        return 1;
    }
}
