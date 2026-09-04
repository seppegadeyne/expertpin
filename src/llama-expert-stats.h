#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ggml_moe_prefetch_stats;

// Per-layer counters following the colibri `ColiExpertStoreStats` contract
// (requests / hits / misses / prefetched / prefetch_hits per layer), plus the
// residency budget and defer/decode timing used by the advisory thresholds.
struct llama_expert_layer_stats {
    uint64_t requests      = 0;
    uint64_t hits          = 0;
    uint64_t misses        = 0;
    uint64_t prefetched    = 0;
    uint64_t prefetch_hits = 0;
};

struct llama_expert_store_stats {
    std::map<uint32_t, llama_expert_layer_stats> layers;

    size_t resident_bytes = 0;
    size_t capacity_bytes = 0;

    uint64_t defer_wait_ns = 0;
    uint64_t decode_ns     = 0;

    void record_hit(uint32_t layer);
    void record_miss(uint32_t layer);
    // A prefetched expert that arrives before its first use counts as a
    // prefetch hit; late arrivals mean the prefetch guessed wrong layers.
    void record_prefetch(uint32_t layer, bool arrived_in_time);

    void add_defer_wait(uint64_t ns) { defer_wait_ns += ns; }
    void add_decode_time(uint64_t ns) { decode_ns += ns; }

    uint64_t total_requests() const;
    uint64_t total_hits() const;
    uint64_t total_misses() const;
    uint64_t total_prefetched() const;
    uint64_t total_prefetch_hits() const;

    void reset();
};

// Advisory evaluation (report, don't gate). Thresholds from issue #1:
//   miss-rate > 70% (hit < 30%)            -> thrashing
//   prefetch_hits / prefetched < 80%       -> prefetch guessing wrong
//   defer_wait / decode_time > 40%         -> I/O-bound
struct llama_expert_stats_advisory {
    bool thrashing            = false;
    bool prefetch_misjudging  = false;
    bool io_bound             = false;

    double miss_rate          = 0.0;
    double prefetch_hit_rate  = 0.0;
    double defer_wait_ratio   = 0.0;

    std::vector<uint32_t> thrashing_layers;
};

llama_expert_stats_advisory llama_expert_stats_evaluate(const llama_expert_store_stats & stats);

// /metrics-style JSON dump: totals, budget, advisories, per-layer counters.
std::string llama_expert_stats_to_json(
        const llama_expert_store_stats & stats,
        const llama_expert_stats_advisory & advisory);

// Serialize the live ggml prefetch counters, including the advisory bounded-LRU
// shadow, for --expert-stats-file.
std::string llama_moe_prefetch_stats_to_json(const ggml_moe_prefetch_stats & stats);
