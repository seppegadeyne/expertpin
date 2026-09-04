#include "llama-expert-stats.h"
#include "ggml-moe-stats.h"

#include <nlohmann/json.hpp>

void llama_expert_store_stats::record_hit(uint32_t layer) {
    auto & entry = layers[layer];
    ++entry.requests;
    ++entry.hits;
}

void llama_expert_store_stats::record_miss(uint32_t layer) {
    auto & entry = layers[layer];
    ++entry.requests;
    ++entry.misses;
}

void llama_expert_store_stats::record_prefetch(uint32_t layer, bool arrived_in_time) {
    auto & entry = layers[layer];
    ++entry.prefetched;
    if (arrived_in_time) {
        ++entry.prefetch_hits;
    }
}

uint64_t llama_expert_store_stats::total_requests() const {
    uint64_t total = 0;
    for (const auto & entry : layers) {
        total += entry.second.requests;
    }
    return total;
}

uint64_t llama_expert_store_stats::total_hits() const {
    uint64_t total = 0;
    for (const auto & entry : layers) {
        total += entry.second.hits;
    }
    return total;
}

uint64_t llama_expert_store_stats::total_misses() const {
    uint64_t total = 0;
    for (const auto & entry : layers) {
        total += entry.second.misses;
    }
    return total;
}

uint64_t llama_expert_store_stats::total_prefetched() const {
    uint64_t total = 0;
    for (const auto & entry : layers) {
        total += entry.second.prefetched;
    }
    return total;
}

uint64_t llama_expert_store_stats::total_prefetch_hits() const {
    uint64_t total = 0;
    for (const auto & entry : layers) {
        total += entry.second.prefetch_hits;
    }
    return total;
}

void llama_expert_store_stats::reset() {
    layers.clear();
    resident_bytes = 0;
    capacity_bytes = 0;
    defer_wait_ns  = 0;
    decode_ns      = 0;
}

llama_expert_stats_advisory llama_expert_stats_evaluate(const llama_expert_store_stats & stats) {
    llama_expert_stats_advisory advisory;

    const uint64_t requests = stats.total_requests();
    const uint64_t misses   = stats.total_misses();

    advisory.miss_rate = requests > 0
            ? static_cast<double>(misses) / static_cast<double>(requests)
            : 0.0;

    // Thrashing: strictly more than 70% misses over all expert activations.
    advisory.thrashing = advisory.miss_rate > 0.70;

    for (const auto & entry : stats.layers) {
        const llama_expert_layer_stats & layer = entry.second;
        if (layer.requests == 0) {
            continue;
        }
        const double layer_miss_rate =
                static_cast<double>(layer.misses) / static_cast<double>(layer.requests);
        if (layer_miss_rate > 0.70) {
            advisory.thrashing_layers.push_back(entry.first);
        }
    }

    const uint64_t prefetched     = stats.total_prefetched();
    const uint64_t prefetch_hits  = stats.total_prefetch_hits();

    advisory.prefetch_hit_rate = prefetched > 0
            ? static_cast<double>(prefetch_hits) / static_cast<double>(prefetched)
            : 0.0;

    // Prefetch misjudging: strictly below 80% of prefetched experts arriving
    // in time. Silent when no prefetch activity was recorded.
    advisory.prefetch_misjudging = prefetched > 0 && advisory.prefetch_hit_rate < 0.80;

    advisory.defer_wait_ratio = stats.decode_ns > 0
            ? static_cast<double>(stats.defer_wait_ns) / static_cast<double>(stats.decode_ns)
            : 0.0;

    // I/O-bound: strictly more than 40% of decode time spent waiting on
    // deferred expert data. Silent until decode time has been recorded.
    advisory.io_bound = stats.decode_ns > 0 && advisory.defer_wait_ratio > 0.40;

    return advisory;
}

std::string llama_expert_stats_to_json(
        const llama_expert_store_stats & stats,
        const llama_expert_stats_advisory & advisory) {
    using json = nlohmann::json;

    json root;
    root["totals"] = {
        {"requests",      stats.total_requests()},
        {"hits",          stats.total_hits()},
        {"misses",        stats.total_misses()},
        {"prefetched",    stats.total_prefetched()},
        {"prefetch_hits", stats.total_prefetch_hits()},
    };
    root["budget"] = {
        {"resident_bytes", stats.resident_bytes},
        {"capacity_bytes", stats.capacity_bytes},
    };

    json advisories = json::array();
    if (advisory.thrashing) {
        advisories.push_back("thrashing");
    }
    if (advisory.prefetch_misjudging) {
        advisories.push_back("prefetch_misjudging");
    }
    if (advisory.io_bound) {
        advisories.push_back("io_bound");
    }
    root["advisories"] = advisories;

    json layers = json::object();
    for (const auto & entry : stats.layers) {
        layers[std::to_string(entry.first)] = {
            {"requests",      entry.second.requests},
            {"hits",          entry.second.hits},
            {"misses",        entry.second.misses},
            {"prefetched",    entry.second.prefetched},
            {"prefetch_hits", entry.second.prefetch_hits},
        };
    }
    root["layers"] = layers;

    return root.dump();
}

std::string llama_moe_prefetch_stats_to_json(const ggml_moe_prefetch_stats & stats) {
    using json = nlohmann::json;

    json root = {
        {"requests", stats.requests},
        {"hits", stats.hits},
        {"misses", stats.misses},
        {"prefetched", stats.prefetched},
        {"prefetch_hits", stats.prefetch_hits},
        {"defer_wait_ns", stats.defer_wait_ns},
        {"resident_bytes", stats.resident_bytes},
        {"capacity_bytes", stats.capacity_bytes},
        {"hit_rate", stats.requests ? static_cast<double>(stats.hits) / stats.requests : 0.0},
        {"prefetch_hit_rate", stats.prefetched ? static_cast<double>(stats.prefetch_hits) / stats.prefetched : 0.0},
        {"defer_wait_ratio", stats.resident_bytes ? static_cast<double>(stats.defer_wait_ns) / 1e9 : 0.0},
    };
    root["cache_sim"] = {
        {"requests", stats.cache_sim_requests},
        {"hits", stats.cache_sim_hits},
        {"misses", stats.cache_sim_misses},
        {"evictions", stats.cache_sim_evictions},
        {"evicted_bytes", stats.cache_sim_evicted_bytes},
        {"bypasses", stats.cache_sim_bypasses},
        {"resident_bytes", stats.cache_sim_resident_bytes},
        {"capacity_bytes", stats.cache_sim_capacity_bytes},
        {"hit_rate", stats.cache_sim_requests
                ? static_cast<double>(stats.cache_sim_hits) / stats.cache_sim_requests
                : 0.0},
    };
    return root.dump(2) + "\n";
}
