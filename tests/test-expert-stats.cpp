#include "llama-expert-stats.h"

#include <nlohmann/json.hpp>

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

void require_advisory(const llama_expert_stats_advisory & advisory, bool thrashing,
        bool prefetch_misjudging, bool io_bound, const std::string & label) {
    require(advisory.thrashing == thrashing,
            label + ": expected thrashing=" + (thrashing ? "true" : "false"));
    require(advisory.prefetch_misjudging == prefetch_misjudging,
            label + ": expected prefetch_misjudging=" + (prefetch_misjudging ? "true" : "false"));
    require(advisory.io_bound == io_bound,
            label + ": expected io_bound=" + (io_bound ? "true" : "false"));
}

// All activations served from resident experts: 100% hit rate, no advisories.
void test_all_resident() {
    llama_expert_store_stats stats;
    stats.capacity_bytes  = 1024;
    stats.resident_bytes  = 512;
    for (int i = 0; i < 100; ++i) {
        stats.record_hit(3);
    }

    require(stats.total_requests() == 100, "all-resident: total requests");
    require(stats.total_hits() == 100, "all-resident: total hits");
    require(stats.total_misses() == 0, "all-resident: total misses");
    require(stats.total_prefetched() == 0, "all-resident: total prefetched");
    require(stats.total_prefetch_hits() == 0, "all-resident: total prefetch hits");

    const auto advisory = llama_expert_stats_evaluate(stats);
    require_advisory(advisory, false, false, false, "all-resident");
    require(advisory.thrashing_layers.empty(), "all-resident: no thrashing layers");
    require(advisory.miss_rate == 0.0, "all-resident: miss rate");
}

// All activations miss residency: thrashing advisory must fire.
void test_all_miss() {
    llama_expert_store_stats stats;
    for (int i = 0; i < 100; ++i) {
        stats.record_miss(5);
    }

    require(stats.total_requests() == 100, "all-miss: total requests");
    require(stats.total_hits() == 0, "all-miss: total hits");
    require(stats.total_misses() == 100, "all-miss: total misses");

    const auto advisory = llama_expert_stats_evaluate(stats);
    require_advisory(advisory, true, false, false, "all-miss");
    require(advisory.thrashing_layers.size() == 1 && advisory.thrashing_layers[0] == 5,
            "all-miss: layer 5 reported as thrashing");
}

// Prefetch-assisted misses: advisory only fires below the 80% in-time ratio.
void test_prefetch_assisted() {
    {
        llama_expert_store_stats stats;
        for (int i = 0; i < 100; ++i) {
            stats.record_miss(0);
        }
        for (int i = 0; i < 100; ++i) {
            stats.record_prefetch(0, i < 85);
        }
        const auto advisory = llama_expert_stats_evaluate(stats);
        require_advisory(advisory, true, false, false, "prefetch 85% in time");
    }
    {
        llama_expert_store_stats stats;
        for (int i = 0; i < 100; ++i) {
            stats.record_miss(0);
        }
        for (int i = 0; i < 100; ++i) {
            stats.record_prefetch(0, i < 79);
        }
        const auto advisory = llama_expert_stats_evaluate(stats);
        require_advisory(advisory, true, true, false, "prefetch 79% in time");
    }
    {
        // Exactly at the 80% boundary: report, don't alarm.
        llama_expert_store_stats stats;
        for (int i = 0; i < 10; ++i) {
            stats.record_miss(0);
        }
        for (int i = 0; i < 10; ++i) {
            stats.record_prefetch(0, i < 8);
        }
        const auto advisory = llama_expert_stats_evaluate(stats);
        require_advisory(advisory, true, false, false, "prefetch exactly 80% in time");
    }
    {
        // No prefetch activity at all: the prefetch advisory must stay silent.
        llama_expert_store_stats stats;
        for (int i = 0; i < 10; ++i) {
            stats.record_miss(0);
        }
        const auto advisory = llama_expert_stats_evaluate(stats);
        require_advisory(advisory, true, false, false, "no prefetch activity");
    }
}

// Miss-rate boundary: exactly 70% misses is not thrashing (advisory is strict).
void test_thrashing_boundary() {
    llama_expert_store_stats stats;
    for (int i = 0; i < 30; ++i) {
        stats.record_hit(1);
    }
    for (int i = 0; i < 70; ++i) {
        stats.record_miss(1);
    }
    const auto advisory = llama_expert_stats_evaluate(stats);
    require_advisory(advisory, false, false, false, "miss rate exactly 70%");
}

// Defer-wait ratio above 40% of decode time flags I/O-bound.
void test_io_bound() {
    {
        llama_expert_store_stats stats;
        stats.add_defer_wait(5'000'000'000ULL);
        stats.add_decode_time(10'000'000'000ULL);
        const auto advisory = llama_expert_stats_evaluate(stats);
        require_advisory(advisory, false, false, true, "defer wait 50% of decode");
    }
    {
        llama_expert_store_stats stats;
        stats.add_defer_wait(4'000'000'000ULL);
        stats.add_decode_time(10'000'000'000ULL);
        const auto advisory = llama_expert_stats_evaluate(stats);
        require_advisory(advisory, false, false, false, "defer wait exactly 40% of decode");
    }
    {
        llama_expert_store_stats stats; // no decode time recorded yet
        stats.add_defer_wait(1'000'000'000ULL);
        const auto advisory = llama_expert_stats_evaluate(stats);
        require_advisory(advisory, false, false, false, "no decode time recorded");
    }
}

// Mixed multi-layer pattern: totals aggregate across layers, per-layer rates hold.
void test_totals_and_rates() {
    llama_expert_store_stats stats;
    for (int i = 0; i < 60; ++i) {
        stats.record_hit(0);
    }
    for (int i = 0; i < 40; ++i) {
        stats.record_miss(0);
    }
    for (int i = 0; i < 90; ++i) {
        stats.record_hit(2);
    }
    for (int i = 0; i < 10; ++i) {
        stats.record_miss(2);
    }

    require(stats.layers.size() == 2, "totals: two layers tracked");
    require(stats.total_requests() == 200, "totals: requests");
    require(stats.total_hits() == 150, "totals: hits");
    require(stats.total_misses() == 50, "totals: misses");

    const auto advisory = llama_expert_stats_evaluate(stats);
    require(advisory.miss_rate > 0.24 && advisory.miss_rate < 0.26,
            "totals: aggregate miss rate should be 25%");
    require(advisory.thrashing_layers.empty(), "totals: no layer below 30% hit rate");
}

// Empty stats must evaluate safely with no advisories and zero rates.
void test_empty() {
    llama_expert_store_stats stats;
    const auto advisory = llama_expert_stats_evaluate(stats);
    require_advisory(advisory, false, false, false, "empty stats");
    require(advisory.miss_rate == 0.0, "empty: miss rate");
    require(advisory.prefetch_hit_rate == 0.0, "empty: prefetch hit rate");
    require(advisory.defer_wait_ratio == 0.0, "empty: defer wait ratio");
}

// JSON dump matches the documented /metrics-style shape and values.
void test_json_dump() {
    llama_expert_store_stats stats;
    stats.capacity_bytes = 4096;
    stats.resident_bytes = 2048;
    for (int i = 0; i < 3; ++i) {
        stats.record_hit(4);
    }
    for (int i = 0; i < 1; ++i) {
        stats.record_miss(4);
    }
    stats.record_prefetch(4, true);
    stats.add_defer_wait(1'000'000'000ULL);
    stats.add_decode_time(10'000'000'000ULL);

    const auto advisory = llama_expert_stats_evaluate(stats);
    const std::string text = llama_expert_stats_to_json(stats, advisory);

    const auto root = nlohmann::json::parse(text);
    require(root.is_object(), "json: root must be an object");

    require(root.at("totals").at("requests") == 4, "json: requests");
    require(root.at("totals").at("hits") == 3, "json: hits");
    require(root.at("totals").at("misses") == 1, "json: misses");
    require(root.at("totals").at("prefetched") == 1, "json: prefetched");
    require(root.at("totals").at("prefetch_hits") == 1, "json: prefetch_hits");

    require(root.at("budget").at("resident_bytes") == 2048, "json: resident_bytes");
    require(root.at("budget").at("capacity_bytes") == 4096, "json: capacity_bytes");

    require(root.at("advisories").is_array() && root.at("advisories").empty(),
            "json: no advisories for this healthy pattern");

    const auto & layer = root.at("layers").at("4");
    require(layer.at("requests") == 4, "json: layer requests");
    require(layer.at("hits") == 3, "json: layer hits");
    require(layer.at("misses") == 1, "json: layer misses");
    require(layer.at("prefetched") == 1, "json: layer prefetched");
    require(layer.at("prefetch_hits") == 1, "json: layer prefetch_hits");

    // Advisories serialize as strings when they fire.
    llama_expert_store_stats thrashing;
    for (int i = 0; i < 100; ++i) {
        thrashing.record_miss(7);
    }
    const auto bad = llama_expert_stats_evaluate(thrashing);
    const auto bad_root = nlohmann::json::parse(llama_expert_stats_to_json(thrashing, bad));
    require(bad_root.at("advisories").size() == 1, "json: one advisory expected");
    require(bad_root.at("advisories").at(0) == "thrashing", "json: advisory name");
}

// reset() returns the store to a pristine state (used between benchmark runs).
void test_reset() {
    llama_expert_store_stats stats;
    stats.capacity_bytes = 4096;
    stats.resident_bytes = 2048;
    stats.record_hit(0);
    stats.record_miss(0);
    stats.record_prefetch(0, true);
    stats.add_defer_wait(1);
    stats.add_decode_time(1);

    stats.reset();
    require(stats.layers.empty(), "reset: layers cleared");
    require(stats.resident_bytes == 0, "reset: resident_bytes cleared");
    require(stats.capacity_bytes == 0, "reset: capacity_bytes cleared");
    require(stats.defer_wait_ns == 0, "reset: defer_wait_ns cleared");
    require(stats.decode_ns == 0, "reset: decode_ns cleared");
    require(stats.total_requests() == 0, "reset: totals cleared");
}

} // namespace

int main() {
    try {
        test_all_resident();
        test_all_miss();
        test_prefetch_assisted();
        test_thrashing_boundary();
        test_io_bound();
        test_totals_and_rates();
        test_empty();
        test_json_dump();
        test_reset();
        std::cout << "expert stats tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-expert-stats: " << error.what() << '\n';
        return 1;
    }
}
