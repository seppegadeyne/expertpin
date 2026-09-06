#!/usr/bin/env python3
"""Bounded-cache feasibility from aggregate shadow stats; never reconstruct routing.

No model load, no payload allocation, and no throughput/I/O prediction. Exact
fitting-cap results are conditional on a fresh cache and immutable key/size pairs.
Smaller-cap results deliberately remain intervals: ordering was not persisted.
"""
import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path

GIB = 1024 ** 3
COUNTERS = ("requests", "hits", "misses", "evictions", "evicted_bytes",
            "bypasses", "resident_bytes", "capacity_bytes")


def validate_stats(document):
    if not isinstance(document, dict) or not isinstance(document.get("cache_sim"), dict):
        raise ValueError("expected a cache_sim object")
    stats = document["cache_sim"]
    for name in COUNTERS:
        if type(stats.get(name)) is not int or stats[name] < 0:
            raise ValueError(f"cache_sim.{name} must be a nonnegative integer")
    if stats["requests"] == 0 or stats["capacity_bytes"] == 0:
        raise ValueError("need a nonempty, enabled shadow observation")
    if stats["requests"] != stats["hits"] + stats["misses"]:
        raise ValueError("requests must equal hits + misses")
    if stats["resident_bytes"] > stats["capacity_bytes"]:
        raise ValueError("resident bytes exceed capacity")
    if stats["bypasses"] > stats["misses"] or stats["evictions"] > stats["misses"] - stats["bypasses"]:
        raise ValueError("more bypasses/evictions than inserted misses")
    if bool(stats["evictions"]) != bool(stats["evicted_bytes"]):
        raise ValueError("eviction count and bytes disagree")
    if stats["evicted_bytes"] < stats["evictions"]:
        raise ValueError("evicted bytes must cover at least one byte per evicted entry")
    rate = stats.get("hit_rate")
    if type(rate) not in (int, float) or not math.isfinite(rate) or not math.isclose(
            rate, stats["hits"] / stats["requests"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("hit_rate must agree with count-based counters")
    return stats


def analyze(document, capacities, *, stable_cold_start=False):
    stats = validate_stats(document)
    if not capacities or any(type(c) is not int or c <= 0 for c in capacities):
        raise ValueError("capacities must be positive integer byte counts")
    if len(set(capacities)) != len(capacities):
        raise ValueError("capacities must be distinct")
    # The runtime may invalidate a key when its byte size changes without
    # recording an eviction. Therefore zero evictions alone is insufficient.
    proven = bool(stable_cold_start and stats["evictions"] == 0 and stats["bypasses"] == 0)
    if proven and (stats["misses"] == 0 or stats["resident_bytes"] < stats["misses"]):
        raise ValueError("fresh stable no-bypass observation needs positive bytes per unique miss")
    if (stable_cold_start and stats["evictions"] > 0 and
            stats["resident_bytes"] + stats["evicted_bytes"] <= stats["capacity_bytes"]):
        raise ValueError("stable cold-start evictions require inserted bytes exceeding capacity")
    working_set = stats["resident_bytes"] if proven else None
    results = []
    for cap in capacities:
        fits = proven and cap >= working_set
        results.append({
            "capacity_bytes": cap,
            "basis": "conditional_exact_fitting_replay" if fits else "ordering_unknown",
            "hits": stats["hits"] if fits else None,
            "misses_lower_bound": stats["misses"] if proven else None,
            "misses_upper_bound": (stats["misses"] if fits else stats["requests"]) if proven else None,
            "logical_working_set_excess_bytes": max(0, working_set - cap) if proven else None,
            "evictions": 0 if fits else None,
            "process_ram_peak_bytes": None,
        })
    return {
        "kind": "aggregate_cache_feasibility_not_model_replay",
        "assumptions": {
            "fresh_cache_and_immutable_key_sizes_asserted": bool(stable_cold_start),
            "scope": "same observed CPU slice stream only; not future prompts or decode-only",
        },
        "observed": {
            "cache_sim": {key: stats[key] for key in COUNTERS},
            "count_hit_rate": stats["hits"] / stats["requests"],
            "unique_slices": stats["misses"] if proven else None,
            "logical_working_set_bytes": working_set,
            "mean_unique_slice_bytes": working_set / stats["misses"] if proven else None,
        },
        "capacities": results,
        "nvme_reload_bytes": None,
        "nvme_reload_ms": None,
        "tok_s": None,
        "vram_peak_bytes": None,
        "limitations": [
            "aggregate counters contain neither expert identities nor access order",
            "count hits do not determine byte hits for unequal slice sizes",
            "shadow misses/evictions are not disk reads or physical page evictions",
            "logical payload capacity excludes allocator, metadata, KV, PLE, scratch and other RAM",
            "a 40-GiB cgroup and measured 28-GiB absolute VRAM remain separate hard gates",
        ],
    }


def replay_example(trace, entries):
    """Tiny equal-sized SYNTHETIC counterexample; not the production policy."""
    cache = OrderedDict()
    hits = evictions = 0
    for key in trace:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            if len(cache) == entries:
                cache.popitem(last=False)
                evictions += 1
            cache[key] = None
    return dict(requests=len(trace), hits=hits, misses=len(trace)-hits, evictions=evictions)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    # Strict decoding: do not let duplicate keys silently replace measurements.
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--capacity-gib", nargs="+", type=int, default=[28, 30, 32, 34])
    parser.add_argument("--assert-stable-cold-start", action="store_true",
                        help="assert fresh cache, fixed tensor identities/strides/slice sizes for this observation")
    args = parser.parse_args()
    try:
        result = analyze(load_json(args.stats), [cap * GIB for cap in args.capacity_gib],
                         stable_cold_start=args.assert_stable_cold_start)
        result["source_stats"] = str(args.stats)
        traces = [["a", "a", "b", "b", "c", "c", "d", "d"],
                  ["a", "b", "c", "d", "a", "b", "c", "d"]]
        result["synthetic_order_counterexample"] = [
            {"trace": trace, "fitting_four_entries": replay_example(trace, 4),
             "smaller_two_entries": replay_example(trace, 2)} for trace in traces]
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, RecursionError) as error:
        parser.exit(2, f"cache analysis: {error}\n")


if __name__ == "__main__":
    main()
