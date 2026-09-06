#!/usr/bin/env python3
"""Verify A/B gates for the cache-shadow experiment (R5, review 2026-09-05).

Gates (fail -> exit 1, details on stdout):
  1. output identity: baseline vs shadow samples must produce identical content
  2. shadow hit-rate >= HIT_RATE_GATE (count-based; byte-based unknown)
  3. shadow decode overhead vs baseline <= OVERHEAD_GATE_PCT (median of per-token ms)
Usage: verify-ab-gates.py RUN_DIR
"""
import json
import math
import sys
from pathlib import Path


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


try:
    if not 2 <= len(sys.argv) <= 4:
        raise ValueError("usage: verify-ab-gates.py RUN_DIR [HIT_RATE [OVERHEAD_PCT]]")
    run_dir = Path(sys.argv[1])
    hit_gate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    overhead_gate_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    if not finite_number(hit_gate) or not 0 <= hit_gate <= 1:
        raise ValueError("hit-rate gate must be finite and in [0, 1]")
    if not finite_number(overhead_gate_pct) or overhead_gate_pct < 0:
        raise ValueError("overhead gate must be finite and nonnegative")
except ValueError as exc:
    print("GATES_FAILED:\n  - %s" % exc)
    sys.exit(1)

failures = []


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def read_object(path):
    try:
        value = json.loads(path.read_text(), object_pairs_hook=unique_object)
    except RecursionError as exc:
        raise ValueError("excessively nested JSON in %s" % path) from exc
    if not isinstance(value, dict):
        raise ValueError("expected JSON object in %s" % path)
    return value


# Read once and validate every sample before computing a median. Otherwise
# a missing/invalid timing can be silently skipped or hidden by other samples.
samples = {}
try:
    for arm in ("shadow-off", "shadow-32g"):
        samples[arm] = []
        for i in (1, 2, 3):
            path = run_dir / arm / ("sample-%d.json" % i)
            sample = read_object(path)
            timing = sample.get("timings")
            if not isinstance(sample.get("content"), str):
                raise ValueError("expected string content in %s" % path)
            if not isinstance(timing, dict):
                raise ValueError("expected timings object in %s" % path)
            n, ms = timing.get("predicted_n"), timing.get("predicted_ms")
            if type(n) is not int or n <= 0:
                raise ValueError("expected positive integer predicted_n in %s" % path)
            if not finite_number(ms) or ms <= 0:
                raise ValueError("expected positive finite predicted_ms in %s" % path)
            per_token = ms / n
            if not finite_number(per_token) or per_token <= 0:
                raise ValueError("invalid per-token timing in %s" % path)
            samples[arm].append(sample)
    stats = read_object(run_dir / "shadow-32g" / "expert-stats.json")
    cs = stats.get("cache_sim")
    if not isinstance(cs, dict):
        raise ValueError("expected cache_sim object")
    reqs, hr = cs.get("requests"), cs.get("hit_rate")
    if type(reqs) is not int or reqs <= 0:
        raise ValueError("cache_sim requests must be a positive integer")
    if not finite_number(hr) or not 0 <= hr <= 1:
        raise ValueError("cache_sim hit_rate must be finite and in [0, 1]")
except (OSError, ValueError, OverflowError) as exc:
    print("GATES_FAILED:\n  - %s" % exc)
    sys.exit(1)


def arm_text(arm):
    return [sample["content"] for sample in samples[arm]]


def med_ms(arm):
    vals = sorted(sample["timings"]["predicted_ms"] / sample["timings"]["predicted_n"]
                  for sample in samples[arm])
    return vals[1]  # Exactly three validated samples per arm.


base_texts = arm_text("shadow-off")
shad_texts = arm_text("shadow-32g")
if base_texts != shad_texts:
    failures.append("output mismatch: shadow content differs from baseline")

print("cache_sim requests=%s hit_rate=%s evictions=%s resident=%s cap=%s"
      % (reqs, hr, cs.get("evictions"), cs.get("resident_bytes"), cs.get("capacity_bytes")))
if hr < hit_gate:
    failures.append("hit_rate %s < gate %s" % (hr, hit_gate))

off_ms = med_ms("shadow-off")
on_ms = med_ms("shadow-32g")
print("median per-token ms: shadow-off=%s shadow-32g=%s" % (off_ms, on_ms))
if off_ms and on_ms:
    overhead_pct = 100.0 * (on_ms / off_ms - 1.0)
    print("shadow overhead = %.2f%% (gate %.1f%%)" % (overhead_pct, overhead_gate_pct))
    if not finite_number(overhead_pct):
        failures.append("non-finite computed overhead")
    elif overhead_pct > overhead_gate_pct:
        failures.append("overhead %.2f%% > gate %.1f%%" % (overhead_pct, overhead_gate_pct))
else:
    failures.append("missing timing data for overhead gate")

if failures:
    print("GATES_FAILED:")
    for f in failures:
        print("  - %s" % f)
    sys.exit(1)
print("GATES_PASSED")
