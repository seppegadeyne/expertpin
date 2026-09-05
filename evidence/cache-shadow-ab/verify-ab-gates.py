#!/usr/bin/env python3
"""Verify A/B gates for the cache-shadow experiment (R5, review 2026-09-05).

Gates (fail -> exit 1, details on stdout):
  1. output identity: baseline vs shadow samples must produce identical content
  2. shadow hit-rate >= HIT_RATE_GATE (count-based; byte-based unknown)
  3. shadow decode overhead vs baseline <= OVERHEAD_GATE_PCT (median of per-token ms)
Usage: verify-ab-gates.py RUN_DIR
"""
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
hit_gate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
overhead_gate_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

failures = []


def arm_text(arm):
    texts = []
    for i in (1, 2, 3):
        f = run_dir / arm / ("sample-%d.json" % i)
        if not f.is_file():
            failures.append("missing %s" % f)
            continue
        texts.append(json.loads(f.read_text())["content"])
    return texts


def med_ms(arm):
    vals = []
    for i in (1, 2, 3):
        f = run_dir / arm / ("sample-%d.json" % i)
        if not f.is_file():
            continue
        t = json.loads(f.read_text())["timings"]
        if t.get("predicted_n"):
            vals.append(t["predicted_ms"] / t["predicted_n"])
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


base_texts = arm_text("shadow-off")
shad_texts = arm_text("shadow-32g")
if len(base_texts) == 3 and len(shad_texts) == 3:
    if base_texts != shad_texts:
        failures.append("output mismatch: shadow content differs from baseline")

stats_f = run_dir / "shadow-32g" / "expert-stats.json"
if not stats_f.is_file():
    failures.append("missing %s" % stats_f)
else:
    stats = json.loads(stats_f.read_text())
    cs = stats.get("cache_sim", {})
    hr = cs.get("hit_rate")
    reqs = cs.get("requests", 0)
    print("cache_sim requests=%s hit_rate=%s evictions=%s resident=%s cap=%s"
          % (reqs, hr, cs.get("evictions"), cs.get("resident_bytes"), cs.get("capacity_bytes")))
    if not reqs:
        failures.append("cache_sim saw no requests")
    elif hr is None or hr < hit_gate:
        failures.append("hit_rate %s < gate %s" % (hr, hit_gate))

off_ms = med_ms("shadow-off")
on_ms = med_ms("shadow-32g")
print("median per-token ms: shadow-off=%s shadow-32g=%s" % (off_ms, on_ms))
if off_ms and on_ms:
    overhead_pct = 100.0 * (on_ms / off_ms - 1.0)
    print("shadow overhead = %.2f%% (gate %.1f%%)" % (overhead_pct, overhead_gate_pct))
    if overhead_pct > overhead_gate_pct:
        failures.append("overhead %.2f%% > gate %.1f%%" % (overhead_pct, overhead_gate_pct))
else:
    failures.append("missing timing data for overhead gate")

if failures:
    print("GATES_FAILED:")
    for f in failures:
        print("  - %s" % f)
    sys.exit(1)
print("GATES_PASSED")
