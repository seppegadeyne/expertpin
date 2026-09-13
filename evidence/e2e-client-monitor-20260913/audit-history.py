#!/usr/bin/env python3
"""Audit the historical E2E sampling gap without running a model."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'hermes-e2e-20260911/run-20260912T035703-e2e/samples.jsonl'
points = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
a, b = max(zip(points, points[1:]), key=lambda pair: pair[1]['elapsed_s'] - pair[0]['elapsed_s'])
result = {
    'source': str(source.relative_to(ROOT.parent)),
    'samples': len(points),
    'longest_gap_seconds': b['elapsed_s'] - a['elapsed_s'],
    'gap_start_label': a['request_label'],
    'gap_end_label': b['request_label'],
    'cgroup_memory_peak_gib': max(int(p['memory.peak']) for p in points) / 1024**3,
    'sampled_device_vram_max_gib': max(p['vram_used_mib'] for p in points) / 1024,
    'limitation': 'Historical sampled VRAM maximum is not a continuous peak; no VRAM bound is proven inside the long client gap. RAM memory.peak is a kernel high-water mark for the server cgroup, not aggregate host/client RAM.',
}
output = Path(__file__).resolve().parent / 'historical-sampling-audit.json'
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
