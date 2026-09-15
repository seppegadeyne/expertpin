#!/usr/bin/env python3
"""Summarize prefill-ubatch run budgets from samples.jsonl."""
import json
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    run = Path(arg)
    samples = run / 'samples.jsonl'
    peak_vram = peak_mem = peak_swap = 0
    n = 0
    last_t = None
    maxgap = 0.0
    oom = False
    for line in samples.read_text().splitlines():
        d = json.loads(line)
        n += 1
        peak_vram = max(peak_vram, d['vram_used_mib'])
        peak_mem = max(peak_mem, int(d['memory.peak']))
        peak_swap = max(peak_swap, int(d['memory.swap.current']))
        if 'max' in d.get('memory.events', ''):
            oom = oom or int(d['memory.events'].split('max ')[1].split()[0]) > 0
        if last_t is not None:
            maxgap = max(maxgap, d['elapsed_s'] - last_t)
        last_t = d['elapsed_s']
    print(f"{run.name}: samples={n} vram_peak={peak_vram/1024:.2f}/28GiB "
          f"mem_peak={peak_mem/1024**3:.2f}/40GiB swap_peak={peak_swap/1024**3:.2f}GiB "
          f"maxgap={maxgap:.1f}s oom={oom}")
