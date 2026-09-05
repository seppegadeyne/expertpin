#!/usr/bin/env python3
"""Reproduce the conservative 35-GiB RAM / 28-GiB VRAM candidate."""

import csv
import glob
import json
import os
import runpy
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/seppe/Projects/expertpin")
MODEL_DIR = Path("/home/seppe/Models/qwen3.8-flash-next/AD-4.27bpw-Q4_K_M-M64")
REFERENCE = ROOT / "evidence/pr2403-gpu-runtime/20260904T081625+0200"
PARSER = Path("/home/seppe/.hermes/profiles/expertpin/skills/devops/gpu-ram-prep/scripts/gguf_tensor_sizes.py")

module = runpy.run_path(str(PARSER))
read_shard = module["read_shard"]
tensor_bytes = module["tensor_bytes"]

expert_per_layer: dict[int, int] = {}
for shard in sorted(glob.glob(str(MODEL_DIR / "*.gguf"))):
    metadata, tensors, data_start = read_shard(shard)
    sizes = tensor_bytes(tensors, data_start, os.path.getsize(shard))
    for name, size in sizes.items():
        lower = name.lower()
        if "exps" not in lower and "experts" not in lower:
            continue
        try:
            layer = int(name.split(".")[1])
        except (IndexError, ValueError):
            continue
        expert_per_layer[layer] = expert_per_layer.get(layer, 0) + size

if len(expert_per_layer) != 48:
    raise RuntimeError(f"expected 48 expert layers, found {len(expert_per_layer)}")

per_layer_bytes = sum(expert_per_layer.values()) // len(expert_per_layer)

# The reference server began draft/slot initialization at 08:17:25. The
# immediately preceding peak therefore includes target weights + target context
# but excludes the 1.8-GiB MTP model and its context.
cutoff = datetime.fromisoformat("2026-09-04T08:17:25+02:00")
target_only_vram_samples: list[int] = []
with (REFERENCE / "runtime-metrics.csv").open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        timestamp = datetime.fromisoformat(row["timestamp"])
        if timestamp >= cutoff:
            break
        target_only_vram_samples.append(int(row["vram_used_mib"].strip()))

if not target_only_vram_samples:
    raise RuntimeError("no target-only VRAM samples found")

reference_target_peak_mib = max(target_only_vram_samples)
extra_gpu_layers = 36 - 30
extra_gpu_mib = extra_gpu_layers * per_layer_bytes / (1024**2)
vram_cap_mib = 28 * 1024
# Conservative: do not subtract the KV/context saving from 32K to 8K.
projected_vram_mib = reference_target_peak_mib + extra_gpu_mib
vram_margin_mib = vram_cap_mib - projected_vram_mib

result = {
    "verdict": "FEASIBLE_WITH_CONFIG",
    "budget": {
        "process_ram_gib": 35,
        "absolute_vram_gib": 28,
        "preload_memavailable_guard_gib": 45,
    },
    "config": {
        "DRAFT": 0,
        "CTX": 8192,
        "NCMOE": 30,
        "NGL": 99,
        "KVT": "q8_0",
        "CACHE_RAM_MIB": 512,
        "RAM_BUDGET_GIB": 35,
        "EXPERT_CACHE_SIM_MIB": 32768,
        "PINNED": 0,
    },
    "exact_model_layout": {
        "expert_layers": len(expert_per_layer),
        "expert_bytes_total": sum(expert_per_layer.values()),
        "expert_gib_total": sum(expert_per_layer.values()) / (1024**3),
        "expert_mib_per_layer": per_layer_bytes / (1024**2),
        "cpu_expert_gib_ncmoe30": 30 * per_layer_bytes / (1024**3),
        "cpu_expert_gib_ncmoe36": 36 * per_layer_bytes / (1024**3),
    },
    "conservative_vram_projection": {
        "reference": str(REFERENCE / "runtime-metrics.csv"),
        "reference_config": "NCMOE=36, CTX=32768, target+MTP",
        "target_only_cutoff": cutoff.isoformat(),
        "measured_absolute_target_only_peak_mib": reference_target_peak_mib,
        "extra_gpu_layers": extra_gpu_layers,
        "extra_gpu_mib": extra_gpu_mib,
        "projected_absolute_peak_mib_without_8k_credit": projected_vram_mib,
        "cap_mib": vram_cap_mib,
        "margin_mib": vram_margin_mib,
    },
    "host_evidence": {
        "reference_cgroup_memory_max_gib": 30,
        "reference_result": "completion_ok",
        "note": "The successful reference included MTP at NCMOE=36; this candidate removes MTP, moves six file-backed expert layers to GPU, and raises MemoryMax to 35 GiB.",
    },
    "unknown_until_guarded_run": [
        "actual peak cgroup MemoryCurrent/MemoryPeak on current commit",
        "actual peak VRAM at NCMOE=30",
        "decode throughput and cache-shadow overhead",
        "shadow hit-byte rate (current telemetry is count-based)",
    ],
}

if vram_margin_mib <= 0:
    raise RuntimeError(f"projected VRAM exceeds cap by {-vram_margin_mib:.1f} MiB")

text = json.dumps(result, indent=2, sort_keys=True) + "\n"
output_path = ROOT / "evidence/cache-shadow-ab/scenario-35g.json"
output_path.write_text(text, encoding="utf-8")
print(text, end="")
