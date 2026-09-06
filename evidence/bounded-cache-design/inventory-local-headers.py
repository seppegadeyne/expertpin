#!/usr/bin/env python3
"""Local header-only intake; external parser path is explicit, not bundled."""
import argparse
import json
import runpy
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--header-reader", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
args = parser.parse_args()
reader = runpy.run_path(str(args.header_reader))
rows = []
shards = sorted(args.model_dir.glob("*.gguf"))
if not shards:
    raise ValueError("no GGUF shards")
for shard in shards:
    meta, tensors, start = reader["read_shard"](str(shard))
    spans = reader["tensor_bytes"](tensors, start, shard.stat().st_size)
    for name, dims, quant, offset in tensors:
        if name.startswith("blk.") and "_exps." in name:
            if len(dims) != 3 or dims[-1] <= 0:
                raise ValueError(f"unexpected expert dimensions: {name}: {dims}")
            rows.append(dict(shard=shard.name, name=name, dims=dims, ggml_type=quant,
                             storage_span_bytes=spans[name], relative_offset=offset))
if not rows or len({r["name"] for r in rows}) != len(rows):
    raise ValueError("missing or duplicate expert tensors")
print(json.dumps({
    "kind": "local_gguf_header_inventory",
    "model_dir": str(args.model_dir),
    "header_reader": str(args.header_reader),
    "shards": len(shards),
    "expert_tensors": len(rows),
    "expert_storage_span_bytes": sum(r["storage_span_bytes"] for r in rows),
    "type_counts": dict(Counter(r["ggml_type"] for r in rows)),
    "limitations": [
        "header reader reads a bounded 64 MiB prefix per shard, no inference or full checkpoint load",
        "offset-derived storage spans INCLUDE alignment/trailing padding; not exact runtime slice payloads",
        "tensor dimension order is GGUF ne[0],ne[1],ne[2]; expert axis is ne[2] for these tensors",
        "inventory spans cannot be substituted for runtime nb[2]/ggml_nbytes without checking layouts",
    ],
    "tensors": rows,
}, indent=2, sort_keys=True))
