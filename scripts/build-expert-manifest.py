#!/usr/bin/env python3
"""Build an expertpin residency manifest from a measured expert-access histogram.

Input: the --expert-stats-file JSON of a serving run with GGML_MOE_HISTOGRAM=1
(advisory lane: distinct routed expert ids per CPU-MoE kernel entry, keyed by
tensor name). Output: a manifest JSON (per-layer expert ids, hottest first)
plus a coverage/sizing report against exact GGUF tensor bytes.

Pure stdlib. GGUF header reading reuses the skill's gguf_tensor_sizes.py.

Usage:
  build-expert-manifest.py expert-stats.json --model-dir /path/UD-Q4_K_XL \
      --budget-gib 28 --out manifest.json --report report.json
  build-expert-manifest.py --self-test
"""
import glob
import json
import os
import re
import runpy
import sys

GGUF_READER = ("/home/seppe/.hermes/profiles/expertpin/skills/devops/"
               "gpu-ram-prep/scripts/gguf_tensor_sizes.py")
LAYER_RE = re.compile(r"^blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight$")


def load_histogram(path):
    """stats JSON -> {layer: {expert_id: count}} (max across up/gate/down)."""
    with open(path) as f:
        root = json.load(f)
    hist = root.get("expert_histogram")
    if not hist or not hist.get("tensors"):
        raise SystemExit(f"{path}: no expert_histogram.tensors — was "
                         "GGML_MOE_HISTOGRAM=1 wired and the server stopped?")
    layers = {}
    for name, counts in hist["tensors"].items():
        m = LAYER_RE.match(name)
        if not m:
            continue  # foreign (e.g. draft-model) tensors: reported by caller
        layer = int(m.group(1))
        slot = layers.setdefault(layer, {})
        for idx, c in enumerate(counts):
            if c:
                slot[idx] = max(slot.get(idx, 0), int(c))
    if not layers:
        raise SystemExit(f"{path}: histogram has no blk.N.ffn_*_exps tensors")
    meta = {"kernel_entries": hist.get("kernel_entries", 0),
            "token_rows": hist.get("token_rows", 0),
            "foreign_tensors": sorted(n for n in hist["tensors"]
                                      if not LAYER_RE.match(n))}
    return layers, meta


def order_experts(layer_counts):
    """Per layer: all expert ids sorted by count desc, then id asc (stable)."""
    return {layer: sorted((eid for eid in counts), key=lambda e: (-counts[e], e))
            for layer, counts in layer_counts.items()}


def layer_expert_bytes(model_dir, layers):
    """Exact per-expert resident bytes per layer (up+gate+down strides)."""
    reader = runpy.run_path(GGUF_READER)
    read_shard, tensor_bytes = reader["read_shard"], reader["tensor_bytes"]
    shards = sorted(glob.glob(os.path.join(model_dir, "*.gguf")))
    if not shards:
        raise SystemExit(f"{model_dir}: no .gguf shards")
    n_experts, sizes = None, {}
    for sh in shards:
        meta, tensors, data_start = read_shard(sh)
        if n_experts is None:
            arch = meta.get("general.architecture", "")
            n_experts = meta.get(f"{arch}.expert_count")
        for name, nb in tensor_bytes(tensors, data_start,
                                     os.path.getsize(sh)).items():
            if LAYER_RE.match(name):
                sizes[name] = nb
    if not n_experts:
        raise SystemExit(f"{model_dir}: expert_count not in GGUF meta")
    out = {}
    for layer in layers:
        total = 0
        for kind in ("up", "gate", "down"):
            name = f"blk.{layer}.ffn_{kind}_exps.weight"
            if name not in sizes:
                raise SystemExit(f"{model_dir}: missing tensor {name}")
            total += sizes[name]
        out[layer] = total // n_experts  # per-expert bytes across 3 tensors
    return out, n_experts


def coverage(orders, counts, k):
    """Fraction of recorded distinct accesses covered by the top-k per layer."""
    covered = total = 0
    for layer, order in orders.items():
        c = counts[layer]
        total += sum(c.values())
        covered += sum(c[e] for e in order[:k])
    return covered / total if total else 0.0


def choose_k(orders, expert_bytes, budget_bytes):
    """Largest K whose total resident bytes fit the budget (K fits all lists)."""
    kmax = min(len(o) for o in orders.values())
    per_k = sum(expert_bytes.values())  # bytes per +1 of K across all layers
    best = 0
    for k in range(1, kmax + 1):
        if k * per_k <= budget_bytes:
            best = k
        else:
            break
    return best, per_k


def build_manifest(orders, k):
    if k < 1:
        raise SystemExit("K=0: budget fits no resident experts; no manifest")
    return {"layers": {str(layer): order[:k] for layer, order in
                       sorted(orders.items())}}


def self_test():
    counts = {8: {0: 10, 1: 5, 2: 5, 3: 1}, 9: {2: 7, 0: 3, 1: 0, 3: 0}}
    orders = order_experts(counts)
    assert orders[8] == [0, 1, 2, 3], orders[8]      # count desc, id-asc ties
    assert orders[9] == [2, 0, 1, 3], orders[9]      # zero-count ids trail
    assert abs(coverage(orders, counts, 1) - 17/31) < 1e-12
    assert abs(coverage(orders, counts, 4) - 1.0) < 1e-12
    expert_bytes = {8: 100, 9: 200}                  # 300 bytes per +1 K
    k, per_k = choose_k(orders, expert_bytes, 900)
    assert (k, per_k) == (3, 300), (k, per_k)
    k, _ = choose_k(orders, expert_bytes, 299)
    assert k == 0
    man = build_manifest(orders, 2)
    assert man == {"layers": {"8": [0, 1], "9": [2, 0]}}, man
    print("build-expert-manifest --self-test: OK")


def main():
    if "--self-test" in sys.argv[1:]:
        self_test()
        return
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        raise SystemExit(__doc__)
    stats_path = args[0]
    opt = lambda name, default=None: (
        args[args.index(name) + 1] if name in args else default)
    model_dir = opt("--model-dir")
    budget_gib = float(opt("--budget-gib", "28"))
    out_path = opt("--out")
    report_path = opt("--report")
    if not model_dir:
        raise SystemExit("--model-dir is required (exact GGUF bytes)")

    layers, meta = load_histogram(stats_path)
    orders = order_experts(layers)
    expert_bytes, n_experts = layer_expert_bytes(model_dir, sorted(layers))
    budget_bytes = int(budget_gib * 1024 ** 3)
    k, per_k = choose_k(orders, expert_bytes, budget_bytes)

    curve = [{"k": kk,
              "resident_gib": round(kk * per_k / 1024 ** 3, 3),
              "coverage": round(coverage(orders, layers, kk), 4)}
             for kk in (32, 64, 128, 192, 256, 320, 384, 448, 512)
             if kk <= min(len(o) for o in orders.values())]
    report = {
        "stats": stats_path, "model_dir": model_dir,
        "kernel_entries": meta["kernel_entries"],
        "token_rows": meta["token_rows"],
        "foreign_tensors_skipped": meta["foreign_tensors"],
        "cpu_moe_layers": sorted(layers),
        "n_experts_per_layer": n_experts,
        "per_expert_bytes_per_layer_avg":
            sum(expert_bytes.values()) // len(expert_bytes),
        "budget_gib": budget_gib, "chosen_k": k,
        "chosen_resident_gib": round(k * per_k / 1024 ** 3, 3),
        "chosen_coverage": round(coverage(orders, layers, k), 4),
        "curve": curve,
    }
    manifest = build_manifest(orders, k)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")
    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
