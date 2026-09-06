# Target-only exact output comparison

Hash scope: canonical content + reasoning_content JSON, as previous compare-mtp-orders.py.

## nmax0 /home/seppe/Projects/expertpin/evidence/clean-mtp-ab/run-20260906T163711-45502ba6d21c48ab95be359f73fb7eac/01-nmax0-response.json

Equal=True; hashes 6b0c803deb1fb4d791efd430f4b1cedaf2c4ec5ca92e7240f554b3ffa5455cb7 / 6b0c803deb1fb4d791efd430f4b1cedaf2c4ec5ca92e7240f554b3ffa5455cb7

{
  "content": {
    "equal": true
  },
  "reasoning_content": {
    "equal": true
  }
}
```diff
```

## nmax0 /home/seppe/Projects/expertpin/evidence/clean-mtp-ab/run-20260906T163711-45502ba6d21c48ab95be359f73fb7eac/02-nmax0-response.json

Equal=True; hashes 6b0c803deb1fb4d791efd430f4b1cedaf2c4ec5ca92e7240f554b3ffa5455cb7 / 6b0c803deb1fb4d791efd430f4b1cedaf2c4ec5ca92e7240f554b3ffa5455cb7

{
  "content": {
    "equal": true
  },
  "reasoning_content": {
    "equal": true
  }
}
```diff
```

## nmax4 /home/seppe/Projects/expertpin/evidence/clean-mtp-ab/run-20260906T161719-6a8bf12c290b44ab9f5cd71f90d8a41c/01-nmax4-response.json

Equal=False; hashes 6b0c803deb1fb4d791efd430f4b1cedaf2c4ec5ca92e7240f554b3ffa5455cb7 / ca94ffb4d26462727b151d7c806c22f75f139e94fbbc76bfad109136d4b38fa4

{
  "content": {
    "equal": true
  },
  "reasoning_content": {
    "equal": false,
    "kind": "token",
    "index_0based": 33,
    "position_1based": 34,
    "target_id": 8404,
    "reference_id": 314,
    "target_ids_context": [
      36,
      318,
      76,
      12370,
      8404,
      17830,
      15089,
      8,
      42903
    ],
    "reference_ids_context": [
      36,
      318,
      76,
      12370,
      314,
      11312,
      8,
      42903,
      11
    ],
    "target_count": 256,
    "reference_count": 256
  }
}
```diff
--- left
+++ right
@@ -1,4 +1,4 @@
 {
   "content": "",
-  "reasoning_content": "We need answer user: \"Explain how a bounded expert cache handles RAM and NVMe misses in detail.\" Need infer context. Likely about MoE (mixture-of-experts) inference, expert cache bounded, RAM and NVMe misses. Need explain mechanism: experts stored in RAM cache bounded size, NVMe backing store, misses: RAM miss -> fetch from NVMe to RAM, evict LRU/least valuable expert, maybe prefetch, overlap compute, handle NVMe miss? If expert not in NVMe? Maybe from model weights? Need detail. Could be about \"bounded expert cache\" in systems like Mixtral? Maybe \"expert cache\" in LLM serving with MoE, offloading experts to NVMe. Need explain RAM and NVMe misses. Let's think.\n\nQuestion: \"Explain how a bounded expert cache handles RAM and NVMe misses in detail.\" It might refer to a specific paper/system? \"bounded expert cache\" maybe in \"MoE-Inference\" or \"Fiddler\"? Need identify. Could be about \"expert cache\" in \"Mixture-of-Experts with bounded cache\" where experts are loaded into GPU? But user says RAM and NVMe misses, not GPU. Maybe cache hierarchy: GPU HBM? RAM"
+  "reasoning_content": "We need answer user: \"Explain how a bounded expert cache handles RAM and NVMe misses in detail.\" Need infer context. Likely about MoE (mixture of experts) inference, expert cache, bounded cache, RAM and NVMe misses. Need explain mechanism: experts stored in RAM cache bounded size, NVMe backing store, GPU? Need detail on miss handling: RAM miss -> fetch from NVMe to RAM, maybe evict LRU, async prefetch, overlap compute, pinning, batching, metadata, page faults? NVMe miss -> fetch from remote storage/object store? Or if NVMe is cache tier? Need be careful.\n\nNeed produce final detailed explanation. Need maybe define bounded expert cache: fixed memory budget for expert weights in RAM, NVMe as secondary cache/tier. Handles misses by loading experts from NVMe into RAM, evicting least recently used, managing concurrency, I/O, GPU transfer. If NVMe miss, fetch from remote storage (S3/HDFS) or recompute? Need explain.\n\nLet's think of possible systems: \"bounded expert cache\" maybe from paper? Could be \"Expert Cache\" in MoE serving, \"bounded expert cache\" handles RAM and NVMe misses. Maybe related to \"Mixtral"
 }
```

## nmax8 /home/seppe/Projects/expertpin/evidence/clean-mtp-ab/run-20260906T161517-2c1eb221aa234812a933801b15ea5d06/01-nmax8-response.json

Equal=False; hashes 6b0c803deb1fb4d791efd430f4b1cedaf2c4ec5ca92e7240f554b3ffa5455cb7 / 35210531f124344251490d21e69495b8e93647b60e9dff44a4d8dfbc18e7a9a8

{
  "content": {
    "equal": true
  },
  "reasoning_content": {
    "equal": false,
    "kind": "token",
    "index_0based": 42,
    "position_1based": 43,
    "target_id": 11,
    "reference_id": 303,
    "target_ids_context": [
      11,
      6009,
      6297,
      59056,
      11,
      21360,
      321,
      23683,
      7575
    ],
    "reference_ids_context": [
      11,
      6009,
      6297,
      59056,
      303,
      21360,
      321,
      23683,
      7575
    ],
    "target_count": 256,
    "reference_count": 256
  }
}
```diff
--- left
+++ right
@@ -1,4 +1,4 @@
 {
   "content": "",
-  "reasoning_content": "We need answer user: \"Explain how a bounded expert cache handles RAM and NVMe misses in detail.\" Need infer context. Likely about MoE (mixture-of-experts) inference, expert cache bounded, RAM and NVMe misses. Need explain mechanism: experts stored in RAM cache bounded size, NVMe backing store, misses: RAM miss -> fetch from NVMe to RAM, evict LRU/least valuable expert, maybe prefetch, overlap compute, handle NVMe miss? If expert not in NVMe? Maybe from model weights? Need detail. Could be about \"bounded expert cache\" in systems like Mixtral? Maybe \"expert cache\" in LLM serving with MoE, offloading experts to NVMe. Need explain RAM and NVMe misses. Let's think.\n\nQuestion: \"Explain how a bounded expert cache handles RAM and NVMe misses in detail.\" It might refer to a specific paper/system? \"bounded expert cache\" maybe in \"MoE-Inference\" or \"Fiddler\"? Need identify. Could be about \"expert cache\" in \"Mixture-of-Experts with bounded cache\" where experts are loaded into GPU? But user says RAM and NVMe misses, not GPU. Maybe cache hierarchy: GPU HBM? RAM"
+  "reasoning_content": "We need answer user: \"Explain how a bounded expert cache handles RAM and NVMe misses in detail.\" Need infer context. Likely about MoE (mixture-of-experts) inference, expert cache bounded in RAM and NVMe? Need explain handling RAM and NVMe misses. Could be from some paper/system? Need provide detailed explanation. Need think.\n\nWe need produce final answer. Need understand \"bounded expert cache\". Maybe in context of LLM MoE serving with expert weights stored in RAM and NVMe, cache limited. Bounded expert cache: cache experts in GPU? RAM? NVMe? Handles misses by fetching from lower tier. Need detail: RAM miss: expert not in RAM, fetch from NVMe to RAM, evict LRU/least valuable expert from RAM, maybe prefetch. NVMe miss: expert not in NVMe? fetch from remote storage/object store? Or if bounded cache only RAM+NVMe, NVMe miss means expert not cached in NVMe, must load from original model repository (disk/network), write into NVMe, then RAM. Need discuss admission policy, eviction, concurrency, overlap, pinning, batching, metadata, capacity constraints, hit rates, latency.\n\nMaybe \"bounded expert cache\" specifically from paper \""
 }
```

## nmax16 /home/seppe/Projects/expertpin/evidence/clean-mtp-ab/run-20260906T161217-e1d70c565cee4ea3a67bfbf72aa153e0/01-nmax16-response.json

Equal=False; hashes 6b0c803deb1fb4d791efd430f4b1cedaf2c4ec5ca92e7240f554b3ffa5455cb7 / 22ad201182312054bcafc9dbecd5c6735939ea60e754373ecaf2b3c18851a3dd

{
  "content": {
    "equal": true
  },
  "reasoning_content": {
    "equal": false,
    "kind": "token",
    "index_0based": 53,
    "position_1based": 54,
    "target_id": 11312,
    "reference_id": 59056,
    "target_ids_context": [
      14235,
      10033,
      16456,
      25,
      11312,
      9476,
      303,
      21360,
      6297
    ],
    "reference_ids_context": [
      14235,
      10033,
      16456,
      25,
      59056,
      6009,
      6297,
      303,
      21966
    ],
    "target_count": 256,
    "reference_count": 256
  }
}
```diff
--- left
+++ right
@@ -1,4 +1,4 @@
 {
   "content": "",
-  "reasoning_content": "We need answer user: \"Explain how a bounded expert cache handles RAM and NVMe misses in detail.\" Need infer context. Likely about MoE (mixture-of-experts) inference, expert cache bounded, RAM and NVMe misses. Need explain mechanism: experts stored in RAM cache bounded size, NVMe backing store, misses: RAM miss -> fetch from NVMe to RAM, evict LRU/least valuable expert, maybe prefetch, overlap compute, handle NVMe miss? If expert not in NVMe? Maybe from model weights? Need detail. Could be about \"bounded expert cache\" in systems like Mixtral? Maybe \"expert cache\" in LLM serving with MoE, offloading experts to NVMe. Need explain RAM and NVMe misses. Let's think.\n\nQuestion: \"Explain how a bounded expert cache handles RAM and NVMe misses in detail.\" It might refer to a specific paper/system? \"bounded expert cache\" maybe in \"MoE-Inference\" or \"Fiddler\"? Need identify. Could be about \"expert cache\" in \"Mixture-of-Experts with bounded cache\" where experts are loaded into GPU? But user says RAM and NVMe misses, not GPU. Maybe cache hierarchy: GPU HBM? RAM"
+  "reasoning_content": "We need answer user: \"Explain how a bounded expert cache handles RAM and NVMe misses in detail.\" Need infer context. Likely about MoE (mixture-of-experts) inference, expert cache bounded, RAM and NVMe misses. Need explain mechanism: bounded expert cache in GPU/CPU/NVMe hierarchy? Maybe from paper? Need detail: when expert requested not in GPU? RAM miss? NVMe miss? Bounded cache means limited capacity, eviction policies, prefetching, async transfers, batching, pinning, LRU/LFU, admission control, writeback? Need be careful.\n\nLet's parse: \"bounded expert cache\" maybe in context of \"Expert Cache\" for MoE serving, e.g., \"Fiddler\", \"MoE-Inference\", \"Mixtral\", \"DeepSpeed-MoE\", \"FlexMoE\", \"MegaScale-Infer\", \"KTransformers\"? Bounded expert cache handles RAM and NVMe misses. Could be from \"MoE-Inference: Efficient Mixture-of-Experts Inference with Expert Cache\"? There is concept: experts stored on NVMe, cached in RAM, GPU. Bounded cache: limited RAM capacity. On miss, load from NVMe to RAM, evict."
 }
```
