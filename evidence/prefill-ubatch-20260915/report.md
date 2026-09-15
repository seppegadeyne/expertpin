# Prefill-ubatch ladder — UD-Q4_K_XL, verified-cold (2026-09-15)

## Question

Seppe's pain point (2026-09-14 ~15:15): cold prefill of the ~19.5K-token
default Hermes system prompt runs at ~19 tok/s on the dev config
(UD-Q4_K_XL, NCMOE=40, DRAFT=1 MTP n_max=4, ubatch default 512) — the
first turn of a server lifetime takes 15-19 minutes. Evidence:
`evidence/hermes-e2e-20260911/run-20260914T104344-e2e/server.log` line 760
(14,213 tokens @ 18.84 tok/s; even 43-51 token follow-up prompts pay
13-26 tok/s).

X-research (2026-09-15, read-only) pointed at `--ubatch-size` as the
standard prefill knob for CPU-MoE offload (community consensus: 1024-2048
sweet spot; 5090 rigs report ~900-1000 tok/s prefill on lighter quants).
Hypothesis: each 512-token ubatch pass streams the routed experts of the
40 CPU-MoE layers; a larger physical batch should amortize that per-pass
cost over more tokens.

## Setup

One ubatch value per server lifetime; whole-checkpoint verified-cold page
cache (0/27,181,314 resident pages, all 4 shards) before every load;
Tier A guard green at readiness (GPU 0%, MemAvailable 42.5 GiB); cgroup
MemoryMax 36G; GGML_CUDA_NO_PINNED=1; qli stopped per Seppe rule 0
(2026-09-14) and restarted+verified after; jetski untouched (paused).
Prompt: tokenizer-calibrated neutral filler, 13,835 tokens (141
paragraphs) — the E2E prompt-size class. 32-token decode follows the
prefill to keep the request pipeline honest. Harness:
`run-guarded.py` (coldcache lineage); tests:
`tests/test-prefill-ubatch-20260915.py` (8/8 CPU-only).

## Results (cold prefill, 13,847 tokens measured)

| ubatch | prefill tok/s | wall      | VRAM peak   | RAM peak  | verdict |
|--------|---------------|-----------|-------------|-----------|---------|
| 512    | 116.93        | 135.2 s   | 24.25/28    | 32.42/40  | completed |
| 1024   | 82.31         | 168.2 s   | 25.49/28    | 32.42/40  | completed (slower) |
| 2048   | —             | —         | 28.00/28 hit| 32.41/40  | fail-closed abort (VRAM cap) |

Server-observed `n_ubatch` matched the requested value in every run
(`ubatch-observed.txt`), so the flag does reach the engine.

## Findings

1. **The community ubatch advice does NOT transfer to this stack.** On
   the neutral filler, 1024 is ~30% SLOWER than 512, and 2048 exceeds the
   28 GiB VRAM budget (compute buffers scale with ubatch) and was killed
   fail-closed by the harness. Default 512 remains the best of the three
   measured values.
2. **The dominant variable is prompt content diversity, not ubatch.**
   The SAME config (ubatch 512) that produces 18.84 tok/s on the real
   Hermes system prompt produced 116.93 tok/s on the neutral filler. The
   filler's small vocabulary (12 subjects × 11 verbs) reuses experts
   heavily within each ubatch; a diverse real-world prompt activates a
   much larger expert union per ubatch, multiplying the per-pass expert
   streaming cost. This reframes the prefill problem: the lever is
   expert-locality/reuse, not physical batch size.
3. **Cold decode after cold prefill is 2.02 tok/s** (495.6 ms/token) —
   expert cache misses after a fresh load dominate the first tokens. This
   matches the E2E observation (5.56 tok/s first decode) and reinforces
   that cold-start experience (prefill + first decode) is the product
   gap, not warm decode (35+ tok/s measured previously).

## Limitations

- Neutral filler is NOT the real Hermes prompt: absolute numbers are an
  upper bound on real-prompt throughput at this config; the 512-vs-1024
  comparison is internally valid (identical prompt, host, cold state).
- n=1 per arm; no variance estimate. Host state was closely matched
  (verified-cold 0.0 ratio in all three arms).
- 2048 abort means "exceeds VRAM budget", not "slower": the compute
  buffer grows with ubatch; no throughput datapoint was collected.
- Decode metrics (2.02 tok/s) are from the same single request; MTP
  acceptance on this filler was 16/22 accepted.

## Rule-0 compliance

qli stopped before GPU work and restarted+verified after EVERY arm
(journal shows epoch-230 shares resuming; `cleanup_errors: []` in the two
completed runs). jetski never touched. headless-chromium restored via
gpu-host-prep --restore (active at close).
