# Checkpoint A/B — PS-IQ2_XXS vs reference Q4_K_M (2026-09-11)

Same-day guarded comparison using the clean-MTP harness with the new
`--checkpoint` flag (commit 3cb0f825). One variable: the model file. Everything
else identical (environment.json diff between the two runs shows only
MODEL/MODEL_DIR + scope UUID).

## Results

| arm | decode req1 (tok/s) | decode req2 (tok/s) | RAM peak | VRAM peak | samples |
|---|---|---|---|---|---|
| ps-iq2xxs (75.2 GB single file) | 31.50 | **50.19** | 32.40 GiB / 40 | 18.63 GiB / 28 | 43 |
| reference (94.5 GB shard set) | 21.86 | 44.33 | 32.40 GiB / 40 | 22.67 GiB / 28 | 53 |

Both arms complete (2x256 tokens each, 68-token prompt, temp 0, seed 42,
n_max 4, acceptance ~0.67 both). Zero cgroup max/oom/oom_kill events; swap
0.24-0.52 GiB; MemoryMax 36G. Host Tier A green at both loads.

## Reading (evidence-first)

1. **20 tok/s goal**: warm decode 50.19 (ps) and 44.33 (ref) — both >= 20
   within 40 GiB RAM / 28 GiB VRAM budgets. Cold-first-request also >= 20
   (31.50 / 21.86).
2. **Same-day A/B**: ps-iq2xxs +13% warm / +44% cold over reference. Real but
   modest; same acceptance regime.
3. **Host-state confound vs history**: the SAME reference config measured
   6.48/8.49 tok/s on 2026-09-06 and 8.37/10.88 later that day. Today it did
   21.86/44.33. The 2.6-6.7x jump vs history is host memory state (MemAvailable
   40.6-48.6 GiB and swap 0.25-0.52 GiB today vs 42-46 GiB / 1.2-2.2 GiB swap
   historically), NOT code (binary SHA 3235529f unchanged since 2026-09-08)
   and NOT the new checkpoint. Do not attribute historical deltas to code.
4. **Not proven**: sustained decode at depth (only 256-token requests),
   64K/264K context, IQ2_XXS quality/tool-calling parity, truly cold page
   cache (files were warm from download), Hermes end-to-end.

## Artifacts

- Runs: `../clean-mtp-ab/run-20260911T031905-*` (ps) and `run-20260911T032256-*` (ref)
- `outcome.json` — machine-readable summary incl. provenance
- `research.md` — X-research + V4.1-watch check (pre-GPU gate)

## Next steps

1. Longer decode (2x512/1024) for sustained throughput on ps-iq2xxs.
2. IQ2_XXS quality gate (tool-calling JSON, needle recall) before any default switch.
3. Controlled cold-cache A/B to bound worst-case first tokens.
4. 64K context benchmark within 40/28.
