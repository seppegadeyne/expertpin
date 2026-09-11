First guarded A/B of the PeasantSmith IQ2_XXS checkpoint (75.2 GB single file) vs the reference AD-4.27bpw Q4_K_M shard set, same day, same binary, one variable (model file — proven by environment.json diff showing only MODEL/MODEL_DIR + scope UUID).

**Results** (nmax4 MTP, 2x256 tokens, 68-tok prompt, temp 0, seed 42):

| arm | decode cold (tok/s) | decode warm (tok/s) | RAM peak | VRAM peak |
|---|---|---|---|---|
| ps-iq2xxs | 31.50 | **50.19** | 32.40 / 40 GiB | 18.63 / 28 GiB |
| reference | 21.86 | 44.33 | 32.40 / 40 GiB | 22.67 / 28 GiB |

Both arms completed with zero cgroup OOM events (MemoryMax 36G) and Tier A green.

Key readings:
- **20 tok/s within 40/28: met on both arms today** (warm 50.19 / 44.33; cold 31.50 / 21.86). n=1 prompt, 2 requests per arm — not a sustained-throughput or long-context proof.
- **Same-day A/B: ps-iq2xxs +13% warm / +44% cold**, same acceptance (~0.67).
- **Host-state confound documented**: the identical reference config measured 6.5-8.5 tok/s on 2026-09-06; today 21.9/44.3. Binary SHA unchanged. The delta vs history is host memory state (MemAvailable/swap), not code or checkpoint. Historical speed comparisons across days remain invalid without host-state control.
- Not measured: IQ2_XXS quality/tool-calling parity, 64K+ context, truly cold page cache, sustained decode at depth.

Harness gained a `--checkpoint {reference,ps-iq2xxs}` flag (commit 3cb0f825) so future A/Bs keep this provenance discipline. Evidence: `evidence/checkpoint-ab-20260911/` + raw runs in `evidence/clean-mtp-ab/run-20260911T*`.

Subagent review note: both gpt-6-astra review attempts failed with HTTP 429 (usage limit) before the code commit; per protocol the slice proceeded with parent self-verification of the review premises (launcher `-m "$MODEL"` verbatim, env.update always overrides MODEL/MODEL_DIR, early is_file check before any host change, CPU-only tests).
