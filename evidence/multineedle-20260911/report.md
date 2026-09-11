# Multi-needle (RULER-style) — 2026-09-11 (14:26–14:35 CEST)

Question: can ps-iq2xxs retrieve and aggregate THREE planted codes from one
haystack in a single answer, at 2K and at 64K context?

## Verdict: PASS at both depths

| depth | prompt tokens | codes recalled | wall | RAM peak | VRAM peak |
|---|---|---|---|---|---|
| ~2K | 2,223 | 7391, 4826, 9517 (all) | 12.3 s | 32.41/40 GiB | (see samples) |
| 64K | 61,478 | 7391, 4826, 9517 (all) | 125.5 s | 32.41/40 GiB | 17.09/28 GiB |

The answer text at both depths is exactly "7391, 4826, 9517" — all three
projects correctly attributed in a single aggregation response. Needles sit
at 15%/45%/75% (early/middle/late), so distractor resistance across the
whole window is exercised, not just the extremes.

## What this adds over the single-needle gates

- Multi-fact retrieval with aggregation (RULER's harder axis) — the model
  must hold three distinct code/project bindings simultaneously and emit
  all of them, not just locate one.
- At 64K this combines long context + multi-needle for the first time on
  this checkpoint.

## Limitations

- n=1 per depth, greedy decoding; one question shape.
- Codes are distinct 4-digit strings; no adversarial transposition
  variants, no variable haystack density, no RULER word-list tasks.

## Artifacts

- outcome.json; runs 142639 (2K) and 143032 (64K). Harness: ad62de55
  (8 CPU-only tests).
