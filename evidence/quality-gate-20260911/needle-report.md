# Needle-recall gate A/B — 2026-09-11 (10:45–11:15 CEST)

Question: does the PS IQ2_XXS checkpoint recall a single four-digit code
planted at start/middle/end of a ~2K-token haystack, compared to the
reference checkpoint, within the 40/28 budgets? This completes the checkpoint
quality gate (tool-calling PASS earlier today + needle-recall PASS here).

## Verdict: PASS on both checkpoints (final scenario d3fe9fc2)

| checkpoint | end | middle | start | RAM peak | VRAM peak |
|---|---|---|---|---|---|
| ps-iq2xxs | 7391 (9.2 s, 102 tok) | 7391 (7.2 s, 120 tok) | 7391 (6.1 s, 102 tok) | 32.41 / 40 GiB | 15.98 / 28 GiB |
| reference | 7391 (25.5 s, 131 tok) | 7391 (13.3 s, 253 tok) | 7391 (9.2 s, 97 tok) | 32.41 / 40 GiB | 20.10 / 28 GiB |

Haystack: deterministic filler paragraphs sized via /tokenize to ~2048 tokens
(2151 measured prompt tokens including the chat template), needle at 10% /
50% / 90%, positions run end->middle->start, temp 0, seed 42, target-only
(DRAFT=0). Swap peaks 0.63 / 1.07 GiB; binary SHA 3235529f... unchanged.

## Gate iterations (design confounds, not checkpoint failures)

1. eb71b914 (256 answer tokens): reference burned its whole budget
   deliberating whether the needle's "only the archivist may share it" clause
   allowed answering — finish_reason=length, fail-closed refusal. Preserved:
   run-20260911T104906.
2. bde348f8 (512 tokens): reference 01-end passed; 02-middle relapsed into
   the same compliance deliberation at length. Preserved:
   run-20260911T110146. Reasoning in both preserved runs shows the code WAS
   located — a scenario artifact, not a retrieval failure.
3. d3fe9fc2 (512 tokens, neutral needle wording — plain internal-audit
   record, no sharing restriction): both checkpoints recall 3/3. Counted.

The ps-iq2xxs arm passed 3/3 under every wording — including the ones the
reference choked on.

## What this does NOT prove

- Single needle at ~2K tokens: no 64K/264K claim, no multi-needle/RULER-style
  aggregation, no variable haystack density.
- n=1 per position, greedy decoding.
- Prefix-cache reuse between positions is partial only (the inserted needle
  breaks the common prefix); wall-times are single-shot warm observations,
  not a throughput benchmark.

## Combined quality-gate status (2026-09-11)

- tool-calling JSON correctness: PASS / PASS (fda74ed1, f188c7dd)
- needle recall ~2K: PASS / PASS (d3fe9fc2)
- Still open before any default-switch: sustained decode depth, cold-cache
  behavior, 64K context within 40/28, Hermes end-to-end.

## Artifacts

- needle-outcome.json, this report; runs run-20260911T110728 (ps),
  run-20260911T110905 (reference); iteration failures preserved in
  run-20260911T104906 and run-20260911T110146 (plus an EADDRINUSE refusal in
  run-20260911T105512). Third 429-blocked review: deleg_a025517a (review.md).
