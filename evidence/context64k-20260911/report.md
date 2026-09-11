# 64K context on ps-iq2xxs — 2026-09-11 (12:22–12:47 CEST)

Question: does ps-iq2xxs serve 64K context (Seppe's product-acceptance
minimum for Hermes) with correct needle recall at start/middle/end, within
the 40/28 budgets?

## Verdict: PASS — 9/9 recall across the 16K -> 32K -> 64K ladder

| step | processed tokens | end | middle | start | RAM peak | VRAM peak |
|---|---|---|---|---|---|---|
| CTX 16384 | 15,269 | 7391 (33.6 s) | 7391 (19.4 s) | 7391 (33.7 s) | 32.42/40 GiB | 16.14/28 GiB |
| CTX 32768 | 30,920 | 7391 (62.2 s) | 7391 (35.7 s) | 7391 (55.1 s) | 32.41/40 GiB | 16.57/28 GiB |
| CTX 65536 | 61,401 | 7391 (123.4 s) | 7391 (85.6 s) | 7391 (133.5 s) | 32.41/40 GiB | 17.14/28 GiB |

Token counts are tokenizer-exact from the server (including chat template
and question), per Seppe's benchmark mandate (K = 1024; the K numbers are
configured context windows). memory.events max/oom/oom_kill all zero at
every step; swap peaks <=0.72 GiB. Same-day ladder on one host, binary SHA
3235529f... unchanged, target-only (DRAFT=0), temp 0, seed 42, one needle
per position, neutral wording (d3fe9fc2 scenario).

## Reading

- 64K served with recall at all three positions: the product-acceptance
  minimum for Hermes context is met on this checkpoint, with ~11 GiB VRAM
  headroom and ~7.6 GiB RAM headroom at the 64K step.
- Wall-times scale roughly with prefill (~500-800 tok/s effective, including
  partial prefix-cache reuse visible in the middle positions).
- This validates CAPACITY + recall at 64K; generation quality at depth,
  multi-needle, and the 264K goal remain separate items.

## Limitations

- Single needle per position, n=1, greedy decoding.
- 264K not tested (needs its own slice; KV q8_0 at 262144 would add ~2.4 GiB
  VRAM per the linear KV scaling seen here: ~0.5 GiB VRAM per 16K tokens).
- Hermes end-to-end at 64K is the next slice.

## Artifacts

- outcome.json (machine-readable ladder), runs 122210 (16K), 122812 (32K),
  123612 (64K). Harness: 468455ca (6 CPU-only tests; needle lineage
  unchanged, 23/23 green).
