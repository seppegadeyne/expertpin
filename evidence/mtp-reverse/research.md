# X research — 2026-09-06, before tests and GPU work

Read-only x_search completed for Qwen3.8 Flash Next / flash-next, AD-4.27bpw, MTP n_max 4/8/16, n-cpu-moe, expert-cache, EXL3 and INT4/W4A16.

Useful leads (external self-reports, NOT local measurements):
- https://x.com/inovelloE/status/2096577582741238212 — expert caching plus MTP and adaptation across requests: motivation to counterbalance measurement order.
- https://x.com/lefu777/status/2096059297046437921 — claimed sensitivity to MTP sampling and cache policy; isolate depth rather than combining changes.
- https://x.com/murasametech/status/2096153986399391892 — reported formatting differences in EXL3 experiments. Different model/runtime/hardware; not proof that our greedy MTP output differences are expected or correct.

No external metrics transferred to expertpin, no browser or local research process launched. No external posts/writes. X search summary mixes 27B and Flash-Next results: do not compare those speeds.

Slice: repeat the existing clean 256-token harness at startup depth 16, then 8, then 4, twice each. Same prompt, binary, seed42/temp0/cache_prompt=false; no tracing, shadow or placement changes. Preserve all output fields and compare exact text hashes and unified diffs across repetitions, depths and forward/reverse order. Differences alone cannot establish cause or quality; greedy speculative decoding does not generally justify arbitrary branching differences. Ranking remains observational unless confounds are actually controlled. Multi-prompt extension deferred to keep this a single matched counter-order slice.
