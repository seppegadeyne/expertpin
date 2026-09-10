# Independent review and parent verification — 2026-09-10

Review `deleg_899b2ef4`, configured gpt-6-astra, completed before commit and any
GPU/service work. Full staged diff supplied as `review.diff` (local, untracked).
Verdict **CHANGES_REQUESTED**, two documentation clarifications; not a code review
approval. No second independent approval claimed. Reviewer reports its optional
inline verification commands were approval-blocked; it completed source review
without them. Parent did not bypass any blocked child command.

## Findings addressed

1. **Request-scoped trace completeness is not initial-state completeness.**
   Parent independently read `ggml/src/ggml-moe-prefetch.cpp:746–773` (access
   precedes record), `ggml/src/ggml-moe-trace.h:28–32` (unscoped filter), and
   `docs/request-scoped-expert-trace.md:16–24` (warmup excluded while shadow
   mutations remain). Parent read source CSV lines 1–8 and 45013–45017 itself:
   the fourth record, seq3, expert20, is a hit on first recorded appearance,
   despite zero dropped/error footer. Independent Python assertions over that
   existing artifact and its hash succeeded, captured in `verification.json`.
   Note now requires a known reset boundary plus all mutations/reset events,
   or justified capacity-specific initial state; existing scoped artifact is
   explicitly not directly replay-ready from empty. A source-cap snapshot is
   insufficient to infer smaller-cap warm state.
2. **Reference demand simulator is not the default runtime policy.**
   Parent independently read `third-party/kimi-k3-in-c/src/core/k3_ops.c:594–623`
   (getmany before get), and already read `src/cache/k3_cache.c:143–163,272,301–304`
   (skip existing hits during reservation, default prefetch, padded slots), plus
   `tools/sim_cache.py:36–48,152` (demand LRU and payload-based capacity).
   Added the explicit policy/recency/capacity mismatch qualification. This is
   source-grounded, not execution of the third-party runtime or a bugfix claim.

## Other parent checks

- Re-read the reference FP contract `src/core/k3_ops.c:1–25` and three invariants
  in README:831–857. These are separate; no universal/numerical-fix claim.
- Re-read production LRU access/bypass/multi-eviction and CPU hook independently;
  shadow accounting is not a physical cache. Tests actually ran in CPU build.
- Full build reached 100%; focused log 12/12; main log 39/41 with historical
  BERT-tokenizer/chat-template failures. No all-green claim.
- Parsed all 18 synthetic JSONL arms, asserted within-cap repeat agreement and
  disabled counters; these are policy fixtures, not model or I/O measurements.
- Re-read new advisor output; blocked/null, 24 missing fields. No reference rate
  substituted. Unknown model tok/s/RAM/VRAM remain null.

Only English docs and evidence are staged. No source, engine, launcher, defaults,
model assets, or reference clone edits. Raw test logs preserve original whitespace;
authored Markdown and JSON pass diff whitespace checks.
