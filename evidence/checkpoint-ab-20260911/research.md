# X-research 2026-09-11 (pre-GPU, mandatory gate)

Tool: x_search (read-only). Query: `qwen3.8-flash-next IQ2_XXS GGUF expert offload speed`.
Search-answer leads only; no external metric verified locally, no browser started,
no posts. Citations are x.com status URLs returned by the search backend.

## Leads taken (ideas only)

- Nvidia + llama.cpp with heavy expert CPU offload on ~2-bit class quants is
  reported at 21-26 tok/s sustained (search-answer aggregation, unverified).
- RTX 3090 Ti + 64 GB host with UD-Q2_K_XL (~IQ2_XXS-class) and --n-cpu-moe 29,
  65K ctx, ~23.5 GB VRAM: ~25 tok/s claim (lead, not reproduced here).
- MTP/speculative on IQ2-class with 94% acceptance reported on Apple Silicon:
  supports keeping the shared MTP drafter in the A/B.

None of these numbers were adopted as claims; our own guarded measurements below
are the only evidence used.

## DeepSeek-V4.1-Flash watch item (cheap check, no GPU)

Web search found `apetersson/DeepSeek-V4.1-Flash-MixedQ2-GGUF` on Hugging Face:
a routed-expert requant of V4.1-Flash rev 2bc89ac5 using llama.cpp PR #28696
quantizers. The "loadable GGUF exists" condition from the 2026-09-10 note is
therefore now met. No download, no integration (293 GB @ 4-bit; 2.0 TB free).
Cost/benefit evaluation deferred until relevant; documented here per the note.

## Order compliance

This research was completed and recorded BEFORE qli stop / model load / any GPU
work. GPU work started at 03:19:05 CEST (first run start), after commit 3cb0f825
was pushed and after both subagent review attempts had failed (HTTP 429).
