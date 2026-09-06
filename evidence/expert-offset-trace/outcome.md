# Guarded attempt outcome — 2026-09-06 12:09 CEST

Code: fe46f1c0025a394cf3cb6fbd5de27771242672c8 (origin/main read back).
Research, independent review/disposition, final CPU tests, CUDA llama-server build,
code commit/push and issue-comment readback completed before host preparation.
No browser was launched by X research. Host cleanup used the mandated helper.

The previous baseline-only GPU restriction is now removed in the active developer
instructions. A DIFFERENT, stricter developer host-memory threshold remains:
MemAvailable must be >45 GiB, while the user's Tier A says >=34 GiB.

Actual post-cleanup/post-qli-stop guard: **41.77951431274414 GiB MemAvailable**, GPU
utilization **0%**. User Tier A passed; developer threshold failed. No FORCE,
no launcher/model load, no actual model trace, and no GPU compute performed.
Therefore new decode tok/s, model run RAM peak against 40 GiB and VRAM peak against
28 GiB are all **unknown/not measured**, not zero. See outcome.json and raw logs
under run-20260906T120915/. These host preflight numbers are NOT run-footprint.

Qli stop request12:09:17.445815; restart request12:09:20.718604 CEST.
Qli active and headless Chromium active verified; journal contains [qli-tune] OK.
No cleanup error. Both services independently rechecked active after harness exit.

Final CPU build: green. CUDA llama-server build: green (compilation, not GPU run).
Targeted CTest12/12; broad main30/32. BERT tokenizer/chat-template failures match
prior committed evidence/expert-gemm/ctest-main.log; no fresh baseline checkout.
Review initial NO-GO, concrete fixes in review-notes.md; no second review GO claim.
Two test-fixture mistakes were fixed (Mock path operator, journal label mismatch).
Raw test logs deliberately retain emitted whitespace (git diff --check warns on
main.log); code diff whitespace check passed before adding raw logs.

Next: actual phase/context attribution and validated runtime-to-GGUF file-offset
join, then a real model capture when the active host guard is met. Physical
NVMe/RAM/PCIe rates and GPU quant-GEMM, advisor input and model-shadow-pressure
validation remain open. This slice provides tested logical CPU observation
instrumentation, not integral completion of phase2 or >=20 tok/s on deep context.
