# Tier A capture harness — review and validation

## Scope

The active developer mandate now permits MemAvailable >=34 GiB and physical
phase-2 GPU measurements. This supersedes the historical >45 GiB refusal; no
FORCE override. This capture selects RAM_BUDGET_GIB=36, so the unmodified launcher
still requires >=40 GiB available (budget+4) and enforces MemoryMax=36 GiB,
stricter than the global 40 GiB RAM ceiling. DRAFT=1, 8K context, 32 new tokens,
8 GiB logical shadow. VRAM abort threshold is now >=28 GiB.

## Independent review

Reviewer deleg_0fa26fc3 (gpt-6-astra), initial NO-GO: old partial-launch signal
window and missing post-kill scope verification; guard change itself consistent.
Parent verified the original Popen-before-scope flag and kill-without-readback
in the source. Fixed in run-guarded.py:131-140 (ownership before spawn; handled
signals deferred until child registration, without inherited signal masks) and
:213-226 (reap launcher, exact scope readback and bounded kill/recheck).
New CPU AST fault injection exercises signals during Popen, Popen failure,
permanently active scope and kill failure. Restore attempts remain isolated;
qli restart is mandatory even if cleanup fails, but that case is explicitly
reported as cleanup_failed, NOT a safety success. No second review GO claimed.

## Evidence limits

memory.peak is the sampled cgroup lifetime high-water mark; memory.current,
swap.current and memory.events are retained. GPU memory is device-wide periodic
nvidia-smi samples, NOT a hard allocation cap or a guaranteed continuous peak.
A short unobserved excursion cannot be ruled out. Trace perturbs inference;
any response timing is diagnostic instrumented throughput, not a new baseline.
No phase/request tags, absolute GGUF offsets or physical bandwidth claims.

## CPU validation before GPU

Six harness test methods pass, including existing exception/restore tests.
Targeted CTest 12/12; broad main 30/32 (same existing BERT tokenizer and chat
validation failures as prior main.log). CPU full build and CUDA llama-server
build pass; the latter is compilation only. First exploratory build-cpu path
was absent; correct existing build is build-cpu-shadow. New cap assertions were
observed failing against old code, then passing; initial host_guard RED was
missing API, not behavioral proof. Real outputs in *-tier-a.log.
