# Clean MTP sweep: source contract and blocking constraint

No model/GPU/service/host actions were run while implementing this harness. The existing research.md was read before CPU tests. The original trace harness is unchanged.

## Startup=4 is not a valid one-load 4/8/16 sweep

- `examples/server/server-context.cpp:1234–1242` explicitly rejects a per-request maximum above the startup maximum when `llama_model_has_recurrent(model)` and speculative decoding are enabled. Its error asks for a restart with higher startup checkpoint capacity.
- `src/llama-arch.cpp:335–342` identifies Qwen4Exp (the actual model architecture) as hybrid; `src/llama-model.cpp:2401–2402` includes hybrid models in `llama_model_has_recurrent`.
- `common/speculative.cpp:1410–1412` allocates checkpoint capacity from startup maximum plus one.

Parent verified server-context.cpp:1234–1242 directly and changed the experimental implementation: run separate guarded invocations `--startup-n-max 4`, then `8`, then `16`. Each repeats twice at the matching startup/request depth, thus preserving a real startup=4 baseline without requesting invalid checkpoint expansion. Reloads and unflushed OS cache/order remain confounds; no extra approval or server change is needed to execute the user's explicitly requested n_max experiments. The earlier proposed one-load sequence was rejected, not run.

## Request override genuinely reaches MTP

- `examples/server/server-common.cpp:933–940`: chat endpoint copies remaining request fields, preserving llama-specific options.
- `examples/server/server-common.h:96–123`: direct dotted keys are supported (as are nested keys). Harness uses the literal `speculative.n_max` key.
- `examples/server/server-context.cpp:1167–1193`: resets speculative parameters per request, reads n_max, and clears startup stage n_max overrides when a flat request override exists. This prevents startup `mtp:n_max=16` from silently winning over request n_max=4.
- `common/speculative.cpp:1605–1631`: resolves runtime stages and passes runtime parameters to the selected implementation.
- `common/speculative.cpp:378–388`: MTP's `mtp_speculative_gen_draft` receives `params.n_max`.
- `scripts/run-qwen38-flash-next.sh:89–93`: guarded launcher uses `--spec-type mtp:n_max=${DRAFT_NMAX}`.

## Completion and acceptance semantics

- `examples/server/server-context.cpp:1594–1595`: `ignore_eos=true` bans the model EOS token when available; other stop behavior is not assumed away. The harness requires exactly 256 completion and predicted tokens, finish_reason=length, and nonblank content/reasoning; any early stop fails.
- `examples/server/server-context.cpp:3587–3593`: draft_n counts drafted tokens actually tested; each tested depth increments its draft counter.
- `examples/server/server-context.cpp:4307–4314`: accepted count is `ids.size()-1`, excluding the extra target token; each accepted prefix depth increments its accepted counter.
- `examples/server/server-context.cpp:662–679` and `examples/server/server-task.cpp:20–39`: emit aggregate and one-indexed per-depth draft/accepted counters. Per-depth acceptance is accepted_at_depth / tested_at_depth, not divided by total output tokens; overall acceptance is draft_n_accepted / draft_n.
- `examples/server/server-task.cpp:395–406`: nonstreaming chat response includes timings. Full response and full timings are retained; aggregate/depth totals are validated.

## Safety and evidence

Environment removes every inherited key with any requested trace prefix and both graph-disable keys; shadow statistics/residency are explicitly disabled. Inherited LLAMA_ARG_* settings are also removed to prevent hidden server parameter overrides. Launcher parameters are fixed, including no-pinned, 8192 context, ncmoe=36, 36 GiB cgroup and 512 MiB prompt-cache budget. No dmon or diagnostic traces are launched.

UUID-owned scope/request cleanup, spawn signal deferral, post-launcher scope verification, mandatory qli restart, and host restoration derive from the original lifecycle. The existing bounded idle gate is reused. Work alarm is 660 seconds, readiness at most 360 seconds, each request at most 180 seconds within the shared work deadline; cleanup commands/child waits are individually bounded with a reserve below the 15-minute total under normal OS timeout delivery. No timing guarantee can survive an uninterruptible kernel operation.

Every request and raw response has an ordinal/depth label; full metrics and append-only budget samples carry that label. VRAM must remain strictly below 28 GiB, recorded cgroup peak at most 40 GiB, actual MemoryMax exactly 36 GiB, and cgroup max/OOM events abort. Tier A >=34 GiB available and <5% GPU utilization is a pre-load idle condition, not an inference-utilization cap. Guarded launcher independently enforces budget+4 GiB headroom. Runs are ordered, caches are not flushed, and repeated fixed-seed greedy requests do not remove warmup/cache/order effects.

CPU verification command: `python3 tests/test-clean-mtp-ab.py -v`. Test fixtures are explicitly synthetic and are not performance evidence. Independent review and any required commit/approval remain prerequisites to a real run; this implementation does not execute them.
