# E2E client budget monitoring — 2026-09-13

## Scope and evidence

The runner previously blocked in `subprocess.run()` throughout both Hermes
turns and the independent artifact probe. Its `run.sample()` calls only
surrounded these calls. `audit-history.py` reproduces the evidence against
`run-20260912T035703-e2e/samples.jsonl`: 37 samples, with a **853.524696-second
gap** across the client invocation.

The historical code/artifact result remains valid, but the reported
24.258789 GiB device VRAM maximum is only a sampled maximum: the 28 GiB bound
was not established inside that gap. The server cgroup's kernel
`memory.peak` was 32.405220 GiB; unlike VRAM sampling, that high-water mark
survives gaps. It does not include the external Hermes client's RAM.

## Change

- One `run_client()` helper for both turns and the artifact probe.
- Calls the existing `Run.sample()` before spawn, during execution, and
  after client exit. Default wait interval is 1 second **plus sample latency**,
  not a hard real-time polling guarantee.
- Uses the earlier of the request deadline and the remaining shared work
  deadline; turn 2 cannot renew the full run budget.
- File-backed stdout/stderr avoid pipe backpressure and retain partial
  transcripts when the client times out or a guard fails.
- Own session/process group, SIGKILL and reap in `finally`, including a
  deferred signal immediately after spawn. Ordinary shell descendants in
  that group are terminated too. Detached sessions/external tool daemons are
  not contained: this is not a sandbox or a new aggregate-client RAM cgroup.
- No model settings, miner handling, RAM caps, or VRAM caps changed.

## Self-review (independent reviewer paused)

No delegation was dispatched: the owner's Astra review prohibition through
approximately September 15 is active; the permanent pin was not changed.
This is self-verification, **not** an independent approval.

Source checks:
- `run-e2e.py:68-109`: deadline, file-backed logs, periodic sample, owned
  process-group cleanup. `:208-210`, `:231-232`, `:245-246`: all three long
  execution sites use the helper (also checked via AST regression).
- `../coldcache-ab-20260911/run-guarded.py:301-313`: spawn defers signals
  until `spawned()`; the helper registers cleanup before releasing that gate.
- Same harness `:323-346`: sample reads NVML + cgroup counters and calls
  `validate_budget`, then checks the server is still alive. Errors propagate
  through the helper's `finally`; no guard is suppressed.
- Remaining limits: periodic sampling can miss brief VRAM spikes; monitor
  command latency is additional; inherited server cgroup does not account
  for external agent/daemon RAM; full model validation is not performed today.

## Executed tests

- Initial API/scaffold RED: 5 tests failed because `run_client` did not exist
  (not claimed as a behavioral regression against the old implementation).
- Final 8/8 CPU regressions pass with **real subprocesses**, mocked guard
  observations only: long/large-output nonzero exit, partial timeout log,
  expired pre-spawn deadline, remaining shared deadline, failing preflight,
  mid-client guard failure with descendant termination, deferred signal
  cleanup, and all three integration call sites.
- CPU build (`GGML_CUDA=OFF`) succeeded.
- Targeted CTest: 6/6 pass, including `test-expert-manifest`.
- Full CPU CTest: 52/55 pass. Existing failures remain:
  `test-tokenizer-0-bert-bge`, `test-chat-template`, `test-eval-callback`
  (last one cannot load stories260K.gguf with this no-libcurl build).
- `git diff --check` passes. Logs are stored alongside this report.

## GPU gate and next step

No model/GPU run. Read-only `nvidia-smi` fails with exit 18,
`Failed to initialize NVML: Driver/library version mismatch`; NVML reports
615.71, installed nvidia-utils/module-on-disk report 615.71.09. The loaded
module version was not established. No driver repair/reboot attempted.
MemAvailable was 43 GiB in `free -g`, but GPU utilization/VRAM cannot be
verified, so the Tier A conjunction is blocked. No services were stopped.
Jetski and headless Chromium remain active; dev serving remains inactive.
qli is paused/not started; systemd actually reports `not-found/inactive`,
not a verified masked unit.

Next: restore a working NVML/driver stack through normal host maintenance,
then repeat the UD code/multi-turn gate with this monitor and an explicit
<=1800-second total run budget. Report aggregate client RAM separately;
do not reuse the historical VRAM maximum as a continuous budget proof.
The current UD checkpoint's historical throughput is 13.26 then 35.44 tok/s
in the two-request evaluation: warm exceeds 20; the first request does not.
No new throughput improvement is claimed by this safety slice.

## Read-only X research (completed before tests)

No directly comparable new 5090/40-GiB measurement found. Community claims
are not local validation and did not justify a checkpoint switch:
- https://x.com/Oluwaphilemon1/status/2098719084066025969 — EXL3 ~4.25 bpw
  and memory hierarchy on a 96-GB RTX PRO 6000 (different budget/hardware).
- https://x.com/Oluwaphilemon1/status/2098517502066377096 — dual-Arc MTP
  and CPU embedding offload (different backend/hardware).
- https://x.com/MiaAI_lab/status/2098712719956361351 — EXL3 installer for a
  27B checkpoint, not this project's checkpoint.
