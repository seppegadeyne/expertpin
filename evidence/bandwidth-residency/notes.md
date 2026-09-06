# CPU buffered-I/O observations — 2026-09-06

## Scope and execution

This closes the **cache-state/accounting observation** sub-slice, not phase 2.
No qwen model load, GPU work, true expert-offset trace or GEMM benchmark.
The fixed developer gate allows GPU work only before the already-completed Step-1
baseline, so the requested model trace / GPU GEMM remains blocked this run.

Final measured script SHA-256:
`89f3381b4b8f16e70b02944920d9b2735b29394eff4d8d46f101359a56ab76c9`.
Python 3.11.15 on Aorus. `findmnt -T` reports `/dev/mapper/root[/@home]`, btrfs,
mounted at `/home`; no assumption about isolated device traffic is made.
No tests/builds/review ran concurrently with these final reads. Miner and other
host tasks remained active. No memory locks, global cache drops or GPU operations.

File for all four measurements:
`/home/seppe/Models/qwen3.8-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00003-of-00033.gguf`

For each sample execute:
```
python3 scripts/expert_bw_calib.py --file <FILE_ABOVE> --length 1024 \
  --chunk-bytes 704000 --pattern random --seed 17 --observe \
  [--cold] --json <OUTPUT.json>
```
Order: cold-1, warm-1, cold-2, warm-2. `--cold` only on cold-*.
These files contain the **final post-review rerun**, replacing the initial
pre-review pair measurements. Not a randomized crossover study or speedup proof.

| Sample | Read-loop seconds | Buffered payload GiB/s | Resident pages before → after | Process storage delta |
|---|---:|---:|---|---:|
| cold-1 | 0.6415760719683021 | 1.559 | 0 → 262144 | 1073741824 B |
| warm-1 | 0.07048709702212363 | 14.187 | 262144 → 262144 | 0 B |
| cold-2 | 0.6548557298956439 | 1.527 | 0 → 262144 | 1073741824 B |
| warm-2 | 0.0685835579643026 | 14.581 | 262144 → 262144 | 0 B |

Each requested prefix is 1073741824 bytes, 262144 pages. Rate rounding verified
with jq against raw JSON. The cold snapshots are now observed nonresident, not
merely inferred from requesting DONTNEED. All residency snapshots are non-atomic;
process read_bytes includes readahead and is not isolated physical NVMe traffic.
Warm buffered copying is not hardware RAM bandwidth. Access order is a shuffled
prefix partition with expert-sized chunks, **not actual expert offsets or routing**.

Advisor rerun: `local-incomplete.json` → `local-advice.json`, exit 2/status blocked,
recommendation null. Same analytical conclusion as `15335f1e`: no defensible
hardware calibration / complete candidate profile yet. No invented input rates.

## Independent review and own verification

Reviewer `deleg_2516dadc`, pinned gpt-6-astra, full initial diff: NO-GO pending
ownership and environment-dependent test fixes. No second review pass.

All four findings processed:
1. **fsuid vs euid masking**: parent inspected Linux `mm/mincore.c` via curl:
   `can_do_mincore` checks file ownership/write permission and otherwise returns
   all-one vectors. Parent independently read kernel `fs/inode.c`
   `inode_owner_or_capable` using `current_fsuid()` (extracted source lines
   2457–2468), plus proc_pid_status(5) defining the fourth Uid field as fsuid.
   Reproduced a mocked divergent-fsuid test RED (`1 is not None`), then GREEN.
   Final `scripts/expert_bw_calib.py:83-115` reads thread fsuid, validates namespace
   euid and rejects unknown/mismatched ownership without changing credentials.
   No live setfsuid/credential manipulation was attempted.
2. **Unstable integration assertions**: real snapshot tests now verify structure
   and bounds, skip explicitly when native observations are unavailable. CLI
   tests accept unavailable observations; no unconditional warm or zero-I/O
   assertion. Deterministic cache-state/counter fixtures remain strict.
3. **Missing ctypes**: reproduced ImportError RED, then caught as null+error in
   `scripts/expert_bw_calib.py:145`; optional test import and native pointer-size
   checks prevent interpreter/target-platform confusion in native fixtures.
4. **Coverage**: added synthetic libc tests for multiple batches, partial final
   page, higher vector bits, late mincore failure, mmap/munmap failure and isolated
   4-GiB cap. Eventlog+fixed clock checks accounting boundaries and rate formula.

These are self-verified fixes, NOT an assertion of reviewer approval after edits.
Initial TDD RED was seven missing-API errors; the review REDs above are genuine
behavioral regression checks. Kernel review source is read-only external research:
- https://raw.githubusercontent.com/torvalds/linux/master/mm/mincore.c
- https://raw.githubusercontent.com/torvalds/linux/master/fs/inode.c
- https://man7.org/linux/man-pages/man5/proc_pid_status.5.html

A web_extract result for mincore incorrectly returned a setfsuid manual page;
parent rejected it and used a direct curl readback of the actual kernel source.

## Verification and remaining work

- Full CPU build `cmake --build build-cpu-shadow -j 8`: PASS, GGML_CUDA=OFF.
- New observation suite: 14/14 on Python 3.11.15 and 3.14.7, no skips locally.
- Existing bandwidth/advisor suite: 12/12 on Python 3.11.15; also CTest PASS.
- Targeted CTest: 9/9, includes expert-manifest; log `ctest-targeted.log` (local).
- Broad main CTest: 27/29; the same recorded baseline failures
  `test-tokenizer-0-bert-bge` and `test-chat-template`; log `ctest-main.log` (local).
- `git diff --check`: PASS. No C++ runtime changes.

No new tok/s, model run RAM footprint or VRAM peak measured. Historical evidence
only: target-only @8K 54–56 tok/s with cgroup MemoryPeak 33.83 GiB <40 and VRAM
26.0 GiB <28; >=20 tok/s at deep context remains unproven.

Next: phase-tagged chronological expert/file-offset capture, trace-matched reads,
RAM/PCIe and quant-specific CPU/GPU GEMM. Model pressure with a smaller shadow cap
and both arm orders remains open. Phase 2 and issue #2 remain OPEN.
