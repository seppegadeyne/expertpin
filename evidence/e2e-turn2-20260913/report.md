# Task-specific E2E second turn — 2026-09-13

## Scope and result

Fixed the code/multi-turn combination: turn 1 creates `/tmp/e2e-codegate.py`, but the previous second turn always requested `/tmp/hermes-e2e-marker.txt`, which the code task never creates. `run_second_turn` now re-executes the existing script for code tasks and retains the marker-file read for marker tasks. It uses the same monitored `run_client`, shared run deadline, isolated environment and file-backed output. Main calls the helper only after the first-turn criteria and code artifact probe succeed.

This is a CPU-verified harness fix, **not a physical code/multi-turn PASS**. Transcript command/output checks remain heuristics, not independent evidence of tool execution, immutable artifact reuse or session identity.

## Tests

- Initial test-first run failed because the new helper API did not exist (seven error reports across six test methods/subtests). This is not claimed as a behavioral RED.
- Final new suite: 8/8 passed, with mocked client results. Covers task/command selection, wrong artifact, query-only output, nonzero exit, invalid task, the existing real marker transcript, and full generated multiline query echoes.
- Existing real-subprocess monitor regressions remain included; the AST wiring check now verifies main's helper call and all three monitored client prefixes.
- CPU-only CMake configure/build (`build-cpu-shadow`, `GGML_CUDA=OFF`) succeeded.
- Final targeted CTest: 7/7 passed, including expert-manifest, client monitor, both turn suites, work budget, contract and coldcache suites (seven CTest registrations, not seven individual Python assertions).
- Full CTest before review follow-up tests: 53/56 passed. Same three failures documented in the prior run: bert-bge tokenizer, chat-template assertion, eval-callback (missing stories260K.gguf / libcurl disabled). Raw log compressed without editing; not an all-green full suite.
- `git diff --check` passed.

## Independent review and parent verification

One independent delegate approved this bounded artifact-selection fix; no blocking findings. It noted the real query echo is multiline and requested better fixture coverage. Parent verified `run-20260911T164123-e2e/hermes-turn2-stdout.txt:1-3,13-21`, corrected the docstring to say **first query-echo line**, and added real-transcript plus full-generated-query tests. Parent verified shared deadline/sampling/output/cleanup in `run-e2e.py:68-109` and task selection/monitored call in `run_second_turn`. The parent reran targeted tests after the follow-up. No second review was requested for the docstring/test-only follow-up.

## GPU and remaining gates

NVML is operational again (615.71.09). Host check at approximately 08:43 CEST: RTX 5090 GPU utilization 100%, MemAvailable 49 GiB. Jetski is active. The existing daytime instruction says to leave Jetski running; no miner stop, host preparation, model load or GPU test was performed. The requested physical rerun remains blocked by the idle guard and has **not** been completed.

No new decode, run RAM or run VRAM measurement. Historical UD-Q4_K_XL first/warm decode is 13.26/35.44 tok/s (not a new measurement). The next physical run still requires explicit <=1800 s total containment, separate external-client RAM accounting plus combined <=40 GiB enforcement, and ongoing <=28 GiB VRAM monitoring. Existing code-task default of 2700 seconds is not permission to run that long; it must not be used under the current mandate. This slice does not resolve those old guard gaps.

Final service check: Jetski and headless-chromium active and untouched; qli inactive/not-found. Qli pause was respected, never started; actual masking is not confirmed by current systemd state.

## Read-only X research (completed before testing)

- https://x.com/scharmueller/status/2098896832189575557 — reported mixed NVFP4/FP8 + MTP on Spark.
- https://x.com/Oluwaphilemon1/status/2098719084066025969 — reported EXL3 / memory hierarchy on PRO 6000.

Community claims surfaced through x_search, not locally verified or directly comparable to this 5090 within 40 GiB system RAM. No defaults changed and no browser started.
