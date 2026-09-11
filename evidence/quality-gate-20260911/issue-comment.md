Progress update 2026-09-11 (morning slice) — checkpoint quality gate, step 1: tool-call JSON correctness.

Follow-up to the 03:00 checkpoint A/B (comment 5628043948): the state file listed "IQ2_XXS quality gate: tool-calling JSON + needle-recall before default-switch" as the next step. The tool-calling half is now done; needle-recall remains open.

What was built (fda74ed1):
- evidence/quality-gate-20260911/run-guarded.py — a copy of the guarded clean-MTP harness with a new --gate toolcall mode: target-only (DRAFT=0), forced single get_weather tool with an enum-constrained schema, tool_choice sent as the string "required" (the OpenAI object form silently degrades to "auto" in this server build because tool_choice is parsed via json_value(..., std::string)), round 2 feeds the canned tool result back and requires a clean stop answer.
- 14 CPU-only regression tests (tests/test-quality-gate-20260911.py, ctest label "targeted"); existing suites unchanged: 16/16 base harness, 9/9 compare-suite, ctest main 39/41 with the two known pre-existing failures.

Physical A/B (same binary SHA 3235529f... and serving config as the 03:00 run; only MODEL/MODEL_DIR differ):
- ps-iq2xxs: round 1 exact get_weather {city: Ghent, unit: celsius} with id (5.1 s, 411 prompt / 71 completion tokens); round 2 "The current weather in Ghent is **17.5°C** and cloudy." (2.1 s). RAM peak 32.41/40 GiB, VRAM peak 16.18/28 GiB.
- reference: round 1 exact same call (13.2 s, 80 completion tokens); round 2 same numeric answer (5.1 s). RAM peak 32.41/40 GiB, VRAM peak 20.09/28 GiB.
- Gate verdict: PASS on both checkpoints. Notably the IQ2_XXS arm answered the tool round in 2.1 s vs 5.1 s on this warm single-shot scenario.

Limitations (unchanged honesty): n=1 forced single-tool scenario, greedy temp 0; no multi-tool selection, parallel calls, streaming deltas, or long trajectories; no needle-recall or 64K/264K context; no Hermes end-to-end. The reference arm needed one retry after an EADDRINUSE refusal (port 8102 TIME-WAIT from the previous arm) that aborted before any host change.

Evidence: evidence/quality-gate-20260911/ (outcome.json, report.md, review.md, research.md, three run dirs including the preserved refusal). Both review attempts (deleg_1901d667, deleg_e1f13ec2) failed with HTTP 429 usage-limit again; core premises were parent-verified against server/launcher sources and documented in review.md.

Next steps from the state file: (1) needle-recall slice to complete the quality gate, (2) sustained decode depth on ps-iq2xxs (2x512/1024) within 40/28, (3) controlled cold-cache A/B, (4) 64K context within 40/28.
