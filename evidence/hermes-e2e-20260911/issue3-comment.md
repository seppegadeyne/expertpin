Milestone 2026-09-11 — Hermes agent runs END-TO-END on the guarded local expertpin endpoint (PASS).

The core of this issue — a real Hermes agent loop against qwen3.8-flash-next-ps-iq2xxs served locally — is proven today (evidence/hermes-e2e-20260911/e2e-outcome.json, run-20260911T152411-e2e):

Session 20260911_152451_5f0772: reasoning -> terminal tool call (printf marker) -> tool result processed -> final answer exactly the command output. 4 messages, 2 tool calls, 68.6 s wall. Budgets held: RAM 32.41/40 GiB, VRAM 19.22/28 GiB, zero OOM events, at CTX=65536.

Proven dev-serving recipe (isolated HERMES_HOME):
- model.provider: custom; model.base_url: http://127.0.0.1:8102/v1
- model.name: the GGUF path (what /v1/models reports)
- model.api_key: placeholder (server ignores auth)
- model.context_length: 262144 (Hermes refuses servers reporting <64K; serve CTX>=65536)

Integration gotchas documented (each cost one preserved iteration): the model_aliases direct-alias path does not feed default-provider startup — use the classic custom provider config; the >=64K reported-context requirement; the query-echo false-positive guard on marker checks.

What this closes from the quality roadmap: with today's tool-call gate, needle ladder (2K..250K), multi-needle, error-path contract, verified-cold and sustained decode — the local checkpoint now has evidence across every axis of the 2026-09-09 mandate except SSE mid-stream error frames (minor, listed in the state file).

Remaining for this issue: turn the runner into a resident dev-serving profile (systemd user unit + keep-alive config around scripts/run-qwen38-flash-next.sh), multi-turn/long-trajectory agent testing, and wiring the isolated home recipe into a documented recipe file in the repo.
