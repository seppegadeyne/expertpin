# Dev-serving recipe: qwen3.8-flash-next (ps-iq2xxs) as the local Hermes backend on Aorus

Status: PROVEN end-to-end on 2026-09-11 — a real Hermes agent completed a
shell-tool task against this exact setup (single-turn AND multi-turn with
session resume). Evidence: evidence/hermes-e2e-20260911/ (e2e-report.md,
multiturn-outcome.json).

## 1. Serve the model (guarded)

```bash
# Miner first (protocol), then:
CTX=65536 scripts/run-qwen38-flash-next.sh   # or higher; see budget notes
```

Why CTX >= 65536: Hermes refuses servers reporting a context window below
64K, even if the model supports more. Budgets measured at 64K on this
checkpoint: RAM ~32.4/40 GiB, VRAM ~17-19/28 GiB (needle-ladder evidence,
2026-09-11). The full trained window (262144) also passed the needle gate
at RAM 32.41/40, VRAM 19.94/28 — with a 12.45 GiB cgroup swap peak during
250K prefill (reclaim pressure, no OOM) and ~5-9 min prefill per fresh
position; prefer 64K for interactive work.

Notes:
- The server ignores Authorization; any placeholder key works client-side.
- The model id IS the GGUF path — exactly what GET /v1/models reports.
- Server errors all surface as HTTP 500 {"error": {...}} (even input
  errors OpenAI would call 400); branch on the message, not the class.
- tool_choice must be the string form ("required"/"auto"/"none"); the
  OpenAI object form silently degrades to "auto" in this build.
- Streaming tool calls arrive as multiple indexed deltas; aggregate by
  index. Mid-stream failures emit `data: {"error": ...}` and the stream
  ends WITHOUT [DONE] (fail closed on truncated streams).

## 2. Configure an isolated Hermes profile (do not touch your daily one)

```bash
export HERMES_HOME=~/.hermes/profiles/expertpin-dev   # any empty dir
hermes config set model.provider custom
hermes config set model.name /home/seppe/Models/qwen3.8-flash-next-ps-iq2xxs/Qwen3.8-Flash-Next-IQ2_XXS.gguf
hermes config set model.base_url http://127.0.0.1:8102/v1
hermes config set model.api_key local-no-key-required
hermes config set model.context_length 262144
```

Gotchas that each cost an iteration (details in e2e-outcome.json):
- Use the CLASSIC custom-provider config above. The `model_aliases`
  direct-alias path does not feed the default-provider startup: the
  client exits with "No API key found for provider 'custom'" in <1 s
  (hermes_cli/runtime_provider.py — bare custom without explicit
  base_url falls through to OpenRouter).
- `context_length` is required because the server reports its CTX (65536)
  and Hermes demands >=64K — fine — but if you serve a smaller CTX Hermes
  refuses to start at all.
- A marker-check on agent output must exclude the echoed query (Hermes
  prints the query first — a substring check on the whole stdout gives
  false positives).

## 3. Use it

```bash
hermes chat -q "your task"          # one-shot
hermes chat --resume latest -q …    # continue the last session
```

Measured on this host (2026-09-11, ps-iq2xxs @ CTX 65536, nmax4/MTP):
- decode 33-52 tok/s (verified-cold 33.3, warm 50+); sustained 37-52 tok/s
  at 512/1024-token generations.
- A 2-turn shell-tool agent task: ~63 s turn 1, ~4 s resumed turn 2
  (prompt-cache reuse).

## 4. Resident serving (BUILT and proven 2026-09-11)

```bash
cp scripts/expertpin-dev.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now expertpin-dev.service   # opt-in; NOT auto-enabled
```

The unit runs scripts/serve-dev-qwen38.sh (dev launcher: same guards,
CTX 65536, no drafter, does NOT stop the miner) under MemoryHigh 33G /
MemoryMax 36G / MemorySwapMax 16G with self-healing restarts. Proven: a
cold agent query takes ~84 s; a warm query on the same instance ~16 s.
Stop with `systemctl --user stop expertpin-dev.service` before benchmark
cycles (the 03:00 cron depends on host headroom). Evidence:
evidence/dev-serving-20260911/.
