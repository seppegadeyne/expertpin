Slotverslag 2 2026-09-11 15:40 — MIJLPAAL: Hermes-agent draait E2E op het lokale expertpin-endpoint

Na multi-needle (3/3 codes @ 2K én 64K, alles binnen 40/28) is nu ook de
kern van issue #3 bewezen: een ECHTE Hermes-agent-run tegen de guarded
lokale ps-iq2xxs-server.

Sessie 20260911_152451_5f0772: reasoning → terminal-tool call (printf)
→ tool-result verwerkt → eindantwoord exact de commando-output.
4 messages, 2 tool calls, 68,6 s; RAM 32,41/40, VRAM 19,22/28, geen OOM,
CTX 65536.

Bewezen dev-serving-recept (geïsoleerde HERMES_HOME):
- model.provider=custom, base_url=http://127.0.0.1:8102/v1
- model.name = GGUF-pad (wat /v1/models rapporteert)
- api_key placeholder; context_length 262144
- server moet >=64K ctx rapporteren (Hermes-minimum)

Vier iteraties met valkuilen bewaard als evidence (alias-path voedt
default-opstart niet; <64K ctx wordt geweigerd; query-echo
vals-positief). Alles in evidence/hermes-e2e-20260911/ (e2e-report.md).

Commits vandaag deze sessie: ad62de55 t/m 8ee22350 — gepusht en exact
herlezen; issue #3-comment 5635193102 geplaatst en herlezen; reviewverbod
gpt-6-astra nageleefd (0 dispatches).

Resterend voor issue #3: resident dev-serving-profiel (systemd unit +
keep-alive), multi-turn/long-trajectory agent-tests, SSE mid-stream
error-frame contract.
