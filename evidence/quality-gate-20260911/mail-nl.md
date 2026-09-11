Tussentijds verslag 2026-09-11 10:15 — toolcalling-kwaliteitsgate AF (PASS beide checkpoints)

Slice: VOLGENDE-item 2 eerste helft — IQ2_XXS-toolcalling-JSON-gate vóór
default-switch. Needle-recall volgt als aparte slice.

Resultaat fysieke A/B (zelfde binary/config als de 03:00-run; één variabele
MODEL/MODEL_DIR):
- ps-iq2xxs: exacte get_weather-call {city: Ghent, unit: celsius} + id (5,1 s);
  round-2 antwoord reflecteert 17,5 °C correct (2,1 s). RAM 32,41/40 GiB,
  VRAM 16,18/28 GiB.
- reference: zelfde exacte call (13,2 s), zelfde numerieke round-2 (5,1 s).
  RAM 32,41/40, VRAM 20,09/28.
- GATE: PASS op beide. IQ2_XXS verliest geen toolcalling-correctheid op dit
  scenario en is hier (warm, single-shot) ook sneller klaar.

Commits: fda74ed1 (harness + 14 tests), f188c7dd (evidence), 815448ae (raw
logs). Alles gepusht naar origin/main en exact herlezen. Issue #2-comment
5631424021 geplaatst en herlezen.

Reviews: beide pogingen (deleg_1901d667, deleg_e1f13ec2) faalden opnieuw met
HTTP 429 usage-limit bij gpt-6-astra. Kernpremises zelf geverifieerd in de
serverbron (o.a. tool_choice-object degradeert stil naar "auto" — vandaar de
string "required" in de gate); zie evidence/quality-gate-20260911/review.md.

Host: qli stond al inactive sinds 09:39 (epoch-230 einde, geen crash); na de
runs qli+Chromium actief geverifieerd, geen llama-server meer actief.

Volgende stappen (state-file): needle-recall-gate, sustained decode op
ps-iq2xxs, koude-cache A/B, 64K-context binnen 40/28.
