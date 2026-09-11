Tussentijds verslag 4 2026-09-11 12:12 — koude-cache A/B AF (advisory); needle-harness-regressie gevangen en hersteld

Slice: koude-cache A/B (fadvise DONTNEED). Resultaat fysiek op ps-iq2xxs
(identieke config, alleen de fadvise verschilt):
- KOUD (drop op exact 75,2 GB): load→ready 36 s; eerste request 35,54 tok/s,
  tweede 49,45 tok/s. RAM 32,41/40, VRAM 18,89/28.
- WARM: load→ready 32 s; 43,56 / 50,40 tok/s. RAM 32,41/40, VRAM 18,81/28.
- Eerlijk: DONTNEED is advisory en er was geen mincore-verificatie — een
  volledig koude load is dus NIET bewezen (mmap raakt mogelijk maar een
  subset aan). Praktisch resultaat staat: traagste geobserveerde request
  35,5 tok/s, ruim boven 20, alle budgetten groen, geen OOM-events.

Belangrijk nevenresultaat: de git-checkout uit het eerdere herstel (3b9c8928)
had ongemerkt de needle-gate-code uit de quality-gate-harness gewist. De
test-suite ving dit (10 falende imports); hersteld in 7d14cc14, alle suites
weer groen (23/23, targeted 2/2, base 16/16).

Commits: 7d14cc14, 4a844ccc, af4a62c0 — gepusht en exact herlezen.
Issue #2-comment 5632834939 geplaatst en herlezen.

Host: qli ononderbroken actief; geen llama-server; tracked tree schoon.

Volgende stappen (state-file): 64K-context binnen 40/28 (prefill-tijd is de
uitdaging: 64K @ ~38 tok/s ≈ 28 min, rakend aan de 30-min GPU-cap — ik
overweeg een gesplitste aanpak), daarna Hermes-E2E.
