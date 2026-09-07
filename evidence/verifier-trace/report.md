# Expertpin — 07-09 dagverslag

**Slice:** bounded echte verifierdiagnostiek gebouwd en gepusht in
`c1791a20` (`feat: trace bounded raw target verifier decisions`). Opt-in trace
legt originele proposal-/target-ID's, raw argmax/top-two-logitmarge, batchinformatie
en acceptancebeslissing vast: maximaal 128 records/request, 4096/proces.
Geen wijziging van acceptance, n_max-default, placement of quantisatie.

**Uitvoering:** CPU- en CUDA-serverbuild geslaagd; targeted tests **20/20**,
analyzer **7/7**, harness **14/14**, target-only tests **6/6**. Main-suite **39/41**:
de bestaande BERT-tokenizer- en chat-templatefailures blijven. Geen alles-groenclaim.

**GPU geblokkeerd vóór modelload:** host Tier A was groen (39,893 GiB beschikbaar,
GPU 2%), maar de extra launcher-guard eist 36 + 4 GiB en gaf `WOULD BLOCK`.
Meteen afgebroken; geen budgetverlaging/forceren/herpoging. Target-only vervolgrun
niet gestart. Er zijn dus **geen nieuwe tok/s, RAM-runpiek of VRAM-runpiek** en nog
geen fysieke verifiertrace. Het preflight-VRAMsample (2,379 GiB) is geen modelrunpiek.
Doel blijft **≥20 tok/s binnen 40 GiB RAM en 28 GiB VRAM**, niet nieuw aangetoond.
Historisch beste schone 256-tokenrequest: 14,26 tok/s; niet opnieuw gemeten.

**Correctness:** greedy n_max-divergentie blijft onverklaard. Nieuwe hooks meten
echte samplerbeslissingen, geen hertokenisatie. Records precederen commit/stop-
afhandeling; niet automatisch bewezen afgeleverde tokens. Numeriek versus
recurrent/KV en runtime-niet-interferentie blijven open.

**Review:** onafhankelijke gpt-6-astra-review vroeg wijzigingen: bronartifact-
validatie, strikte raw/rowchecks en aparte requestcap bij rewinds. Zelf tegen code
geverifieerd; tien echte RED-checkfailures, daarna fixes en tests groen. Geen tweede
onafhankelijke APPROVE geclaimd. Parent verifieerde bias/samplingplaatsing in
common/sampling.cpp635–657, root/draftmapping in server-context.cpp3616–3648 en
rewindgedrag4637–4640. Details: `evidence/verifier-trace/review.md`.

**Herstel:** qli stop/start 03:16:13–03:16:17 (4,112 s tussen verzoeken).
`qli.service` en `headless-chromium.service` beide **active**; `[qli-tune] OK`
in journal; geen llama-server; cleanup_errors leeg.

**Volgende stap:** bij voldoende launcher-headroom guarded MTP4 en target0,
ieder 2×256 tokens, trace vergelijken en exacte outputs toetsen aan eerdere runs.
Daarna identiek-prefixreplay om batchnumeriek versus recurrent/KV te isoleren;
geen enginefix of snelheidstuning vóór deze correctness-diagnose.

X-research afgerond vóór tests/review/GPU; alleen ideeën, geen externe cijfers
als lokaal bewijs overgenomen. Bronnen:
- https://x.com/inovelloE/status/2096577582741238212 (batched-cache/MTP-lead)
- https://x.com/geldeki/status/2096738747547013337 (EXL3/offload-lead)
- https://x.com/Tech2Wild/status/2096754559263617169 (W4A16/MTP-lead)

Bewijs: `evidence/verifier-trace/outcome.json`, build-/testlogs en de originele
weigering onder `evidence/clean-mtp-ab/run-20260907T031610-cd4123bcf611408a88757c3dc37eb554/`.
