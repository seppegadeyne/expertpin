# Expertpin — 06-09, schone MTP-A/B

Uitgevoerd: drie aparte guarded modelloads, telkens twee identieke requests van 68 prompttokens + 256 decodetokens. Vaste prompt/seed42, temperatuur0, CTX8192, NCMOE36, geen traces/shadow; productie-graphs niet uitgeschakeld. Alleen n_max (en de daarvan afhankelijke recurrente checkpointcapaciteit) gewijzigd.

| n_max | Decode tok/s, eerste → herhaling | MTP-acceptance | RAM-piek /40 GiB | VRAM-samplepiek /28 GiB |
|---|---|---|---|---|
| 4 | 6,48 → 8,49 | 66,48% | 32,405 | 22,652 |
| 8 | 6,13 → 7,80 | 62,11% | 32,407 | 22,658 |
| 16 | 7,87 → 9,83 | 59,60% | 32,405 | 22,658 |

**Doel ≥20 tok/s niet gehaald.** Hoogste waarneming 9,834792 tok/s; tekort 10,165208 tok/s. n_max16 is in deze geordende proef het snelst, maar geen bewezen causale winnaar: alle herhalingen versnellen, OS-cache/swap is niet gestabiliseerd en de gegenereerde tekst verschilt tussen n_max-waarden. Binnen elke n_max zijn beide tekstantwoorden hash-identiek. Geen kwaliteits- of lange-contextclaim en geen definitieve defaultwijziging.

Acceptance per positie is volledig bewaard. Bij n_max16: positie1 78/137; positie2 16/23; positie3 10/16; positie4 4/7; positie5 3/4; posities6–12 elk1/1; posities13–16 elk0/1. De hoge posities hebben dus nauwelijks waarnemingen; geen stevige acceptancecurveclaim. De historische 95% uit 32 tokens houdt hier niet stand.

Budgetten: MemoryMax36 GiB, hoogste gelezen cgroup memory.peak zoals tabel; finale scopejournals bevestigen afgerond32,4 GiB. VRAM is device-wide gesampled, geen continu maximum. Swap-samplepieken respectievelijk2,214/2,203/2,195 GiB apart. Alle memory.events max/oom/oom_kill0. HostTierA44,24–44,91 GiB, GPU2/0/0%; DRYguards groen. Iedere GPU-run ruim onder30 minuten. Qli na iedere run herstart; eigen scopes leeg/weg geverifieerd, cleanup_errors[]. Qli en headless Chromium finaal active; [qli-tune] OK om15:51:26. Geen llama-server over.

Codecommit `812bbba8` gepusht en origin-readback bevestigd. Tests: nieuwe unit14/14, bestaande targeted16/16, CUDA-serverbuild groen. Main37/39: bekende BERT-tokenizer/chat-templatefailures blijven bestaan, dus niet alles groen.

Onafhankelijke gpt-6-astra-review vond een cleanupblokker: unitstatus alleen bewijst geen lege cgroup. Parent heeft dit zelf bevestigd en hersteld met cgroup-readback/escalatie plus regressietests; geen tweede onafhankelijke APPROVE geclaimd. Parent controleerde ook server-context.cpp1167–1193/1234–1242: recurrente checkpoints vereisen aparte starts voor4/8/16. Implementatiedelegatie gaf420s-tooltimeout maar liet werkende bestanden achter; review afgerond vóórGPU.

X-research vooraf, alleen als ideeënbron: https://x.com/ItsmeAjayKV/status/2094895868621345162 (n_max-sweep); https://x.com/inovelloE/status/2096577582741238212 (cache+MTP). Geen externe prestatiecijfers overgenomen; geen browser gestart.

Volgende stap: tegenvolgorde/herhaalde4↔16 op meerdere vaste prompts, inclusief controle waarom greedy outputs verschillen; pas daarna een defaultkeuze of één placement-A/B. Model-shadow-pressuregate en matched fysieke advisor-inputs blijven open. Geen requantfallback gestart.

Evidence: `evidence/clean-mtp-ab/outcome.json`, drie `run-*`-directories met requests/responses, per-depth counters, budgetsamplen en servicelogs. Ruwe response-timings en totals door parent nagerekend. Mailstatus volgt in state-file, inboxaflevering wordt niet aangenomen.
