# Expertpin — tegenvolgorde MTP, 06-09-2026

**De n_max-ranking houdt niet stand. Geen nieuwe default gerechtvaardigd.**

Eén slice: bestaande schone 256-token benchmark herhaald in volgorde **16 → 8 → 4**, twee requests per modelload. Zelfde prompt, seed42/temp0, cache_prompt=false, DRAFT1, CTX8192, NCMOE36; traces/shadow uit. Geen engine-, placement- of modelwijziging. De serverbinary-SHA256 is vóór/na gelijk aan de eerdere forward-run.

| n_max | Decode tok/s, herhaling 1 / 2 | RAM-piek GiB / 40 | VRAM-samplepiek GiB / 28 |
|---|---:|---:|---:|
| 16 | 4,857 / 8,257 | 32,405 | 22,658 |
| 8 | 10,569 / 14,260 | 32,405 | 22,658 |
| 4 | 8,370 / 10,883 | 32,405 | 22,652 |

RAM = hoogste gelezen cgroup memory.peak; finale scopejournals bevestigen afgerond 32,4 GiB. MemoryMax=36G, max/oom/oom_kill=0. VRAM = device-wide steekproeven, geen continue piekgarantie (138/86/98 samples). Swap apart: 2,188 / 2,188 / 2,195 GiB. Tier A vóór laden: 44,832 / 45,518 / 46,060 GiB beschikbaar, GPU 0/2/2%; DRY-guards groen, GGML_CUDA_NO_PINNED=1.

Beste nieuwe request **14,260 tok/s**, nog **5,740 tok/s onder 20**. Pooled decode-rate over beide herhalingen: eerder n16=8,741 > n4=7,348 > n8=6,862; nu n8=12,140 > n4=9,463 > n16=6,117. Alle tweede requests zijn sneller. Dit weerlegt een stabiele winnaar in deze proef, maar identificeert cache/swap/hostbelasting niet causaal. **Bestaande n_max=4 behouden als conservatieve referentie, niet als bewezen snelste. n_max=16 niet promoveren tot default-kandidaat.**

**Outputcontrole:** alle twaalf teksten systematisch vergeleken: 66 paren, 18 gelijk en 48 verschillend. Binnen elke diepte zijn beide herhalingen én forward/reverse exact gelijk; tussen dieptes verschillen alle teksten. SHA256 en unified diffs staan in `outcome.json`/`output-differences.md`. Hash betreft content+reasoning_content, niet token-ID's. Het zijn verschillen in de reasoningtekst, niet alleen formatting; geen kwaliteitsbeoordeling. Parent broncontrole `common/sampling.cpp:546–548,790–807` bevestigt greedy targetkeuze en draftvergelijking: dit is **niet bewezen alleen normale speculatieve branching**. Numeriek batchgedrag/recurrente staat of een defect blijven onbekende oorzaken. Acceptance blijft per diepte identiek aan forward: 118/198, 118/190, 119/179; volledige positiecurves bewaard.

**Tests/review:** nieuwe analyzer 9/9, bestaande harness 14/14, targeted 16/16 groen. Main 37/39: historische BERT-tokenizer/chat-template-failures blijven. Review `deleg_cc888a59` vroeg wijzigingen: duplicate runs/requests en ontbrekende launch-provenancechecks. Parent reproduceerde drie rode regressies en herstelde gates plus tie-ranking; daarna groen. Geen tweede onafhankelijke APPROVE geclaimd. Codecommit **d0a1d8c6**, gepusht en exact origin/main herlezen.

**Herstel:** qli na elke run gestart; scopes leeg/verdwenen geverifieerd, cleanup_errors=[]; laatste herstart 16:19:31, `[qli-tune] OK`; qli en Chromium active. Stop/starttijden staan in metrics.json. Geen llama-server achtergebleven.

**Volgende:** eerst diepte-afhankelijke greedy outputdivergentie lokaliseren met target-only/tokenvergelijking; daarna gerandomiseerde herhalingen/meerdere vaste prompts om een diepte te kiezen vóór placement-A/B. Multi-prompt vandaag niet uitgevoerd. Model-shadow-pressuregate/advisor-inputs blijven open; requant niet gestart.

X-research vóór tests/GPU: [Inovello](https://x.com/inovelloE/status/2096577582741238212), [lefu777](https://x.com/lefu777/status/2096059297046437921), [murasametech](https://x.com/murasametech/status/2096153986399391892). Alleen ideeën over cacheadaptatie/sampling/outputcontrole; externe modellen en snelheden niet met onze metingen gelijkgesteld. Geen browser gestart of externe posts geplaatst.

Evidence: `evidence/mtp-reverse/`, raw runs onder `evidence/clean-mtp-ab/run-20260906T161*`.
