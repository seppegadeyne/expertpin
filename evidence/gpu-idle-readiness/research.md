# X research — 2026-09-06, vóór tests en service-stop

Read-only x_search afgerond. Zoektermen: Qwen3.8 Flash Next, AD-4.27bpw,
n-cpu-moe, expert-cache, EXL3, MTP draft, INT4/W4A16, pageable PCIe staging.
Geen browsers/tabbladen of lokale browserprocessen gestart door deze tool.

Leads uit zoekantwoord (geen geverifieerde benchmarkdata):
- https://x.com/SamSchieds/status/2096508476499562784 — expert-cache als idee;
  AMD-resultaten zijn geen RTX 5090-metingen.
- https://x.com/JonathanLeaders/status/2095715911072207016 — AD/MTP-afstemming;
  geen nieuwe snelheid of budgetclaim overgenomen.
- https://x.com/laputa76557045/status/2096378476500771202 — architectuurspecifieke
  MTP-compleetheid; lokale reeds gevalideerde draft niet vervangen.

Zoekantwoord vermengt modelgroottes/backends en geeft niet-geverifieerde scripts.
Die niet uitvoeren. Suggestie pinned-memory expliciet VERWORPEN wegens verplicht
GGML_CUDA_NO_PINNED=1. Bruikbaar principe: transfer/dispatch apart meten;
hot GPU-matvec is geen PCIe- of modelroutingmeting.

Slice: bounded GPU-idle-readiness (max 60s, ongewijzigde <5%-guard), daarna de
bestaande synthetische GPU-bench werkelijk uitvoeren binnen 40G/28GiB. Echte
request-GPU-routing en matched pageable staging blijven een afzonderlijke slice.
