# X-research 2026-09-06, vóór effectief testen

x_search afgerond. Geen browser gestart, geen processen/tabbladen te sluiten.
Leads (zoek-samenvattingen, geen verbatim verificatie/lokale metingen):
- https://x.com/JonathanLeaders/status/2095715911072207016 : AD-3.84bpw/MTP 5090, claim 24 tok/s; geen gemeten 40-GiB-runfootprint.
- https://x.com/TeksEdge/status/2096297314595864698 : RAM expert-cache/PCIe overlap; end-to-end secondhand claims, geen geïsoleerde calibratie.
Geen bruikbare exact-quant RAM/PCIe/random-NVMe expertlatenties gevonden in deze zoekopdracht. De zoek-samenvatting herhaalt onterecht '63GB expertbank op vermoedelijk 64GB host' en verwart PLE-NVMe met expert-I/O; NIET overgenomen (eerdere bron-audit in state).

Slice: herstel calibratie-inputcorrectheid en voeg een fail-closed budgetadvisor toe, met expliciete gemeten kandidaat-inputs en onbekende GPU-metingen. Geen GPU-werk: vaste developer-gate staat GPU-slice alleen toe als Stap-1-baseline nog niet draaide; die draaide al. Fase 2 integraal blijft dus open; geen gefingeerde PCIe/GPU-GEMM-metingen of optimale budgetclaim.
