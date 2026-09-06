# X research — 2026-09-06, vóór tests

Read-only x_search afgerond. Geen browser/tabbladen gestart.
- https://x.com/DoingFedTime/status/2096079245437001737 — zoekantwoord noemt ncmoe/transferkosten en paging als bottleneck. Bruikbaar idee: meet echte offsetset, niet prefix-bandbreedte of theoretische FLOPs.
- https://x.com/JonathanLeaders/status/2095715911072207016 — AD-3.84 + MTP lead; ander quant, geen eigen meetbewijs.
- https://x.com/edyCryptoNFT/status/2096142267333656736 — AD4.27 contextclaim, onvoldoende budget/provenance voor vergelijking.
Geen reproduceerbare trace-offset koud/warm of quant-specifieke GPU-GEMM-data gevonden in dit zoekantwoord. Architectuur/PLE/modelvarianten in samenvatting niet overgenomen. Geen externe metrics als lokale metingen of advisor-input gebruikt.

Slice: fysieke buffered reads op exact gejoinde CPU-requestoffsets; gecontroleerde random unique-offsetset, DONTNEED + mincore + /proc/self/io, daarna matched warme herlezing. GPU-GEMM indien haalbaar apart gemeten; complete GPU-routing en placement-advisor blijven open zolang inputs ontbreken.
