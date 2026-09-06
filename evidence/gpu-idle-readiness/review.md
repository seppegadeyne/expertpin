# Onafhankelijke review en parentverificatie

Review deleg_62d0fb75 (geconfigureerd gpt-6-astra) las volledige diff; eindoordeel
BLOCK met twee concrete must-fixes. Afgerond vóór GPU-stop.

1. Nieuwe outputvalidatie stond binnen lifecycle try/finally: afwijzing kon
   eerdere execution-/cleanuplogs alsnog overschrijven. Parent heeft zelf
   controlflow bevestigd; validate_output nu vóór lifecycle (run-physical.py
   153–163), vereist lege bestaande directory binnen evidence. Nieuwe mocks
   bewijzen geen commando/writes bij bestaande gpu/execution/readinessbestanden
   of ongeldige bestemming.
2. Bestaande stop_scope liet observation/logfouten de beëindiging overslaan.
   Parent bevestigde query vóór stop en onbeschermde state-query. Nu zijn
   observaties, stop en kill afzonderlijk beschermd (120–150), twee stoppogingen
   en één SIGKILL blijven bereikbaar. Foutinjecties op cgroup/state/kill-log
   uitgevoerd; onbewijsbare beëindiging blijft fout, verplichte minerrestore
   behouden. MemorySwapMax=0 wordt nu ook binnen scope teruggelezen.

Geen tweede onafhankelijke PASS geclaimd; fixes door parent uitgevoerd en getest.
Readinesspremisse (<5%, >=34GiB, <28GiB, deadline incl commandtijd) door reviewer
bevestigd. Parent bench-expert-gpu.cpp 44–75/110–138/174–176 zelf gelezen:
synthetische fixed-shape CUDA MUL_MAT, dispatch+sync-walltiming, geen routingmodel.

Tests: initieel RED met 5 ontbrekende-API-errors (geen gedrags-REDclaim).
Na implementatie+reviewfixes 13 readiness/lifecycle-tests groen; bestaande fysieke
cleanup 3 mockscenario's groen; fullCPUbuild+CUDAbenchbuild groen; targeted15/15.
Main33/35; exact dezelfde BERTtokenizer/chattemplatefailures als gecommitteerde
vorige run (trace-matched-bandwidth/main.log). Geen nieuwe suitefailure.
