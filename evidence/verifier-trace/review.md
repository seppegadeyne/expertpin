# Onafhankelijke review en parent-verificatie

Review deleg_79811d89 (gpt-6-astra), volledig afgerond vóór GPU/servicewijziging.
Eindoordeel REQUEST_CHANGES, drie concrete bevindingen; geen tweede onafhankelijke
APPROVE geclaimd. Volledige oorspronkelijke review is lokaal beschikbaar in de
Hermes-cache; deze samenvatting is voor de repo.

1. P1 analyzer vertrouwde summary/log zonder bronrequest/responsevalidatie.
   Parent zelf bevestigd in oorspronkelijke analyze(): summary requests=[] werd
   niet afgewezen. Nieuwe echte RED-regressie gaf ValueError-not-raised. Fix:
   precies twee originele payloads exact gelijk aan vaste harnesspayload (alleen
   nmax0/4 verschilt), originele responses via validate_completion, gelijkheid
   met summarytimings, target-only versus echte MTP-proposalbeslissingen vereist.
2. P2 raw- en rowinvarianten ontbraken. Parent met negen mutaties werkelijk RED
   (tezamen met #1 tien failures); nu exacte ID/integer/finite/margin/top-two/
   proposal/row-validatie. Volledige producerachtige synthetische fixtures;
   tests voor verkeerde payload, ontbrekende response en onechte MTP toegevoegd.
3. P2 rewind kan n_decoded terugzetten en 128-positievenster opnieuw doorlopen.
   Parent server-context.cpp4637–4640 en4681 zelf gelezen. Afzonderlijke
   per-slot/task-admissioncounter toegevoegd, test128-cap en reset bij nieuwe task.
   Analyzer weigert dubbele/ontbrekende posities; geen delivered-tokenclaim.

Kernpremissen zelf herlezen: apply_server_biases zet uitsluitend een pointer;
common/sampling.cpp635–657 muteert logits pas in sampler; snapshot-hooks staan
ervoor. Server-context.cpp3616–3648 bouwt root+draft rowindices, common/sampling.cpp
790–818 matcht target-ID en draft-ID met eerste-mismatchbreak. Acceptanceregel is
ongewijzigd. Raw argmax is NIET per definitie post-penalty/grammar geselecteerd ID.
Assembled batchgrootte is NIET kernel microbatchgrootte. Numeriek versus hidden/
recurrent/KV-causaliteit blijft open tot identiek-prefixreplay/state-isolatie.

Uitvoering na fixes: analyzer7/7, harness14/14, target-only6/6; CPUfullbuild en
CUDAserverbuild exit0; targetedCTest20/20 inclusief expert-manifest; main39/41
met historische BERT-tokenizer/chat-templatefailures. Geen alles-groenclaim.
GPU-runtime/niet-interferentie worden pas na de echte metingen gerapporteerd.
