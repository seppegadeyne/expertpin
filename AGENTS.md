# AGENTS.md — expertpin

## Repo-identiteit (LEES EERST)

- Dit is **seppegadeyne/expertpin**: https://github.com/seppegadeyne/expertpin — een VOLLEDIG ZELFSTANDIG project (Seppe-mandaat 2026-09-04).
- Geen upstream-relatie meer: alle externe remotes (`upstream`, `fork`) zijn verwijderd; alleen `origin` rest. Er is géén ander publiek repo waar we aan bijdragen, PR's opvolgen of discussiëren. Alle gh-activiteit (issues, commits, pushes) uitsluitend op seppegadeyne/expertpin.
- **`gh` default repo is vastgezet met `gh repo set-default seppegadeyne/expertpin`.** Controleer bij twijfel met `gh repo set-default --view`.
- De codebase (oorsprong ik_llama.cpp-fork) is volledig eigendom van dit project: vrij breken, hernoemen, herschrijven. Compatibiliteit met andere forks/upstreams is GEEN criterium.
- Andere repo's (club-3090, Whamp/vLLM, FreeToken, buun, EXL3, ...) alléén READ-ONLY als inspiratie (klonen naar `third-party/`, buiten git, mag); eigen implementaties schrijven; nooit contribueren of posten.

## Werkafspraken

- Commits: Engels, conventional style (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Push naar `origin main`; nooit force-push op main (op eigen feature-branches mag geschiedenis herschreven worden).
- Tests: bestaande unit tests (o.a. `test-expert-manifest`) moeten altijd groen blijven; nieuw gedrag krijgt eigen tests (TDD waar haalbaar).
- GPU-werk op Aorus: qubic-miner is user-service `qli.service` (`systemctl --user stop/start qli.service`); vóór modelload guards uit `scripts/run-qwen38-flash-next.sh` respecteren (DRY=1 eerst; MemAvailable-check; cgroup MemoryMax); miner ná afloop altijd herstarten, ook bij falen.
- Budgetten (Seppe, 2026-09-04): run-footprint max 40 GiB systeemRAM én max 28 GiB VRAM — elke GPU-run tegen beide meten en rapporteren.
- Modellen/assets staan buiten de repo in `/home/seppe/Models/qwen3.8-flash-next/` — nooit committen.

## Voorbeeld-launcher

Zie `scripts/run-qwen38-flash-next.sh` (RAM/VRAM-guards, cgroup-cap, expertpin-vlaggen `--expert-manifest`/`--resident-experts`).
