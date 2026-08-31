# AGENTS.md — expertpin

## Repo-identiteit (LEES EERST)

- Dit is **seppegadeyne/expertpin**: https://github.com/seppegadeyne/expertpin
- De clone heeft twee remotes: `origin` = seppegadeyne/expertpin (werkrepo), `upstream` = ikawrakow/ik_llama.cpp (alleen-lezen referentie).
- **`gh` default repo is vastgezet met `gh repo set-default seppegadeyne/expertpin`.** Controleer bij twijfel met `gh repo set-default --view`. Gebruik nóóit gh-commando's (issues/PRs/releases) gericht op `ikawrakow/*` — dat is de upstream van de auteur, niet onze werkrepo.
- Bij nieuwe clones van deze repo op andere machines: altijd direct `gh repo set-default seppegadeyne/expertpin` uitvoeren vóór enig gh-gebruik, want gh resolve anders naar de eerste remote (upstream) en issues/PRs landen op de verkeerde repo.

## Werkafspraken

- Commits: Engels, conventional style (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Push naar `origin main`; nooit force-push; nooit de `upstream` remote beschrijven/pushen.
- Tests: bestaande unit tests (o.a. `test-expert-manifest`) moeten altijd groen blijven; nieuw gedrag krijgt eigen tests (TDD waar haalbaar).
- `--resident-experts 0` moet bit-identiek zijn aan upstream-gedrag — regressie hierop is een blocker.
- GPU-werk op Aorus: qubic-miner is user-service `qli.service` (`systemctl --user stop/start qli.service`); vóór modelload guards uit `scripts/run-qwen38-flash-next.sh` respecteren (DRY=1 eerst; MemAvailable-check; cgroup MemoryMax); miner ná afloop altijd herstarten, ook bij falen.
- Modellen/assets staan buiten de repo in `/home/seppe/Models/qwen3.8-flash-next/` — nooit committen.

## Voorbeeld-launcher

Zie `scripts/run-qwen38-flash-next.sh` (RAM/VRAM-guards, cgroup-cap, expertpin-vlaggen `--expert-manifest`/`--resident-experts`).
