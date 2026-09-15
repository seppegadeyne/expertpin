#!/usr/bin/env python3
"""Guarded cold prefill ubatch ladder for UD-Q4_K_XL (2026-09-15).

Seppe's pain point (2026-09-14): cold prefill of the ~19.5K-token default
Hermes system prompt runs at ~19 tok/s on the dev config (UD-Q4_K_XL,
NCMOE=40, DRAFT=1, ubatch default 512) — first turn of a server lifetime
takes 15-19 minutes. Evidence: evidence/hermes-e2e-20260911/
run-20260914T104344-e2e/server.log line 760 (14213 tokens @ 18.84 tok/s;
even 43-51 token prompts pay 13-26 tok/s = full per-ubatch expert
streaming over the CPU MoE layers).

Hypothesis (X-research 2026-09-15: ubatch 1024-2048 is the standard
prefill knob for CPU-MoE offload; community 5090 prefill ~900-1000 tok/s):
a larger physical batch amortizes the per-pass expert streaming over more
tokens, so the same cold prompt should prefill several times faster.

This harness measures ONE ubatch value per server lifetime, with a
verified-cold page cache before load (whole-checkpoint, all shards) and a
deterministic ~14K-token neutral filler prompt that reproduces the E2E
prompt-size class. Decode is NOT the target here; a short 32-token decode
follows only to keep the request pipeline honest.

Fail-closed throughout; budgets 40 GiB RAM / 28 GiB VRAM; GGML_CUDA_NO_PINNED=1.
Miner handling per Seppe 2026-09-14 rule 0: qli.service (active again since
2026-09-14) is stopped before GPU work and restarted+verified after; jetski
stays untouched (paused).
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREP = Path('/home/seppe/.hermes/profiles/expertpin/scripts/gpu-host-prep.sh')

# Bound: whole run incl. cleanup must fit the 30-min GPU cap (Seppe).
# The 512 control arm reproduces the E2E pain: 14.2K tokens @ ~19 tok/s
# ≈ 755s prefill alone, plus cold drop/snapshot (~2-3 min) and load.
WORK_SECONDS = 1500
RAM_BUDGET_GIB = 36
PROMPT_TARGET_TOKENS = 14000
DECODE_TOKENS = 32
LADDER = (512, 1024, 2048)

CHECKPOINTS = {
    'ud-q4kxl': ('/home/seppe/Models/qwen3.8-flash-next/UD-Q4_K_XL',
                 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'),
}
DEFAULT_CHECKPOINT = 'ud-q4kxl'


def validate_plan(ubatch, checkpoint=DEFAULT_CHECKPOINT):
    if type(ubatch) is not int or ubatch not in LADDER:
        raise ValueError(f'ubatch must be one of {LADDER}')
    if checkpoint not in CHECKPOINTS:
        raise ValueError('unknown checkpoint: ' + repr(checkpoint))


def clean_environment(inherited, scope, ubatch, checkpoint=DEFAULT_CHECKPOINT):
    env = {k: v for k, v in inherited.items()
           if not k.startswith(('GGML_MOE_TRACE', 'GGML_CUDA_TRANSFER_TRACE'))
           and k not in ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_NO_GRAPHS',
                         'EXPERTPIN_VERIFIER_TRACE')
           and not k.startswith('LLAMA_ARG_')}
    validate_plan(ubatch, checkpoint)
    model_dir, model_name = CHECKPOINTS[checkpoint]
    env.update(GGML_CUDA_NO_PINNED='1', DRAFT='1', DRAFT_NMAX='4',
               DRAFT_MODEL='/home/seppe/Models/qwen3.8-flash-next/mtp-drafter/'
                           'mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf',
               MODEL_DIR=model_dir, MODEL=model_dir + '/' + model_name,
               CTX='65536', NCMOE='40', NGL='99', UBATCH=str(ubatch),
               THREADS='16', KVT='q8_0',
               RAM_BUDGET_GIB=str(RAM_BUDGET_GIB), GPU_NEED_GIB='24',
               CACHE_RAM_MIB='512',
               PORT='8102', BIN_DIR=str(ROOT / 'build-sm120/bin'), FORCE='0',
               PINNED='0', EXPERT_CACHE_SIM_MIB='0', EXPERT_STATS_FILE='',
               MANIFEST='', RESIDENT='0',
               EXPERTPIN_SCOPE_UNIT=scope, DRY='0', REASONING='off')
    return env


def neutral_filler(paragraphs: int, words_per_paragraph: int = 60) -> str:
    """Deterministic neutral filler (no names, no needle content). Reuses the
    multi-needle lineage's haystack generator style: numbered neutral
    sentences. Exact token count is measured via /tokenize at runtime."""
    rng = random.Random(20260915)
    subjects = ['archive', 'calendar', 'corridor', 'engine', 'harbor',
                'library', 'machine', 'orchard', 'pipeline', 'signal',
                'terminal', 'warehouse']
    verbs = ['adjusts', 'buffers', 'catalogs', 'dispatches', 'encodes',
             'filters', 'gathers', 'indexes', 'logs', 'measures', 'notifies']
    adjectives = ['analog', 'bounded', 'cyclic', 'dedicated', 'external',
                  'formal', 'grouped', 'incremental', 'joint', 'kinetic']
    parts = []
    for i in range(paragraphs):
        words = [f'{rng.choice(subjects)}-{rng.choice(verbs)}'
                 if rng.random() < 0.2 else rng.choice(subjects)
                 for _ in range(words_per_paragraph)]
        parts.append(f'Note {i}: ' + ' '.join(words) + '.')
    return ' '.join(parts)


def payload(prompt, max_tokens, nmax=4):
    return {'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': 0.0, 'seed': 42,
            'stream': False, 'cache_prompt': False, 'ignore_eos': True,
            'speculative.n_max': nmax}


def validate_completion(data, expect_tokens):
    if not isinstance(data, dict) or 'error' in data:
        raise RuntimeError('invalid/error completion')
    usage = data.get('usage')
    timings = data.get('timings')
    if not isinstance(usage, dict) or type(usage.get('completion_tokens')) is not int:
        raise RuntimeError('missing usage')
    # Prefill is the metric; decode may stop early via EOS -> accept stop.
    choices = data.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or choices[0].get(
            'finish_reason') not in ('length', 'stop'):
        raise RuntimeError('unexpected finish')
    if not isinstance(timings, dict):
        raise RuntimeError('missing timings')
    for key in ('prompt_ms', 'predicted_ms', 'predicted_per_second'):
        value = timings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise RuntimeError('invalid timing: ' + key)
    return {'usage': usage, 'timings': timings}


def validate_budget(point):
    if not 0 <= point['vram_used_mib'] < 28 * 1024:
        raise RuntimeError('VRAM cap exceeded/invalid')
    if not 0 <= point['gpu_util_pct'] <= 100:
        raise RuntimeError('invalid GPU utilization')
    if int(point['memory.max']) != RAM_BUDGET_GIB * 1024**3:
        raise RuntimeError('unexpected cgroup MemoryMax')
    if not 0 <= int(point['memory.current']) <= int(point['memory.peak']) <= 40 * 1024**3:
        raise RuntimeError('RAM cap exceeded/invalid')
    events = dict(line.split() for line in point['memory.events'].splitlines())
    if any(int(events.get(k, 0)) for k in ('max', 'oom', 'oom_kill', 'oom_group_kill')):
        raise RuntimeError('cgroup memory pressure/OOM event')


# ---- verified-cold + guarded-run machinery reused verbatim from the
# coldcache lineage (2026-09-14 whole-checkpoint version) ----
spec = importlib.util.spec_from_file_location(
    'coldcache', ROOT / 'evidence/coldcache-ab-20260911/run-guarded.py')
assert spec is not None and spec.loader is not None
coldcache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coldcache)
drop_model_cache = coldcache.drop_model_cache
residency_snapshot = coldcache.residency_snapshot
verified_cold_report = coldcache.verified_cold_report
terminate = coldcache.terminate


class Run(coldcache.Run):
    """Same guarded lifecycle; miner handling per 2026-09-14 rule 0: qli
    stop before GPU work / start+verify after; jetski never touched."""

    # Rule 0 (Seppe, 2026-09-14): qli is the ONLY miner this run touches.
    # jetski.service stays paused — never stopped-or-started here. The
    # inherited lineage restored every unit it stopped; that would boot a
    # paused jetski after GPU work, so the candidate list is pinned.
    MINERS = ('qli.service',)

    def _miner_stop_uses(self, service):
        if service not in self.MINERS:
            return None
        return super()._miner_stop_uses(service)

    def __init__(self, out, ubatch, checkpoint=DEFAULT_CHECKPOINT):
        validate_plan(ubatch, checkpoint)
        # coldcache.Run.__init__(startup_nmax=4) keeps DRAFT=1 lineage.
        super().__init__(out, 4, checkpoint, cold_cache=True)
        self.ubatch = ubatch
        # 2026-09-12 lineage trap: the inherited sample() resolves the module
        # constant of the PARENT module (coldcache WORK_SECONDS=660) unless
        # the instance override is set. Pin this harness's own budget.
        self.work_seconds = WORK_SECONDS
        self.summary.update(ubatch=ubatch, harness='prefill-ubatch-20260915')

    def execute(self):
        self.command(['git', 'rev-parse', 'HEAD'], 'commit')
        check = subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True, timeout=10)
        if check.returncode != 1:
            raise RuntimeError('pre-existing llama-server or failed process observation')
        units = self.command(['systemctl', '--user', 'list-units', '--type=scope',
                              '--state=running', '--no-legend', '--plain',
                              'expertpin-test-*.scope'], 'preexisting-scopes')
        if units:
            raise RuntimeError('pre-existing expertpin scope')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 8102))
        env = clean_environment(os.environ, self.scope, self.ubatch, self.checkpoint)
        self.save('environment.json', {k: v for k, v in env.items()
                                       if k not in os.environ or os.environ[k] != v})
        if not Path(env['MODEL']).is_file():
            raise RuntimeError('checkpoint model file missing: ' + env['MODEL'])
        if not Path(env['DRAFT_MODEL']).is_file():
            raise RuntimeError('required draft model missing')
        coldcache.model_files(env['MODEL'])
        self.prepared = True
        self.command(['bash', PREP], 'host-prep', timeout=45)
        self.stop_miners()          # rule 0: qli stop (jetski skip is a no-op)
        self.command(['nvidia-smi'], 'nvidia-before')
        self.command(['free', '-g'], 'free-before')
        gate_spec = importlib.util.spec_from_file_location(
            'idle_gate', ROOT / 'evidence/trace-matched-bandwidth/run-physical.py')
        assert gate_spec is not None and gate_spec.loader is not None
        gate = importlib.util.module_from_spec(gate_spec)
        gate_spec.loader.exec_module(gate)
        gate.OUT = self.out
        idle = gate.wait_gpu_idle('model-readiness')
        self.summary['guard'] = idle
        if not (0 <= idle['gpu_util_pct'] < 5 and idle['mem_available_bytes'] >= 34 * 1024**3):
            raise RuntimeError('Tier A host guard blocked')
        report = verified_cold_report(env['MODEL'])
        self.summary['cold_cache_report'] = report
        self.summary['cache_drop'] = report['drop']
        self.summary['residency'] = report['snapshot']
        self.summary['residency_verdict'] = {
            'verdict': report['verdict'], 'resident_ratio': report.get('resident_ratio')}
        self.save('summary.json', self.summary)
        if not report['verified_cold']:
            raise RuntimeError('whole-checkpoint cold-cache verification failed')
        launch = ['bash', str(ROOT / 'scripts/run-qwen38-flash-next.sh')]
        dry = subprocess.run(launch, env=dict(env, DRY='1'), capture_output=True,
                             text=True, timeout=30)
        (self.out / 'dry.log').write_text(dry.stdout + dry.stderr)
        if dry.returncode or 'WOULD BLOCK' in dry.stdout + dry.stderr \
                or 'guards      : OK' not in dry.stdout:
            raise RuntimeError('launcher dry guard blocked/unverified')
        if f'ubatch={self.ubatch}' not in dry.stdout:
            raise RuntimeError('dry plan does not reflect the requested ubatch')
        self.server_log = (self.out / 'server.log').open('x')
        self.scope_launched = True
        self.proc = self.spawn(launch, env=env, stdout=self.server_log, stderr=subprocess.STDOUT)
        self.spawned()
        for _ in range(20):
            if self.command(['systemctl', '--user', 'is-active', self.scope],
                            'scope-ready', check=False) == 'active':
                break
            if self.proc.poll() is not None:
                raise RuntimeError('launcher exited before scope readiness')
            time.sleep(0.25)
        else:
            raise RuntimeError('own scope never became active')
        self.label = 'model-load'
        deadline = min(self.start + self.work_budget(), time.monotonic() + 360)
        while time.monotonic() < deadline:
            self.sample()
            try:
                with urllib.request.urlopen('http://127.0.0.1:8102/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        else:
            raise TimeoutError('health readiness timeout')
        self.summary['model_loaded'] = True
        # Fail closed if the server did not adopt the requested ubatch.
        log = (self.out / 'server.log').read_text()
        observed = [line for line in log.splitlines() if 'n_ubatch' in line]
        self.save('ubatch-observed.txt', '\n'.join(observed) + '\n')
        if not any(f'= {self.ubatch}' in line for line in observed):
            raise RuntimeError('server did not adopt the requested ubatch')
        # Tokenizer-exact prompt sized to the E2E pain class (~14K tokens).
        filler, count, paragraphs = self.calibrate_prompt()
        self.summary['prompt_filler_paragraphs'] = paragraphs
        self.summary['prompt_tokens'] = count
        self.label = f'ubatch{self.ubatch}'
        request_path = self.out / 'request.json'
        response_path = self.out / 'response.json'
        self.save(request_path.name, payload(filler, DECODE_TOKENS))
        began = time.monotonic()
        self.sample()
        with response_path.open('x') as response, (self.out / 'curl.log').open('x') as error:
            self.request = self.spawn(['curl', '--silent', '--show-error', '--fail-with-body',
                                       '--connect-timeout', '5', '--max-time', '900',
                                       '-H', 'Content-Type: application/json',
                                       '--data-binary', '@' + str(request_path),
                                       'http://127.0.0.1:8102/v1/chat/completions'],
                                      stdout=response, stderr=error)
            self.spawned()
            while self.request.poll() is None:
                self.sample()
                if time.monotonic() - began >= 900:
                    raise TimeoutError('request deadline')
                time.sleep(1)
            if self.request.wait(timeout=5):
                raise RuntimeError('completion HTTP/curl failure')
        self.sample()
        result = validate_completion(json.loads(response_path.read_text()), DECODE_TOKENS)
        result.update(ubatch=self.ubatch, prompt_tokens=count,
                      wall_seconds=time.monotonic() - began)
        self.save('metrics.json', result)
        self.summary['requests'] = [result]
        self.summary['status'] = 'completed'

    def calibrate_prompt(self, target=PROMPT_TARGET_TOKENS, tolerance=600):
        """Size the neutral filler to the E2E pain class (~14K tokens) via
        live /tokenize calibration: probe, scale, refine, fail closed."""
        probe_paragraphs = 40
        per_para = self.tokenize_count(neutral_filler(probe_paragraphs)) / probe_paragraphs
        paragraphs = max(1, round(target / per_para))
        filler = neutral_filler(paragraphs)
        count = self.tokenize_count(filler)
        if abs(count - target) > tolerance:
            paragraphs = max(1, round(paragraphs * target / count))
            filler = neutral_filler(paragraphs)
            count = self.tokenize_count(filler)
        if abs(count - target) > tolerance:
            raise RuntimeError(f'prompt calibration failed: {count} vs target {target}')
        return filler, count, paragraphs

    def tokenize_count(self, text):
        body = json.dumps({'content': text, 'add_special': False}).encode()
        request = urllib.request.Request('http://127.0.0.1:8102/tokenize',
                                         data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            tokens = json.load(response)['tokens']
        if not isinstance(tokens, list) or not tokens:
            raise RuntimeError('invalid tokenization')
        return len(tokens)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ubatch', type=int, choices=LADDER, required=True)
    parser.add_argument('--checkpoint', choices=sorted(CHECKPOINTS), default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)
    out = Path(__file__).resolve().parent / (
        'run-' + datetime.now().strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex)
    out.mkdir()
    run = Run(out, args.ubatch, args.checkpoint)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    try:
        for sig in previous:
            signal.signal(sig, run.interrupt)
        signal.alarm(WORK_SECONDS)
        run.execute()
    except BaseException as error:
        run.summary['error'] = repr(error)
    finally:
        signal.alarm(0)
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            run.cleanup()
            run.summary['finished'] = datetime.now().astimezone().isoformat()
            run.summary['elapsed_seconds'] = time.monotonic() - run.start
            run.save('summary.json', run.summary)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    print(out)
    print(json.dumps(run.summary, indent=2))
    return 0 if run.summary['status'] == 'completed' else 1


if __name__ == '__main__':
    main()
