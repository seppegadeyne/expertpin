#!/usr/bin/env python3
"""Clean repeated MTP measurement. Import/--help are CPU-only.

Run separately at startup depths 4, 8, 16: recurrent checkpoints are startup-bounded.
Each invocation repeats the same 256-token request twice at its startup depth.
Depth 0 disables the draft model entirely for target-only correctness isolation.
Lifecycle derived from ../expert-offset-trace/run-guarded.py; no tracing/dmon.
"""
import argparse
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import posix
import signal
import socket
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
PREP = '/home/seppe/.hermes/profiles/expertpin/scripts/gpu-host-prep.sh'
DEPTHS = (4, 8, 16)
TOKENS = int(os.environ.get('SUSTAINED_TOKENS', '256'))
WORK_SECONDS = 660  # Reserve >3 minutes of the 15-minute bound for bounded cleanup.
REQUEST_SECONDS = 180
RAM_BUDGET_GIB = 36
PROMPT = 'Explain how a bounded expert cache handles RAM and NVMe misses in detail.'
TRACE_PREFIXES = ('GGML_MOE_TRACE', 'GGML_MOE_GPU_TRACE', 'GGML_CUDA_TRANSFER_TRACE')
GRAPH_DISABLE = ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_NO_GRAPHS')
# Checkpoint aliases for A/B: shared drafter, identical serving configuration;
# only MODEL_DIR/MODEL differ. The reference is the 197 GiB AD-4.27bpw Q4_K_M
# shard set; ps-iq2xxs is PeasantSmith's community 75.2 GiB IQ2_XXS GGUF
# (SHA-256 verified 2026-09-11 against the HF model card before first load).
CHECKPOINTS = {
    'reference': ('/home/seppe/Models/qwen3.8-flash-next/AD-4.27bpw-Q4_K_M-M64',
                  'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf'),
    'ps-iq2xxs': ('/home/seppe/Models/qwen3.8-flash-next-ps-iq2xxs',
                  'Qwen3.8-Flash-Next-IQ2_XXS.gguf'),
    'ud-q4kxl': ('/home/seppe/Models/qwen3.8-flash-next/UD-Q4_K_XL',
                 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'),
}
DEFAULT_CHECKPOINT = 'reference'


def validate_plan(startup_nmax, checkpoint=DEFAULT_CHECKPOINT):
    if type(startup_nmax) is not int or startup_nmax not in (0, 4, 8, 16):
        raise ValueError('startup n_max must be 0 (target-only), 4, 8 or 16')
    if checkpoint not in CHECKPOINTS:
        raise ValueError('unknown checkpoint: ' + repr(checkpoint))


def clean_environment(inherited, scope, startup_nmax=4, checkpoint=DEFAULT_CHECKPOINT):
    env = {k: v for k, v in inherited.items()
           if not k.startswith(TRACE_PREFIXES) and k not in GRAPH_DISABLE and k != 'EXPERTPIN_VERIFIER_TRACE'
           and not k.startswith('LLAMA_ARG_')}
    validate_plan(startup_nmax, checkpoint)
    model_dir, model_name = CHECKPOINTS[checkpoint]
    env.update(GGML_CUDA_NO_PINNED='1', DRAFT=str(int(startup_nmax != 0)), DRAFT_NMAX=str(startup_nmax),
               DRAFT_MODEL='/home/seppe/Models/qwen3.8-flash-next/mtp-drafter/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf',
               MODEL_DIR=model_dir, MODEL=model_dir + '/' + model_name,
               CTX='8192', NCMOE='36', NGL='99', THREADS='16', KVT='q8_0',
               RAM_BUDGET_GIB='36', GPU_NEED_GIB='24', CACHE_RAM_MIB='512',
               PORT='8102', BIN_DIR=str(ROOT / 'build-sm120/bin'), FORCE='0', PINNED='0',
               EXPERT_CACHE_SIM_MIB='0', EXPERT_STATS_FILE='', MANIFEST='', RESIDENT='0',
               EXPERTPIN_SCOPE_UNIT=scope, DRY='0')
    return env


def payload(nmax):
    validate_plan(nmax)
    result = {'messages': [{'role': 'user', 'content': PROMPT}], 'max_tokens': TOKENS,
            'temperature': 0.0, 'seed': 42, 'stream': False, 'cache_prompt': False,
            'ignore_eos': True}
    if nmax:
        result['speculative.n_max'] = nmax
    return result


def output_fields(data):
    message = data['choices'][0]['message']
    result = {k: message.get(k) for k in ('content', 'reasoning_content')}
    if any(v is not None and not isinstance(v, str) for v in result.values()):
        raise ValueError('non-text response field')
    return result


def validate_completion(data, nmax):
    """Fail closed on EOS, malformed metrics, absent MTP, or inconsistent totals."""
    if not isinstance(data, dict) or 'error' in data:
        raise RuntimeError('invalid/error completion')
    usage, choices, timings = data.get('usage'), data.get('choices'), data.get('timings')
    if (not isinstance(usage, dict) or type(usage.get('completion_tokens')) is not int
            or usage['completion_tokens'] != TOKENS or not isinstance(choices, list)
            or len(choices) != 1 or not isinstance(choices[0], dict)
            or choices[0].get('finish_reason') != 'length'):
        raise RuntimeError('request did not produce the intended 256-token length-limited decode')
    message = choices[0].get('message')
    if not isinstance(message, dict) or not any(
            isinstance(message.get(k), str) and message[k].strip()
            for k in ('content', 'reasoning_content')):
        raise RuntimeError('request produced no text')
    if not isinstance(timings, dict) or type(timings.get('predicted_n')) is not int or timings['predicted_n'] != TOKENS:
        raise RuntimeError('missing/mismatched predicted count')
    for key in ('prompt_ms', 'predicted_ms', 'predicted_per_second'):
        value = timings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError('invalid timing: ' + key)
    rows = timings.get('draft_by_depth')
    if nmax == 0:
        if (timings.get('draft_n', 0) != 0 or timings.get('draft_n_accepted', 0) != 0
                or rows not in (None, [])):
            raise RuntimeError('unexpected drafting in target-only mode')
        return {'usage': usage, 'timings': timings, 'acceptance_fraction': None}
    if not isinstance(rows, list) or not rows:
        raise RuntimeError('missing MTP acceptance depth counters')
    drafted = accepted = 0
    for depth, row in enumerate(rows, 1):
        if (not isinstance(row, dict) or type(row.get('depth')) is not int
                or row['depth'] != depth or depth > nmax
                or type(row.get('draft_n')) is not int or row['draft_n'] <= 0
                or type(row.get('draft_n_accepted')) is not int
                or not 0 <= row['draft_n_accepted'] <= row['draft_n']):
            raise RuntimeError('invalid MTP depth counters')
        drafted += row['draft_n']
        accepted += row['draft_n_accepted']
    if (type(timings.get('draft_n')) is not int or timings['draft_n'] != drafted
            or type(timings.get('draft_n_accepted')) is not int
            or timings['draft_n_accepted'] != accepted or accepted > TOKENS):
        raise RuntimeError('inconsistent MTP acceptance totals')
    return {'usage': usage, 'timings': timings, 'acceptance_fraction': accepted / drafted}


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


def drop_model_cache(path):
    """Best-effort whole-file POSIX_FADV_DONTNEED via a read-only fd.

    Requests the kernel to evict clean page-cache pages of the model file so
    the following load reads from storage. DONTNEED is advisory only: the
    cold state is NOT verified (see the mincore discussion in
    scripts/expert_bw_calib.py). Fails closed on a missing/irregular file;
    the caller must abort before host changes on any OSError."""
    path = Path(path)
    if not path.is_file():
        raise ValueError('model file missing for cache drop: ' + str(path))
    size = path.stat().st_size
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, size, posix.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return {'fadvise_completed': True, 'bytes': size, 'file': str(path)}


def residency_snapshot(path, batch_pages=65536):
    """Bounded whole-file mincore residency snapshot (pages resident in the
    page cache), using the proven expert_bw_calib technique: PROT_NONE
    MAP_SHARED mapping + batched mincore vectors. Snapshot only — not atomic;
    residency can change during/after the syscalls. Requires an owned regular
    file (Linux may mask mincore for foreign files)."""
    path = Path(path)
    if not path.is_file():
        raise ValueError('model file missing for residency snapshot: ' + str(path))
    import ctypes
    import stat
    page = os.sysconf('SC_PAGESIZE')
    size = path.stat().st_size
    fd = os.open(path, os.O_RDONLY)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError('residency snapshot requires an owned regular file')
        pages = (size + page - 1) // page
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.mmap.restype = ctypes.c_void_p
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        libc.mincore.restype = ctypes.c_int
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munmap.restype = ctypes.c_int
        addr = libc.mmap(None, size, 0, 1, fd, 0)  # PROT_NONE, MAP_SHARED
        if addr == ctypes.c_void_p(-1).value:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        resident = 0
        try:
            for start in range(0, pages, batch_pages):
                count = min(batch_pages, pages - start)
                vec = (ctypes.c_ubyte * count)()
                if libc.mincore(addr + start * page, count * page, vec):
                    errno = ctypes.get_errno()
                    raise OSError(errno, os.strerror(errno))
                resident += sum(value & 1 for value in vec)
        finally:
            if libc.munmap(addr, size):
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
        return {'method': 'mincore PROT_NONE MAP_SHARED (whole file, batched)',
                'pages': pages, 'resident_pages': resident, 'error': None,
                'page_size': page, 'file': str(path)}
    finally:
        os.close(fd)


def classify_residency(resident_pages, pages, max_resident_ratio):
    if pages == 0:
        return {'verdict': 'EMPTY', 'resident_ratio': None}
    ratio = resident_pages / pages
    return {'verdict': 'COLD' if ratio <= max_resident_ratio else 'NOT_COLD',
            'resident_ratio': round(ratio, 6)}


def verified_cold_report(path, max_resident_ratio=0.10):
    """Drop + snapshot + classify: converts an advisory DONTNEED into a
    measured cold-state report. NEVER silently claims coldness: the verdict
    reflects the measured ratio against the threshold."""
    dropped = drop_model_cache(path)
    snapshot = residency_snapshot(path)
    if snapshot['error'] is not None:
        return {'verified_cold': False, 'verdict': 'UNKNOWN', 'drop': dropped,
                'snapshot': snapshot}
    verdict = classify_residency(snapshot['resident_pages'], snapshot['pages'],
                                 max_resident_ratio)
    return {'verified_cold': verdict['verdict'] == 'COLD', 'verdict': verdict['verdict'],
            'resident_ratio': verdict['resident_ratio'], 'drop': dropped,
            'snapshot': snapshot}


def terminate(child):
    if child is not None:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        else:
            child.wait(timeout=5)


class Run:
    """Small ownership container: no subprocesses or writes until execute()."""
    def __init__(self, out, startup_nmax, checkpoint=DEFAULT_CHECKPOINT, cold_cache=False):
        validate_plan(startup_nmax, checkpoint)
        self.out = out
        self.startup_nmax = startup_nmax
        self.checkpoint = checkpoint
        self.cold_cache = cold_cache
        self.sequence = (startup_nmax, startup_nmax)
        self.scope = 'expertpin-test-' + uuid.uuid4().hex + '.scope'
        self.proc = self.request = self.server_log = None
        self.cgroup_path = None
        self.scope_launched = self.prepared = self.launching = False
        self.pending_signal = None
        self.tokenize_references = []
        self.verifier_trace = False
        self.start = time.monotonic()
        self.label = 'preflight'
        self.summary = {'started': datetime.now().astimezone().isoformat(), 'scope': self.scope,
                        'model_loaded': False, 'startup_n_max': startup_nmax,
                        'checkpoint': checkpoint,
                        'cold_cache': cold_cache,
                        'sequence': list(self.sequence), 'requests': [], 'status': 'blocked_or_failed'}

    def save(self, name, data):
        (self.out / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

    def command(self, args, name, check=True, timeout=10):
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        (self.out / (name + '.log')).write_text(result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError(f'{name}: rc={result.returncode}: {result.stderr}')
        return result.stdout.strip()

    def interrupt(self, signum, frame):
        if self.launching:
            self.pending_signal = signum
        else:
            raise TimeoutError(f'signal {signum}')

    def spawn(self, args, **kwargs):
        # Defer handled signals until the child handle is registered by caller.
        self.launching = True
        try:
            return subprocess.Popen(args, **kwargs)
        except BaseException:
            self.launching = False
            raise

    def spawned(self):
        self.launching = False
        if self.pending_signal is not None:
            self.interrupt(self.pending_signal, None)

    def sample(self):
        if time.monotonic() - self.start >= WORK_SECONDS:
            raise TimeoutError('work deadline; cleanup reserve begins')
        raw = self.command(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                            '--format=csv,noheader,nounits'], 'gpu-latest')
        if len(raw.splitlines()) != 1:
            raise RuntimeError('expected exactly one GPU')
        util, used = map(int, raw.split(','))
        cg = self.command(['systemctl', '--user', 'show', self.scope, '-p', 'ControlGroup', '--value'], 'cgroup')
        if not cg or not cg.endswith('/' + self.scope):
            raise RuntimeError('missing or mismatched own scope')
        base = Path('/sys/fs/cgroup') / cg.lstrip('/')
        self.cgroup_path = base
        point: dict[str, Any] = {k: (base / k).read_text().strip() for k in
                 ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current', 'memory.events')}
        point.update(elapsed_s=time.monotonic() - self.start, request_label=self.label,
                     vram_used_mib=used, gpu_util_pct=util,
                     mem_available_bytes=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                                                  if s.startswith('MemAvailable:'))) * 1024)
        with (self.out / 'samples.jsonl').open('a') as output:
            output.write(json.dumps(point) + '\n')
        validate_budget(point)
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError('server process missing or exited')

    def execute(self):
        self.command(['git', 'rev-parse', 'HEAD'], 'commit')
        check = subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True, timeout=10)
        if check.returncode != 1:
            raise RuntimeError('pre-existing llama-server or failed process observation')
        units = self.command(['systemctl', '--user', 'list-units', '--type=scope', '--state=running',
                              '--no-legend', '--plain', 'expertpin-test-*.scope'], 'preexisting-scopes')
        if units:
            raise RuntimeError('pre-existing expertpin scope')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 8102))  # Refuse an occupied endpoint before host changes.
        env = clean_environment(os.environ, self.scope, self.startup_nmax, self.checkpoint)
        if self.verifier_trace:
            env['EXPERTPIN_VERIFIER_TRACE'] = '1'
        self.summary['verifier_trace'] = self.verifier_trace
        self.save('environment.json', {k: v for k, v in env.items()
                                      if k not in os.environ or os.environ[k] != v})
        if not Path(env['MODEL']).is_file():
            raise RuntimeError('checkpoint model file missing: ' + env['MODEL'])
        if self.startup_nmax and not Path(env['DRAFT_MODEL']).is_file():
            raise RuntimeError('required draft model missing')
        self.prepared = True  # Restore even a partially failed preparation.
        self.command(['bash', PREP], 'host-prep', timeout=45)
        self.stop_miners()
        self.command(['nvidia-smi'], 'nvidia-before')
        self.command(['free', '-g'], 'free-before')
        spec = importlib.util.spec_from_file_location('clean_mtp_idle_gate', ROOT / 'evidence/trace-matched-bandwidth/run-physical.py')
        assert spec is not None and spec.loader is not None
        gate: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        gate.OUT = self.out
        idle = gate.wait_gpu_idle('model-readiness')
        self.summary['guard'] = idle
        if not (0 <= idle['gpu_util_pct'] < 5 and idle['mem_available_bytes'] >= 34 * 1024**3):
            raise RuntimeError('Tier A host guard blocked')
        if self.cold_cache:
            # After the GPU guard is green and BEFORE the dry check/scope
            # launch: drop the model file from the page cache so the model
            # load below reads from storage, then MEASURE the residency so
            # the coldness claim is verified rather than advisory.
            self.summary['cache_drop'] = drop_model_cache(env['MODEL'])
            snapshot = residency_snapshot(env['MODEL'])
            self.summary['residency'] = snapshot
            verdict = classify_residency(snapshot['resident_pages'], snapshot['pages'],
                                         max_resident_ratio=0.10)
            self.summary['residency_verdict'] = verdict
            self.save('summary.json', self.summary)
        launch = ['bash', str(ROOT / 'scripts/run-qwen38-flash-next.sh')]
        dry = subprocess.run(launch, env=dict(env, DRY='1'), capture_output=True, text=True, timeout=30)
        (self.out / 'dry.log').write_text(dry.stdout + dry.stderr)
        if dry.returncode or 'WOULD BLOCK' in dry.stdout + dry.stderr or 'guards      : OK' not in dry.stdout:
            raise RuntimeError('launcher dry guard blocked/unverified')
        self.server_log = (self.out / 'server.log').open('x')
        self.scope_launched = True
        self.proc = self.spawn(launch, env=env, stdout=self.server_log, stderr=subprocess.STDOUT)
        self.spawned()
        for _ in range(20):
            if self.command(['systemctl', '--user', 'is-active', self.scope], 'scope-ready', check=False) == 'active':
                break
            if self.proc.poll() is not None:
                raise RuntimeError('launcher exited before scope readiness')
            time.sleep(0.25)
        else:
            raise RuntimeError('own scope never became active')
        self.label = 'model-load'
        deadline = min(self.start + WORK_SECONDS, time.monotonic() + 360)
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
        for index, nmax in enumerate(self.sequence, 1):
            self.label = f'{index:02d}-nmax{nmax}'
            remaining = min(REQUEST_SECONDS, self.start + WORK_SECONDS - time.monotonic())
            if remaining <= 1:
                raise TimeoutError('no request budget remains')
            request_path = self.out / (self.label + '-request.json')
            response_path = self.out / (self.label + '-response.json')
            self.save(request_path.name, payload(nmax))
            began = time.monotonic()
            self.sample()
            with response_path.open('x') as response, (self.out / (self.label + '-curl.log')).open('x') as error:
                self.request = self.spawn(['curl', '--silent', '--show-error', '--fail-with-body',
                                           '--connect-timeout', '5', '--max-time', str(remaining),
                                           '-H', 'Content-Type: application/json', '--data-binary', '@' + str(request_path),
                                           'http://127.0.0.1:8102/v1/chat/completions'], stdout=response, stderr=error)
                self.spawned()
                while self.request.poll() is None:
                    self.sample()
                    if time.monotonic() - began >= remaining:
                        raise TimeoutError('request deadline')
                    time.sleep(1)
                if self.request.wait(timeout=5):
                    raise RuntimeError('completion HTTP/curl failure: ' + self.label)
            self.sample()
            result = validate_completion(json.loads(response_path.read_text()), nmax)
            result.update(label=self.label, n_max=nmax, wall_seconds=time.monotonic() - began)
            self.save(self.label + '-metrics.json', result)
            self.summary['requests'].append(result)
            self.save('summary.json', self.summary)
        if self.tokenize_references:
            self.capture_tokenizations()
        self.summary['status'] = 'completed'

    def capture_tokenizations(self):
        """Retokenized response fields, NOT observed decode or draft token IDs."""
        paths = [self.out / (r['label'] + '-response.json') for r in self.summary['requests']]
        paths += self.tokenize_references
        records = []
        def post(endpoint, body):
            self.sample()  # Enforce work deadline and budgets for every request.
            request = urllib.request.Request('http://127.0.0.1:8102/' + endpoint,
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)
        for path in paths:
            fields = output_fields(json.loads(path.read_text()))
            tokens = {}
            for key, text in fields.items():
                if text is None:
                    tokens[key] = None
                    continue
                result = post('tokenize', {'content': text, 'add_special': False})['tokens']
                if not isinstance(result, list) or any(type(t) is not int for t in result):
                    raise RuntimeError('invalid token IDs')
                if post('detokenize', {'tokens': result})['content'] != text:
                    raise RuntimeError('retokenization failed exact round-trip')
                tokens[key] = result
            records.append({'response_path': str(path.resolve()), 'fields': fields, 'tokens': tokens})
        # Do not detokenize individual IDs: byte-fallback tokens may be invalid
        # UTF-8 in isolation. Compare complete ID sequences offline, including
        # prefix/length and null differences; retain exact source fields here.
        self.save('retokenized.json', {'kind': 'retokenized_fields_not_decode_trace',
                                      'records': records})

    def own_scope_empty(self):
        cg = self.command(['systemctl', '--user', 'show', self.scope, '-p', 'ControlGroup', '--value'],
                          'cleanup-cgroup', check=False)
        if cg:
            if not cg.endswith('/' + self.scope):
                raise RuntimeError('unexpected cleanup cgroup')
            self.cgroup_path = Path('/sys/fs/cgroup') / cg.lstrip('/')
        base = self.cgroup_path
        if base is None:
            raise RuntimeError('owned cgroup never observed; cannot prove emptiness')
        return not base.exists() or 'populated 0' in (base / 'cgroup.events').read_text()

    # QLI-PAUSE (Seppe, 2026-09-11): qli.service is masked and must NEVER be
    # started or restarted — not before GPU work (its stop line below is a
    # harmless no-op on a masked unit), and not in cleanup. Jetski is the
    # active miner: the night cron stops it before GPU work and restores it
    # after (Seppe mandate 2026-09-11 ~20:45).
    QLI_MASKED_HINT = 'not-found'   # systemctl is-enabled qli.service while masked

    def _miner_stop_uses(self, service):
        """Return the systemctl action used to stop a miner, or None when the
        unit is masked/absent (qli) — stopping those is a documented no-op."""
        enabled = self.command(['systemctl', '--user', 'is-enabled', service],
                               service + '-enabled', check=False)
        return None if enabled.strip() in ('not-found', 'masked') else 'stop'

    def stop_miners(self):
        stopped = {}
        for service in ('qli.service', 'jetski.service'):
            action = self._miner_stop_uses(service)
            if action is None:
                self.summary[service + '_skip'] = 'masked/not-found — not stopped (pause mandate)'
                continue
            self.summary[service + '_stop_requested'] = datetime.now().astimezone().isoformat()
            self.command(['systemctl', '--user', action, service], service.replace('.', '-') + '-stop')
            stopped[service] = True
        self.summary['miners_stopped'] = list(stopped)

    def restore_miners(self, errors, attempt):
        """Pause-conform miner restoration: restart exactly what THIS run
        stopped (Jetski), never qli (masked/paused)."""
        for service in self.summary.get('miners_stopped') or []:
            attempt(service.replace('.', '-') + '-start',
                    lambda s=service: self.command(['systemctl', '--user', 'start', s],
                                                   s.replace('.', '-') + '-start'))
            state = attempt(service.replace('.', '-') + '-active',
                            lambda s=service: self.command(['systemctl', '--user', 'is-active', s],
                                                           s.replace('.', '-') + '-active', check=False))
            if state != 'active':
                errors.append(service + ' not verified active')

    def cleanup(self):
        """Each action bounded; failures never suppress mandatory restoration."""
        errors = []
        def attempt(name, fn):
            try:
                return fn()
            except BaseException as error:
                errors.append(f'{name}: {error}')
                return None
        attempt('request-stop', lambda: terminate(self.request))
        if self.scope_launched:
            attempt('scope-stop', lambda: self.command(['systemctl', '--user', 'stop', self.scope], 'scope-stop'))
            state = attempt('scope-state', lambda: self.command(['systemctl', '--user', 'is-active', self.scope], 'scope-state', check=False))
            empty = attempt('scope-empty', self.own_scope_empty)
            if state not in ('inactive', 'failed', 'unknown') or empty is not True:
                attempt('scope-kill', lambda: self.command(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', self.scope], 'scope-kill'))
        attempt('launcher-stop', lambda: terminate(self.proc))
        if self.scope_launched:
            def verify():
                for _ in range(3):
                    state = self.command(['systemctl', '--user', 'is-active', self.scope], 'scope-final-state', check=False)
                    if state in ('inactive', 'failed', 'unknown') and self.own_scope_empty():
                        self.summary['scope_stopped_verified'] = True
                        return
                    self.command(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', self.scope], 'scope-final-kill', check=False)
                    time.sleep(0.25)
                raise RuntimeError('own scope termination unverified')
            attempt('scope-final-verification', verify)
        if self.server_log:
            attempt('server-log-close', self.server_log.close)
        # Pause-conform restoration: only what this run stopped comes back
        # (Jetski); qli stays masked/paused per the 2026-09-11 mandate.
        self.restore_miners(errors, attempt)
        if self.prepared:
            attempt('host-restore', lambda: self.command(['bash', PREP, '--restore'], 'host-restore', timeout=30))
            state = attempt('headless-active', lambda: self.command(['systemctl', '--user', 'is-active', 'headless-chromium.service'], 'headless-active'))
            if state != 'active':
                errors.append('headless host not verified active')
        self.summary['cleanup_errors'] = errors
        if errors:
            self.summary['status'] = 'cleanup_failed'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--startup-n-max', type=int, choices=(0, 4, 8, 16), default=4,
                        help='repeat twice at this startup/request depth; reload for a different depth')
    parser.add_argument('--checkpoint', choices=sorted(CHECKPOINTS), default=DEFAULT_CHECKPOINT,
                        help='A/B checkpoint alias: reference 197 GiB Q4_K_M shards vs ps-iq2xxs 75.2 GiB IQ2_XXS')
    parser.add_argument('--cold-cache', action='store_true',
                        help='request POSIX_FADV_DONTNEED on the model file after the GPU guard, '
                             'before launch (advisory; cold state NOT verified)')
    parser.add_argument('--tokenize-reference', type=Path, action='append', default=[],
                        help='retokenize saved response fields after generation; not a decode trace')
    # Explicit opt-in only: clean baselines must not inherit diagnostic tracing.
    parser.add_argument('--verifier-trace', action='store_true', help='diagnostic first-128 verifier decisions, NOT a clean throughput baseline')
    args = parser.parse_args(argv)
    try:
        validate_plan(args.startup_n_max)
        for path in args.tokenize_reference:
            output_fields(json.loads(path.read_text()))
    except ValueError as error:
        print(error)
        return 2
    out = Path(__file__).resolve().parent / ('run-' + datetime.now().strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex)
    out.mkdir()
    run = Run(out, args.startup_n_max, args.checkpoint, cold_cache=args.cold_cache)
    run.tokenize_references = args.tokenize_reference
    run.verifier_trace = args.verifier_trace
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
    raise SystemExit(main())
