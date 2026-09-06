"""One bounded model trace attempt, after research/review/build. No guard overrides."""
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import urllib.request
import urllib.error
import importlib.util

# Reuse the already regression-tested bounded idle gate; importing has no lifecycle actions.
_gate_path = Path(__file__).resolve().parents[1] / 'trace-matched-bandwidth/run-physical.py'
_gate_spec = importlib.util.spec_from_file_location('physical_gate', _gate_path)
assert _gate_spec is not None and _gate_spec.loader is not None
_gate = importlib.util.module_from_spec(_gate_spec)
_gate_spec.loader.exec_module(_gate)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / datetime.now().strftime('run-%Y%m%dT%H%M%S')
OUT.mkdir()
PREP = '/home/seppe/.hermes/profiles/expertpin/scripts/gpu-host-prep.sh'
start = time.monotonic()
summary = {'started': datetime.now().astimezone().isoformat(), 'model_loaded': False}
proc = None
scope = 'expertpin-test-' + uuid.uuid4().hex + '.scope'
scope_launched = False
request = None
pcie = None
pcie_log = None
prepared = False
server_log = None
samples = []
launching = False
pending_signal = None
RAM_BUDGET_GIB = 36  # Measurement cap; launcher still requires budget + 4 GiB.


def host_guard(util, mem):
    """Tier A host gate; independent of the stricter launcher headroom gate."""
    return 0 <= util < 5 and mem >= 34 * 1024**3


def validate_completion(data):
    """Require the intended bounded decode, not merely an HTTP 200 or EOS."""
    usage = data.get('usage', {})
    choices = data.get('choices', [])
    if (type(usage.get('completion_tokens')) is not int or usage['completion_tokens'] != 32
            or len(choices) != 1 or choices[0].get('finish_reason') != 'length'):
        raise RuntimeError('request did not produce the intended 32-token decode')
    message = choices[0].get('message', {})
    if not any(isinstance(message.get(k), str) and message[k].strip()
               for k in ('content', 'reasoning_content')):
        raise RuntimeError('request produced no text')
    return usage


def command(args, name, check=True):
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    (OUT / (name + '.log')).write_text(result.stdout + result.stderr)
    if check and result.returncode:
        raise RuntimeError(f'{name}: rc={result.returncode}: {result.stderr}')
    return result.stdout.strip()


def gpu():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = result.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError('expected exactly one GPU')
    return [int(n.strip()) for n in rows[0].split(',')]


def available():
    return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                    if s.startswith('MemAvailable:'))) * 1024


def sample():
    util, used, _ = gpu()
    point = {'elapsed_s': time.monotonic() - start, 'vram_used_mib': used,
             'gpu_util_pct': util, 'mem_available_bytes': available()}
    if scope_launched:
        cg = subprocess.check_output(['systemctl', '--user', 'show', scope, '-p', 'ControlGroup', '--value'], text=True, timeout=10).strip()
        if not cg or not cg.endswith('/' + scope):
            raise RuntimeError('missing or mismatched own scope')
        base = Path('/sys/fs/cgroup') / cg.lstrip('/')
        for key in ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current', 'memory.events'):
            point[key] = (base / key).read_text().strip()
        if int(point['memory.max']) != RAM_BUDGET_GIB * 1024**3:
            raise RuntimeError('unexpected cgroup MemoryMax')
    samples.append(point)
    (OUT / 'samples.json').write_text(json.dumps(samples, indent=2) + '\n')
    if used >= 28 * 1024:
        raise RuntimeError('VRAM cap exceeded')
    if time.monotonic() - start > 900:
        raise TimeoutError('15 minute run limit')
    if proc is None or proc.poll() is not None:
        raise RuntimeError('server process missing or exited')


def interrupt(signum, frame):
    global pending_signal
    if launching:
        pending_signal = signum
        return
    raise TimeoutError(f'signal {signum}')


signal.signal(signal.SIGTERM, interrupt)
signal.signal(signal.SIGINT, interrupt)
signal.signal(signal.SIGALRM, interrupt)
signal.alarm(960)
try:
    command(['git', 'rev-parse', 'HEAD'], 'commit')
    if subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True).returncode == 0:
        raise RuntimeError('pre-existing llama-server')
    units = command(['systemctl', '--user', 'list-units', '--type=scope', '--state=running',
                     '--no-legend', '--plain', 'expertpin-test-*.scope'], 'preexisting-scopes')
    if units:
        raise RuntimeError('pre-existing expertpin scope')
    prepared = True
    command(['bash', PREP], 'host-prep')
    summary['qli_stop_requested'] = datetime.now().astimezone().isoformat()
    command(['systemctl', '--user', 'stop', 'qli.service'], 'qli-stop')
    command(['nvidia-smi'], 'nvidia-before')
    command(['free', '-g'], 'free-before')
    _gate.OUT = OUT
    idle = _gate.wait_gpu_idle('model-readiness')
    util, mem = idle['gpu_util_pct'], idle['mem_available_bytes']
    summary['guard'] = {'gpu_util_pct': util, 'mem_available_bytes': mem,
                        'required_mem_available': '>=34 GiB (Tier A)',
                        'ram_budget_gib': RAM_BUDGET_GIB,
                        'launcher_headroom_gib': 4}
    if not host_guard(util, mem):
        raise RuntimeError('host guard blocked; no model load')
    env = dict(os.environ, GGML_CUDA_NO_PINNED='1', DRAFT='1', DRAFT_NMAX='4',
               DRAFT_MODEL='/home/seppe/Models/qwen3.8-flash-next/mtp-drafter/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf',
               MODEL_DIR='/home/seppe/Models/qwen3.8-flash-next/AD-4.27bpw-Q4_K_M-M64',
               CTX='8192', NCMOE='36', NGL='99', RAM_BUDGET_GIB=str(RAM_BUDGET_GIB), CACHE_RAM_MIB='512',
               PORT='8102', BIN_DIR=str(ROOT / 'build-sm120/bin'), FORCE='0', PINNED='0',
               EXPERT_CACHE_SIM_MIB='8192', EXPERT_STATS_FILE=str(OUT / 'expert-stats.json'),
               GGML_MOE_TRACE_FILE=str(OUT / 'trace.csv'), GGML_MOE_TRACE_REQUEST_ONLY='1',
               GGML_MOE_GPU_TRACE_FILE=str(OUT / 'gpu-trace.csv'))
    env['EXPERTPIN_SCOPE_UNIT'] = scope
    if not Path(env['DRAFT_MODEL']).is_file():
        raise RuntimeError('required draft model missing')
    launch = ['bash', str(ROOT / 'scripts/run-qwen38-flash-next.sh')]
    dry = subprocess.run(launch, env=dict(env, DRY='1'), capture_output=True, text=True, timeout=30)
    (OUT / 'dry.log').write_text(dry.stdout + dry.stderr)
    if dry.returncode or 'WOULD BLOCK' in dry.stdout:
        raise RuntimeError('launcher dry guard blocked')
    server_log = (OUT / 'server.log').open('w')
    scope_launched = True
    # Register ownership before spawn and defer handled signals until the child
    # handle is stored; cleanup can then reap the launcher and its exact scope.
    launching = True
    try:
        proc = subprocess.Popen(launch, env=dict(env, DRY='0'), stdout=server_log, stderr=subprocess.STDOUT)
    finally:
        launching = False
    if pending_signal is not None:
        interrupt(pending_signal, None)
    # Bounded readiness check for this exact unit; never infer ownership from wildcards.
    for _ in range(20):
        result = subprocess.run(['systemctl', '--user', 'is-active', scope], capture_output=True, text=True, timeout=5)
        if result.stdout.strip() == 'active':
            break
        if proc.poll() is not None:
            raise RuntimeError('launcher exited before scope readiness')
        time.sleep(0.25)
    else:
        raise RuntimeError('own scope never became active')
    ready = False
    for _ in range(600):
        sample()
        try:
            with urllib.request.urlopen('http://127.0.0.1:8102/health', timeout=1) as response:
                ready = response.status == 200
        except (OSError, urllib.error.URLError):
            pass
        if ready:
            break
        time.sleep(1)
    if not ready:
        raise TimeoutError('health readiness timeout')
    summary['model_loaded'] = True
    payload = {'messages': [{'role': 'user', 'content':
                'Explain how a bounded expert cache handles RAM and NVMe misses in detail.'}],
               'max_tokens': 32, 'temperature': 0.0, 'seed': 42, 'stream': False, 'cache_prompt': False}
    (OUT / 'request.json').write_text(json.dumps(payload) + '\n')
    # Device-wide PCIe throughput samples; NOT pageable-only or per-process bytes.
    # Includes our IDs readback, other GPU clients, and driver traffic. Keep raw units.
    summary['pcie_observation_start'] = datetime.now().astimezone().isoformat()
    pcie_log = (OUT / 'pcie-dmon.log').open('x')
    launching = True
    try:
        pcie = subprocess.Popen(['nvidia-smi', 'dmon', '-s', 't', '-d', '1', '-c', '300', '-o', 'DT'],
                                stdout=pcie_log, stderr=subprocess.STDOUT)
    finally:
        launching = False
    if pending_signal is not None:
        interrupt(pending_signal, None)
    summary['request_start'] = datetime.now().astimezone().isoformat()
    # curl runs separately so memory/VRAM continue to be sampled during inference.
    with (OUT / 'response.json').open('w') as response:
        request = subprocess.Popen(['curl', '--silent', '--show-error', '--fail-with-body', '--max-time', '300',
                                    '-H', 'Content-Type: application/json', '--data-binary', '@' + str(OUT / 'request.json'),
                                    'http://127.0.0.1:8102/v1/chat/completions'], stdout=response)
        while request.poll() is None:
            sample()
            time.sleep(1)
        if request.returncode:
            raise RuntimeError('completion failed')
    sample()
    summary['request_end'] = datetime.now().astimezone().isoformat()
    summary['pcie_exit_before_cleanup'] = pcie.poll()
    summary['completion_usage'] = validate_completion(json.loads((OUT / 'response.json').read_text()))
    summary['status'] = 'request_completed_pending_evidence_validation'
except Exception as error:
    summary['status'] = 'blocked_or_failed'
    summary['error'] = str(error)
finally:
    signal.alarm(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cleanup_errors = []

    def attempt(name, fn):
        try:
            return fn()
        except BaseException as error:
            cleanup_errors.append(f'{name}: {error}')
            print(cleanup_errors[-1], flush=True)
            return None

    def terminate(child):
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)

    attempt('request-stop', lambda: terminate(request))
    attempt('pcie-stop', lambda: terminate(pcie))
    if pcie_log:
        attempt('pcie-log-close', pcie_log.close)
    if scope_launched:
        attempt('scope-stop', lambda: command(['systemctl', '--user', 'stop', scope], 'scope-stop'))
        active = attempt('scope-state', lambda: command(['systemctl', '--user', 'is-active', scope], 'scope-state', check=False))
        if active not in ('inactive', 'failed', 'unknown'):
            attempt('scope-kill', lambda: command(['systemctl', '--user', 'kill', '--signal=KILL', scope], 'scope-kill'))
            cleanup_errors.append('scope-stop not verified inactive')
    attempt('launcher-stop', lambda: terminate(proc))
    if scope_launched:
        # Recheck after reaping the launcher, which can no longer create a late
        # scope. A successful kill request alone is not proof of termination.
        def verify_scope_stopped():
            for retry in range(3):
                state = command(['systemctl', '--user', 'is-active', scope], 'scope-final-state', check=False)
                if state in ('inactive', 'failed', 'unknown'):
                    summary['scope_stopped_verified'] = True
                    return
                command(['systemctl', '--user', 'kill', '--signal=KILL', scope], 'scope-final-kill', check=False)
                time.sleep(0.25)
            raise RuntimeError('own scope still active or unverified before mandatory qli restart')
        attempt('scope-final-verification', verify_scope_stopped)
    if server_log:
        attempt('log-close', server_log.close)
    summary['qli_start_requested'] = datetime.now().astimezone().isoformat()
    # Required by the developer protocol, also on a preflight refusal.
    attempt('qli-start', lambda: command(['systemctl', '--user', 'start', 'qli.service'], 'qli-start'))
    summary['qli_status'] = attempt('qli-active', lambda: command(['systemctl', '--user', 'is-active', 'qli.service'], 'qli-active'))
    if prepared:
        attempt('host-restore', lambda: command(['bash', PREP, '--restore'], 'host-restore'))
    summary['headless_status'] = attempt('headless-active', lambda: command(['systemctl', '--user', 'is-active', 'headless-chromium.service'], 'headless-active'))
    attempt('journal', lambda: command(['journalctl', '--user', '-u', 'qli.service', '--since', summary['qli_start_requested'], '--no-pager'], 'qli-journal'))
    summary['cleanup_errors'] = cleanup_errors
    if cleanup_errors:
        summary['status'] = 'cleanup_failed'
    summary['finished'] = datetime.now().astimezone().isoformat()
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(OUT)
    print(json.dumps(summary, indent=2))
