"""Bounded physical probes only, after all research/review/build work. No model load."""
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('EXPERTPIN_PHYSICAL_OUT', str(Path(__file__).resolve().parent))).resolve()
PREP = '/home/seppe/.hermes/profiles/expertpin/scripts/gpu-host-prep.sh'


def cmd(args, name, check=True, timeout: float = 60):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    (OUT / (name + '.log')).write_text(p.stdout + p.stderr)
    if check and p.returncode:
        raise RuntimeError(f'{name}: {p.returncode}: {p.stderr}')
    return p.stdout.strip()


def guard_sample(name, timeout: float = 10) -> dict:
    cmd(['nvidia-smi'], name + '-nvidia', timeout=timeout)
    cmd(['free', '-g'], name + '-free', timeout=timeout)
    line = cmd(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], name + '-gpu', timeout=timeout)
    util, vram = map(int, line.split(','))
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
    if not (0 <= util <= 100 and 0 <= vram < 28 * 1024 and available >= 34 * 1024**3):
        raise RuntimeError('Tier A RAM/VRAM guard blocked')
    return dict(mem_available_bytes=available, gpu_util_pct=util, vram_mib=vram)


def guard(name):
    value = guard_sample(name)
    if value['gpu_util_pct'] >= 5:
        raise RuntimeError('GPU guard blocked')
    return value


def wait_gpu_idle(name, timeout=60, interval=2):
    """Retry only busy utilization, never RAM/VRAM or observation failures."""
    if not (0 < timeout <= 90 and 0 < interval <= timeout):
        raise ValueError('invalid readiness bounds')
    start = time.monotonic()
    observations = []
    try:
        while True:
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise RuntimeError('GPU idle readiness deadline exceeded')
            # Three commands per observation; their combined timeout fits remaining.
            value = guard_sample(f'{name}-{len(observations):02d}', timeout=min(10, remaining / 3))
            value['elapsed_s'] = time.monotonic() - start
            observations.append(value)
            if value['elapsed_s'] >= timeout:
                raise RuntimeError('GPU idle readiness deadline exceeded')
            # Defensive revalidation also keeps the polling contract independently testable.
            if not (value['mem_available_bytes'] >= 34 * 1024**3 and 0 <= value['vram_mib'] < 28 * 1024):
                raise RuntimeError('Tier A RAM/VRAM guard blocked')
            if not 0 <= value['gpu_util_pct'] <= 100:
                raise RuntimeError('invalid GPU utilization')
            if value['gpu_util_pct'] < 5:
                return value
            time.sleep(min(interval, timeout - value['elapsed_s']))
    finally:
        (OUT / (name + '-readiness.json')).write_text(json.dumps(observations, indent=2) + '\n')


def inner():
    cg = next(x.split(':', 2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    base = Path('/sys/fs/cgroup') / cg.lstrip('/')
    if int((base / 'memory.max').read_text()) != 40 * 1024**3:
        raise RuntimeError('MemoryMax mismatch')
    if int((base / 'memory.swap.max').read_text()) != 0:
        raise RuntimeError('MemorySwapMax mismatch')
    data = {'samples': [], 'model_load': False, 'started': datetime.now().astimezone().isoformat()}
    def sample():
        values = {k: (base / k).read_text().strip() for k in ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current', 'memory.events')}
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10)
        values.update(vram_mib=int(raw.strip()), timestamp=datetime.now().astimezone().isoformat())
        data['samples'].append(values)
        if values['vram_mib'] >= 28 * 1024 or int(values['memory.peak']) > 40 * 1024**3:
            raise RuntimeError('budget exceeded')
    try:
        sample()
        if os.environ.get('EXPERTPIN_GPU_ONLY') != '1':
            command = [sys.executable, str(ROOT / 'scripts/bench-trace-bandwidth.py'), '--model-root', '/home/seppe/Models/qwen3.8-flash-next/AD-4.27bpw-Q4_K_M-M64', '--sample-mib', '512', '--seed', '17', '--json', str(OUT / 'bandwidth.json')]
            cmd(command, 'bandwidth-stdout', timeout=180)
            sample()
        data['gpu_guard'] = guard('before-gemm')
        gpu_cmd = [str(ROOT / 'build-sm120/bin/bench-expert-gpu'), '101']
        with (OUT / 'gpu.json').open('x') as output, (OUT / 'gpu.stderr.log').open('x') as error:
            p = subprocess.Popen(gpu_cmd, stdout=output, stderr=error, env=dict(os.environ, GGML_CUDA_NO_PINNED='1', CUDA_VISIBLE_DEVICES='0'))
            try:
                deadline = time.monotonic() + 120
                while p.poll() is None:
                    sample()
                    if time.monotonic() >= deadline:
                        raise TimeoutError('GPU probe limit')
                    time.sleep(0.05)
                if p.returncode:
                    raise RuntimeError(f'GPU probe rc={p.returncode}')
            finally:
                if p.poll() is None:
                    p.kill()
                p.wait(timeout=10)
        sample()
        data['success'] = True
    except BaseException as e:
        data['error'] = repr(e)
        raise
    finally:
        data['finished'] = datetime.now().astimezone().isoformat()
        (OUT / 'budget.json').write_text(json.dumps(data, indent=2) + '\n')


def stop_scope(scope):
    """Escalate failed stop, then verify the exact unit has no populated cgroup."""
    base = None
    for attempt in range(2):
        observed = False
        try:
            cg = cmd(['systemctl', '--user', 'show', scope, '-p', 'ControlGroup', '--value'], 'cleanup-cgroup', check=False)
            if cg and not cg.endswith('/' + scope):
                raise RuntimeError('unexpected scope cgroup')
            if cg:
                base = Path('/sys/fs/cgroup') / cg.lstrip('/')
            observed = True
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            pass  # Observation/log failure must never skip termination attempts.
        if attempt:
            try:
                cmd(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', scope], 'scope-kill', check=False)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                pass
        try:
            cmd(['systemctl', '--user', 'stop', scope], 'scope-stop', check=False, timeout=20)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            pass
        try:
            state = cmd(['systemctl', '--user', 'is-active', scope], 'scope-state', check=False)
            populated = base is not None and base.exists() and 'populated 1' in (base / 'cgroup.events').read_text()
            if observed and state in ('inactive', 'failed', 'unknown') and not populated:
                return state
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            pass
    raise RuntimeError('scope termination unverified after SIGKILL escalation')


def validate_output():
    """Reject invalid/reused run destinations before lifecycle actions or writes."""
    if not OUT.is_relative_to(ROOT / 'evidence') or not OUT.is_dir():
        raise RuntimeError('output must be an existing evidence directory')
    # Require a separate empty run directory, not a directory of source/research.
    if any(OUT.iterdir()):
        raise RuntimeError('output directory must be empty')


def outer():
    validate_output()
    scope = 'expertpin-physical-' + uuid.uuid4().hex + '.scope'
    data = {'scope': scope, 'cleanup_errors': [], 'started': datetime.now().astimezone().isoformat()}
    def interrupt(signum, frame):
        raise TimeoutError(f'signal {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupt)
    signal.alarm(600)
    try:
        cmd(['bash', PREP], 'host-prep')
        data['qli_stop_requested'] = datetime.now().astimezone().isoformat()
        cmd(['systemctl', '--user', 'stop', 'qli.service'], 'qli-stop')
        data['guard'] = wait_gpu_idle('before-probes')
        cmd(['systemd-run', '--user', '--scope', '--unit=' + scope, '-p', 'MemoryMax=40G', '-p', 'MemorySwapMax=0', 'timeout', '--signal=TERM', '--kill-after=10', '300', sys.executable, str(Path(__file__).resolve()), '--inner'], 'scope', timeout=330)
        data['success'] = True
    except BaseException as e:
        data['error'] = repr(e)
    finally:
        signal.alarm(0)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        try:
            data['scope-state'] = stop_scope(scope)
        except BaseException as e:
            # Unconditional qli restoration is a standing operator mandate;
            # failed termination must remain an explicit failed run, never success.
            data['cleanup_errors'].append(f'scope termination: {e!r}')
        actions = [
            ('qli-start', ['systemctl', '--user', 'start', 'qli.service'], True),
            ('qli-active', ['systemctl', '--user', 'is-active', 'qli.service'], True),
            ('host-restore', ['bash', PREP, '--restore'], True),
            ('chromium-active', ['systemctl', '--user', 'is-active', 'headless-chromium.service'], True),
            ('qli-journal', ['journalctl', '--user', '-u', 'qli.service', '--since', data['started'], '-n', '30', '--no-pager'], True),
        ]
        for name, command, check in actions:
            try:
                if name == 'qli-start':
                    data['qli_start_requested'] = datetime.now().astimezone().isoformat()
                result = cmd(command, name, check=check)
                data[name] = result
                if name == 'scope-state' and result not in ('inactive', 'failed', 'unknown'):
                    raise RuntimeError('scope not verified stopped')
            except BaseException as e:
                data['cleanup_errors'].append(f'{name}: {e!r}')
        (OUT / 'execution.json').write_text(json.dumps(data, indent=2) + '\n')
    return 0 if data.get('success') and not data['cleanup_errors'] else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--inner']:
        inner()
    else:
        raise SystemExit(outer())
