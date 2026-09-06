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
OUT = Path(__file__).resolve().parent
PREP = '/home/seppe/.hermes/profiles/expertpin/scripts/gpu-host-prep.sh'


def cmd(args, name, check=True, timeout=60):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    (OUT / (name + '.log')).write_text(p.stdout + p.stderr)
    if check and p.returncode:
        raise RuntimeError(f'{name}: {p.returncode}: {p.stderr}')
    return p.stdout.strip()


def guard(name):
    cmd(['nvidia-smi'], name + '-nvidia')
    cmd(['free', '-g'], name + '-free')
    line = cmd(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], name + '-gpu')
    util, vram = map(int, line.split(','))
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
    if not (0 <= util < 5 and vram < 28 * 1024 and available >= 34 * 1024**3):
        raise RuntimeError('Tier A/GPU guard blocked')
    return dict(mem_available_bytes=available, gpu_util_pct=util, vram_mib=vram)


def inner():
    cg = next(x.split(':', 2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    base = Path('/sys/fs/cgroup') / cg.lstrip('/')
    if int((base / 'memory.max').read_text()) != 40 * 1024**3:
        raise RuntimeError('MemoryMax mismatch')
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
    cg = cmd(['systemctl', '--user', 'show', scope, '-p', 'ControlGroup', '--value'], 'cleanup-cgroup', check=False)
    base = Path('/sys/fs/cgroup') / cg.lstrip('/') if cg and cg.endswith('/' + scope) else None
    for attempt in range(2):
        try:
            if attempt:
                cmd(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', scope], 'scope-kill', check=False)
            cmd(['systemctl', '--user', 'stop', scope], 'scope-stop', check=False, timeout=20)
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        state = cmd(['systemctl', '--user', 'is-active', scope], 'scope-state', check=False)
        populated = base is not None and base.exists() and 'populated 1' in (base / 'cgroup.events').read_text()
        if state in ('inactive', 'failed', 'unknown') and not populated:
            return state
    raise RuntimeError('scope termination unverified after SIGKILL escalation')


def outer():
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
        data['guard'] = guard('before-probes')
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
