#!/usr/bin/env python3
"""Hermes E2E: serve ps-iq2xxs under the guarded lifecycle, run a real
`hermes chat -q` task (shell tool) against the local endpoint from an
ISOLATED HERMES_HOME, capture the transcript, then the standard cleanup.
Seppe's active Hermes provider is never touched. Durations follow the
extended mandate; RAM/VRAM caps and every guard unchanged.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('harness', HERE.parent / 'coldcache-ab-20260911/run-guarded.py')
assert spec is not None and spec.loader is not None
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

E2E_HOME = (HERE.parent.parent / '.hermes-work/e2e-home').resolve()
TASK = ('Use the shell tool to run exactly this command: printf "hermes-e2e-ok"\n'
        'Then reply with exactly the output you saw and nothing else.')
WORK_SECONDS = int(os.environ.get('E2E_WORK_SECONDS', '1800'))
REQUEST_SECONDS = WORK_SECONDS - 120


def main():
    out = HERE / ('run-' + time.strftime('%Y%m%dT%H%M%S') + '-e2e')
    out.mkdir()
    run = harness.Run(out, 4, 'ps-iq2xxs')
    run.work_seconds = WORK_SECONDS
    run.request_seconds = REQUEST_SECONDS
    summary = {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'kind': 'hermes-e2e-agent-run', 'task': TASK,
               'hermes_home': str(E2E_HOME), 'status': 'failed'}
    (out / 'e2e-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    try:
        # --- guarded lifecycle up to model readiness (mirrors execute()) ---
        run.command(['git', 'rev-parse', 'HEAD'], 'commit')
        if subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True, timeout=10).returncode != 1:
            raise RuntimeError('pre-existing llama-server')
        if run.command(['systemctl', '--user', 'list-units', '--type=scope', '--state=running',
                        '--no-legend', '--plain', 'expertpin-test-*.scope'], 'preexisting-scopes'):
            raise RuntimeError('pre-existing expertpin scope')
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 8102))
        env = harness.clean_environment(os.environ, run.scope, 4, 'ps-iq2xxs')
        run.save('environment.json', {k: v for k, v in env.items()
                                      if k not in os.environ or os.environ[k] != v})
        run.prepared = True
        run.command(['bash', harness.PREP], 'host-prep', timeout=45)
        run.summary['qli_stop_requested'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        run.command(['systemctl', '--user', 'stop', 'qli.service'], 'qli-stop')
        run.command(['nvidia-smi'], 'nvidia-before')
        run.command(['free', '-g'], 'free-before')
        gate_spec = importlib.util.spec_from_file_location('gate', harness.ROOT / 'evidence/trace-matched-bandwidth/run-physical.py')
        gate = importlib.util.module_from_spec(gate_spec)
        gate_spec.loader.exec_module(gate)
        gate.OUT = out
        idle = gate.wait_gpu_idle('model-readiness')
        run.summary['guard'] = idle
        if not (0 <= idle['gpu_util_pct'] < 5 and idle['mem_available_bytes'] >= 34 * 1024**3):
            raise RuntimeError('Tier A host guard blocked')
        launch = ['bash', str(harness.ROOT / 'scripts/run-qwen38-flash-next.sh')]
        dry = subprocess.run(launch, env=dict(env, DRY='1'), capture_output=True, text=True, timeout=30)
        (out / 'dry.log').write_text(dry.stdout + dry.stderr)
        if dry.returncode or 'WOULD BLOCK' in dry.stdout + dry.stderr or 'guards      : OK' not in dry.stdout:
            raise RuntimeError('launcher dry guard blocked/unverified')
        run.server_log = (out / 'server.log').open('x')
        run.scope_launched = True
        run.proc = run.spawn(launch, env=env, stdout=run.server_log, stderr=subprocess.STDOUT)
        run.spawned()
        import urllib.request
        deadline = min(run.start + run.work_seconds, time.monotonic() + 900)
        while time.monotonic() < deadline:
            run.sample()
            try:
                with urllib.request.urlopen('http://127.0.0.1:8102/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError('health readiness timeout')
        run.summary['model_loaded'] = True
        summary['model_loaded'] = True

        # --- the actual E2E step: a real hermes agent run against the model ---
        summary['hermes_invocation'] = 'hermes chat -q <shell-tool task>'
        (out / 'task.txt').write_text(TASK)
        client_env = dict(os.environ, HERMES_HOME=str(E2E_HOME))
        began = time.monotonic()
        run.label = 'hermes-e2e'
        run.sample()
        client = subprocess.run(['hermes', 'chat', '--yolo', '-q', TASK],
                                capture_output=True, text=True,
                                timeout=REQUEST_SECONDS, env=client_env, cwd=str(out))
        wall = time.monotonic() - began
        run.sample()
        (out / 'hermes-stdout.txt').write_text(client.stdout)
        (out / 'hermes-stderr.txt').write_text(client.stderr)
        (out / 'hermes-exit.txt').write_text(str(client.returncode))
        summary['hermes_returncode'] = client.returncode
        summary['hermes_wall_seconds'] = round(wall, 1)
        summary['hermes_stdout_has_marker'] = 'hermes-e2e-ok' in client.stdout
        summary['status'] = 'completed' if client.returncode == 0 and summary['hermes_stdout_has_marker'] else 'client_failed'
        (out / 'e2e-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    finally:
        try:
            run.cleanup()
        finally:
            run.summary['e2e'] = summary
            run.summary['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            run.save('summary.json', run.summary)
    print(out)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get('status') == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
