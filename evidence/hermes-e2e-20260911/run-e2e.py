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
# The code-precision scenario that IQ2_XXS failed in real use (2026-09-11
# loop/typo spiral): write a small exact-syntax script AND run it AND report
# its real output. PASS requires the script to exist, execute cleanly, and
# produce the expected output.
CODE_TASK = ('Write a small Python script to /tmp/e2e-codegate.py that prints exactly:\n'
             'code-gate-ok-7391\n'
             'Requirements: use a for-loop over range(3) summing those iteration numbers, and print'
             ' the literal string above once at the end. Use the shell tool to write the file, then'
             ' run it with python3, and reply with exactly the output it printed.')
TASK = ('Use the shell tool to run exactly this command: printf "hermes-e2e-ok" > /tmp/hermes-e2e-marker.txt && cat /tmp/hermes-e2e-marker.txt\n'
        'Then reply with exactly the output you saw and nothing else.')
# Multi-turn variant: turn 2 must REUSE the turn-1 result (a file the agent
# itself created) — proves tool-result persistence across turns, not just a
# second independent query. Resume via `hermes chat -c`.
TASK_TURN2 = ('Now use the shell tool to run exactly: cat /tmp/hermes-e2e-marker.txt\n'
              'Reply with exactly what it printed and nothing else.')
MULTI_TURN = os.environ.get('E2E_MULTI_TURN', '0') == '1'
E2E_CHECKPOINT = os.environ.get('E2E_CHECKPOINT', 'ps-iq2xxs')
E2E_TASK_KIND = os.environ.get('E2E_TASK', 'marker')  # marker | code
E2E_NCMOE = os.environ.get('E2E_NCMOE', '36')
# Dev default is --reasoning off (IQ2_XXS code loops); the gate mirrors the
# dev setting so we test the configuration users will actually run.
E2E_REASONING = os.environ.get('E2E_REASONING', 'off')
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
        env = harness.clean_environment(os.environ, run.scope, 4, E2E_CHECKPOINT)
        env['NCMOE'] = E2E_NCMOE
        # Mirror the dev default: reasoning off (both launchers accept the
        # REASONING env; auto = model behavior for benchmarks).
        env['REASONING'] = E2E_REASONING
        # Hermes requires a >=64K reported context; 64K is proven within
        # budget on this checkpoint (context ladder evidence 2026-09-11).
        env['CTX'] = '65536'
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
        for _ in range(20):
            if run.command(['systemctl', '--user', 'is-active', run.scope], 'scope-ready', check=False) == 'active':
                break
            if run.proc.poll() is not None:
                raise RuntimeError('launcher exited before scope readiness')
            time.sleep(0.25)
        else:
            raise RuntimeError('own scope never became active')
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
        client_env = dict(os.environ, HERMES_HOME=str(E2E_HOME),
                          EXPERTPIN_LOCAL_KEY='local-e2e-dummy')
        began = time.monotonic()
        run.label = 'hermes-e2e'
        run.sample()
        client = subprocess.run(['hermes', 'chat', '--yolo', '-q', CODE_TASK if E2E_TASK_KIND == 'code' else TASK],
                                capture_output=True, text=True,
                                timeout=REQUEST_SECONDS, env=client_env, cwd=str(out))
        wall = time.monotonic() - began
        run.sample()
        (out / 'hermes-stdout.txt').write_text(client.stdout)
        (out / 'hermes-stderr.txt').write_text(client.stderr)
        (out / 'hermes-exit.txt').write_text(str(client.returncode))
        summary['hermes_returncode'] = client.returncode
        summary['hermes_wall_seconds'] = round(wall, 1)
        stdout = client.stdout
        # Guard against the false positive where the marker only appears in
        # the echoed Query: require it AFTER the query echo, plus evidence of
        # an actual tool execution (shell) in the transcript.
        after_query = stdout.split('\n', 1)[1] if '\n' in stdout else ''
        summary['hermes_stdout_has_marker'] = 'hermes-e2e-ok' in after_query
        summary['hermes_ran_shell_tool'] = ('shell' in stdout.lower() and
                                            ('printf' in stdout or 'command' in stdout.lower()))
        ok_turn1 = (client.returncode == 0 and summary['hermes_stdout_has_marker']
                    and summary['hermes_ran_shell_tool'])
        if E2E_TASK_KIND == 'code' and ok_turn1:
            # Code-precision gate: verify the artifact independently of the
            # agent's self-report — the script must exist, run cleanly, and
            # print exactly the required line.
            import subprocess as _sp
            probe = _sp.run(['python3', '/tmp/e2e-codegate.py'], capture_output=True, text=True, timeout=30)
            summary['codegate_script_rc'] = probe.returncode
            summary['codegate_script_output'] = probe.stdout.strip()
            summary['codegate_script_expected'] = 'code-gate-ok-7391'
            summary['codegate_pass'] = (probe.returncode == 0
                                        and probe.stdout.strip() == 'code-gate-ok-7391')
            ok_turn1 = ok_turn1 and summary['codegate_pass']
        if MULTI_TURN and ok_turn1:
            import time as _t
            _t.sleep(2)
            began2 = time.monotonic()
            run.label = 'hermes-e2e-turn2'
            run.sample()
            turn2 = subprocess.run(['hermes', 'chat', '--yolo', '--resume', 'latest', '-q', TASK_TURN2],
                                   capture_output=True, text=True,
                                   timeout=REQUEST_SECONDS, env=client_env, cwd=str(out))
            wall2 = time.monotonic() - began2
            run.sample()
            (out / 'hermes-turn2-stdout.txt').write_text(turn2.stdout)
            (out / 'hermes-turn2-stderr.txt').write_text(turn2.stderr)
            summary['turn2_returncode'] = turn2.returncode
            summary['turn2_wall_seconds'] = round(wall2, 1)
            after_query2 = turn2.stdout.split('\n', 1)[1] if '\n' in turn2.stdout else ''
            summary['turn2_stdout_has_marker'] = 'hermes-e2e-ok' in after_query2
            summary['turn2_ran_shell_tool'] = 'cat /tmp/hermes-e2e-marker.txt' in turn2.stdout
            summary['multi_turn'] = True
            summary['status'] = 'completed' if (turn2.returncode == 0
                                               and summary['turn2_stdout_has_marker']
                                               and summary['turn2_ran_shell_tool']) else 'client_failed'
        else:
            summary['multi_turn'] = False
            summary['status'] = 'completed' if ok_turn1 else 'client_failed'
        (out / 'e2e-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    except BaseException as error:
        summary['error'] = repr(error)
        summary['status'] = 'failed'
        try:
            (out / 'e2e-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        except OSError:
            pass
        raise
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
