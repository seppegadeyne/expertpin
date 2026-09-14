#!/usr/bin/env python3
"""Hermes E2E: serve the selected checkpoint under the guarded lifecycle, run
a real `hermes chat -q` task (shell tool) against the local endpoint from an
ISOLATED HERMES_HOME, capture the transcript, then the standard cleanup.
Seppe's active Hermes provider is never touched. RAM/VRAM caps and every
guard unchanged. The runner itself must execute INSIDE a dedicated client
cgroup scope (E2E_CLIENT_UNIT, 4 GiB MemoryMax) so harness + hermes client +
curl RAM is kernel-accounted and every sample checks the combined
server+client kernel peaks against the 40 GiB budget (pattern proven in
.hermes-work/daily-20260914/run-ud.py). Total work deadline is hard-capped
at 1800 s (Seppe 2026-09-13: the retired 2700 s code default stays retired).
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
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
if E2E_CHECKPOINT not in harness.CHECKPOINTS:
    raise ValueError('unknown E2E_CHECKPOINT: ' + repr(E2E_CHECKPOINT))
if E2E_TASK_KIND not in ('marker', 'code'):
    raise ValueError('unknown E2E task kind: ' + repr(E2E_TASK_KIND))
# Mirror the dev default per checkpoint: UD-Q4_K_XL serves with NCMOE 40
# (dev-serving default), the A/B checkpoints ran NCMOE 36.
E2E_NCMOE = os.environ.get('E2E_NCMOE', '40' if E2E_CHECKPOINT == 'ud-q4kxl' else '36')
# Dev default is --reasoning off (IQ2_XXS code loops); the gate mirrors the
# dev setting so we test the configuration users will actually run.
E2E_REASONING = os.environ.get('E2E_REASONING', 'off')
# Seppe 2026-09-13 08:45: enforce an explicit <=1800 s TOTAL deadline for the
# UD code/multi-turn rerun. The old 2700 s code default is retired: an env
# override above the cap (or below the cleanup reserve) fails closed HERE, at
# import, before any host change.
WORK_CAP_SECONDS = 1800
_requested = os.environ.get('E2E_WORK_SECONDS')
WORK_SECONDS = int(_requested) if _requested else WORK_CAP_SECONDS
if not 120 <= WORK_SECONDS <= WORK_CAP_SECONDS:
    raise ValueError(f'E2E_WORK_SECONDS={WORK_SECONDS} outside 120..{WORK_CAP_SECONDS}; '
                     'the 2700 s code default is retired, total run must stay <=1800 s')
REQUEST_SECONDS = WORK_SECONDS - 120

# Client-inclusive cgroup accounting (Seppe steering 2026-09-13 + the proven
# 2026-09-14 nightly pattern): the runner — and with it the hermes client,
# probes and curl — must run inside a dedicated systemd scope with a 4 GiB
# MemoryMax, so the CLIENT side of the 40 GiB budget is kernel-accounted
# rather than inferred. Sum of caps: 36G server + 4G client = 40G.
CLIENT_CAP_GIB = 4
CLIENT_UNIT = os.environ.get('E2E_CLIENT_UNIT', '')
CLIENT_UNIT_PATTERN = re.compile(r'expertpin-e2e-client-[A-Za-z0-9_-]+[.]scope')


def select_task(kind):
    """The actual prompt for the configured task kind (summary must record
    THIS, not the marker default, so metadata matches the executed run)."""
    if kind == 'code':
        return CODE_TASK
    if kind == 'marker':
        return TASK
    raise ValueError('unknown E2E task kind: ' + repr(kind))


def client_scope_point(unit=None, proc_cgroup=None, sys_base=Path('/sys/fs/cgroup')):
    """Validate the runner's own client scope and read its memory counters.

    Fail-closed: no unit configured, an unexpected unit name, a foreign
    cgroup, a MemoryMax other than the agreed 4 GiB cap, peak above cap, or
    any max/oom event all refuse BEFORE (or during) the run. Split
    parameters keep this CPU-testable without touching /proc or /sys.
    """
    unit = CLIENT_UNIT if unit is None else unit
    if not unit:
        raise RuntimeError('E2E_CLIENT_UNIT required: launch via the guarded wrapper '
                           '(dedicated 4G client scope); raw runs are refused')
    if not CLIENT_UNIT_PATTERN.fullmatch(unit):
        raise ValueError('unexpected client unit name: ' + repr(unit))
    if proc_cgroup is None:
        proc_cgroup = Path('/proc/self/cgroup').read_text()
    cgroup = next(s[3:] for s in proc_cgroup.splitlines() if s.startswith('0::'))
    if not cgroup.endswith('/' + unit):
        raise RuntimeError('runner must be inside the dedicated client scope: ' + unit)
    base = Path(sys_base) / cgroup.lstrip('/')
    point = {k: (base / k).read_text().strip() for k in
             ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current', 'memory.events')}
    cap = CLIENT_CAP_GIB * 1024**3
    if int(point['memory.max']) != cap:
        raise RuntimeError('client MemoryMax must be exactly %dG' % CLIENT_CAP_GIB)
    if not 0 <= int(point['memory.current']) <= int(point['memory.peak']) <= cap:
        raise RuntimeError('client RAM cap exceeded')
    events = dict(line.split() for line in point['memory.events'].splitlines())
    if any(int(events.get(k, 0)) for k in ('max', 'oom', 'oom_kill', 'oom_group_kill')):
        raise RuntimeError('client cap pressure/OOM event')
    point['cgroup'] = cgroup
    return point


class E2ERun(harness.Run):
    """Guarded Run + client-inclusive accounting + the qli pause guard."""

    def command(self, args, *rest, **kwargs):
        if (args[:2] == ['systemctl', '--user'] and 'qli.service' in args
                and any(a in args for a in ('start', 'restart', 'stop'))):
            raise RuntimeError('qli pause: no qli.service mutations permitted')
        return super().command(args, *rest, **kwargs)

    def client_scope_point(self):
        return client_scope_point()

    def sample(self):
        super().sample()
        assert self.cgroup_path is not None
        server = {k: (self.cgroup_path / k).read_text().strip() for k in
                  ('memory.current', 'memory.peak', 'memory.max')}
        if int(server['memory.max']) != harness.RAM_BUDGET_GIB * 1024**3:
            raise RuntimeError('server MemoryMax mismatch')
        client = self.client_scope_point()
        # Conservative upper bound for the client-inclusive footprint: the two
        # kernel peaks need not have occurred simultaneously, so their sum can
        # overstate the true concurrent peak. Never present it as exact.
        upper = int(server['memory.peak']) + int(client['memory.peak'])
        record = dict(elapsed_s=time.monotonic() - self.start, client=client, server=server,
                      sum_of_kernel_peaks_bytes=upper)
        with (self.out / 'inclusive-ram.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')
        if upper > 40 * 1024**3:
            raise RuntimeError('combined server+client RAM cap exceeded')


def turn1_checks(task_kind, stdout, after_query):
    """Task-kind specific client-success criteria (2026-09-12 fix): the code
    task must show its expected output after the query echo plus evidence the
    script was written/run; the marker task keeps the printf-marker criteria.
    Guards against the query-echo false positive in both cases."""
    if task_kind == 'code':
        return {'hermes_stdout_has_marker': 'code-gate-ok-7391' in after_query,
                'hermes_ran_shell_tool': ('e2e-codegate.py' in stdout
                                          and ('python3' in stdout or 'EOF' in stdout))}
    return {'hermes_stdout_has_marker': 'hermes-e2e-ok' in after_query,
            'hermes_ran_shell_tool': ('shell' in stdout.lower() and
                                      ('printf' in stdout or 'command' in stdout.lower()))}


def run_client(run, args, *, out, prefix, env, timeout, poll_seconds=1.0):
    """Sample the existing RAM/VRAM guards while a client or probe runs.

    File-backed output avoids pipe backpressure and preserves partial logs on
    failure. Each invocation shares the run's deadline; later turns cannot
    reset it. Kill the owned process group on every exit so ordinary shell
    tool descendants cannot outlive an aborted client. Detached sessions are
    not contained by this process-group boundary (this is not a sandbox).
    """
    if timeout <= 0 or poll_seconds <= 0:
        raise ValueError('client timeout and poll interval must be positive')
    deadline = min(time.monotonic() + timeout, run.start + run.work_budget())
    stdout_path = out / (prefix + '-stdout.txt')
    stderr_path = out / (prefix + '-stderr.txt')
    if time.monotonic() >= deadline:
        raise subprocess.TimeoutExpired(args, timeout)
    run.sample()  # refuse a failing guard before spawning any client
    with stdout_path.open('x') as stdout, stderr_path.open('x') as stderr:
        client = run.spawn(args, stdout=stdout, stderr=stderr, env=env,
                           cwd=str(out), start_new_session=True)
        try:
            run.spawned()  # a deferred signal must still pass through finally
            while True:
                run.sample()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    client.wait(timeout=min(poll_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
            run.sample()
        finally:
            try:
                os.killpg(client.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                client.wait(timeout=5)
    return subprocess.CompletedProcess(args, client.returncode,
                                       stdout_path.read_text(), stderr_path.read_text())


def run_second_turn(run, out, client_env, task_kind):
    """Resume the session and reuse the artifact produced by the first task.

    Transcript checks are heuristics, not proof of tool execution: require
    both the command and expected output after the first query-echo line.
    Both prompts place the entire command on that line; their continuation
    contains neither the command nor the expected output marker.
    The existing monitored client preserves the shared deadline and logs.
    """
    if task_kind == 'code':
        command, expected = 'python3 /tmp/e2e-codegate.py', 'code-gate-ok-7391'
        task = ('Now use the shell tool to run exactly: ' + command + '\n'
                'Reuse the existing script without rewriting it. '
                'Reply with exactly what it printed and nothing else.')
    elif task_kind == 'marker':
        command, expected = 'cat /tmp/hermes-e2e-marker.txt', 'hermes-e2e-ok'
        task = TASK_TURN2
    else:
        raise ValueError('unknown E2E task kind: ' + repr(task_kind))
    began = time.monotonic()
    run.label = 'hermes-e2e-turn2'
    client = run_client(run, ['hermes', 'chat', '--yolo', '--resume', 'latest', '-q', task],
                        out=out, prefix='hermes-turn2', timeout=REQUEST_SECONDS, env=client_env)
    after_query = client.stdout.split('\n', 1)[1] if '\n' in client.stdout else ''
    has_marker = expected in after_query
    ran_tool = command in after_query
    return {'turn2_returncode': client.returncode,
            'turn2_wall_seconds': round(time.monotonic() - began, 1),
            'turn2_task': task,
            'turn2_stdout_has_marker': has_marker,
            'turn2_ran_shell_tool': ran_tool,
            'multi_turn': True,
            'status': 'completed' if client.returncode == 0 and has_marker and ran_tool else 'client_failed'}


def main():
    # Fail closed BEFORE creating artifacts or touching any service: the
    # runner (and with it the hermes client/probes/curl) must live inside
    # the dedicated 4G client scope so client RAM is kernel-accounted.
    client_scope_point()
    task = select_task(E2E_TASK_KIND)
    out = HERE / ('run-' + time.strftime('%Y%m%dT%H%M%S') + '-e2e')
    out.mkdir()
    run = E2ERun(out, 4, E2E_CHECKPOINT)
    run.work_seconds = WORK_SECONDS
    run.request_seconds = REQUEST_SECONDS
    # The Run constructor stamps self.start at construction time; with the
    # GPU idle-wait (up to 15 min) in between, the work deadline would be
    # half-spent before the server is even ready. Re-arm the clock at the
    # moment real work begins.
    run.start = time.monotonic()
    summary = {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'kind': 'hermes-e2e-agent-run', 'task': task,
               'task_kind': E2E_TASK_KIND, 'checkpoint': E2E_CHECKPOINT,
               'startup_n_max': 4, 'work_seconds': WORK_SECONDS,
               'client_unit': CLIENT_UNIT, 'client_cap_gib': CLIENT_CAP_GIB,
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
        run.stop_miners()  # qli masked = no-op skip; Jetski stopped for GPU work
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
        (out / 'task.txt').write_text(task)
        # Fail-closed against STALE artifacts from earlier runs: if a previous
        # gate left /tmp/e2e-codegate.py (or the marker) behind, the probe
        # below could pass without this run's agent ever writing anything.
        for stale in ('/tmp/e2e-codegate.py', '/tmp/hermes-e2e-marker.txt'):
            Path(stale).unlink(missing_ok=True)
            summary['stale_removed_' + stale.rsplit('/', 1)[1]] = True
        client_env = dict(os.environ, HERMES_HOME=str(E2E_HOME),
                          EXPERTPIN_LOCAL_KEY='local-e2e-dummy')
        began = time.monotonic()
        run.label = 'hermes-e2e'
        run.sample()
        client = run_client(run, ['hermes', 'chat', '--yolo', '-q', task],
                            out=out, prefix='hermes', timeout=REQUEST_SECONDS, env=client_env)
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
        # an actual tool execution in the transcript. Criteria are task-kind
        # specific (see turn1_checks; 2026-09-12 fix).
        after_query = stdout.split('\n', 1)[1] if '\n' in stdout else ''
        summary.update(turn1_checks(E2E_TASK_KIND, stdout, after_query))
        ok_turn1 = (client.returncode == 0 and summary['hermes_stdout_has_marker']
                    and summary['hermes_ran_shell_tool'])
        if E2E_TASK_KIND == 'code' and ok_turn1:
            # Code-precision gate: verify the artifact independently of the
            # agent's self-report — the script must exist, run cleanly, and
            # print exactly the required line.
            probe = run_client(run, ['python3', '/tmp/e2e-codegate.py'],
                               out=out, prefix='codegate-probe', timeout=30, env=client_env)
            summary['codegate_script_rc'] = probe.returncode
            summary['codegate_script_output'] = probe.stdout.strip()
            summary['codegate_script_expected'] = 'code-gate-ok-7391'
            summary['codegate_pass'] = (probe.returncode == 0
                                        and probe.stdout.strip() == 'code-gate-ok-7391')
            ok_turn1 = ok_turn1 and summary['codegate_pass']
        if MULTI_TURN and ok_turn1:
            summary.update(run_second_turn(run, out, client_env, E2E_TASK_KIND))
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
            try:
                (out / 'client-final.json').write_text(
                    json.dumps(client_scope_point(), indent=2) + '\n')
            except (OSError, RuntimeError, ValueError) as error:
                summary['client_final_read_error'] = repr(error)
            run.summary['e2e'] = summary
            run.summary['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            run.save('summary.json', run.summary)
    print(out)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get('status') == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
