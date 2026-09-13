#!/usr/bin/env python3
"""CPU-only client lifecycle tests; real subprocesses, no model or services."""
import importlib.util
import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'e2e_monitor', ROOT / 'evidence/hermes-e2e-20260911/run-e2e.py')
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ClientMonitorTests(unittest.TestCase):
    def setUp(self):
        (ROOT / '.hermes-work').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.hermes-work')
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.lifecycle = runner.harness.Run(self.out, 4, 'ps-iq2xxs')
        self.lifecycle.work_seconds = 10
        self.lifecycle.sample = mock.Mock()

    def invoke(self, source, timeout=3.0):
        script = self.out / 'client.py'
        script.write_text(source)
        return runner.run_client(self.lifecycle, [sys.executable, str(script)],
                                 out=self.out, prefix='client', env=dict(os.environ),
                                 timeout=timeout, poll_seconds=0.03)

    def test_samples_during_client_and_preserves_large_output_and_exit(self):
        result = self.invoke('import sys, time\nprint("x" * 200000, flush=True)\n'
                             'print("diagnostic", file=sys.stderr)\ntime.sleep(.15)\nsys.exit(7)\n')
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, 'x' * 200000 + '\n')
        self.assertEqual(result.stderr, 'diagnostic\n')
        self.assertGreaterEqual(self.lifecycle.sample.call_count, 3)
        self.assertEqual((self.out / 'client-stdout.txt').read_text(), result.stdout)

    def test_timeout_preserves_partial_transcript(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke('import time\nprint("started", flush=True)\ntime.sleep(10)\n', timeout=.2)
        self.assertEqual((self.out / 'client-stdout.txt').read_text(), 'started\n')

    def test_shared_deadline_prevents_spawn(self):
        self.lifecycle.start = time.monotonic() - 20
        with mock.patch.object(self.lifecycle, 'spawn', side_effect=AssertionError('spawned')):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.invoke('raise AssertionError("should not run")')

    def test_guard_failure_kills_client_and_ordinary_descendant(self):
        child_pid = self.out / 'child.pid'
        parent_pid = self.out / 'parent.pid'
        source = ('import os, pathlib, subprocess, sys, time\n'
                  f'pathlib.Path({str(parent_pid)!r}).write_text(str(os.getpid()))\n'
                  'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])\n'
                  f'pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n'
                  'time.sleep(10)\n')
        def sample():
            if child_pid.exists():
                raise RuntimeError('simulated VRAM breach')
        self.lifecycle.sample.side_effect = sample
        with self.assertRaisesRegex(RuntimeError, 'simulated VRAM breach'):
            self.invoke(source)
        for path in (parent_pid, child_pid):
            pid = int(path.read_text())
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                stat = Path(f'/proc/{pid}/stat')
                try:
                    state = stat.read_text().rsplit(')', 1)[1].split()[0]
                except FileNotFoundError:
                    break
                if state == 'Z':  # dead descendant awaiting init's reap
                    break
                time.sleep(.01)
            else:
                self.fail(f'client process {pid} survived guard failure')

    def test_pending_signal_after_spawn_still_reaps_client(self):
        self.lifecycle.pending_signal = 15
        processes = []
        spawn_original = self.lifecycle.spawn
        def spawn(*args, **kwargs):
            process = spawn_original(*args, **kwargs)
            processes.append(process)
            return process
        with mock.patch.object(self.lifecycle, 'spawn', side_effect=spawn):
            with self.assertRaisesRegex(TimeoutError, 'signal 15'):
                self.invoke('import time\ntime.sleep(10)\n')
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)
        self.assertFalse(self.lifecycle.launching)

    def test_failing_preflight_never_spawns(self):
        self.lifecycle.sample.side_effect = RuntimeError('NVML unavailable')
        with mock.patch.object(self.lifecycle, 'spawn', side_effect=AssertionError('spawned')):
            with self.assertRaisesRegex(RuntimeError, 'NVML unavailable'):
                self.invoke('raise AssertionError("should not run")')

    def test_running_client_uses_remaining_shared_budget(self):
        self.lifecycle.work_seconds = .2
        began = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke('import time\ntime.sleep(10)\n', timeout=3)
        self.assertLess(time.monotonic() - began, 1.5)

    def test_both_turns_and_artifact_probe_are_wired_to_monitor(self):
        tree = ast.parse(inspect.getsource(runner.main))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'run_client']
        prefixes = {kw.value.value for call in calls for kw in call.keywords
                    if kw.arg == 'prefix' and isinstance(kw.value, ast.Constant)}
        self.assertEqual(prefixes, {'hermes', 'hermes-turn2', 'codegate-probe'})


if __name__ == '__main__':
    unittest.main()
