#!/usr/bin/env python3
"""CPU-only regression tests: the e2e runner's work-budget wiring.

2026-09-12 incident (run-20260912T031812-e2e): the runner sets
run.work_seconds, but harness.sample() consulted the MODULE constant
WORK_SECONDS (=660) — the deadline fired at ~660 s after the re-armed
start while the agent was mid-run (the agent itself completed fine and
the artifact probe passed post-hoc). Third timing trap in this lineage
after the two 2026-09-11 ones (construction-time start, GPU idle-wait).
"""
import importlib.util
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]


def load_harness():
    spec = importlib.util.spec_from_file_location(
        'budget_harness', HERE / 'evidence/coldcache-ab-20260911/run-guarded.py')
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    with mock.patch('subprocess.run', side_effect=AssertionError('no subprocess')), \
         mock.patch('subprocess.Popen', side_effect=AssertionError('no spawn')), \
         mock.patch.object(Path, 'mkdir', side_effect=AssertionError('no write')):
        spec.loader.exec_module(module)
    return module


def load_runner():
    spec = importlib.util.spec_from_file_location(
        'e2erun', HERE / 'evidence/hermes-e2e-20260911/run-e2e.py')
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules['e2erun'] = module          # so __name__ != '__main__'
    with mock.patch.dict(os.environ, {'E2E_TASK': 'code'}, clear=False):
        spec.loader.exec_module(module)
    return module


class WorkBudgetTests(unittest.TestCase):
    def test_sample_deadline_uses_runner_budget_not_module_constant(self):
        h = load_harness()
        run = h.Run(Path('/unused'), 4, 'ps-iq2xxs')
        # Exactly what the runner wires after construction (7d57edcc re-arm):
        run.work_seconds = 2700
        run.start = time.monotonic()
        # Simulate the incident condition: module constant already exceeded
        # (660 s after start) but the runner budget NOT.
        run.start -= h.WORK_SECONDS + 30          # elapsed = 690 s > 660 module
        self.assertLess(run.start + h.WORK_SECONDS, time.monotonic())  # module deadline passed
        self.assertGreater(run.start + run.work_seconds, time.monotonic())  # runner deadline not
        # sample() must NOT raise TimeoutError in that window (pre-fix it did).
        def command(args, name, **kwargs):
            if name == 'gpu-latest':
                return '0, 0'
            if name == 'cgroup':
                return '/slice/' + run.scope
            return 'ok'
        meminfo = 'MemTotal:    62000000 kB\nMemFree:      10000000 kB\nMemAvailable: 50000000 kB\n'
        cgroup_values = {
            'memory.current': str(10 * 1024**3),
            'memory.peak': str(10 * 1024**3),
            'memory.max': str(h.RAM_BUDGET_GIB * 1024**3),
            'memory.swap.current': '0',
            'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\nsock_throttled 0\n',
        }
        def fake_read(self, *args, **kwargs):
            name = self.name
            if name in cgroup_values:
                return cgroup_values[name]
            return meminfo
        run.proc = mock.Mock(poll=mock.Mock(return_value=None))
        with mock.patch.object(run, 'command', side_effect=command), \
             mock.patch.object(h.Path, 'read_text', fake_read), \
             mock.patch.object(Path, 'open', mock.mock_open()):
            run.sample()   # pre-fix: TimeoutError('work deadline; cleanup reserve begins')

    def test_sample_still_raises_past_runner_budget(self):
        h = load_harness()
        run = h.Run(Path('/unused'), 4, 'ps-iq2xxs')
        run.work_seconds = 2700
        run.start = time.monotonic() - 2750       # past even the runner budget
        with self.assertRaises(TimeoutError):
            run.sample()

    def test_runner_wires_budget_into_health_deadline_too(self):
        # run-e2e.py plans min(run.start + run.work_seconds, +900) for health;
        # the harness-side health loop must honor the same override so a long
        # cold model load (e.g. 75-100 s UD shard load + slow cold prefill)
        # cannot be cut by the module constant.
        h = load_harness()
        module = load_runner()
        self.assertGreater(module.WORK_SECONDS, h.WORK_SECONDS)  # runner budget exceeds module


if __name__ == '__main__':
    unittest.main()
