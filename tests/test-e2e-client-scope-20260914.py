#!/usr/bin/env python3
"""CPU-only regression tests for the 2026-09-14 E2E runner hardening.

Seppe steering 2026-09-13 08:45 + night-run 2026-09-14 follow-up:
 1. the UD code/multi-turn rerun must enforce an explicit <=1800 s total
    deadline — the old 2700 s code-task default is retired, env overrides
    above the cap fail closed at import;
 2. the runner must execute inside a dedicated client cgroup scope (4G) so
    harness + hermes client + curl RAM is kernel-accounted and the combined
    server+client kernel peaks are checked against the 40 GiB budget during
    every sample (adopted from .hermes-work/daily-20260914/run-ud.py);
 3. the summary metadata must reflect the ACTUAL run: E2E_CHECKPOINT (not a
    hardcoded ps-iq2xxs), the selected task text/kind, and the work budget.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'evidence/hermes-e2e-20260911/run-e2e.py'


def load_runner(env=None, name='e2e_scope_test'):
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    try:
        with mock.patch.dict(os.environ, env or {}, clear=False):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class WorkCapTests(unittest.TestCase):
    def test_default_is_1800_for_both_task_kinds(self):
        for kind in ('marker', 'code'):
            with mock.patch.dict(os.environ, {'E2E_WORK_SECONDS': ''}, clear=False):
                os.environ.pop('E2E_WORK_SECONDS', None)
                module = load_runner({'E2E_TASK': kind}, name='e2e_cap_' + kind)
            self.assertEqual(module.WORK_SECONDS, 1800)
            self.assertEqual(module.REQUEST_SECONDS, 1680)

    def test_env_above_cap_fails_closed(self):
        # The retired code default must not be resurrectable via env.
        with self.assertRaises(ValueError):
            load_runner({'E2E_WORK_SECONDS': '2700'}, name='e2e_cap_over')
        with self.assertRaises(ValueError):
            load_runner({'E2E_WORK_SECONDS': '1801'}, name='e2e_cap_over2')

    def test_env_within_cap_accepted(self):
        module = load_runner({'E2E_WORK_SECONDS': '900'}, name='e2e_cap_ok')
        self.assertEqual(module.WORK_SECONDS, 900)
        self.assertEqual(module.REQUEST_SECONDS, 780)

    def test_env_below_reserve_rejected(self):
        with self.assertRaises(ValueError):
            load_runner({'E2E_WORK_SECONDS': '60'}, name='e2e_cap_low')


class MetadataTests(unittest.TestCase):
    def test_unknown_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            load_runner({'E2E_CHECKPOINT': 'bogus-quant'}, name='e2e_meta_ckpt')

    def test_unknown_task_kind_rejected(self):
        with self.assertRaises(ValueError):
            load_runner({'E2E_TASK': 'bogus'}, name='e2e_meta_task')

    def test_select_task_returns_the_actual_prompt(self):
        module = load_runner(name='e2e_meta_select')
        self.assertIs(module.select_task('code'), module.CODE_TASK)
        self.assertIs(module.select_task('marker'), module.TASK)
        with self.assertRaises(ValueError):
            module.select_task('bogus')

    def test_ncmoe_default_follows_dev_checkpoint_default(self):
        # Dev default is UD-Q4_K_XL with NCMOE 40; ps/reference ran NCMOE 36.
        self.assertEqual(load_runner({'E2E_CHECKPOINT': 'ud-q4kxl'},
                                     name='e2e_meta_ud').E2E_NCMOE, '40')
        self.assertEqual(load_runner({'E2E_CHECKPOINT': 'ps-iq2xxs'},
                                     name='e2e_meta_ps').E2E_NCMOE, '36')
        self.assertEqual(load_runner({'E2E_CHECKPOINT': 'ud-q4kxl', 'E2E_NCMOE': '44'},
                                     name='e2e_meta_ovr').E2E_NCMOE, '44')


class ClientScopeTests(unittest.TestCase):
    PROC = '0::/user.slice/user-1000.slice/expertpin-e2e-client-t.scope\n'
    UNIT = 'expertpin-e2e-client-t.scope'

    def setUp(self):
        self.runner = load_runner(name='e2e_scope_mod')
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.hermes-work')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / 'sys/fs/cgroup/user.slice/user-1000.slice' / self.UNIT
        self.base.mkdir(parents=True)
        self.write_cgroup(max_gib=4)

    def write_cgroup(self, max_gib=4, current='0', peak='0', events='max 0\noom 0\noom_kill 0\noom_group_kill 0\n'):
        (self.base / 'memory.max').write_text(str(max_gib * 1024**3) + '\n')
        (self.base / 'memory.current').write_text(current + '\n')
        (self.base / 'memory.peak').write_text(peak + '\n')
        (self.base / 'memory.swap.current').write_text('0\n')
        (self.base / 'memory.events').write_text(events)
        return self.base.parents[2]  # the fake /sys/fs/cgroup root

    def point(self, **kwargs):
        return self.runner.client_scope_point(unit=self.UNIT, proc_cgroup=self.PROC,
                                              sys_base=self.write_cgroup(**kwargs))

    def test_requires_unit(self):
        with self.assertRaises(RuntimeError):
            self.runner.client_scope_point(unit='', proc_cgroup=self.PROC,
                                           sys_base=Path('/unused'))

    def test_rejects_unexpected_unit_name(self):
        with self.assertRaises(ValueError):
            self.runner.client_scope_point(unit='expertpin-test-abc.scope',
                                           proc_cgroup=self.PROC, sys_base=Path('/unused'))

    def test_rejects_foreign_cgroup(self):
        with self.assertRaises(RuntimeError):
            self.runner.client_scope_point(unit=self.UNIT,
                                           proc_cgroup='0::/user.slice/other.scope\n',
                                           sys_base=Path('/unused'))

    def test_valid_scope_returns_point(self):
        point = self.point(current=str(1024**3), peak=str(2 * 1024**3))
        self.assertEqual(point['memory.max'], str(4 * 1024**3))
        self.assertTrue(point['cgroup'].endswith('/' + self.UNIT))

    def test_rejects_wrong_cap(self):
        with self.assertRaises(RuntimeError):
            self.point(max_gib=5)

    def test_rejects_events(self):
        with self.assertRaises(RuntimeError):
            self.point(events='max 1\noom 0\noom_kill 0\noom_group_kill 0\n')

    def test_rejects_peak_above_cap(self):
        with self.assertRaises(RuntimeError):
            self.point(current=str(5 * 1024**3), peak=str(5 * 1024**3))


class InclusiveSampleTests(unittest.TestCase):
    """E2ERun.sample: combined server+client kernel peaks vs the 40 GiB budget."""

    def make_run(self, client_peak_gib, server_peak_gib=32):
        module = load_runner(name='e2e_sample_mod')
        run = module.E2ERun(Path('/unused'), 4, 'ps-iq2xxs')
        run.work_seconds = 1800
        run.start = __import__('time').monotonic()
        h = module.harness
        cgroup_values = {
            'memory.current': str(server_peak_gib * 1024**3),
            'memory.peak': str(server_peak_gib * 1024**3),
            'memory.max': str(h.RAM_BUDGET_GIB * 1024**3),
            'memory.swap.current': '0',
            'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n',
        }
        meminfo = 'MemTotal: 62000000 kB\nMemAvailable: 50000000 kB\n'

        def command(args, name, **kwargs):
            if name == 'gpu-latest':
                return '0, 0'
            if name == 'cgroup':
                return '/slice/' + run.scope
            return 'ok'

        def fake_read(self, *args, **kwargs):
            return cgroup_values.get(self.name, meminfo)

        client = {'memory.current': '0', 'memory.peak': str(client_peak_gib * 1024**3),
                  'memory.max': str(4 * 1024**3), 'memory.swap.current': '0',
                  'memory.events': 'max 0\noom 0\noom_kill 0\noom_group_kill 0\n',
                  'cgroup': '/slice/expertpin-e2e-client-t.scope'}
        run.proc = mock.Mock(poll=mock.Mock(return_value=None))
        patches = [mock.patch.object(run, 'command', side_effect=command),
                   mock.patch.object(h.Path, 'read_text', fake_read),
                   mock.patch.object(Path, 'open', mock.mock_open()),
                   mock.patch.object(run, 'client_scope_point', return_value=client)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return run

    def test_combined_within_budget_passes_and_records(self):
        run = self.make_run(client_peak_gib=1, server_peak_gib=32)
        run.sample()  # 33 GiB combined: within 40
        handle = Path.open  # mock_open recorded the append
        written = ''.join(c.args[0] for c in handle.return_value.write.call_args_list)
        self.assertIn('sum_of_kernel_peaks_bytes', written)

    def test_combined_above_budget_fails(self):
        run = self.make_run(client_peak_gib=4, server_peak_gib=37)
        with self.assertRaises(RuntimeError):
            run.sample()


class QliGuardTests(unittest.TestCase):
    def test_qli_mutations_blocked_before_subprocess(self):
        module = load_runner(name='e2e_qli_mod')
        run = module.E2ERun(Path('/unused'), 4, 'ps-iq2xxs')
        with mock.patch('subprocess.run', side_effect=AssertionError('subprocess ran')):
            for action in ('start', 'restart', 'stop'):
                with self.assertRaises(RuntimeError):
                    run.command(['systemctl', '--user', action, 'qli.service'], 'qli')


if __name__ == '__main__':
    unittest.main()
