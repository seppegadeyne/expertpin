#!/usr/bin/env python3
"""CPU-only regression tests for the cold-cache A/B harness; no GPU/launcher/service."""
import importlib.util
import os
import tempfile
from pathlib import Path
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'evidence/coldcache-ab-20260911/run-guarded.py'


def load():
    with mock.patch('subprocess.run', side_effect=AssertionError('import subprocess')), \
         mock.patch('subprocess.Popen', side_effect=AssertionError('import spawn')), \
         mock.patch.object(Path, 'mkdir', side_effect=AssertionError('import write')):
        spec = importlib.util.spec_from_file_location('coldcache', PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class ColdCacheTests(unittest.TestCase):
    def setUp(self):
        self.harness = load()

    def test_drop_model_cache_requires_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'model.gguf'
            target.write_bytes(b'x' * 65536)
            dropped = self.harness.drop_model_cache(target)
            self.assertTrue(dropped['fadvise_completed'])
            self.assertEqual(dropped['bytes'], 65536)
            self.assertTrue(dropped['file'])
            with self.assertRaises(ValueError):
                self.harness.drop_model_cache(Path(directory) / 'missing.gguf')

    def test_drop_uses_whole_file_single_call(self):
        module = self.harness
        with mock.patch.object(module.Path, 'is_file', return_value=True), \
             mock.patch.object(module.os, 'stat') as stat, \
             mock.patch.object(module.os, 'open', return_value=9) as opener, \
             mock.patch.object(module.os, 'close') as closer, \
             mock.patch.object(module.os, 'posix_fadvise') as advise:
            stat.return_value = mock.Mock(st_size=123456)
            dropped = module.drop_model_cache(Path('/models/x.gguf'))
        opener.assert_called_once()
        closer.assert_called_once_with(9)
        advise.assert_called_once_with(9, 0, 123456, module.posix.POSIX_FADV_DONTNEED)
        self.assertEqual(dropped['bytes'], 123456)

    def test_cold_mode_wires_drop_between_prep_and_launcher(self):
        run = self.harness.Run(Path('/unused'), 4, 'ps-iq2xxs', cold_cache=True)
        self.assertTrue(run.cold_cache)
        self.assertIn('cold_cache', run.summary)
        self.assertTrue(run.summary['cold_cache'])
        default = self.harness.Run(Path('/unused'), 4, 'ps-iq2xxs')
        self.assertFalse(default.cold_cache)

    def _mock_execute_stack(self, run, drop_result):
        """Common mocks to drive execute() through the guard into the drop."""
        import types
        order = []
        def fake_exec(module):
            module.wait_gpu_idle = lambda label: (
                order.append('guard'), {'gpu_util_pct': 0,
                                        'mem_available_bytes': 40 * 1024**3})[1]
        fake_spec = mock.MagicMock(name='spec')
        fake_spec.name = 'gate-mock'
        fake_spec.loader.exec_module.side_effect = fake_exec
        def drop_effect(path):
            order.append('drop')
            if isinstance(drop_result, Exception):
                raise drop_result
            return drop_result
        def command(args, name, **kwargs):
            order.append('command:' + name)
            if name == 'preexisting-scopes':
                return ''
            if name == 'cgroup':
                return '/slice/' + run.scope
            if name in ('scope-state', 'scope-final-state'):
                return 'inactive'
            return 'ok'
        def sub_run(args, **kwargs):
            if args and args[0] == 'pgrep':
                return mock.Mock(returncode=1)
            order.append('dry-launch')
            raise AssertionError('launcher dry reached')
        return order, (
            mock.patch.object(run, 'command', side_effect=command),
            mock.patch.object(run, 'save'),
            mock.patch.object(self.harness, 'drop_model_cache', side_effect=drop_effect),
            mock.patch.object(self.harness.subprocess, 'run', side_effect=sub_run),
            mock.patch.object(self.harness.subprocess, 'Popen', side_effect=AssertionError('no spawn in test')),
            mock.patch.object(self.harness.Path, 'is_file', return_value=True),
            mock.patch.object(self.harness.socket, 'socket'),
            mock.patch.object(run, 'sample'),
            mock.patch.object(self.harness.importlib.util, 'spec_from_file_location', return_value=fake_spec),
            mock.patch.object(self.harness.importlib.util, 'module_from_spec', return_value=types.ModuleType('gate-mock')))

    def test_execute_calls_drop_between_guard_and_launch(self):
        run = self.harness.Run(Path('/unused'), 4, 'ps-iq2xxs', cold_cache=True)
        order, stack = self._mock_execute_stack(run, {'fadvise_completed': True, 'bytes': 1, 'file': 'x'})
        with stack[0], stack[1], stack[2], stack[3], stack[4], stack[5], stack[6], stack[7], stack[8], stack[9]:
            with self.assertRaises(AssertionError):
                run.execute()
        self.assertEqual(order[-2:], ['drop', 'dry-launch'])

    def test_drop_failure_never_launches_or_spawns(self):
        run = self.harness.Run(Path('/unused'), 4, 'ps-iq2xxs', cold_cache=True)
        order, stack = self._mock_execute_stack(run, OSError('injected'))
        with stack[0], stack[1], stack[2], stack[3], stack[4], stack[5], stack[6], stack[7], stack[8], stack[9]:
            with self.assertRaises(OSError):
                run.execute()
        self.assertIn('drop', order)
        self.assertNotIn('dry-launch', order)
        self.assertFalse(run.scope_launched)


class MinerPauseTests(unittest.TestCase):
    """Seppe 2026-09-11: qli masked/paused — never start/restart; Jetski is
    the day miner (night cron stops it for GPU work, restores after)."""

    def setUp(self):
        self.harness = load()

    def _run(self):
        run = self.harness.Run(Path('/unused'), 4, 'ps-iq2xxs')
        run.prepared = False
        return run

    def test_qli_masked_is_never_stopped_or_started(self):
        run = self._run()
        calls = []
        def command(args, name, **kwargs):
            calls.append((name, tuple(args)))
            if name.endswith('-enabled'):
                return 'not-found'   # masked qli on this host
            return 'ok'
        with mock.patch.object(run, 'command', side_effect=command):
            run.stop_miners()
            errors = []
            run.restore_miners(errors, lambda n, f: f())
        started = [c for c in calls if 'start' in c[1]]
        self.assertEqual(started, [])   # pause mandate: no qli start anywhere
        self.assertEqual(run.summary['qli.service_skip'],
                         'masked/not-found — not stopped (pause mandate)')
        self.assertNotIn('qli.service', run.summary['miners_stopped'])
        self.assertEqual(errors, [])

    def test_jetski_active_is_stopped_then_restored_and_verified(self):
        run = self._run()
        calls = []
        def command(args, name, **kwargs):
            calls.append((name, tuple(args)))
            if name == 'qli.service-enabled':
                return 'not-found'
            if name == 'jetski.service-enabled':
                return 'enabled'
            if name == 'jetski-service-active':
                return 'active'
            return 'ok'
        with mock.patch.object(run, 'command', side_effect=command):
            run.stop_miners()
            self.assertEqual(run.summary['miners_stopped'], ['jetski.service'])
            errors = []
            run.restore_miners(errors, lambda n, f: f())
        self.assertEqual(errors, [])
        self.assertIn(('jetski-service-stop', ('systemctl', '--user', 'stop', 'jetski.service')), calls)
        self.assertIn(('jetski-service-start', ('systemctl', '--user', 'start', 'jetski.service')), calls)

    def test_jetski_restore_failure_is_reported_not_swallowed(self):
        run = self._run()
        def command(args, name, **kwargs):
            if name == 'qli.service-enabled':
                return 'not-found'
            if name == 'jetski.service-enabled':
                return 'enabled'
            if name == 'jetski-service-active':
                return 'failed'
            return 'ok'
        with mock.patch.object(run, 'command', side_effect=command):
            run.stop_miners()
            errors = []
            run.restore_miners(errors, lambda n, f: f())
        self.assertEqual(errors, ['jetski.service not verified active'])

    def test_cleanup_masked_qli_no_longer_fails_the_run(self):
        run = self._run()
        def command(args, name, **kwargs):
            if name.endswith('-enabled'):
                return 'not-found'
            if name in ('scope-state', 'scope-final-state'):
                return 'inactive'
            if name == 'cgroup':
                return '/slice/' + run.scope
            return 'ok'
        with mock.patch.object(run, 'command', side_effect=command), \
             mock.patch.object(run, 'own_scope_empty', return_value=True):
            run.cleanup()
        self.assertEqual(run.summary.get('cleanup_errors'), [])   # pre-fix this was ['qli not verified active', ...]
        self.assertNotIn('qli_status', run.summary)


if __name__ == '__main__':
    unittest.main()
