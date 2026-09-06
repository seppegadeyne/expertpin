"""CPU-only readiness tests; no services, nvidia-smi, or GPU calls."""
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'evidence/trace-matched-bandwidth/run-physical.py'
spec = importlib.util.spec_from_file_location('physical', SOURCE)
assert spec is not None and spec.loader is not None
physical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physical)


class ReadinessTests(unittest.TestCase):
    def run_samples(self, values, timeout=6, vram=100):
        now = [0.0]
        calls = []
        def sample(name, **kwargs):
            calls.append(name)
            return dict(mem_available_bytes=values[min(len(calls)-1, len(values)-1)][1],
                        gpu_util_pct=values[min(len(calls)-1, len(values)-1)][0], vram_mib=vram)
        def sleep(seconds):
            now[0] += seconds
        with tempfile.TemporaryDirectory() as d, patch.object(physical, 'OUT', Path(d)), \
                patch.object(physical, 'guard_sample', side_effect=sample), \
                patch.object(physical.time, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(physical.time, 'sleep', side_effect=sleep):
            return physical.wait_gpu_idle('test', timeout=timeout, interval=2), calls

    def test_stale_busy_sample_recovers_without_relaxing_guard(self):
        result, calls = self.run_samples([(29, 34*1024**3), (5, 34*1024**3), (4, 34*1024**3)])
        self.assertEqual(result['gpu_util_pct'], 4)
        self.assertEqual(len(calls), 3)

    def test_persistent_busy_is_bounded(self):
        with self.assertRaisesRegex(RuntimeError, 'idle readiness deadline'):
            self.run_samples([(5, 50*1024**3)])

    def test_ram_failure_is_not_retried(self):
        with self.assertRaisesRegex(RuntimeError, 'RAM/VRAM guard'):
            self.run_samples([(29, 34*1024**3-1), (0, 50*1024**3)])

    def test_immediate_idle(self):
        result, calls = self.run_samples([(0, 34*1024**3)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['mem_available_bytes'], 34*1024**3)

    def test_observation_failure_propagates(self):
        with tempfile.TemporaryDirectory() as d, patch.object(physical, 'OUT', Path(d)), \
                patch.object(physical, 'guard_sample', side_effect=OSError('nvidia failed')):
            with self.assertRaises(OSError):
                physical.wait_gpu_idle('test')

    def test_vram_boundary_blocks(self):
        with self.assertRaisesRegex(RuntimeError, 'RAM/VRAM guard'):
            self.run_samples([(0, 50*1024**3)], vram=28*1024)

    def test_invalid_utilization_blocks(self):
        for util in (-1, 101):
            with self.assertRaisesRegex(RuntimeError, 'invalid GPU utilization'):
                self.run_samples([(util, 50*1024**3)])

    def test_deadline_includes_observation_time(self):
        with tempfile.TemporaryDirectory() as d, patch.object(physical, 'OUT', Path(d)), \
                patch.object(physical.time, 'monotonic', side_effect=[0, 0, 60]), \
                patch.object(physical, 'guard_sample', return_value={'gpu_util_pct': 0}):
            with self.assertRaisesRegex(RuntimeError, 'idle readiness deadline'):
                physical.wait_gpu_idle('test')

    def test_invalid_bounds(self):
        for timeout, interval in ((0, 2), (91, 2), (60, 0), (1, 2)):
            with self.assertRaises(ValueError):
                physical.wait_gpu_idle('test', timeout=timeout, interval=interval)


class LifecycleTests(unittest.TestCase):
    def test_reused_destination_is_unchanged_and_no_services(self):
        for previous in ('gpu.json', 'execution.json', 'before-probes-readiness.json'):
            with tempfile.TemporaryDirectory(dir=SOURCE.parents[1]) as d:
                out = Path(d)
                (out / previous).write_text('previous evidence')
                with patch.object(physical, 'OUT', out), patch.object(physical, 'cmd') as cmd:
                    with self.assertRaisesRegex(RuntimeError, 'must be empty'):
                        physical.outer()
                    cmd.assert_not_called()
                    self.assertEqual(list(out.iterdir()), [out / previous])
                    self.assertEqual((out / previous).read_text(), 'previous evidence')

    def test_invalid_destination_no_writes_or_services(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parents[2] / 'tests') as d:
            out = Path(d)
            with patch.object(physical, 'OUT', out), patch.object(physical, 'cmd') as cmd:
                with self.assertRaisesRegex(RuntimeError, 'existing evidence directory'):
                    physical.outer()
                cmd.assert_not_called()
                self.assertEqual(list(out.iterdir()), [])

    def test_observation_errors_cannot_skip_termination(self):
        for failed in ('cleanup-cgroup', 'scope-state'):
            for error in (OSError('log failure'), subprocess.TimeoutExpired('mock', 10)):
                calls = []
                def cmd(args, name, **kwargs):
                    calls.append(name)
                    if name == failed:
                        raise error
                    return '' if name == 'cleanup-cgroup' else 'inactive'
                with patch.object(physical, 'cmd', side_effect=cmd):
                    with self.assertRaisesRegex(RuntimeError, 'termination unverified'):
                        physical.stop_scope('owned.scope')
                self.assertEqual(calls.count('scope-stop'), 2)
                self.assertEqual(calls.count('scope-kill'), 1)

    def test_kill_log_error_cannot_skip_second_stop(self):
        calls = []
        def cmd(args, name, **kwargs):
            calls.append(name)
            if name == 'scope-kill':
                raise OSError('log failure')
            if name == 'scope-state':
                return 'active' if calls.count(name) == 1 else 'inactive'
            return ''
        with patch.object(physical, 'cmd', side_effect=cmd):
            self.assertEqual(physical.stop_scope('owned.scope'), 'inactive')
        self.assertEqual(calls.count('scope-stop'), 2)


if __name__ == '__main__':
    unittest.main()
