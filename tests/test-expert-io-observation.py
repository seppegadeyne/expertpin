#!/usr/bin/env python3
"""CPU-only I/O observation tests. Counter fixtures are synthetic, not benchmarks."""
import importlib.util
try:
    import ctypes
except ImportError:
    ctypes = None
import json
import os
from pathlib import Path
import subprocess
import sys
import stat
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bw', ROOT / 'scripts/expert_bw_calib.py')
assert spec is not None and spec.loader is not None
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)


class ObservationTests(unittest.TestCase):
    def test_counter_parser_requires_exact_nonnegative_field(self):
        self.assertEqual(bw.parse_read_bytes('rchar: 456\nread_bytes: 123\n'), 123)
        for text in ('rchar: 123', 'read_bytes: -1', 'read_bytes: 1.2',
                     'read_bytes: 1\nread_bytes: 2', 'read_bytes: +1'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                bw.parse_read_bytes(text)

    def test_fsuid_mismatch_must_not_look_resident(self):
        with tempfile.TemporaryFile(dir=ROOT) as f:
            f.write(b'x' * 4096)
            f.flush()
            uid = os.geteuid()
            status = f'Uid:\t{uid}\t{uid}\t{uid}\t{uid + 1}\n'
            with patch.object(Path, 'read_text', return_value=status):
                self.assertIsNone(bw.cache_snapshot(f.fileno(), 4096)['resident_pages'])

    def test_missing_ctypes_is_unknown(self):
        import builtins
        real_import = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == 'ctypes':
                raise ImportError('no ctypes')
            return real_import(name, *args, **kwargs)
        with patch.object(builtins, '__import__', side_effect=missing):
            self.assertIsNone(bw.cache_snapshot(-1, 4096)['resident_pages'])

    def test_invalid_fsuid_source_fails_closed(self):
        for text in ('', 'Uid: 1 2 3', 'Uid: 1 2 3 -1', 'Uid: 1 2 3 4\nUid: 1 2 3 4'):
            with patch.object(Path, 'read_text', return_value=text), self.assertRaises(ValueError):
                bw.filesystem_uid()
        with patch.object(Path, 'read_text', side_effect=OSError('no proc')):
            with tempfile.TemporaryFile(dir=ROOT) as f:
                f.write(b'x')
                f.flush()
                self.assertIsNone(bw.cache_snapshot(f.fileno(), 1)['resident_pages'])

    def test_counter_timing_order_and_rate(self):
        events = []
        def called(name, value):
            events.append(name)
            return value
        obs = {}
        with patch.object(bw, 'drop_cache', side_effect=lambda *a: called('drop', None)), \
             patch.object(bw, 'cache_snapshot', side_effect=lambda *a: called('cache', dict(pages=1, resident_pages=0))), \
             patch.object(bw, 'storage_snapshot') as storage, \
             patch.object(bw.time, 'perf_counter') as clock, \
             patch.object(os, 'preadv', side_effect=lambda *a: called('read', 4)):
            counters = iter([7, 7 + bw.GIB])
            ticks = iter([100.0, 102.0])
            storage.side_effect = lambda: called('storage', dict(read_bytes=next(counters), error=None))
            clock.side_effect = lambda: called('clock', next(ticks))
            self.assertEqual(bw.measure(-1, 8, 4, True, observation=obs), (8, 2.0))
        self.assertEqual(events, ['drop', 'cache', 'storage', 'clock', 'read', 'read', 'clock', 'storage', 'cache'])
        self.assertEqual(obs['storage_accounted_gib_per_s'], 0.5)

    def test_batched_libc_contract_and_failures(self):
        if ctypes is None or not sys.platform.startswith('linux') or ctypes.sizeof(ctypes.c_void_p) != 8:
            self.skipTest('requires 64-bit Linux and ctypes')
        page = os.sysconf('SC_PAGESIZE')
        length = 65536 * page + 123
        info = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=os.geteuid(), st_size=8 * bw.GIB)
        for failure in (None, 'mmap', 'late_mincore', 'munmap', 'cap'):
            fake = Mock()
            fake.mmap.return_value = ctypes.c_void_p(-1).value if failure == 'mmap' else 0x100000
            fake.munmap.return_value = -1 if failure == 'munmap' else 0
            calls = []
            def mincore(addr, size, vec):
                calls.append((addr, size, len(vec)))
                if failure == 'late_mincore' and len(calls) == 2:
                    return -1
                vec[:] = [2 + (index % 2) for index in range(len(vec))]
                if len(vec) == 1:
                    vec[0] = 3
                return 0
            fake.mincore.side_effect = mincore
            with patch.object(ctypes, 'CDLL', return_value=fake), \
                 patch.object(os, 'fstat', return_value=info), \
                 patch.object(bw, 'filesystem_uid', return_value=info.st_uid):
                sample = bw.cache_snapshot(-1, 4 * bw.GIB + 1 if failure == 'cap' else length)
            with self.subTest(failure=failure):
                if failure is None:
                    self.assertEqual(sample['pages'], 65537)
                    self.assertEqual(sample['resident_pages'], 32769)
                    self.assertEqual(calls, [(0x100000, 65536 * page, 65536), (0x100000 + 65536 * page, 123, 1)])
                else:
                    self.assertIsNone(sample['resident_pages'])
                    self.assertIsNotNone(sample['error'])
                self.assertEqual(fake.munmap.call_count, 0 if failure in ('mmap', 'cap') else 1)
                if failure == 'cap':
                    fake.mmap.assert_not_called()

    def test_counter_failures_are_unknown_not_zero(self):
        with patch.object(Path, 'read_text', side_effect=OSError('denied')):
            self.assertIsNone(bw.storage_snapshot()['read_bytes'])
        with patch.object(Path, 'read_text', return_value='rchar: 1'):
            self.assertIsNone(bw.storage_snapshot()['read_bytes'])

    def test_cache_classification_is_snapshot_only(self):
        self.assertEqual(bw.cache_snapshot_state(dict(pages=3, resident_pages=0)), 'all_nonresident')
        self.assertEqual(bw.cache_snapshot_state(dict(pages=3, resident_pages=3)), 'all_resident')
        self.assertEqual(bw.cache_snapshot_state(dict(pages=3, resident_pages=1)), 'mixed')
        self.assertEqual(bw.cache_snapshot_state(dict(pages=None, resident_pages=None)), 'unavailable')

    def test_accounting_cannot_claim_hardware_rates(self):
        with tempfile.TemporaryFile(dir=ROOT) as f:
            f.write(b'x' * 7000)
            f.flush()
            obs = {}
            with patch.object(bw, 'storage_snapshot', side_effect=[
                    dict(read_bytes=4096, error=None), dict(read_bytes=12288, error=None)]):
                total, seconds = bw.measure(f.fileno(), 7000, 123, False, 'random', 17, obs)
            self.assertEqual(total, 7000)
            self.assertGreater(seconds, 0)
            self.assertEqual(obs['storage_read_bytes_delta'], 8192)
            self.assertIsNone(obs['nvme_bytes_per_s'])
            self.assertIsNone(obs['ram_bytes_per_s'])
            if obs['cache_after']['error'] is None:
                self.assertLessEqual(obs['cache_after']['resident_pages'], obs['cache_after']['pages'])
            self.assertEqual(obs['cache_before']['method'], 'mincore PROT_NONE MAP_SHARED')

    def test_unavailable_or_regressing_counter_blocks_delta(self):
        with tempfile.TemporaryFile(dir=ROOT) as f:
            f.write(b'x' * 4096)
            f.flush()
            for before, after in [(None, 4), (4, None), (5, 4)]:
                obs = {}
                with patch.object(bw, 'storage_snapshot', side_effect=[
                        dict(read_bytes=before, error=None), dict(read_bytes=after, error=None)]):
                    bw.measure(f.fileno(), 4096, 4096, False, observation=obs)
                self.assertIsNone(obs['storage_read_bytes_delta'])
                self.assertIsNone(obs['storage_accounted_gib_per_s'])

    def test_real_residency_unaligned_tail_and_invalid_fd(self):
        with tempfile.TemporaryFile(dir=ROOT) as f:
            f.write(b'x' * (os.sysconf('SC_PAGESIZE') + 123))
            f.flush()
            sample = bw.cache_snapshot(f.fileno(), f.tell())
            invalid = bw.cache_snapshot(-1, f.tell())
            self.assertIsNone(invalid['resident_pages'])
            self.assertIsNotNone(invalid['error'])
            self.assertIsNone(bw.cache_snapshot(f.fileno(), 0)['resident_pages'])
            self.assertIsNone(bw.cache_snapshot(f.fileno(), f.tell() + 1)['resident_pages'])
            if sample['error'] is not None:
                self.skipTest('residency unavailable: ' + sample['error'])
            self.assertEqual(sample['pages'], 2)
            self.assertGreaterEqual(sample['resident_pages'], 0)
            self.assertLessEqual(sample['resident_pages'], 2)

    def test_legacy_measure_does_not_observe(self):
        with tempfile.TemporaryFile(dir=ROOT) as f:
            f.write(b'x' * 100)
            f.flush()
            with patch.object(bw, 'cache_snapshot', side_effect=AssertionError('unexpected')):
                self.assertEqual(bw.measure(f.fileno(), 100, 100, False)[0], 100)

    def test_unowned_and_excessive_prefix_not_observed(self):
        with tempfile.TemporaryFile(dir=ROOT) as f:
            f.write(b'x' * 4096)
            f.flush()
            with patch.object(os, 'geteuid', return_value=os.geteuid() + 1):
                self.assertIsNone(bw.cache_snapshot(f.fileno(), 4096)['resident_pages'])
            self.assertIsNone(bw.cache_snapshot(f.fileno(), 4 * bw.GIB + 1)['resident_pages'])

    def test_cli_opt_in_and_json_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            source = Path(tmp) / 'sample.bin'
            output = Path(tmp) / 'result.json'
            source.write_bytes(b'x' * 7000)
            for flags in ([], ['--observe']):
                run = subprocess.run([sys.executable, str(ROOT / 'scripts/expert_bw_calib.py'),
                                      '--file', str(source), '--length', '1', '--chunk-bytes', '123',
                                      '--pattern', 'random', '--seed', '17', '--json', str(output)] + flags,
                                     capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr)
                result = json.loads(run.stdout)
                self.assertEqual(result, json.loads(output.read_text()))
                self.assertEqual(result['bytes'], 7000)
                self.assertEqual('observation' in result, bool(flags))
                if flags:
                    obs = result['observation']
                    self.assertIn(obs['cache_before_state'], ('all_resident', 'all_nonresident', 'mixed', 'unavailable'))
                    self.assertIsNone(obs['nvme_bytes_per_s'])
                    self.assertIsNone(obs['ram_bytes_per_s'])
                    if obs['storage_read_bytes_delta'] is not None:
                        self.assertGreaterEqual(obs['storage_read_bytes_delta'], 0)


if __name__ == '__main__':
    unittest.main()
