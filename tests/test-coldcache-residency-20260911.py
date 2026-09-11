#!/usr/bin/env python3
"""CPU-only regression tests for the verified-cold mincore residency check."""
import importlib.util
import os
import tempfile
from pathlib import Path
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'evidence/coldcache-ab-20260911/run-guarded.py'
with mock.patch('subprocess.run', side_effect=AssertionError('import subprocess')), \
     mock.patch('subprocess.Popen', side_effect=AssertionError('import spawn')), \
     mock.patch.object(Path, 'mkdir', side_effect=AssertionError('import write')):
    spec = importlib.util.spec_from_file_location('coldcache_vc', PATH)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)


class ResidencyTests(unittest.TestCase):
    def test_residency_counts_real_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'blob.bin'
            page = os.sysconf('SC_PAGESIZE')
            target.write_bytes(b'\0' * (page * 64))
            result = harness.residency_snapshot(target)
            self.assertIsNone(result['error'], result)
            self.assertEqual(result['pages'], 64)
            self.assertIsInstance(result['resident_pages'], int)
            self.assertTrue(0 <= result['resident_pages'] <= 64)

    def test_residency_detects_drop_effect_on_small_file(self):
        # Touch the file into the cache, snapshot, drop, snapshot again: the
        # resident count must not increase after DONTNEED (kernel may keep some
        # pages, but never gains) and the snapshot must be structurally valid.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'blob.bin'
            target.write_bytes(os.urandom(1 << 20))
            Path(target).read_bytes()  # fault the whole file in
            before = harness.residency_snapshot(target)
            dropped = harness.drop_model_cache(target)
            self.assertTrue(dropped['fadvise_completed'])
            after = harness.residency_snapshot(target)
            self.assertIsNone(before['error'])
            self.assertIsNone(after['error'])
            self.assertLessEqual(after['resident_pages'], before['resident_pages'])

    def test_residency_rejects_missing_file(self):
        with self.assertRaises(ValueError):
            harness.residency_snapshot(Path('/nonexistent/model.gguf'))

    def test_verified_cold_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'model.gguf'
            target.write_bytes(os.urandom(1 << 20))
            Path(target).read_bytes()
            harness.drop_model_cache(target)
            report = harness.verified_cold_report(target, max_resident_ratio=0.10)
            self.assertIn(report['verdict'], ('COLD', 'NOT_COLD'))
            self.assertLessEqual(report['resident_ratio'], 1.0)
            self.assertFalse(report['verified_cold'] if report['verdict'] == 'NOT_COLD' else True)

    def test_verified_cold_verdicts(self):
        self.assertEqual(harness.classify_residency(0, 100, 0.10)['verdict'], 'COLD')
        self.assertEqual(harness.classify_residency(5, 100, 0.10)['verdict'], 'COLD')
        self.assertEqual(harness.classify_residency(50, 100, 0.10)['verdict'], 'NOT_COLD')
        self.assertEqual(harness.classify_residency(100, 0, 0.10)['verdict'], 'EMPTY')


if __name__ == '__main__':
    unittest.main()
