#!/usr/bin/env python3
"""CPU-only whole-checkpoint cold-cache tests; no model load or services."""
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cold_shards', ROOT / 'evidence/coldcache-ab-20260911/run-guarded.py')
assert spec is not None and spec.loader is not None
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class ShardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.paths = [Path(self.temp.name) / f'model-{i:05d}-of-00004.gguf' for i in range(1, 5)]
        for path in self.paths:
            path.write_bytes(b'x' * 4096)

    def snapshot(self, path):
        return dict(file=str(path), pages=1, resident_pages=int(Path(path) == self.paths[-1]),
                    page_size=4096, error=None, method='mock mincore')

    def test_warm_last_shard_prevents_false_cold(self):
        with mock.patch.object(harness, 'drop_model_cache') as drop, \
             mock.patch.object(harness, 'residency_snapshot', side_effect=self.snapshot):
            drop.side_effect = lambda p: dict(file=str(p), bytes=4096, fadvise_completed=True)
            result = harness.verified_cold_report(self.paths[0])
        self.assertFalse(result['verified_cold'])
        self.assertEqual(result['snapshot']['resident_pages'], 1)
        self.assertEqual(result['snapshot']['pages'], 4)
        self.assertEqual(result['drop']['bytes'], 16384)
        self.assertEqual(drop.call_count, 4)

    def test_validate_all_shards_before_any_drop(self):
        self.paths[2].unlink()
        with mock.patch.object(harness, 'drop_model_cache') as drop:
            with self.assertRaises(ValueError):
                harness.verified_cold_report(self.paths[0])
            drop.assert_not_called()

    def test_missing_empty_or_directory_shard_rejected(self):
        self.paths[3].write_bytes(b'')
        with self.assertRaises(ValueError):
            harness.model_files(self.paths[0])
        self.paths[3].unlink()
        self.paths[3].mkdir()
        with self.assertRaises(ValueError):
            harness.model_files(self.paths[0])

    def test_non_first_or_invalid_split_rejected(self):
        for name in (self.paths[1], Path(self.temp.name) / 'model-00001-of-00000.gguf'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                harness.model_files(name)

    def test_all_drops_precede_all_snapshots(self):
        order = []
        def drop(path):
            order.append(('drop', Path(path)))
            return dict(file=str(path), bytes=4096, fadvise_completed=True)
        def snapshot(path):
            order.append(('snapshot', Path(path)))
            return self.snapshot(path)
        with mock.patch.object(harness, 'drop_model_cache', side_effect=drop), \
             mock.patch.object(harness, 'residency_snapshot', side_effect=snapshot):
            harness.verified_cold_report(self.paths[0])
        self.assertEqual(order, [('drop', p) for p in self.paths] + [('snapshot', p) for p in self.paths])

    def test_snapshot_error_does_not_claim_cold(self):
        def snapshot(path):
            result = self.snapshot(path)
            if Path(path) == self.paths[-1]:
                result['error'] = 'injected'
            return result
        with mock.patch.object(harness, 'residency_snapshot', side_effect=snapshot):
            result = harness.verified_cold_report(self.paths[0])
        self.assertFalse(result['verified_cold'])
        self.assertEqual(result['verdict'], 'UNKNOWN')

    def test_small_warm_shard_cannot_hide_in_aggregate(self):
        def snapshot(path):
            result = self.snapshot(path)
            if Path(path) != self.paths[-1]:
                result['pages'] = 1000
            return result
        with mock.patch.object(harness, 'residency_snapshot', side_effect=snapshot):
            result = harness.verified_cold_report(self.paths[0])
        self.assertLess(result['resident_ratio'], 0.10)
        self.assertFalse(result['verified_cold'])

    def test_single_file_still_measures_real_residency(self):
        path = Path(self.temp.name) / 'single.gguf'
        path.write_bytes(b'x' * 4096)
        result = harness.verified_cold_report(path)
        self.assertEqual(result['snapshot']['pages'], 1)
        self.assertEqual(result['snapshot']['file'], str(path))
        self.assertEqual(result['drop']['bytes'], 4096)

    def test_all_zero_shards_are_verified_cold(self):
        def snapshot(path):
            return dict(self.snapshot(path), resident_pages=0)
        with mock.patch.object(harness, 'residency_snapshot', side_effect=snapshot):
            result = harness.verified_cold_report(self.paths[0], max_resident_ratio=0.0)
        self.assertTrue(result['verified_cold'])
        self.assertEqual(result['snapshot']['resident_pages'], 0)
        self.assertEqual(len(result['snapshot']['files']), 4)

    def test_late_drop_failure_prevents_any_snapshot(self):
        with mock.patch.object(harness, 'drop_model_cache', side_effect=[{}, {}, OSError('injected')]), \
             mock.patch.object(harness, 'residency_snapshot') as snapshot:
            with self.assertRaises(OSError):
                harness.verified_cold_report(self.paths[0])
        snapshot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
