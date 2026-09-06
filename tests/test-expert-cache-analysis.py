"""CPU-only tests for bounded-cache feasibility from aggregate evidence."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis", ROOT / "scripts/analyze-expert-cache.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def stats(**overrides):
    value = dict(requests=8, hits=4, misses=4, evictions=0, evicted_bytes=0,
                 bypasses=0, resident_bytes=40, capacity_bytes=64, hit_rate=0.5)
    value.update(overrides)
    return {"cache_sim": value}


class CacheAnalysisTests(unittest.TestCase):
    def test_fitting_and_smaller_caps(self):
        result = analysis.analyze(stats(), [32, 40, 64], stable_cold_start=True)
        self.assertEqual(result["observed"]["unique_slices"], 4)
        self.assertEqual(result["observed"]["mean_unique_slice_bytes"], 10)
        small, fit, larger = result["capacities"]
        self.assertIsNone(small["hits"])
        self.assertEqual(small["misses_lower_bound"], 4)
        self.assertEqual(small["misses_upper_bound"], 8)
        self.assertEqual(fit["hits"], 4)
        self.assertEqual(larger["hits"], 4)
        self.assertEqual(small["logical_working_set_excess_bytes"], 8)
        self.assertIsNone(result["nvme_reload_bytes"])
        self.assertIsNone(result["tok_s"])

    def test_assumptions_must_be_explicit(self):
        result = analysis.analyze(stats(), [64])
        self.assertIsNone(result["observed"]["unique_slices"])
        self.assertIsNone(result["capacities"][0]["hits"])
        self.assertIsNone(result["capacities"][0]["misses_lower_bound"])

    def test_pressure_or_bypass_does_not_prove_unique_count(self):
        for change in (dict(evictions=1, evicted_bytes=30), dict(bypasses=1)):
            with self.subTest(change=change):
                result = analysis.analyze(stats(**change), [64], stable_cold_start=True)
                self.assertIsNone(result["observed"]["unique_slices"])
                self.assertIsNone(result["capacities"][0]["hits"])

    def test_bad_counters_rejected(self):
        for change in (dict(hits=5), dict(requests=True), dict(misses=-1),
                       dict(resident_bytes=65), dict(evictions=5), dict(bypasses=5),
                       dict(evicted_bytes=10), dict(evictions=1), dict(hit_rate=float("nan")),
                       dict(hit_rate=0.6), dict(capacity_bytes=0), dict(requests=8.0)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                analysis.analyze(stats(**change), [32])

    def test_empty_observer_rejected(self):
        with self.assertRaises(ValueError):
            analysis.analyze(stats(requests=0, hits=0, misses=0, resident_bytes=0, hit_rate=0), [32])

    def test_eviction_bytes_must_cover_evicted_entries(self):
        with self.assertRaises(ValueError):
            analysis.analyze(stats(evictions=2, evicted_bytes=1), [32])

    def test_stable_cold_pressure_requires_enough_inserted_bytes(self):
        for evicted in (10, 24):
            with self.subTest(evicted=evicted), self.assertRaises(ValueError):
                analysis.analyze(stats(evictions=1, evicted_bytes=evicted), [64],
                                 stable_cold_start=True)

    def test_unknown_layout_does_not_assume_byte_conservation(self):
        result = analysis.analyze(stats(evictions=1, evicted_bytes=10), [64])
        self.assertIsNone(result["observed"]["unique_slices"])

    def test_repository_evidence_regression(self):
        path = ROOT / "evidence/cache-shadow-ab/20260905T032156+0200/shadow-32g/expert-stats.json"
        result = analysis.analyze(analysis.load_json(path), [28*analysis.GIB, 30*analysis.GIB],
                                  stable_cold_start=True)
        self.assertEqual(result["observed"]["logical_working_set_bytes"], 31745638400)
        self.assertEqual(result["capacities"][0]["logical_working_set_excess_bytes"], 1680867328)
        self.assertEqual(result["capacities"][1]["hits"], 424383)

    def test_bad_capacities_rejected(self):
        for caps in ([], [True], [0], [-1], [1.5], [32, 32]):
            with self.subTest(caps=caps), self.assertRaises(ValueError):
                analysis.analyze(stats(), caps)

    def test_aggregate_order_is_not_recoverable(self):
        # Same frequency histogram and fitting-cache totals, different small LRU.
        a = ["a", "a", "b", "b", "c", "c", "d", "d"]
        b = ["a", "b", "c", "d", "a", "b", "c", "d"]
        self.assertEqual(analysis.replay_example(a, 4), analysis.replay_example(b, 4))
        self.assertEqual(analysis.replay_example(a, 2)["hits"], 4)
        self.assertEqual(analysis.replay_example(b, 2)["hits"], 0)


if __name__ == "__main__":
    unittest.main()
