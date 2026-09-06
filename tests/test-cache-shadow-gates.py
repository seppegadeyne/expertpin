"""CPU-only regression tests; generated data are fixtures, not benchmarks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "evidence/cache-shadow-ab/verify-ab-gates.py"


class GatesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        for arm in ("shadow-off", "shadow-32g"):
            (self.run / arm).mkdir()
            for i in range(1, 4):
                self.save(arm, f"sample-{i}.json", {
                    "content": "fixture output",
                    "timings": {"predicted_n": 128, "predicted_ms": 2560},
                })
        self.stats = {"cache_sim": {"requests": 100, "hit_rate": 0.95}}
        self.save("shadow-32g", "expert-stats.json", self.stats)

    def save(self, arm, name, value):
        (self.run / arm / name).write_text(json.dumps(value))

    def check(self, success, *args):
        proc = subprocess.run([sys.executable, str(SCRIPT), str(self.run), *args],
                              text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0 if success else 1, proc.stdout + proc.stderr)
        self.assertIn("GATES_PASSED" if success else "GATES_FAILED", proc.stdout)
        self.assertNotIn("Traceback", proc.stderr)

    def test_valid_and_thresholds(self):
        self.check(True)
        self.check(False, "0.96")

    def test_nonfinite_or_invalid_hit_rate(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.95", None):
            with self.subTest(value=value):
                self.stats["cache_sim"]["hit_rate"] = value
                self.save("shadow-32g", "expert-stats.json", self.stats)
                self.check(False)

    def test_invalid_request_count(self):
        for value in (-1, 0, True, 1.5, "100", None):
            with self.subTest(value=value):
                self.stats["cache_sim"]["requests"] = value
                self.save("shadow-32g", "expert-stats.json", self.stats)
                self.check(False)

    def test_each_sample_requires_valid_timing(self):
        for field, values in (("predicted_n", (0, -1, True, 1.5, "128", None)),
                              ("predicted_ms", (0, -1, True, float("nan"),
                                                float("inf"), "2560", None))):
            for value in values:
                with self.subTest(field=field, value=value):
                    sample = {"content": "fixture output",
                              "timings": {"predicted_n": 128, "predicted_ms": 2560}}
                    sample["timings"][field] = value
                    self.save("shadow-32g", "sample-2.json", sample)
                    self.check(False)

    def test_malformed_or_incomplete_sample(self):
        for value in ([], {}, {"content": None}, {"content": "fixture output", "timings": []}):
            self.save("shadow-off", "sample-1.json", value)
            self.check(False)
        path = self.run / "shadow-off/sample-1.json"
        path.write_text("{broken")
        self.check(False)
        path.unlink()
        self.check(False)

    def test_malformed_stats(self):
        for value in ([], {}, {"cache_sim": None}, {"cache_sim": []}):
            self.save("shadow-32g", "expert-stats.json", value)
            self.check(False)
        path = self.run / "shadow-32g/expert-stats.json"
        path.write_text("{broken")
        self.check(False)
        path.write_bytes(b"\xff")
        self.check(False)
        path.unlink()
        self.check(False)

    def test_invalid_thresholds(self):
        for args in (("nan",), ("inf",), ("-1",), ("1.1",), ("abc",),
                     ("0.9", "nan"), ("0.9", "inf"), ("0.9", "-1")):
            with self.subTest(args=args):
                self.check(False, *args)

    def test_output_mismatch_and_overhead(self):
        for content, ms in (("different", 2560), ("fixture output", 3000)):
            for i in range(1, 4):
                self.save("shadow-32g", f"sample-{i}.json", {
                    "content": content, "timings": {"predicted_n": 128, "predicted_ms": ms}})
            self.check(False)

    def test_extreme_timing_arithmetic(self):
        for n, ms in ((10 ** 400, 1.0), (128, 10 ** 400), (128, 5e-324)):
            self.save("shadow-off", "sample-1.json", {
                "content": "fixture output", "timings": {"predicted_n": n, "predicted_ms": ms}})
            self.check(False)
        for arm, ms in (("shadow-off", 1e-300), ("shadow-32g", 1e300)):
            for i in range(1, 4):
                self.save(arm, f"sample-{i}.json", {
                    "content": "fixture output", "timings": {"predicted_n": 128, "predicted_ms": ms}})
        self.check(False)

    def test_duplicate_stats_keys(self):
        path = self.run / "shadow-32g/expert-stats.json"
        for raw in ('{"cache_sim":{"requests":100,"hit_rate":NaN,"hit_rate":0.95}}',
                    '{"cache_sim":null,"cache_sim":{"requests":100,"hit_rate":0.95}}'):
            with self.subTest(raw=raw):
                path.write_text(raw)
                self.check(False)

    def test_duplicate_sample_keys(self):
        path = self.run / "shadow-off/sample-1.json"
        for raw in ('{"content":"fixture output","timings":{"predicted_n":128,'
                    '"predicted_ms":NaN,"predicted_ms":2560}}',
                    '{"content":"fixture output","timings":null,'
                    '"timings":{"predicted_n":128,"predicted_ms":2560}}'):
            with self.subTest(raw=raw):
                path.write_text(raw)
                self.check(False)

    def test_excessively_nested_json(self):
        for arm, name in (("shadow-off", "sample-1.json"), ("shadow-32g", "expert-stats.json")):
            with self.subTest(arm=arm):
                path = self.run / arm / name
                original = path.read_bytes()
                path.write_text('{"unused":' + '[' * 2000 + '0' + ']' * 2000 + '}')
                try:
                    self.check(False)
                finally:
                    path.write_bytes(original)

    def test_committed_measurements_still_pass(self):
        self.run = ROOT / "evidence/cache-shadow-ab/20260905T032156+0200"
        self.check(True)


if __name__ == "__main__":
    unittest.main()
