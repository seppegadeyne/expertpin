#!/usr/bin/env python3
"""CPU-only calibration regression tests; all advisor fixtures are SYNTHETIC."""
import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

bw = load('expert_bw_calib')

class ProbeTests(unittest.TestCase):
    def test_exact_unaligned_coverage(self):
        for length, chunk in [(20000, 5000), (20000, 123), (17, 20), (0, 3)]:
            with self.subTest(length=length, chunk=chunk):
                offset = 0
                for start, size in bw.chunk_plan(length, chunk):
                    self.assertEqual(start, offset)
                    self.assertGreater(size, 0)
                    self.assertLessEqual(size, chunk)
                    offset += size
                self.assertEqual(offset, length)

    def test_random_permutation(self):
        sequential = bw.read_plan(20000, 5000, 'sequential', 17)
        random = bw.read_plan(20000, 5000, 'random', 17)
        self.assertCountEqual(sequential, random)
        self.assertEqual(random, bw.read_plan(20000, 5000, 'random', 17))
        self.assertNotEqual(sequential, random)
        with self.assertRaises(ValueError):
            bw.read_plan(10, 2, 'typo', 0)

    def test_short_read(self):
        with tempfile.TemporaryFile(dir=ROOT) as f:
            f.write(b'x' * 7000)
            f.flush()
            with self.assertRaises(OSError):
                bw.measure(f.fileno(), 8000, 123, False)
            self.assertEqual(bw.measure(f.fileno(), 7000, 123, False)[0], 7000)

class AdvisorTests(unittest.TestCase):
    def setUp(self):
        self.advisor = load('advise-expert-cache')
        self.profile = {
            'schema_version': 1, 'workload': 'SYNTHETIC unit test only',
            'ram_bytes_per_s': 1000, 'nvme_bytes_per_s': 100,
            'pcie_bytes_per_s': 500,
            'candidates': [self.candidate('small', 100, 200), self.candidate('large', 0, 300)]}

    def candidate(self, name, disk, host):
        return dict(id=name, host_cache_bytes=host, gpu_cache_bytes=100,
                    host_other_bytes=50, gpu_other_bytes=50,
                    ram_bytes_per_token=100, nvme_bytes_per_token=disk,
                    pcie_bytes_per_token=100, cpu_compute_seconds_per_token=0.1,
                    gpu_compute_seconds_per_token=0.2)

    def test_ranking_and_deterministic_tie(self):
        out = self.advisor.advise(self.profile)
        self.assertEqual(out['recommendation']['id'], 'large')
        self.assertAlmostEqual(out['recommendation']['serial_seconds_per_token'], 0.6)
        self.assertNotIn('env', out)
        self.profile['candidates'][0]['nvme_bytes_per_token'] = 0
        self.assertEqual(self.advisor.advise(self.profile)['recommendation']['id'], 'small')
        self.profile['candidates'].reverse()
        self.assertEqual(self.advisor.advise(self.profile)['recommendation']['id'], 'small')

    def test_zero_cost_candidate(self):
        zero = self.profile['candidates'][0]
        for key in self.advisor.TRAFFIC + self.advisor.COMPUTE:
            zero[key] = 0
        out = self.advisor.advise(self.profile)
        self.assertEqual(out['recommendation']['id'], zero['id'])
        self.assertEqual(out['recommendation']['serial_seconds_per_token'], 0)

    def test_missing_calibration_blocks(self):
        for key in ('ram_bytes_per_s', 'nvme_bytes_per_s', 'pcie_bytes_per_s'):
            p = copy.deepcopy(self.profile)
            p[key] = None
            out = self.advisor.advise(p)
            self.assertEqual(out['status'], 'blocked')
            self.assertIsNone(out['recommendation'])

    def test_budget_includes_other_memory(self):
        for prefix, cap in [('host', 40), ('gpu', 28)]:
            p = copy.deepcopy(self.profile)
            p['candidates'] = [p['candidates'][0]]
            c = p['candidates'][0]
            c[prefix + '_cache_bytes'] = cap * 2**30 - 50
            self.assertEqual(self.advisor.advise(p)['status'], 'advisory')
            c[prefix + '_other_bytes'] += 1
            self.assertEqual(self.advisor.advise(p)['status'], 'blocked')

    def test_invalid_numbers_and_duplicate_ids(self):
        for value in (True, -1, 0, float('nan'), float('inf'), '100', 10**400):
            p = copy.deepcopy(self.profile)
            p['pcie_bytes_per_s'] = value
            with self.subTest(value=str(value)), self.assertRaises(ValueError):
                self.advisor.advise(p)
        self.profile['candidates'][1]['id'] = 'small'
        with self.assertRaises(ValueError):
            self.advisor.advise(self.profile)

    def test_unknown_and_bad_shape_rejected(self):
        for p in ([], {}, dict(self.profile, unexpected=1), dict(self.profile, schema_version=True)):
            with self.assertRaises(ValueError):
                self.advisor.advise(p)
        self.profile['candidates'][0]['cpu_compute_seconds_per_token'] = None
        self.assertEqual(self.advisor.advise(self.profile)['status'], 'blocked')

    def test_strict_json(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '[' * 2000 + '0' + ']' * 2000):
            with self.subTest(text=text[:40]), self.assertRaises(ValueError):
                self.advisor.parse_profile(text)

    def test_cli_success_blocked_and_invalid(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            source = pathlib.Path(tmp) / 'profile.json'
            output = pathlib.Path(tmp) / 'result.json'
            for value, rc, status in [(self.profile, 0, 'advisory'),
                                      (dict(self.profile, pcie_bytes_per_s=None), 2, 'blocked'),
                                      ({}, 2, None)]:
                source.write_text(json.dumps(value))
                run = subprocess.run([sys.executable, str(ROOT / 'scripts/advise-expert-cache.py'),
                                      str(source), '--json', str(output)],
                                     capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, rc, run.stderr)
                if status:
                    self.assertEqual(json.loads(run.stdout)['status'], status)
                    self.assertEqual(output.read_text(), run.stdout)
                else:
                    self.assertIn('invalid fields', run.stderr)

    def test_all_candidate_numbers_and_incomplete_loser(self):
        c = self.profile['candidates'][0]
        for key in self.advisor.MEMORY + self.advisor.TRAFFIC + self.advisor.COMPUTE:
            for value in (-1, True, float('nan'), float('inf'), '0'):
                p = copy.deepcopy(self.profile)
                p['candidates'][0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.advisor.advise(p)
        c['host_cache_bytes'] = 41 * 2**30
        c['cpu_compute_seconds_per_token'] = None
        self.assertEqual(self.advisor.advise(self.profile)['status'], 'blocked')

if __name__ == '__main__':
    unittest.main()
