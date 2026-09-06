#!/usr/bin/env python3
"""Exercise the real CPU calibration executable; no model or GPU required."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import unittest

BINARY = str(Path(sys.argv.pop(1)).resolve())


class CpuBenchTest(unittest.TestCase):
    def test_cli_rejects_invalid_or_unbounded_work(self):
        for args in ([], ['0', '3'], ['1', '0'], ['257', '3'], ['1', '10001'],
                     ['1x', '3'], ['1', '3', 'extra'], ['-1', '3'], ['', '3'],
                     ['9' * 1000, '3'], ['1', '9' * 1000], ['1', '']):
            with self.subTest(args=args):
                result = subprocess.run([BINARY, *args], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')

    def test_even_median_and_multiple_threads(self):
        result = subprocess.run([BINARY, '2', '4'], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['threads'], 2)
        self.assertEqual([case['quant'] for case in data['cases']], ['iq2_s', 'iq3_s', 'iq4_nl'])
        for case in data['cases']:
            self.assertEqual(case['validated_expert_ids'], [0, 1])
            self.assertLessEqual(case['max_relative_l2_error'], 0.03)
            self.assertEqual(len(case['samples_us']), 4)
            self.assertTrue(all(0 < x < float('inf') for x in case['samples_us']))
            self.assertAlmostEqual(case['median_us'], statistics.median(case['samples_us']))

    def test_real_kernels_and_output_contract(self):
        result = subprocess.run([BINARY, '1', '3'], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['schema_version'], 1)
        self.assertEqual(data['backend'], 'cpu')
        self.assertEqual(data['operation'], 'MUL_MAT_ID')
        self.assertEqual(data['payload'], 'synthetic_quantized')
        self.assertEqual(data['cache_condition'], 'repeated_hot_two_expert_bank')
        self.assertIsNone(data['model_tokens_per_second'])
        self.assertIsNone(data['physical_ram_gib_s'])
        self.assertEqual(data['threads'], 1)
        self.assertEqual(data['warmups'], 3)
        self.assertEqual(len(data['cases']), 3)
        for case, expected in zip(data['cases'], [('iq2_s', 2560, 640, 524800),
                                                 ('iq3_s', 2560, 640, 704000),
                                                 ('iq4_nl', 640, 2560, 921600)]):
            self.assertEqual((case['quant'], case['input_columns'], case['output_rows'],
                              case['slice_bytes']), expected)
            self.assertEqual(case['bank_bytes'], 2 * case['slice_bytes'])
            self.assertEqual(case['experts_in_bank'], 2)
            self.assertEqual(case['selected_expert'], 1)
            self.assertEqual(case['tokens'], 1)
            self.assertEqual(case['validated_expert_ids'], [0, 1])
            self.assertLessEqual(case['max_relative_l2_error'], 0.03)
            samples = case['samples_us']
            self.assertEqual(len(samples), 3)
            self.assertTrue(all(0 < x < float('inf') for x in samples))
            self.assertAlmostEqual(case['median_us'], statistics.median(samples))
            self.assertEqual(case['minimum_us'], min(samples))
            self.assertEqual(case['maximum_us'], max(samples))


if __name__ == '__main__':
    unittest.main()
