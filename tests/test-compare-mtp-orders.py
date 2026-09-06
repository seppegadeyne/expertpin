#!/usr/bin/env python3
"""CPU-only tests. All rates/text in mocks are synthetic, not measurement evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_orders', ROOT / 'scripts/compare-mtp-orders.py')
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ComparisonTests(unittest.TestCase):
    def test_both_channels_and_exact_unicode_whitespace(self):
        a = m.text_fields({'content': 'é\n', 'reasoning_content': 'think'})
        self.assertTrue(m.compare(a, dict(a))['equal'])
        for b in [dict(a, content='é'), dict(a, reasoning_content='different'), dict(a, content='e\u0301\n')]:
            result = m.compare(a, b)
            self.assertFalse(result['equal'])
            self.assertNotEqual(result['left_sha256'], result['right_sha256'])
            self.assertTrue(result['diff'])

    def test_null_is_not_empty(self):
        self.assertFalse(m.compare(m.text_fields({'content': 'x'}),
                                   m.text_fields({'content': 'x', 'reasoning_content': ''}))['equal'])

    def test_rejects_nontext_or_empty(self):
        for value in [None, {}, {'content': []}, {'content': '  '}]:
            with self.assertRaises(ValueError):
                m.text_fields(value)

    def fixtures(self):
        return [{'n_max': n, 'started': f'2026-09-06T10:0{i}:00+02:00',
                 'finished': f'2026-09-06T10:0{i}:30+02:00', 'scope': f'scope-{i}',
                 'directory': str(ROOT / f'fixture-{i}'), 'environment': {'CTX': '8192'},
                 'requests': [
            {'payload': {'seed': 42}, 'metadata': {'model': 'fixture'}, 'text': {'content': 'fixture'}, 'timings': {'predicted_ms': 1000.0 + i}}
            for _ in range(2)]} for i, n in enumerate((4, 8, 16, 16, 8, 4))]

    def test_complete_pairs_and_no_causal_claim(self):
        with patch.object(m, 'load_run', side_effect=self.fixtures()):
            result = m.analyze([1, 2, 3], [4, 5, 6])
        self.assertEqual(len(result['comparisons']), 66)
        self.assertTrue(all(p['equal'] for p in result['comparisons']))
        self.assertFalse(result['causal_ranking_proven'])
        self.assertFalse(result['default_changed'])
        self.assertFalse(result['same_observed_ranking'])
        self.assertEqual(result['rankings']['forward']['pooled_tok_s'][4], 256)

    def test_refuses_wrong_depth_time_or_workload(self):
        for kind in ('depth', 'time', 'payload', 'metadata'):
            rows = copy.deepcopy(self.fixtures())
            if kind == 'depth': rows[3]['n_max'] = 4
            if kind == 'time': rows[3]['started'] = '0'
            if kind == 'payload': rows[3]['requests'][0]['payload']['seed'] = 99
            if kind == 'metadata': rows[3]['requests'][0]['metadata']['model'] = 'different'
            with patch.object(m, 'load_run', side_effect=rows), self.assertRaises(ValueError):
                m.analyze([1, 2, 3], [4, 5, 6])

    def test_rejects_duplicate_or_overlapping_runs_and_environment_changes(self):
        for key, value in [('directory', str(ROOT / 'fixture-2')), ('scope', 'scope-2'),
                           ('started', '2026-09-06T10:02:20+02:00'), ('environment', {'CTX': '4096'})]:
            rows = self.fixtures()
            rows[3][key] = value
            with patch.object(m, 'load_run', side_effect=rows), self.assertRaises(ValueError):
                m.analyze([1, 2, 3], [4, 5, 6])

    def test_exact_ties_have_order_independent_groups(self):
        rows = self.fixtures()
        for row in rows:
            for q in row['requests']:
                q['timings']['predicted_ms'] = 1000.0
        with patch.object(m, 'load_run', side_effect=rows):
            result = m.analyze([1, 2, 3], [4, 5, 6])
        self.assertTrue(result['same_observed_ranking'])
        self.assertEqual(result['rankings']['forward']['tie_groups'], [[4, 8, 16]])

    def test_duplicate_raw_request_labels_rejected(self):
        import json
        directory = sorted((ROOT / 'evidence/clean-mtp-ab').glob('run-20260906T15*/summary.json'))[0].parent
        original = Path.read_text
        def read(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path.name == 'summary.json':
                data = json.loads(text)
                data['requests'][1] = copy.deepcopy(data['requests'][0])
                return json.dumps(data)
            return text
        with patch.object(Path, 'read_text', read), self.assertRaises(ValueError):
            m.load_run(directory)

    def test_existing_forward_raw_evidence(self):
        paths = sorted((ROOT / 'evidence/clean-mtp-ab').glob('run-20260906T15*/summary.json'))
        self.assertEqual(len(paths), 3)
        self.assertEqual([m.load_run(p.parent)['n_max'] for p in paths], [4, 8, 16])


if __name__ == '__main__':
    unittest.main()
