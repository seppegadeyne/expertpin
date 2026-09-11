#!/usr/bin/env python3
"""CPU-only regression tests for the multi-needle (RULER-style) gate; no GPU/launcher/service."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'evidence/multineedle-20260911/run-guarded.py'
with mock.patch('subprocess.run', side_effect=AssertionError('import subprocess')), \
     mock.patch('subprocess.Popen', side_effect=AssertionError('import spawn')), \
     mock.patch.object(Path, 'mkdir', side_effect=AssertionError('import write')):
    spec = importlib.util.spec_from_file_location('multineedle', PATH)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)


class MultiNeedleTests(unittest.TestCase):
    def test_needles_have_unique_codes_and_projects(self):
        codes = [n['code'] for n in harness.MULTI_NEEDLES]
        projects = [n['project'] for n in harness.MULTI_NEEDLES]
        self.assertEqual(len(set(codes)), len(codes))
        self.assertEqual(len(set(projects)), len(projects))
        for needle in harness.MULTI_NEEDLES:
            self.assertNotIn(needle['code'], json.dumps(harness.build_haystack(50)))
            self.assertNotIn(needle['project'], ' '.join(harness.build_haystack(50)))

    def test_insert_all_positions_unique(self):
        paragraphs = harness.build_haystack(40)
        with_needles, positions = harness.insert_all_needles(paragraphs)
        self.assertEqual(len(with_needles), len(paragraphs) + len(harness.MULTI_NEEDLES))
        for needle, index in zip(harness.MULTI_NEEDLES, positions):
            self.assertEqual(with_needles[index], needle['paragraph'])
        codes_in_text = [n['code'] for n in harness.MULTI_NEEDLES]
        text = '\n\n'.join(with_needles)
        for code in codes_in_text:
            self.assertEqual(text.count(code), 1)
        # ordering: positions strictly increasing
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(set(positions)), len(positions))

    def test_insert_positions_follow_fractions(self):
        paragraphs = harness.build_haystack(100)
        with_needles, positions = harness.insert_all_needles(paragraphs)
        total = len(with_needles)
        for needle, index in zip(harness.MULTI_NEEDLES, positions):
            self.assertAlmostEqual(index / total, needle['fraction'], delta=0.08)

    def test_multi_payload_mentions_all_projects(self):
        paragraphs, _ = harness.insert_all_needles(harness.build_haystack(30))
        payload = harness.multi_needle_payload(paragraphs)
        content = payload['messages'][0]['content']
        for needle in harness.MULTI_NEEDLES:
            self.assertIn(needle['project'], content)
            self.assertIn(needle['paragraph'], content)
        self.assertEqual(payload['max_tokens'], harness.NEEDLE_TOKENS)
        self.assertEqual(payload['temperature'], 0.0)
        self.assertEqual(payload['seed'], 42)
        self.assertIs(payload['stream'], False)
        self.assertIs(payload['cache_prompt'], True)

    def test_validate_multi_accepts_full_recall(self):
        good = {'usage': {'completion_tokens': 40},
                'choices': [{'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': 'Aurora 7391, Borealis 4826, Cascade 9517.'}}],
                'timings': {'prompt_ms': 5.0, 'predicted_ms': 1.0, 'predicted_per_second': 1.0}}
        result = harness.validate_multi_needle(good)
        self.assertEqual(result['recalled_codes'], sorted(n['code'] for n in harness.MULTI_NEEDLES))
        self.assertTrue(result['all_recalled'])

    def test_validate_multi_counts_partial_and_rejects_malformed(self):
        base = {'usage': {'completion_tokens': 40},
                'choices': [{'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': 'Aurora 7391 and Borealis 4826.'}}],
                'timings': {'prompt_ms': 5.0, 'predicted_ms': 1.0, 'predicted_per_second': 1.0}}
        partial = harness.validate_multi_needle(json.loads(json.dumps(base)))
        self.assertEqual(len(partial['recalled_codes']), 2)
        self.assertFalse(partial['all_recalled'])
        for mutate in [
            lambda d: d['choices'][0].update(finish_reason='length'),
            lambda d: d['choices'][0]['message'].update(content=''),
            lambda d: d['choices'][0]['message'].update(content=None),
            lambda d: d.update(error='x'),
        ]:
            with self.subTest(mutate=mutate):
                data = json.loads(json.dumps(base))
                mutate(data)
                with self.assertRaises(RuntimeError):
                    harness.validate_multi_needle(data)

    def test_multi_verdict_requires_all_codes(self):
        def result(codes):
            return {'recalled_codes': codes, 'all_recalled': len(codes) == 3, 'content': ' '.join(codes)}
        all_codes = sorted(n['code'] for n in harness.MULTI_NEEDLES)
        self.assertEqual(harness.multi_needle_verdict({'single': result(all_codes)})['gate'], 'PASS')
        verdict = harness.multi_needle_verdict({'single': result(all_codes[:2])})
        self.assertEqual(verdict['gate'], 'FAIL')
        self.assertTrue(verdict['failures'])

    def test_run_multi_mode(self):
        run = harness.Run(Path('/unused'), 0, 'ps-iq2xxs', gate='multi')
        self.assertEqual(run.sequence, ('single',))
        self.assertEqual(run.summary['gate'], 'multi')
        harness.validate_plan(0, 'reference', 'multi')
        with self.assertRaises(ValueError):
            harness.validate_plan(4, 'reference', 'multi')


if __name__ == '__main__':
    unittest.main()
