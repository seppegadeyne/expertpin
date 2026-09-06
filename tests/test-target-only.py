#!/usr/bin/env python3
"""Synthetic CPU fixtures, not model evidence."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('harness', ROOT / 'evidence/clean-mtp-ab/run-guarded.py')
assert spec is not None and spec.loader is not None
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

class TargetOnly(unittest.TestCase):
    def test_target_plan(self):
        h.validate_plan(0)
        env = h.clean_environment({'DRAFT': '1'}, 'test.scope', 0)
        self.assertEqual(env['DRAFT'], '0')
        self.assertEqual(env['DRAFT_NMAX'], '0')
        target = h.payload(0)
        mtp = h.payload(4)
        mtp.pop('speculative.n_max')
        self.assertEqual(target, mtp)
        self.assertNotIn('speculative.n_max', target)

    def fixture(self):
        return {'usage': {'completion_tokens': 256}, 'choices': [
            {'finish_reason': 'length', 'message': {'reasoning_content': 'synthetic'}}],
            'timings': {'predicted_n': 256, 'prompt_ms': 1, 'predicted_ms': 1000,
                        'predicted_per_second': 256}}

    def test_target_counters_absent_or_zero(self):
        self.assertIsNone(h.validate_completion(self.fixture(), 0)['acceptance_fraction'])
        d = self.fixture()
        d['timings'].update(draft_n=0, draft_n_accepted=0, draft_by_depth=[])
        self.assertIsNone(h.validate_completion(d, 0)['acceptance_fraction'])

    def test_target_rejects_drafting_and_short_response(self):
        for change in ({'draft_n': 1}, {'draft_n_accepted': 1}, {'draft_by_depth': [{}]}, {'predicted_n': 255}):
            d = self.fixture()
            d['timings'].update(change)
            with self.assertRaises(RuntimeError):
                h.validate_completion(d, 0)

    def test_reference_validation(self):
        with self.assertRaises(ValueError):
            h.output_fields({'choices': [{'message': {'content': 3}}]})
        self.assertEqual(h.output_fields({'choices': [{'message': {'content': '', 'reasoning_content': 'abc'}}]}),
                         {'content': '', 'reasoning_content': 'abc'})

    def capture(self, mode='ok'):
        with tempfile.TemporaryDirectory(dir=ROOT / 'evidence/target-only') as directory:
            out = Path(directory)
            run = h.Run(out, 0)
            run.summary['requests'] = [{'label': '01'}, {'label': '02'}]
            run.tokenize_references = [out / 'ref.json']
            for name, text in [('01-response.json', 'é \n'), ('02-response.json', 'é \n'), ('ref.json', 'é \nx')]:
                run.save(name, {'choices': [{'message': {'content': None, 'reasoning_content': text}}]})
            calls = []
            def urlopen(request, timeout):
                self.assertEqual(timeout, 5)
                body = json.loads(request.data)
                calls.append((request.full_url, body))
                if mode == 'http':
                    raise OSError('synthetic HTTP failure')
                if request.full_url.endswith('/tokenize'):
                    self.assertIs(body['add_special'], False)
                    # Deliberately byte-based synthetic tokenizer: individual é
                    # bytes cannot be UTF-8 decoded, complete sequences can.
                    data = {'tokens': [True] if mode == 'ids' else list(body['content'].encode())}
                else:
                    data = {'content': 'wrong' if mode == 'roundtrip' else bytes(body['tokens']).decode()}
                return io.StringIO(json.dumps(data))
            with mock.patch.object(run, 'sample', side_effect=TimeoutError('deadline') if mode == 'deadline' else None) as sample, \
                 mock.patch.object(h.urllib.request, 'urlopen', side_effect=urlopen):
                run.capture_tokenizations()
            result = json.loads((out / 'retokenized.json').read_text())
            self.assertEqual(len(result['records']), 3)
            self.assertEqual(sample.call_count, len(calls))
            self.assertEqual(len(calls), 6)
            self.assertIsNone(result['records'][0]['tokens']['content'])
            self.assertEqual(result['records'][0]['fields']['reasoning_content'], 'é \n')
            self.assertEqual(result['kind'], 'retokenized_fields_not_decode_trace')

    def test_capture_exact_roundtrip_and_reference_without_single_byte_decode(self):
        self.capture()

    def test_capture_fail_closed(self):
        for mode, error in [('http', OSError), ('ids', RuntimeError), ('roundtrip', RuntimeError), ('deadline', TimeoutError)]:
            with self.subTest(mode=mode), self.assertRaises(error):
                self.capture(mode)

if __name__ == '__main__':
    unittest.main()
