#!/usr/bin/env python3
"""CPU-only regression tests for the error-path contract gate; no GPU/launcher/service."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'evidence/hermes-e2e-20260911/run-contract.py'
with mock.patch('subprocess.run', side_effect=AssertionError('import subprocess')), \
     mock.patch('subprocess.Popen', side_effect=AssertionError('import spawn')), \
     mock.patch.object(Path, 'mkdir', side_effect=AssertionError('import write')):
    spec = importlib.util.spec_from_file_location('contract_errors', PATH)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)


class ErrorPathTests(unittest.TestCase):
    def test_http_error_shape(self):
        result = harness.contract_check_http_error(400, {'error': {'code': 400, 'message': 'prompt too long', 'type': 'invalid_request_error'}})
        self.assertEqual(result['status'], 400)
        self.assertEqual(result['error_type'], 'invalid_request_error')
        # This server classifies overflow as 500 (server-context.cpp:3983).
        accepted500 = harness.contract_check_http_error(500, {'error': {'code': 500, 'message': 'the request exceeds the available context size', 'type': 'server_error'}})
        self.assertEqual(accepted500['status'], 500)

    def test_http_error_rejects_success_and_malformed(self):
        for status, body in (
            (200, {'error': {'code': 400, 'message': 'x'}}),
            (400, {'message': 'no error envelope'}),
            (400, {'error': 'not an object'}),
            (400, {}),
            (302, {'error': {'code': 302, 'message': 'redirect is not an error envelope'}}),
        ):
            with self.subTest(status=status):
                with self.assertRaises(RuntimeError):
                    harness.contract_check_http_error(status, body)

    def test_overflow_payload_exceeds_context(self):
        payload = harness.overflow_payload()
        text = payload['messages'][0]['content']
        self.assertGreater(len(text.split()) * 2, 8192)  # crude token lower bound
        self.assertGreater(payload['max_tokens'], 0)

    def test_bad_tool_choice_payload(self):
        payload = harness.bad_tool_choice_payload()
        self.assertEqual(payload['tool_choice'], 'bogus-choice')
        self.assertIn('tools', payload)

    def test_missing_messages_payload(self):
        payload = harness.missing_messages_payload()
        self.assertNotIn('messages', payload)
        self.assertIn('max_tokens', payload)

    def test_errors_mode_sequence(self):
        run = harness.Run(Path('/unused'), 0, 'ps-iq2xxs', gate='errors')
        self.assertEqual(run.sequence, ('overflow', 'bad-tool-choice', 'missing-messages'))
        self.assertEqual(run.summary['gate'], 'errors')
        harness.validate_plan(0, 'reference', 'errors')
        with self.assertRaises(ValueError):
            harness.validate_plan(4, 'reference', 'errors')

    def test_error_verdict_requires_all_three(self):
        ok = {'overflow': {'status': 500}, 'bad-tool-choice': {'status': 400}, 'missing-messages': {'status': 400}}
        self.assertEqual(harness.errors_verdict(ok)['gate'], 'PASS')
        partial = dict(ok)
        partial['missing-messages'] = {'status': 200}  # a 2xx body is never a valid error observation
        verdict = harness.errors_verdict(partial)
        self.assertEqual(verdict['gate'], 'FAIL')
        self.assertTrue(any('missing-messages' in f for f in verdict['failures']))


if __name__ == '__main__':
    unittest.main()
