#!/usr/bin/env python3
"""CPU-only regression tests for the Hermes E2E contract gate; no GPU/launcher/service."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'evidence/hermes-e2e-20260911/run-contract.py'
with mock.patch('subprocess.run', side_effect=AssertionError('import subprocess')), \
     mock.patch('subprocess.Popen', side_effect=AssertionError('import spawn')), \
     mock.patch.object(Path, 'mkdir', side_effect=AssertionError('import write')):
    spec = importlib.util.spec_from_file_location('contract', PATH)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)


def chunk(delta, finish=None):
    return {'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}


def completion_plain(content='ready', finish='stop'):
    return {'object': 'chat.completion', 'model': 'x', 'choices': [
        {'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': finish}],
        'usage': {'completion_tokens': 5}}


def completion_tool():
    return {'object': 'chat.completion', 'model': 'x', 'choices': [
        {'index': 0, 'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': None,
         'tool_calls': [{'index': 0, 'id': 'call-1', 'type': 'function',
                         'function': {'name': 'get_weather', 'arguments': '{"city": "Ghent", "unit": "celsius"}'}}]}}],
        'usage': {'completion_tokens': 30}}


class ContractGateTests(unittest.TestCase):
    def test_models_check(self):
        result = harness.contract_check_models({'data': [{'id': 'expertpin'}, {'id': 'other'}]})
        self.assertEqual(result['model_ids'], ['expertpin', 'other'])
        for bad in ({}, {'data': []}, {'data': [{'no_id': 1}]}, {'data': 'x'}, None):
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    harness.contract_check_models(bad)

    def test_nonstreaming_plain_and_tool(self):
        result = harness.contract_check_nonstreaming(completion_plain(), expect_tool_calls=False)
        self.assertEqual(result['finish_reason'], 'stop')
        harness.contract_check_nonstreaming(completion_tool(), expect_tool_calls=True)
        for mutate in [
            lambda d: d.update(object='x'),
            lambda d: d['choices'].clear(),
            lambda d: d['choices'][0].update(index=1),
            lambda d: d['choices'][0]['message'].update(role='user'),
            lambda d: d['choices'][0].update(finish_reason='stop') if False else d['choices'][0].update(finish_reason='tool_calls'),
            lambda d: d.pop('usage'),
        ]:
            with self.subTest(mutate=mutate):
                data = completion_plain()
                mutate(data)
                with self.assertRaises(RuntimeError):
                    harness.contract_check_nonstreaming(data, False)
        wrong_expect = completion_tool()
        with self.assertRaises(RuntimeError):
            harness.contract_check_nonstreaming(wrong_expect, False)

    def test_streaming_plain(self):
        events = [chunk({'role': 'assistant'}), chunk({'content': 're'}), chunk({'content': 'ady'}, finish='stop')]
        result = harness.contract_check_streaming(events, expect_tool_calls=False)
        self.assertEqual(result['finish_reason'], 'stop')
        for bad in [
            [],
            [chunk({'role': 'user'})],
            [chunk({'content': 'x'})],  # no terminal finish
            [chunk({'content': 'x'}, finish='stop'), chunk({'content': 'y'}, finish='stop')],
            [chunk({'content': '   '}, finish='stop')],
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    harness.contract_check_streaming(bad, False)

    def test_streaming_tool_calls(self):
        events = [chunk({'role': 'assistant'}),
                  chunk({'tool_calls': [{'index': 0, 'id': 'call-1', 'type': 'function',
                                         'function': {'name': 'get_weather', 'arguments': ''}}]}),
                  chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '{"city"'}}]}),
                  chunk({}, finish='tool_calls')]
        result = harness.contract_check_streaming(events, expect_tool_calls=True)
        self.assertEqual(result['finish_reason'], 'tool_calls')
        self.assertEqual(result['streamed_tool_calls'], 2)
        no_tool = [chunk({'role': 'assistant'}, finish='stop')]
        with self.assertRaises(RuntimeError):
            harness.contract_check_streaming(no_tool, True)
        missing_index = [chunk({'tool_calls': [{'function': {'name': 'x'}}]}, finish='tool_calls')]
        with self.assertRaises(RuntimeError):
            harness.contract_check_streaming(missing_index, True)

    def test_tool_round_trip(self):
        round1 = completion_tool()
        round2 = completion_plain(content='It is 17.5C in Ghent.')
        trip = {'round1': round1,
                'round2_request': {'messages': [
                    {'role': 'user', 'content': 'q'},
                    {'role': 'assistant', 'content': None, 'tool_calls': round1['choices'][0]['message']['tool_calls']},
                    {'role': 'tool', 'tool_call_id': 'call-1', 'content': '{"t": 17.5}'}]},
                'round2': round2}
        result = harness.contract_check_tool_round(trip)
        self.assertEqual(result['call_id'], 'call-1')
        self.assertEqual(result['round1_finish'], 'tool_calls')
        self.assertEqual(result['round2_finish'], 'stop')
        for mutate in [
            lambda t: t['round2_request']['messages'][2].update(tool_call_id='wrong'),
            lambda t: t['round2_request']['messages'][2].update(role='user'),
            lambda t: t['round2_request']['messages'].pop(2),
            lambda t: t['round1']['choices'][0]['message']['tool_calls'][0]['function'].update(name='other'),
        ]:
            with self.subTest(mutate=mutate):
                data = json.loads(json.dumps(trip))
                mutate(data)
                with self.assertRaises(RuntimeError):
                    harness.contract_check_tool_round(data)

    def test_run_contract_mode(self):
        run = harness.Run(Path('/unused'), 0, 'ps-iq2xxs', gate='contract')
        self.assertEqual(run.sequence, ('models', 'plain', 'stream-plain', 'stream-tool', 'tool-round'))
        self.assertEqual(run.summary['gate'], 'contract')
        self.assertEqual(run.request_seconds, harness.NEEDLE_REQUEST_SECONDS)
        harness.validate_plan(0, 'reference', 'contract')
        with self.assertRaises(ValueError):
            harness.validate_plan(4, 'reference', 'contract')


if __name__ == '__main__':
    unittest.main()
