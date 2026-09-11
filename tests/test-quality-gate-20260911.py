#!/usr/bin/env python3
"""CPU-only regression tests for the tool-call JSON quality gate; no GPU/launcher/service."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'evidence/quality-gate-20260911/run-guarded.py'
with mock.patch('subprocess.run', side_effect=AssertionError('import subprocess')), \
     mock.patch('subprocess.Popen', side_effect=AssertionError('import spawn')), \
     mock.patch.object(Path, 'mkdir', side_effect=AssertionError('import write')):
    spec = importlib.util.spec_from_file_location('quality_gate', PATH)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)


def tool_completion():
    # Synthetic unit fixture, not benchmark evidence.
    return {'usage': {'completion_tokens': 40, 'prompt_tokens': 120},
            'choices': [{'finish_reason': 'tool_calls',
                         'message': {'role': 'assistant', 'content': None,
                                     'tool_calls': [{'index': 0, 'id': 'qfc-abc123',
                                                     'type': 'function',
                                                     'function': {'name': 'get_weather',
                                                                  'arguments': '{"city": "Ghent", "unit": "celsius"}'}}]}}],
            'timings': {'prompt_ms': 100.0, 'predicted_n': 40, 'predicted_ms': 2000.0,
                        'predicted_per_second': 20.0}}


class ToolCallGateTests(unittest.TestCase):
    def test_validate_plan_gate_requires_target_only(self):
        for bad in (4, 8, 16):
            with self.assertRaises(ValueError):
                harness.validate_plan(bad, 'ps-iq2xxs', 'toolcall')
        harness.validate_plan(0, 'ps-iq2xxs', 'toolcall')
        harness.validate_plan(0, 'reference', 'toolcall')
        with self.assertRaises(ValueError):
            harness.validate_plan(0, 'bogus', 'toolcall')
        with self.assertRaises(ValueError):
            harness.validate_plan(4, 'ps-iq2xxs', 'needle')

    def test_payload_binds_tools_and_forced_choice(self):
        payload = harness.toolcall_payload()
        self.assertEqual(payload['tool_choice'], 'required')
        self.assertEqual(payload['max_tokens'], harness.TOOLCALL_TOKENS)
        self.assertEqual(payload['temperature'], 0.0)
        self.assertEqual(payload['seed'], 42)
        self.assertIs(payload['stream'], False)
        self.assertNotIn('speculative.n_max', payload)
        self.assertIs(payload['cache_prompt'], False)
        tools = payload['tools']
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]['type'], 'function')
        function = tools[0]['function']
        self.assertEqual(function['name'], 'get_weather')
        schema = function['parameters']
        self.assertEqual(schema['type'], 'object')
        self.assertEqual(sorted(schema['properties']), ['city', 'unit'])
        self.assertIn('city', schema['required'])
        self.assertEqual(schema['properties']['city']['type'], 'string')
        self.assertEqual(schema['properties']['unit']['enum'], ['celsius', 'fahrenheit'])

    def test_payload_prompt_has_task_and_city(self):
        content = harness.toolcall_payload()['messages'][0]['content']
        self.assertIn('current weather', content.lower())
        self.assertIn('Ghent', content)
        self.assertIn('get_weather', content)

    def test_gate_environment_disables_draft(self):
    # kept minimal: environment reuse is covered by throughput tests; gate forces DRAFT=0
        env = harness.clean_environment({'PATH': '/cpu-only'}, 's.scope', 0, 'ps-iq2xxs')
        self.assertEqual(env['DRAFT'], '0')

    def test_round2_appends_tool_result_and_drops_tools(self):
        payload = harness.toolcall_payload()
        second = harness.toolcall_round2_payload(payload, tool_call_id='qfc-abc123',
                                                 arguments='{"city": "Ghent", "unit": "celsius"}',
                                                 tool_result_json={'city': 'Ghent', 'unit': 'celsius',
                                                                   'temperature_c': 17.5, 'condition': 'cloudy'})
        roles = [m['role'] for m in second['messages']]
        self.assertEqual(roles, ['user', 'assistant', 'tool'])
        assistant = second['messages'][1]
        self.assertEqual(assistant['role'], 'assistant')
        self.assertEqual(assistant['tool_calls'][0]['id'], 'qfc-abc123')
        self.assertEqual(assistant['tool_calls'][0]['type'], 'function')
        self.assertEqual(assistant['tool_calls'][0]['function']['name'], 'get_weather')
        self.assertEqual(assistant['tool_calls'][0]['function']['arguments'],
                         '{"city": "Ghent", "unit": "celsius"}')
        tool = second['messages'][2]
        self.assertEqual(tool['role'], 'tool')
        self.assertEqual(tool['tool_call_id'], 'qfc-abc123')
        self.assertIn('17.5', tool['content'])
        self.assertNotIn('tools', second)
        self.assertNotIn('tool_choice', second)
        self.assertEqual(second['max_tokens'], harness.TOOLCALL_TOKENS_ROUND2)
        self.assertEqual(second['temperature'], 0.0)
        self.assertEqual(second['seed'], 42)
        # Source payload must stay unmutated.
        self.assertEqual(len(payload['messages']), 1)

    def test_round2_rejects_mismatched_inputs(self):
        payload = harness.toolcall_payload()
        for kwargs in (
            {'tool_call_id': '', 'arguments': '{}', 'tool_result_json': {}},
            {'tool_call_id': 'x', 'arguments': 'not json', 'tool_result_json': {}},
            {'tool_call_id': 'x', 'arguments': '{"city": "Ghent"}', 'tool_result_json': {'city': 'Ghent', 'temperature_c': 1.0}},
            {'tool_call_id': 'x', 'arguments': '{"city": "Antwerp"}', 'tool_result_json': {'city': 'Ghent', 'temperature_c': 1.0}},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    harness.toolcall_round2_payload(payload, **kwargs)

    def test_validate_accepts_exact_schema_tool_call(self):
        result = harness.validate_toolcall(tool_completion())
        self.assertEqual(result['tool_name'], 'get_weather')
        self.assertEqual(result['arguments']['city'], 'Ghent')
        self.assertEqual(result['arguments']['unit'], 'celsius')
        self.assertEqual(result['tool_call_id'], 'qfc-abc123')

    def test_validate_rejects_bad_completions(self):
        mutations = [
            lambda d: d['choices'][0].update(finish_reason='stop'),
            lambda d: d['choices'][0].update(finish_reason='length'),
            lambda d: d['choices'][0]['message'].update(tool_calls=[]),
            lambda d: d['choices'][0]['message'].pop('tool_calls'),
            lambda d: d['choices'][0]['message']['tool_calls'][0]['function'].update(name='other_tool'),
            lambda d: d['choices'][0]['message']['tool_calls'][0]['function'].update(
                arguments='{"city": "Antwerp", "unit": "celsius"}'),
            lambda d: d['choices'][0]['message']['tool_calls'][0]['function'].update(
                arguments='{"city": "Ghent"}'),
            lambda d: d['choices'][0]['message']['tool_calls'][0]['function'].update(
                arguments='{"city": "Ghent", "unit": "kelvin"}'),
            lambda d: d['choices'][0]['message']['tool_calls'][0]['function'].update(
                arguments='{"city": "Ghent", "unit": "celsius", "extra": 1}'),
            lambda d: d['choices'][0]['message']['tool_calls'][0]['function'].update(arguments='not json'),
            lambda d: d['choices'][0]['message']['tool_calls'][0]['function'].update(arguments=''),
            lambda d: d['choices'][0]['message']['tool_calls'][0].update(id=''),
            lambda d: d['choices'][0]['message']['tool_calls'][0].update(type='custom'),
            lambda d: d['choices'][0]['message']['tool_calls'].append(
                json.loads(json.dumps(d['choices'][0]['message']['tool_calls'][0])) | {'id': 'qfc-second'}),
            lambda d: d.update(error='failed'),
            lambda d: d.pop('usage'),
            lambda d: d['usage'].update(completion_tokens='40'),
            lambda d: d.pop('timings'),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = tool_completion()
                mutate(data)
                with self.assertRaises(RuntimeError):
                    harness.validate_toolcall(data)

    def test_validate_round2_requires_clean_stop(self):
        base = {'usage': {'completion_tokens': 64},
                'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Ghent is 17.5 degrees.'}}],
                'timings': {'prompt_ms': 1.0, 'predicted_n': 64, 'predicted_ms': 1.0, 'predicted_per_second': 1.0}}
        result = harness.validate_round2(base)
        self.assertIn('17.5', result['content'])
        for mutate in [
            lambda d: d['choices'][0].update(finish_reason='length'),
            lambda d: d['choices'][0]['message'].update(content='   '),
            lambda d: d['choices'][0]['message'].update(content=None),
            lambda d: d['choices'][0]['message'].update(tool_calls=[]),
            lambda d: d['choices'].clear(),
            lambda d: d.update(error='x'),
        ]:
            with self.subTest(mutate=mutate):
                data = json.loads(json.dumps(base))
                mutate(data)
                with self.assertRaises(RuntimeError):
                    harness.validate_round2(data)

    def test_gate_verdict_logic(self):
        ok = {'round1': {'tool_name': 'get_weather', 'arguments': {'city': 'Ghent', 'unit': 'celsius'},
                         'tool_call_id': 'qfc-abc123'},
              'round2': {'content': 'Ghent is 17.5 degrees.'}}
        both = json.loads(json.dumps(ok))
        self.assertEqual(harness.gate_verdict({'ps-iq2xxs': json.loads(json.dumps(ok)),
                                               'reference': json.loads(json.dumps(ok))})['gate'], 'PASS')
        mutations = [
            lambda v: v['ps-iq2xxs'].pop('round2'),
            lambda v: v['ps-iq2xxs'].pop('round1'),
            lambda v: v['ps-iq2xxs']['round1'].update(tool_name='wrong'),
            lambda v: v['reference']['round1'].update(tool_name='wrong'),
            lambda v: v['reference'].pop('round2'),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = {'ps-iq2xxs': json.loads(json.dumps(ok)),
                        'reference': json.loads(json.dumps(ok))}
                mutate(data)
                self.assertEqual(harness.gate_verdict(data)['gate'], 'FAIL')

    def test_run_records_gate_mode_and_sequence(self):
        run = harness.Run(Path('/unused'), 0, 'ps-iq2xxs', gate='toolcall')
        self.assertEqual(run.summary['gate'], 'toolcall')
        self.assertEqual(run.sequence, ('round1', 'round2'))
        self.assertEqual(run.summary['sequence'], ['round1', 'round2'])
        for bad in ('needle', 'bogus', None, 4):
            with self.assertRaises(ValueError):
                harness.Run(Path('/unused'), 0, 'ps-iq2xxs', gate=bad)

    def test_run_default_mode_unchanged(self):
        run = harness.Run(Path('/unused'), 4, 'ps-iq2xxs')
        self.assertEqual(run.summary['gate'], 'throughput')
        self.assertEqual(run.sequence, (4, 4))
        with self.assertRaises(ValueError):
            harness.Run(Path('/unused'), 4, 'ps-iq2xxs', gate='toolcall')

    def test_main_gate_flag_reaches_run(self):
        run = harness.Run(Path('/unused'), 0, 'ps-iq2xxs', gate='toolcall')
        with tempfile.TemporaryDirectory(dir=PATH.parent) as directory:
            run.out = Path(directory)
            with mock.patch.object(harness, 'Run', return_value=run), \
                 mock.patch.object(Path, 'mkdir'), \
                 mock.patch.object(harness, 'signal'), \
                 mock.patch.object(run, 'execute', side_effect=RuntimeError('injected gate failure')), \
                 mock.patch.object(run, 'command', return_value='active'):
                self.assertEqual(harness.main(['--gate', 'toolcall', '--checkpoint', 'ps-iq2xxs',
                                               '--startup-n-max', '0']), 1)
            summary = json.loads((run.out / 'summary.json').read_text())
            self.assertIn('injected gate failure', summary['error'])
            self.assertEqual(summary['gate'], 'toolcall')
            self.assertEqual(summary['checkpoint'], 'ps-iq2xxs')

    def test_main_gate_rejects_draft_mode_and_unknown_gate(self):
        for argv, code in ((['--gate', 'toolcall'], 2), (['--gate', 'toolcall', '--startup-n-max', '4'], 2)):
            with self.subTest(argv=argv):
                # validate_plan failure is reported via return code, not SystemExit
                self.assertEqual(harness.main(argv), code)
        with self.assertRaises(SystemExit) as caught:
            harness.main(['--gate', 'bogus'])
        self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
