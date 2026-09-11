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
        for bad in ('bogus', None, 4):
            with self.assertRaises(ValueError):
                harness.Run(Path('/unused'), 0, 'ps-iq2xxs', gate=bad)
        # 'needle' at target-only is a valid gate mode now (separate sequence).
        self.assertEqual(harness.Run(Path('/unused'), 0, 'ps-iq2xxs', gate='needle').sequence,
                         ('end', 'middle', 'start'))

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


def json_copy(value):
    import json
    return json.loads(json.dumps(value))


class NeedleGateTests(unittest.TestCase):
    def test_haystack_deterministic_and_code_free(self):
        first = harness.build_haystack(60)
        second = harness.build_haystack(60)
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 60)
        needle_paragraph = harness.NEEDLE_PARAGRAPH
        self.assertIn('7391', needle_paragraph)
        for paragraph in first:
            self.assertNotIn('7391', paragraph)
            self.assertNotIn('Aurora', paragraph)

    def test_needle_positions_insert_once(self):
        paragraphs = harness.build_haystack(30)
        for position, fraction in harness.NEEDLE_POSITIONS.items():
            with_paragraphs, index = harness.insert_needle(list(paragraphs), position)
            self.assertEqual(with_paragraphs[index], harness.NEEDLE_PARAGRAPH)
            self.assertEqual(with_paragraphs.count(harness.NEEDLE_PARAGRAPH), 1)
            self.assertEqual(len(with_paragraphs), len(paragraphs) + 1)
            self.assertAlmostEqual(index / len(with_paragraphs), fraction, delta=0.12)

    def test_insert_needle_rejects_unknown_position(self):
        with self.assertRaises(ValueError):
            harness.insert_needle(list(harness.build_haystack(10)), 'bogus')

    def test_size_haystack_bounded_and_converging(self):
        calls = []
        def post(endpoint, body):
            calls.append(endpoint)
            # ~10 tokens per paragraph synthetic tokenizer
            return {'tokens': list(range(10 * len(body['content'].split('\n\n'))))}
        paragraphs = harness.size_haystack(post, target_tokens=200)
        self.assertTrue(180 <= len(paragraphs) * 10 <= 220, len(paragraphs))
        self.assertLessEqual(len(calls), 6)
        for bad_target in (0, -5):
            with self.assertRaises(ValueError):
                harness.size_haystack(post, target_tokens=bad_target)

    def test_needle_payload_shape(self):
        paragraphs, _ = harness.insert_needle(harness.build_haystack(5), 'middle')
        payload = harness.needle_payload(paragraphs)
        self.assertEqual(len(payload['messages']), 1)
        content = payload['messages'][0]['content']
        self.assertIn(paragraphs[0], content)
        self.assertIn(harness.NEEDLE_PARAGRAPH, content)
        self.assertIn(harness.NEEDLE_QUESTION, content)
        self.assertEqual(payload['max_tokens'], harness.NEEDLE_TOKENS)
        self.assertEqual(payload['temperature'], 0.0)
        self.assertEqual(payload['seed'], 42)
        self.assertIs(payload['stream'], False)
        self.assertIs(payload['cache_prompt'], True)
        self.assertNotIn('tools', payload)

    def test_validate_needle_accepts_and_rejects(self):
        good = {'usage': {'completion_tokens': 8},
                'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '7391'}}],
                'timings': {'prompt_ms': 10.0, 'predicted_ms': 1.0, 'predicted_per_second': 8.0}}
        result = harness.validate_needle(good)
        self.assertTrue(result['recalled'])
        self.assertEqual(result['content'], '7391')
        # A wrong answer is a valid completion: recall fails, verdict catches it.
        wrong = json_copy(good)
        wrong['choices'][0]['message']['content'] = 'I do not know.'
        self.assertFalse(harness.validate_needle(wrong)['recalled'])
        for mutate in [
            lambda d: d['choices'][0]['message'].update(content=''),
            lambda d: d['choices'][0]['message'].update(content=None),
            lambda d: d['choices'][0].update(finish_reason='length'),
            lambda d: d['choices'][0]['message'].update(tool_calls=[]),
            lambda d: d.update(error='x'),
        ]:
            with self.subTest(mutate=mutate):
                data = json_copy(good)
                mutate(data)
                with self.assertRaises(RuntimeError):
                    harness.validate_needle(data)

    def test_needle_verdict_requires_all_positions(self):
        def result(recalled):
            return {p: {'recalled': recalled, 'content': '7391'} for p in ('start', 'middle', 'end')}
        self.assertEqual(harness.needle_verdict(result(True))['gate'], 'PASS')
        self.assertEqual(harness.needle_verdict(result(False))['gate'], 'FAIL')
        mixed = result(True)
        mixed['middle'] = {'recalled': False, 'content': 'no'}
        verdict = harness.needle_verdict(mixed)
        self.assertEqual(verdict['gate'], 'FAIL')
        self.assertIn('middle', verdict['failures'][0])

    def test_run_needle_mode_and_plan_rules(self):
        run = harness.Run(Path('/unused'), 0, 'ps-iq2xxs', gate='needle')
        self.assertEqual(run.sequence, ('end', 'middle', 'start'))
        self.assertEqual(run.summary['gate'], 'needle')
        harness.validate_plan(0, 'reference', 'needle')
        for bad_nmax in (4, 8):
            with self.assertRaises(ValueError):
                harness.validate_plan(bad_nmax, 'ps-iq2xxs', 'needle')

    def test_needle_budgets_within_gpu_protocol(self):
        self.assertLessEqual(harness.NEEDLE_REQUEST_SECONDS, 300)
        self.assertLessEqual(harness.NEEDLE_WORK_SECONDS, 960)
        self.assertLess(harness.NEEDLE_WORK_SECONDS, 1800)


if __name__ == '__main__':
    unittest.main()
