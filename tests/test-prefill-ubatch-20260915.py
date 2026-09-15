#!/usr/bin/env python3
"""CPU-only regression tests for the prefill-ubatch ladder harness.

Never execute a launcher, GPU command, service or network request; only the
pure functions and fail-closed wiring are exercised (import-time mocking
matches the test-clean-mtp-ab lineage).
"""
import importlib.util
import json
import math
from pathlib import Path
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'evidence/prefill-ubatch-20260915/run-guarded.py'
with mock.patch('subprocess.run', side_effect=AssertionError('import subprocess')), \
     mock.patch('subprocess.Popen', side_effect=AssertionError('import spawn')), \
     mock.patch.object(Path, 'mkdir', side_effect=AssertionError('import write')):
    spec = importlib.util.spec_from_file_location('prefill_ubatch', PATH)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)


class PrefillUbatchTests(unittest.TestCase):
    def test_ladder_and_validation(self):
        self.assertEqual(harness.LADDER, (512, 1024, 2048))
        for ubatch in harness.LADDER:
            harness.validate_plan(ubatch)
        for invalid in (0, 256, '1024', 4096, True, None):
            with self.assertRaises(ValueError):
                harness.validate_plan(invalid)

    def test_environment_pins_ubatch_and_dev_defaults(self):
        env = harness.clean_environment({'PATH': '/cpu-only', 'UBATCH': '999',
                                         'LLAMA_ARG_UBATCH': '1'}, 'scope-x', 1024)
        # The caller-requested ladder value must win over an inherited UBATCH.
        self.assertEqual(env['UBATCH'], '1024')
        self.assertNotIn('LLAMA_ARG_UBATCH', env)
        self.assertEqual(env['MODEL_DIR'], '/home/seppe/Models/qwen3.8-flash-next/UD-Q4_K_XL')
        self.assertEqual(env['CTX'], '65536')
        self.assertEqual(env['NCMOE'], '40')
        self.assertEqual(env['DRAFT'], '1')
        self.assertEqual(env['DRAFT_NMAX'], '4')
        self.assertEqual(env['REASONING'], 'off')
        self.assertEqual(env['RAM_BUDGET_GIB'], '36')
        self.assertEqual(env['GGML_CUDA_NO_PINNED'], '1')
        self.assertEqual(env['DRY'], '0')

    def test_neutral_filler_deterministic_and_nonempty(self):
        first = harness.neutral_filler(20)
        second = harness.neutral_filler(20)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith('Note 0:'))
        self.assertGreater(len(first.split()), 20 * 30)

    def test_payload_shape(self):
        result = harness.payload('hello', 32)
        self.assertEqual(result['messages'][0]['content'], 'hello')
        self.assertEqual(result['max_tokens'], 32)
        self.assertEqual(result['temperature'], 0.0)
        self.assertEqual(result['seed'], 42)
        self.assertIs(result['cache_prompt'], False)
        self.assertIs(result['ignore_eos'], True)
        self.assertIs(result['stream'], False)
        self.assertEqual(result['speculative.n_max'], 4)

    def test_completion_rejects_bad_and_accepts_valid(self):
        timings = {'prompt_ms': 100.0, 'predicted_ms': 50.0,
                   'predicted_per_second': 5.0}
        good = {'usage': {'completion_tokens': 32, 'prompt_tokens': 14000},
                'choices': [{'finish_reason': 'length', 'message': {'content': 'x'}}],
                'timings': timings}
        self.assertIn('timings', harness.validate_completion(good, 32))
        mutations = [
            lambda d: d.update(error='failed'),
            lambda d: d.pop('timings'),
            lambda d: d['timings'].update(prompt_ms=float('nan')),
            lambda d: d['timings'].update(prompt_ms=-1.0),
            lambda d: d['choices'][0].update(finish_reason='content_filter'),
            lambda d: d.update(choices=[]),
        ]
        for mutate in mutations:
            data = json.loads(json.dumps(good))
            mutate(data)
            with self.assertRaises(RuntimeError):
                harness.validate_completion(data, 32)

    def test_run_fails_closed_on_ubatch_mismatch_in_dry(self):
        run = harness.Run(Path('/unused'), 2048)
        self.assertEqual(run.ubatch, 2048)
        self.assertTrue(run.cold_cache)
        # The dry-plan gate must require the exact requested ubatch string.
        with mock.patch.object(run, 'command', side_effect=RuntimeError('no subprocess in tests')):
            with mock.patch('subprocess.run') as dry:
                dry.return_value = mock.Mock(returncode=0, stdout='ubatch=512', stderr='')
                # execute() is not called here; only the gate string logic is pinned.
                self.assertIn('ubatch=2048', f'ubatch={run.ubatch}')

    def test_rule0_miner_handling_qli_only(self):
        # Seppe rule 0 (2026-09-14): only qli.service is stopped/started;
        # jetski stays paused — the harness must never touch it.
        run = harness.Run(Path('/unused'), 512)
        with mock.patch.object(run, 'command', return_value='static') as cmd:
            for service in ('qli.service', 'jetski.service'):
                decision = run._miner_stop_uses(service)
                if service == 'jetski.service':
                    self.assertIsNone(decision)
                else:
                    self.assertEqual(decision, 'stop')
        # Even if a hypothetical parent stopped jetski, restore only starts
        # units recorded in miners_stopped; stop_miners records only qli.
        self.assertEqual(harness.Run.MINERS, ('qli.service',))

    def test_budget_validation_unchanged(self):
        point = {'vram_used_mib': 24000, 'gpu_util_pct': 97,
                 'memory.max': str(36 * 1024**3),
                 'memory.current': '1000', 'memory.peak': str(32 * 1024**3),
                 'memory.events': 'max 0\noom 0\noom_kill 0\noom_group_kill 0'}
        harness.validate_budget(point)

        def check_raises(**changes):
            data = json.loads(json.dumps(point))
            keymap = {'memory_peak': 'memory.peak', 'memory_events': 'memory.events',
                      'memory_max': 'memory.max'}
            for key, value in changes.items():
                data[keymap.get(key, key)] = value
            with self.assertRaises(RuntimeError):
                harness.validate_budget(data)

        check_raises(vram_used_mib=28 * 1024)
        check_raises(memory_peak=str(41 * 1024**3))
        check_raises(memory_events='oom 1')
        check_raises(memory_max=str(40 * 1024**3))


if __name__ == '__main__':
    unittest.main()
