#!/usr/bin/env python3
"""CPU-only regression tests; never execute a launcher, GPU command or service."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'evidence/clean-mtp-ab/run-guarded.py'
with mock.patch('subprocess.run', side_effect=AssertionError('import subprocess')), \
     mock.patch('subprocess.Popen', side_effect=AssertionError('import spawn')), \
     mock.patch.object(Path, 'mkdir', side_effect=AssertionError('import write')):
    spec = importlib.util.spec_from_file_location('clean_mtp', PATH)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)


def completion():
    # Deliberately synthetic unit fixture, not benchmark evidence.
    return {'usage': {'completion_tokens': 256},
            'choices': [{'finish_reason': 'length', 'message': {'reasoning_content': 'fixture text'}}],
            'timings': {'predicted_n': 256, 'prompt_ms': 20, 'predicted_ms': 5000,
                        'predicted_per_second': 51.2, 'draft_n': 30, 'draft_n_accepted': 12,
                        'draft_by_depth': [{'depth': 1, 'draft_n': 20, 'draft_n_accepted': 10},
                                           {'depth': 2, 'draft_n': 10, 'draft_n_accepted': 2}]}}


class CleanMTPTests(unittest.TestCase):
    def test_sanitizes_every_prefix_and_graph_disable(self):
        inherited = {'PATH': '/cpu-only', 'FORCE': '1', 'MANIFEST': 'bad', 'RESIDENT': '99',
                     'EXPERT_STATS_FILE': 'bad', 'EXPERT_CACHE_SIM_MIB': '999',
                     'LLAMA_ARG_SPEC_AUTOTUNE': '1'}
        for prefix in harness.TRACE_PREFIXES:
            for suffix in ('', '_FILE', '_REQUEST_ONLY', '_UNKNOWN_FUTURE_FLAG'):
                inherited[prefix + suffix] = '1'
        inherited.update({k: '0' for k in harness.GRAPH_DISABLE})
        before = dict(inherited)
        env = harness.clean_environment(inherited, 'expertpin-test-test.scope')
        self.assertEqual(inherited, before)
        self.assertFalse(any(k.startswith(harness.TRACE_PREFIXES) for k in env))
        self.assertFalse(set(harness.GRAPH_DISABLE) & set(env))
        self.assertNotIn('LLAMA_ARG_SPEC_AUTOTUNE', env)
        expected = dict(GGML_CUDA_NO_PINNED='1', DRAFT='1', DRAFT_NMAX='4', CTX='8192',
                        NCMOE='36', RAM_BUDGET_GIB='36', CACHE_RAM_MIB='512', FORCE='0',
                        MANIFEST='', RESIDENT='0', EXPERT_CACHE_SIM_MIB='0', EXPERT_STATS_FILE='')
        for key, value in expected.items():
            self.assertEqual(env[key], value)
        self.assertEqual(env['PATH'], '/cpu-only')

    def test_sequence_only_nmax_changes(self):
        self.assertEqual(harness.DEPTHS, (4, 8, 16))
        baseline = harness.payload(4)
        for nmax in harness.DEPTHS:
            candidate = harness.payload(nmax)
            self.assertEqual(candidate.pop('speculative.n_max'), nmax)
            self.assertEqual(candidate, {k: v for k, v in baseline.items() if k != 'speculative.n_max'})
        self.assertEqual(baseline['max_tokens'], 256)
        self.assertEqual(baseline['temperature'], 0)
        self.assertEqual(baseline['seed'], 42)
        self.assertIs(baseline['cache_prompt'], False)
        self.assertIs(baseline['ignore_eos'], True)
        self.assertIs(baseline['stream'], False)
        baseline['messages'][0]['content'] = 'mutation'
        self.assertEqual(harness.payload(4)['messages'][0]['content'], harness.PROMPT)

    def test_matching_startup_and_request_depth(self):
        for nmax in (4, 8, 16):
            harness.validate_plan(nmax)
            run = harness.Run(Path('/unused'), nmax)
            self.assertEqual(run.sequence, (nmax, nmax))
            self.assertEqual(run.summary['sequence'], [nmax, nmax])
        for invalid in (True, 3, 32):
            with self.assertRaises(ValueError):
                harness.validate_plan(invalid)

    def test_valid_completion_retains_full_timings(self):
        data = completion()
        self.assertIs(harness.validate_completion(data, 4)['timings'], data['timings'])
        self.assertEqual(harness.validate_completion(data, 4)['acceptance_fraction'], 0.4)

    def test_rejects_short_long_eos_empty_and_bad_timings(self):
        mutations = [lambda d: d['usage'].update(completion_tokens=255),
                     lambda d: d['usage'].update(completion_tokens=257),
                     lambda d: d['usage'].update(completion_tokens=True),
                     lambda d: d['choices'][0].update(finish_reason='stop'),
                     lambda d: d['choices'][0].update(message={'content': ' '}),
                     lambda d: d.update(error='failed'),
                     lambda d: d.pop('timings'),
                     lambda d: d['timings'].update(predicted_n=255),
                     lambda d: d['timings'].update(predicted_ms=float('nan')),
                     lambda d: d['timings'].update(predicted_per_second=float('inf')),
                     lambda d: d['timings'].update(draft_by_depth=[]),
                     lambda d: d['timings'].update(draft_n=31),
                     lambda d: d['timings'].update(draft_n_accepted=11),
                     lambda d: d['timings']['draft_by_depth'][0].update(depth=5),
                     lambda d: d['timings']['draft_by_depth'][0].update(draft_n_accepted=21)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = completion()
                mutate(data)
                with self.assertRaises(RuntimeError):
                    harness.validate_completion(data, 4)

    def test_budget_limits_and_oom(self):
        point = {'vram_used_mib': 27 * 1024, 'gpu_util_pct': 99,
                 'memory.max': str(36 * 1024**3), 'memory.current': '100',
                 'memory.peak': str(40 * 1024**3), 'memory.events': 'max 0\noom 0\noom_kill 0'}
        harness.validate_budget(point)
        for key, value in [('vram_used_mib', 28 * 1024), ('memory.max', str(40 * 1024**3)),
                           ('memory.peak', str(40 * 1024**3 + 1)), ('memory.events', 'oom_kill 1')]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                harness.validate_budget(dict(point, **{key: value}))

    def test_terminate_escalates_and_reaps(self):
        child = mock.Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('fixture', 5), 0]
        harness.terminate(child)
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count, 2)

    def test_cleanup_exact_scope_and_restore_despite_failures(self):
        run = harness.Run(Path('/unused'), 16)
        run.scope_launched = run.prepared = True
        run.request, run.proc = mock.Mock(), mock.Mock()
        calls = []
        def command(args, name, **kwargs):
            calls.append((args, name))
            if name in ('scope-stop', 'qli-start'):
                raise RuntimeError('injected failure')
            if name in ('scope-state', 'scope-final-state'):
                return 'active' if name == 'scope-state' else 'inactive'
            return 'active'
        with mock.patch.object(run, 'command', side_effect=command), \
             mock.patch.object(run, 'own_scope_empty', return_value=True), \
             mock.patch.object(harness, 'terminate') as terminate:
            run.cleanup()
        self.assertEqual(terminate.call_args_list, [mock.call(run.request), mock.call(run.proc)])
        names = [name for args, name in calls]
        self.assertIn('scope-kill', names)
        self.assertIn('scope-final-state', names)
        self.assertIn('qli-active', names)
        self.assertIn('host-restore', names)
        self.assertIn('headless-active', names)
        self.assertTrue(run.summary['scope_stopped_verified'])
        self.assertEqual(run.summary['status'], 'cleanup_failed')
        for args, name in calls:
            if name.startswith('scope-'):
                self.assertIn(run.scope, args)
                self.assertFalse(any('*' in item for item in args))

    def test_spawn_defers_signal_until_handle_registered(self):
        run = harness.Run(Path('/unused'), 4)
        run.scope_launched = True
        def command(args, name, **kwargs):
            return 'failed' if 'state' in name else 'active'
        with mock.patch.object(run, 'command', side_effect=command) as cmd, \
             mock.patch.object(run, 'own_scope_empty', side_effect=[False, False, True]):
            run.cleanup()
        self.assertTrue(run.summary['scope_stopped_verified'])
        names = [call.args[1] for call in cmd.call_args_list]
        self.assertIn('scope-kill', names)
        self.assertIn('scope-final-kill', names)

    def test_cgroup_population_readback(self):
        with tempfile.TemporaryDirectory(dir=PATH.parent) as directory:
            run = harness.Run(Path('/unused'), 4)
            run.cgroup_path = Path(directory)
            events = run.cgroup_path / 'cgroup.events'
            with mock.patch.object(run, 'command', return_value=''):
                events.write_text('populated 1\nfrozen 0\n')
                self.assertFalse(run.own_scope_empty())
                events.write_text('populated 0\nfrozen 0\n')
                self.assertTrue(run.own_scope_empty())

    def test_signal_registration(self):
        run = harness.Run(Path('/unused'), 16)
        child = mock.Mock()
        def spawn(*args, **kwargs):
            run.interrupt(15, None)
            return child
        with mock.patch.object(harness.subprocess, 'Popen', side_effect=spawn):
            run.proc = run.spawn(['cpu-mock'])
        self.assertIs(run.proc, child)
        with self.assertRaises(TimeoutError):
            run.spawned()

    def test_main_failure_runs_finally_and_saves_failure(self):
        with tempfile.TemporaryDirectory(dir=PATH.parent) as directory:
            run = harness.Run(Path(directory), 16)
            with mock.patch.object(harness, 'Run', return_value=run), \
                 mock.patch.object(Path, 'mkdir'), \
                 mock.patch.object(harness, 'signal'), \
                 mock.patch.object(run, 'execute', side_effect=TimeoutError('injected work deadline')), \
                 mock.patch.object(run, 'command', return_value='active') as command:
                self.assertEqual(harness.main(['--startup-n-max', '16']), 1)
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertIn('injected work deadline', summary['error'])
            self.assertEqual(summary['qli_status'], 'active')
            self.assertEqual(summary['cleanup_errors'], [])
            self.assertIn(mock.call(['systemctl', '--user', 'start', 'qli.service'], 'qli-start'), command.call_args_list)

    def test_deadline_blocks_before_budget_commands(self):
        run = harness.Run(Path('/unused'), 16)
        with mock.patch.object(harness.time, 'monotonic', return_value=run.start + harness.WORK_SECONDS), \
             mock.patch.object(run, 'command', side_effect=AssertionError('late GPU command')):
            with self.assertRaises(TimeoutError):
                run.sample()

    def test_uuid_scope_and_work_bound(self):
        a, b = harness.Run(Path('/unused'), 16), harness.Run(Path('/unused'), 16)
        self.assertNotEqual(a.scope, b.scope)
        self.assertRegex(a.scope, r'^expertpin-test-[0-9a-f]{32}\.scope$')
        self.assertLessEqual(harness.WORK_SECONDS, 660)
        self.assertLessEqual(harness.REQUEST_SECONDS, 180)


if __name__ == '__main__':
    unittest.main()
