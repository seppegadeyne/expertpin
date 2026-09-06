"""CPU-only fault injection executes the actual harness finally AST with fake services."""
import ast
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock

SOURCE = Path(__file__).resolve().parents[1] / 'evidence/expert-offset-trace/run-guarded.py'
TREE = ast.parse(SOURCE.read_text())
FINAL = next(node.finalbody for node in TREE.body if isinstance(node, ast.Try))
CODE = compile(ast.Module(body=FINAL, type_ignores=[]), str(SOURCE), 'exec')


class GuardTests(unittest.TestCase):
    def test_completion_requires_real_bounded_decode(self):
        function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'validate_completion')
        ns = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), ns)
        validate = ns['validate_completion']
        for count, reason, text, valid in ((32, 'length', 'answer', True),
                (1, 'stop', '', False), (31, 'length', 'answer', False),
                (32.0, 'length', 'answer', False), (32, 'stop', 'answer', False),
                (32, 'length', '   ', False)):
            data = {'usage': {'completion_tokens': count}, 'choices': [
                {'finish_reason': reason, 'message': {'content': text}}]}
            if valid:
                self.assertEqual(validate(data)['completion_tokens'], 32)
            else:
                with self.assertRaises(RuntimeError): validate(data)
        with self.assertRaises(RuntimeError): validate({})
        data = {'usage': {'completion_tokens': 32}, 'choices': [
            {'finish_reason': 'length', 'message': {'reasoning_content': 'reasoning'}}]}
        self.assertEqual(validate(data)['completion_tokens'], 32)

    def test_tier_a_boundaries(self):
        # Compile only the pure guard: importing the harness would start services.
        function = next(node for node in TREE.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'host_guard')
        ns = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), ns)
        guard = ns['host_guard']
        gib = 1024**3
        for util, mem, expected in ((0, 34*gib, True), (4, 34*gib, True),
                                    (5, 50*gib, False), (0, 34*gib-1, False)):
            with self.subTest(util=util, mem=mem):
                self.assertIs(guard(util, mem), expected)

    def test_sample_checks_configured_cap_and_vram_boundary(self):
        function = next(node for node in TREE.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'sample')
        for cap, vram, expected in ((36, 28671, None), (40, 100, RuntimeError),
                                    (36, 28672, RuntimeError)):
            with self.subTest(cap=cap, vram=vram):
                base = MagicMock()
                values = {'memory.current': '123', 'memory.peak': '456',
                          'memory.max': str(cap * 1024**3),
                          'memory.swap.current': '0', 'memory.events': 'oom 0'}
                base.__truediv__.side_effect = lambda key: SimpleNamespace(read_text=lambda: values[key])
                path = MagicMock()
                path.__truediv__.return_value = base
                ns = dict(gpu=lambda: (0, vram, 0), time=SimpleNamespace(monotonic=lambda: 1),
                          start=0, available=lambda: 50*1024**3, scope_launched=True,
                          subprocess=SimpleNamespace(check_output=lambda *a, **k: '/test/owned.scope'),
                          scope='owned.scope', Path=lambda *a: path, samples=[], OUT=MagicMock(),
                          json=json, proc=SimpleNamespace(poll=lambda: None), RAM_BUDGET_GIB=36)
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), ns)
                if expected:
                    with self.assertRaises(expected):
                        ns['sample']()
                else:
                    ns['sample']()
                    self.assertEqual(len(ns['samples']), 1)


class CleanupTests(unittest.TestCase):
    def test_signal_during_spawn_preserves_child_for_cleanup(self):
        body = next(node.body for node in TREE.body if isinstance(node, ast.Try))
        first = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'scope_launched' for t in n.targets))
        last = next(i for i in range(first, len(body)) if isinstance(body[i], ast.If)
                    and ast.unparse(body[i].test) == 'pending_signal is not None')
        handler = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'interrupt')
        for failure in (False, True):
            child = Mock()
            ns = dict(launching=False, pending_signal=None, proc=None, scope_launched=False,
                      launch=[], env={}, server_log=None)
            exec(compile(ast.Module(body=[handler], type_ignores=[]), str(SOURCE), 'exec'), ns)
            def spawn(*a, **k):
                self.assertTrue(ns['scope_launched'])
                ns['interrupt'](signal.SIGTERM, None)
                if failure:
                    raise OSError('spawn failed')
                return child
            ns['subprocess'] = SimpleNamespace(Popen=spawn, STDOUT=-2)
            with self.assertRaises(OSError if failure else TimeoutError):
                exec(compile(ast.Module(body=body[first:last+1], type_ignores=[]), str(SOURCE), 'exec'), ns)
            self.assertFalse(ns['launching'])
            self.assertIs(ns['proc'], None if failure else child)
            self.assertTrue(ns['scope_launched'])

    def test_final_scope_recheck_precedes_mandatory_miner_restart(self):
        for stuck, kill_fails in ((False, False), (True, False), (True, True)):
            calls = []
            def command(args, name, check=True):
                calls.append(name)
                if name == 'scope-final-state':
                    return 'active' if stuck else 'inactive'
                if kill_fails and name == 'scope-final-kill':
                    raise OSError('kill failed')
                return 'inactive' if name == 'scope-state' else 'active'
            ns = dict(signal=Mock(), subprocess=subprocess, command=command, summary={},
                      request=None, proc=None, scope_launched=True, scope='owned.scope',
                      server_log=None, datetime=datetime, prepared=True, PREP='fake',
                      OUT=MagicMock(), json=json, time=SimpleNamespace(sleep=lambda _: None))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(CODE, ns)
            self.assertLess(calls.index('scope-final-state'), calls.index('qli-start'))
            self.assertEqual(ns['summary'].get('scope_stopped_verified', False), not stuck)
            self.assertEqual(bool(ns['summary']['cleanup_errors']), stuck)
            self.assertIn('host-restore', calls)

    def test_service_failures_never_skip_later_restore(self):
        for kind in (OSError('log write error'), subprocess.TimeoutExpired('injected', 60), KeyboardInterrupt()):
            for failure in ('scope-stop', 'scope-state', 'qli-start', 'qli-active', 'host-restore', 'headless-active', 'qli-journal'):
                with self.subTest(failure=failure, kind=type(kind).__name__):
                    calls = []

                    def command(args, name, check=True):
                        calls.append(name)
                        if name == failure:
                            raise kind
                        return 'inactive' if name in ('scope-state', 'scope-final-state') else 'active'

                    ns = dict(signal=Mock(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, SIG_IGN=signal.SIG_IGN),
                              subprocess=subprocess, command=command, summary={}, request=None, proc=None,
                              scope_launched=True, scope='expertpin-test-fake.scope', server_log=None,
                              datetime=datetime, prepared=True, PREP='fake-prep', OUT=MagicMock(), json=json)
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(CODE, ns)
                    for needed in ('qli-start', 'qli-active', 'host-restore', 'headless-active', 'qli-journal'):
                        self.assertIn(needed, calls)
                    self.assertEqual(ns['summary']['status'], 'cleanup_failed')
                    self.assertTrue(ns['summary']['cleanup_errors'])

    def test_children_are_reaped_even_after_terminate_timeout(self):
        calls = []
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('child', 10), 0]
        ns = dict(signal=Mock(), subprocess=subprocess,
                  command=lambda args, name, check=True: calls.append(name) or 'active',
                  summary={}, request=child, proc=None, scope_launched=False, server_log=None,
                  datetime=datetime, prepared=True, PREP='fake-prep', OUT=MagicMock(), json=json)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(CODE, ns)
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count, 2)
        self.assertIn('host-restore', calls)


if __name__ == '__main__':
    unittest.main()
