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


class CleanupTests(unittest.TestCase):
    def test_service_failures_never_skip_later_restore(self):
        for kind in (OSError('log write error'), subprocess.TimeoutExpired('injected', 60), KeyboardInterrupt()):
            for failure in ('scope-stop', 'scope-state', 'qli-start', 'qli-active', 'host-restore', 'headless-active', 'qli-journal'):
                with self.subTest(failure=failure, kind=type(kind).__name__):
                    calls = []

                    def command(args, name, check=True):
                        calls.append(name)
                        if name == failure:
                            raise kind
                        return 'inactive' if name == 'scope-state' else 'active'

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
