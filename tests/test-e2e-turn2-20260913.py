#!/usr/bin/env python3
"""CPU-only second-turn regression: reuse the artifact made by this task."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('e2e_turn2', ROOT / 'evidence/hermes-e2e-20260911/run-e2e.py')
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Turn2Tests(unittest.TestCase):
    def execute(self, kind, stdout, rc=0):
        run = mock.Mock()
        out = Path('/unused')
        env = {'HERMES_HOME': '/isolated'}
        result = subprocess.CompletedProcess([], rc, stdout, '')
        with mock.patch.object(runner, 'run_client', return_value=result) as client:
            checks = runner.run_second_turn(run, out, env, kind)
        args, kwargs = client.call_args
        self.assertIs(args[0], run)
        self.assertEqual(args[1][:6], ['hermes', 'chat', '--yolo', '--resume', 'latest', '-q'])
        self.assertEqual(kwargs['out'], out)
        self.assertEqual(kwargs['env'], env)
        self.assertEqual(kwargs['prefix'], 'hermes-turn2')
        self.assertEqual(kwargs['timeout'], runner.REQUEST_SECONDS)
        return checks, args[1][-1]

    def test_code_reexecutes_code_artifact(self):
        checks, task = self.execute('code', 'Query: retry\nterminal python3 /tmp/e2e-codegate.py\ncode-gate-ok-7391\n')
        self.assertIn('python3 /tmp/e2e-codegate.py', task)
        self.assertNotIn('hermes-e2e-marker.txt', task)
        self.assertEqual(checks['status'], 'completed')
        self.assertTrue(checks['multi_turn'])

    def test_marker_still_reads_marker_artifact(self):
        checks, task = self.execute('marker', 'Query: retry\nterminal cat /tmp/hermes-e2e-marker.txt\nhermes-e2e-ok\n')
        self.assertEqual(task, runner.TASK_TURN2)
        self.assertEqual(checks['status'], 'completed')

    def test_real_marker_transcript_with_multiline_query(self):
        path = ROOT / 'evidence/hermes-e2e-20260911/run-20260911T164123-e2e/hermes-turn2-stdout.txt'
        checks, _ = self.execute('marker', path.read_text())
        self.assertEqual(checks['status'], 'completed')

    def test_full_generated_query_without_execution_fails(self):
        for kind in ('code', 'marker'):
            with self.subTest(kind=kind):
                _, task = self.execute(kind, '')
                checks, _ = self.execute(kind, 'Query: ' + task + '\nInitializing agent...\n')
                self.assertFalse(checks['turn2_ran_shell_tool'])
                self.assertFalse(checks['turn2_stdout_has_marker'])
                self.assertEqual(checks['status'], 'client_failed')

    def test_wrong_artifact_cannot_pass_code_turn(self):
        checks, _ = self.execute('code', 'Query: retry\nterminal cat /tmp/hermes-e2e-marker.txt\nhermes-e2e-ok\n')
        self.assertEqual(checks['status'], 'client_failed')

    def test_echoed_command_is_not_execution_evidence(self):
        for kind, command, marker in [('code', 'python3 /tmp/e2e-codegate.py', 'code-gate-ok-7391'),
                                      ('marker', 'cat /tmp/hermes-e2e-marker.txt', 'hermes-e2e-ok')]:
            with self.subTest(kind=kind):
                checks, _ = self.execute(kind, f'Query: {command}\n{marker}\n')
                self.assertFalse(checks['turn2_ran_shell_tool'])
                self.assertEqual(checks['status'], 'client_failed')

    def test_failed_client_cannot_pass(self):
        checks, _ = self.execute('code', 'Query: retry\npython3 /tmp/e2e-codegate.py\ncode-gate-ok-7391\n', rc=1)
        self.assertEqual(checks['status'], 'client_failed')

    def test_unknown_task_rejected_before_client(self):
        with mock.patch.object(runner, 'run_client') as client:
            with self.assertRaises(ValueError):
                runner.run_second_turn(mock.Mock(), Path('/unused'), {}, 'typo')
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
