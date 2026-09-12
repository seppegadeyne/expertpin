#!/usr/bin/env python3
"""CPU-only regression tests: task-kind specific turn-1 criteria.

2026-09-12 incident (run-20260912T034046-e2e): the shared marker criteria
('hermes-e2e-ok' + printf) evaluated a CODE task — the agent had written
and run the script and answered exactly `code-gate-ok-7391` (12m35s, 4
messages), yet status was client_failed and the artifact probe was never
reached (it sits behind ok_turn1). Fourth harness trap in this lineage.
Tests use the REAL captured stdout of that run.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_STDOUT = ROOT / 'evidence/hermes-e2e-20260911/run-20260912T034046-e2e/hermes-stdout.txt'


def load_runner():
    spec = importlib.util.spec_from_file_location(
        'e2erun_turn1', ROOT / 'evidence/hermes-e2e-20260911/run-e2e.py')
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules['e2erun_turn1'] = module
    spec.loader.exec_module(module)
    return module


class Turn1CriteriaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stdout = RUN_STDOUT.read_text()
        cls.after_query = cls.stdout.split('\n', 1)[1] if '\n' in cls.stdout else ''

    def test_real_code_run_stdout_now_passes_criteria(self):
        m = load_runner()
        checks = m.turn1_checks('code', self.stdout, self.after_query)
        # Pre-fix (marker criteria on a code task): both False → client_failed.
        self.assertTrue(checks['hermes_stdout_has_marker'])
        self.assertTrue(checks['hermes_ran_shell_tool'])

    def test_marker_criteria_unchanged_for_marker_task(self):
        m = load_runner()
        # Shaped like the real PASS run 20260911T152411: shell-tool execution
        # line + printf + final marker answer after the query echo.
        marker_stdout = ('Query: Use the shell tool to run exactly this command: printf "hermes-e2e-ok" > /tmp/hermes-e2e-marker.txt && cat /tmp/hermes-e2e-marker.txt\n'
                         ' preparing terminal $ printf "hermes-e2e-ok" > /tmp/hermes-e2e-marker.txt && cat /tmp/hermes-e2e-marker.txt\n'
                         'hermes-e2e-ok')
        after = marker_stdout.split('\n', 1)[1]
        checks = m.turn1_checks('marker', marker_stdout, after)
        self.assertTrue(checks['hermes_stdout_has_marker'])
        self.assertTrue(checks['hermes_ran_shell_tool'])
        # Query-echo false positive stays excluded:
        echo_only = m.turn1_checks('marker', 'Query: hermes-e2e-ok\nsome other text', 'some other text')
        self.assertFalse(echo_only['hermes_stdout_has_marker'])

    def test_code_query_echo_false_positive_excluded(self):
        m = load_runner()
        # Task text echoed in the Query contains the expected output; a
        # transcript that never shows real execution must NOT pass.
        bogus = ('Query: write a script printing code-gate-ok-7391\n'
                 'I could not complete this.')
        after = bogus.split('\n', 1)[1]
        checks = m.turn1_checks('code', bogus, after)
        self.assertFalse(checks['hermes_stdout_has_marker'])   # output only in query echo
        self.assertFalse(checks['hermes_ran_shell_tool'])      # no script name in transcript


if __name__ == '__main__':
    unittest.main()
