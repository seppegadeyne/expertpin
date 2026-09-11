#!/usr/bin/env python3
"""CPU-only regression tests for the 64K context slice; no GPU/launcher/service."""
import importlib.util
import os
import unittest
from unittest import mock
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'evidence/context64k-20260911/run-guarded.py'


def load(env=None):
    saved = {k: os.environ.get(k) for k in ('SERVER_CTX', 'NEEDLE_TARGET_TOKENS',
                                            'NEEDLE_REQUEST_SECONDS', 'NEEDLE_WORK_SECONDS',
                                            'MODEL_LOAD_DEADLINE')}
    try:
        for key, value in (env or {}).items():
            os.environ[key] = value
        with mock.patch('subprocess.run', side_effect=AssertionError('import subprocess')), \
             mock.patch('subprocess.Popen', side_effect=AssertionError('import spawn')), \
             mock.patch.object(Path, 'mkdir', side_effect=AssertionError('import write')):
            spec = importlib.util.spec_from_file_location('ctx64k', PATH)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ContextSliceTests(unittest.TestCase):
    def test_defaults_unchanged(self):
        h = load()
        self.assertEqual(h.SERVER_CTX, '8192')
        self.assertEqual(h.NEEDLE_TARGET_TOKENS, 2048)
        self.assertEqual(h.NEEDLE_REQUEST_SECONDS, 300)
        self.assertEqual(h.NEEDLE_WORK_SECONDS, 960)
        self.assertEqual(h.MODEL_LOAD_DEADLINE, 360)

    def test_ctx_override_reaches_launcher_env(self):
        h = load({'SERVER_CTX': '65536'})
        env = h.clean_environment({'PATH': '/cpu'}, 's.scope', 0, 'ps-iq2xxs')
        self.assertEqual(env['CTX'], '65536')
        self.assertEqual(h.SERVER_CTX, '65536')

    def test_needle_target_override_scales_haystack(self):
        h = load({'NEEDLE_TARGET_TOKENS': '4096'})
        self.assertEqual(h.NEEDLE_TARGET_TOKENS, 4096)
        calls = []
        def post(endpoint, body):
            calls.append(1)
            # ~30 tokens/paragraph synthetic rate scales to the new target
            n = len(body['content'].split('\n\n'))
            return {'tokens': list(range(30 * n))}
        paragraphs = h.size_haystack(post, target_tokens=h.NEEDLE_TARGET_TOKENS)
        self.assertTrue(0.9 * 4096 <= 30 * len(paragraphs) <= 1.1 * 4096, len(paragraphs))
        self.assertLessEqual(len(calls), 8)

    def test_extended_budgets_within_mandate(self):
        h = load({'NEEDLE_REQUEST_SECONDS': '2400', 'NEEDLE_WORK_SECONDS': '3000',
                  'MODEL_LOAD_DEADLINE': '900'})
        run = h.Run(Path('/unused'), 0, 'ps-iq2xxs', gate='needle')
        self.assertEqual(run.request_seconds, 2400)
        self.assertEqual(run.work_seconds, 3000)
        self.assertEqual(h.MODEL_LOAD_DEADLINE, 900)

    def test_invalid_env_fails_closed_at_import(self):
        with self.assertRaises(ValueError):
            load({'NEEDLE_WORK_SECONDS': 'abc'})

    def test_needle_functions_inherited_unchanged(self):
        h = load()
        paragraphs, index = h.insert_needle(h.build_haystack(20), 'middle')
        self.assertEqual(paragraphs[index], h.NEEDLE_PARAGRAPH)
        self.assertIn('7391', h.NEEDLE_PARAGRAPH)


if __name__ == '__main__':
    unittest.main()
