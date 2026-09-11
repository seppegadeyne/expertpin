#!/usr/bin/env python3
"""CPU-only tests: client handling of a mid-stream SSE error frame.

The server contract (examples/server/server.cpp, verified 2026-09-11):
- an error BEFORE the first chunk is sent as a normal non-stream HTTP error
  response (line ~1215, matching OpenAI behavior);
- an error AFTER the first chunk is sent as one SSE event
  `data: {"error": {...}}` and the stream then ENDS WITHOUT the [DONE]
  sentinel (lines ~1264-1267: server_sent_event(sink, {"error": ...});
  sink.done(); return false).

A physical mid-stream failure cannot be triggered deterministically
without injecting a fault into the engine (out of scope for a test), so
this documents the client-side expectation with synthetic frames.
"""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'evidence/hermes-e2e-20260911/run-contract.py'
with mock.patch('subprocess.run', side_effect=AssertionError('import subprocess')), \
     mock.patch('subprocess.Popen', side_effect=AssertionError('import spawn')), \
     mock.patch.object(Path, 'mkdir', side_effect=AssertionError('import write')):
    spec = importlib.util.spec_from_file_location('contract_sse', PATH)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)


def chunk(delta, finish=None):
    return {'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}


class MidStreamErrorTests(unittest.TestCase):
    def test_parser_rejects_error_frame_fail_closed(self):
        # A mid-stream error frame is NOT a chat.completion.chunk; the
        # validator must reject it (the client surfaces the error upstream
        # rather than treating the stream as a valid completion).
        events = [chunk({'role': 'assistant'}), chunk({'content': 'par'}),
                  {'error': {'code': 500, 'message': 'decode failed', 'type': 'server_error'}}]
        with self.assertRaises(RuntimeError):
            harness.contract_check_streaming(events, expect_tool_calls=False)

    def test_error_frame_shape_documented(self):
        frame = {'error': {'code': 500, 'message': 'decode failed', 'type': 'server_error'}}
        self.assertIsInstance(frame['error'], dict)
        self.assertIn('message', frame['error'])

    def test_truncated_stream_without_done_or_finish_fails(self):
        # After a mid-stream error the stream ends with NO [DONE] and no
        # terminal finish_reason — a client must treat that as an error,
        # never as a completed answer.
        events = [chunk({'role': 'assistant'}), chunk({'content': 'par'})]
        with self.assertRaises(RuntimeError):
            harness.contract_check_streaming(events, expect_tool_calls=False)


if __name__ == '__main__':
    unittest.main()
