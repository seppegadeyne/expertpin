#!/usr/bin/env python3
"""CPU-only synthetic GGUF/CSV fixtures, always inside the repository; no model assets."""
import copy
import csv
import ctypes
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('trace_bw', ROOT / 'scripts/bench-trace-bandwidth.py')
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
bw = probe.bw


class TraceBandwidthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT, prefix='.trace-bw-test-')
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.models = self.directory / 'models'
        self.models.mkdir()
        self.csv = self.directory / 'joined.csv'
        self.rows = []
        for quant in probe.QUANTS:
            name = f'q{quant}.gguf'
            (self.models / name).write_bytes(b'GGUF\x03\x00\x00\x00' + bytes(range(256)) * 256)
            for expert in range(5):
                size = 1234 + quant
                self.rows.append(dict(seq=len(self.rows), weight_entry=quant, epoch=1, ids_rows=1,
                                      tensor=f'blk.0.ffn_up_exps.weight', ggml_type=quant, expert=expert,
                                      stride=size, relative_offset=expert * size, bytes=size,
                                      shadow_hit=0, shadow_bypass=0, context='target', request_id=27,
                                      seq_id=0, phase='decode', pos_min=63, pos_max=63,
                                      shard=name, absolute_offset=8192 + expert * 8192 + 123))
        self.rows.append(dict(self.rows[0], seq=len(self.rows)))  # exact repeated demand
        self.rows.append(dict(self.rows[1], seq=len(self.rows), phase='prefill'))
        self.write_trace()

    def write_trace(self, footer=None):
        text = io.StringIO(newline='')
        writer = csv.DictWriter(text, fieldnames=probe.COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(self.rows)
        text.write(footer if footer is not None else f'# end written={len(self.rows)} dropped=0 error=0\n')
        self.csv.write_text(text.getvalue())

    def test_filter_exact_dedup_and_hash(self):
        result = probe.read_trace(self.csv)
        self.assertEqual(len(result['slices']), 15)
        self.assertEqual(result['slices'][0]['trace_occurrences'], 2)
        self.assertEqual(result['slices'][0]['first_csv_line'], 2)
        self.assertEqual(result['provenance']['filtered_rows'], 16)
        self.assertEqual(result['provenance']['csv_rows'], 17)
        self.assertEqual(result['provenance']['joined_sha256'], hashlib.sha256(self.csv.read_bytes()).hexdigest())
        self.assertEqual(len(probe.read_trace(self.csv, 'prefill')['slices']), 1)
        self.assertEqual(probe.read_trace(self.csv, 'all')['provenance']['filtered_rows'], 17)

    def test_seeded_stratification_caps_and_coverage(self):
        slices = probe.read_trace(self.csv)['slices']
        plan = probe.sample_plan(slices, 8000, 17)
        self.assertEqual(plan, probe.sample_plan(list(reversed(slices)), 8000, 17))
        self.assertNotEqual(plan, probe.sample_plan(slices, 8000, 18))
        self.assertEqual(set(plan), set(probe.QUANTS))
        flat = [x for group in plan.values() for x in group]
        self.assertLessEqual(sum(x['bytes'] for x in flat), 8000)
        self.assertEqual(len(flat), len({(x['shard'], x['absolute_offset'], x['bytes']) for x in flat}))
        with patch.object(probe, 'MAX_SLICES', 3):
            self.assertEqual(sum(map(len, probe.sample_plan(slices, 80000, 17).values())), 3)
        for budget in (0, -1, bw.GIB + 1, 1, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                probe.sample_plan(slices, budget, 17)
        result = probe.probe(self.csv, None, 8000, plan_only=True)
        self.assertEqual(result['coverage']['duplicate_rows'], 1)
        self.assertLess(result['coverage']['sampled_byte_fraction'], 1)
        self.assertFalse(result['full_trace_replay'])
        self.assertEqual(result['coverage']['sampled_bytes'], sum(x['coverage']['sampled_bytes'] for x in result['quant_results']))

    def test_plan_only_never_touches_assets(self):
        with patch.object(os, 'open', side_effect=AssertionError('asset open')), \
             patch.object(os, 'pread', side_effect=AssertionError('asset read')), \
             patch.object(os, 'preadv', side_effect=AssertionError('asset read')):
            result = probe.probe(self.csv, self.directory / 'nonexistent', 8000, plan_only=True)
        self.assertEqual(result['mode'], 'plan_only')
        self.assertFalse(result['model_files_verified'])
        self.assertIsNone(result['total'])
        self.assertEqual(result['provenance']['plan_sha256'], probe.probe(self.csv, None, 8000, plan_only=True)['provenance']['plan_sha256'])

    def test_malformed_csv_fails_closed(self):
        original = copy.deepcopy(self.rows)
        cases = [('bytes', '0'), ('absolute_offset', '-1'), ('absolute_offset', '+1'),
                 ('absolute_offset', '01'), ('absolute_offset', '١'), ('absolute_offset', str(1 << 63)),
                 ('ggml_type', '99'), ('relative_offset', '10'), ('stride', '10'),
                 ('phase', ''), ('pos_min', '70'), ('shadow_hit', '2'), ('seq', '1'),
                 ('shard', '../escape.gguf'), ('shard', '/absolute.gguf'), ('shard', 'sub/a.gguf'),
                 ('shard', 'a\\b.gguf'), ('shard', 'a.bin'), ('tensor', 'not-an-expert')]
        for key, value in cases:
            self.rows = copy.deepcopy(original)
            self.rows[0][key] = value
            self.write_trace()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                probe.read_trace(self.csv)
        self.rows = copy.deepcopy(original)
        self.rows[-2]['ggml_type'] = 21
        self.write_trace()
        with self.assertRaisesRegex(ValueError, 'conflicting quant'):
            probe.read_trace(self.csv)

    def test_footer_header_and_input_limits(self):
        for footer in ('', '# end written=17 dropped=0 error=0', '# end written=18 dropped=0 error=0\n',
                       '# end written=17 dropped=1 error=0\n', '# end written=17 dropped=0 error=1\n',
                       '# end written=17 dropped=0 error=0\njunk\n'):
            self.write_trace(footer)
            with self.subTest(footer=footer), self.assertRaises(ValueError):
                probe.read_trace(self.csv)
        self.write_trace()
        valid = self.csv.read_text()
        self.csv.write_text(valid.replace('absolute_offset\n', 'offset\n', 1))
        with self.assertRaises(ValueError):
            probe.read_trace(self.csv)
        for key, value in (('MAX_LINE', 10), ('MAX_ROWS', 3), ('MAX_TRACE_BYTES', 10), ('MAX_SHARDS', 2)):
            self.write_trace()
            with self.subTest(cap=key), patch.object(probe, key, value), self.assertRaises(ValueError):
                probe.read_trace(self.csv)
        self.rows = [dict(self.rows[0], phase='prefill')]
        self.write_trace()
        with self.assertRaisesRegex(ValueError, 'no slices'):
            probe.read_trace(self.csv)

    def test_page_union_keeps_holes_and_rounds_boundaries(self):
        page = os.sysconf('SC_PAGESIZE')
        slices = [dict(shard='a.gguf', absolute_offset=page + 123, bytes=page),
                  dict(shard='a.gguf', absolute_offset=2 * page + 40, bytes=1),
                  dict(shard='a.gguf', absolute_offset=5 * page + 1, bytes=2),
                  dict(shard='b.gguf', absolute_offset=page, bytes=1)]
        self.assertEqual(probe.page_ranges(slices), [dict(shard='a.gguf', offset=page, length=2 * page),
                                                    dict(shard='a.gguf', offset=5 * page, length=page),
                                                    dict(shard='b.gguf', offset=page, length=page)])

    def test_offset_mincore_nonfaulting_libc_contract(self):
        page = os.sysconf('SC_PAGESIZE')
        fake = Mock()
        fake.mmap.return_value, fake.munmap.return_value = 0x100000, 0
        def mincore(addr, size, vec):
            vec[:] = [1, 0]
            return 0
        fake.mincore.side_effect = mincore
        info = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=os.geteuid(), st_size=10 * page)
        with patch.object(ctypes, 'CDLL', return_value=fake), patch.object(os, 'fstat', return_value=info), \
             patch.object(bw, 'filesystem_uid', return_value=info.st_uid):
            result = bw.cache_snapshot(-1, page, offset=3 * page + 123)
        fake.mmap.assert_called_once_with(None, page + 123, 0, 1, -1, 3 * page)
        fake.munmap.assert_called_once_with(0x100000, page + 123)
        self.assertEqual(result['pages'], 2)
        self.assertEqual(result['resident_pages'], 1)
        self.assertEqual((result['page_offset'], result['page_length']), (3 * page, 2 * page))

    def test_real_offset_snapshot_and_eof(self):
        with (self.models / 'q20.gguf').open('rb') as source:
            result = bw.cache_snapshot(source.fileno(), 4096, offset=8192 + 123)
            self.assertIsNone(result['error'], result)
            self.assertEqual(result['page_offset'], 8192)
            self.assertEqual(result['pages'], 2)
            for offset, length in ((-1, 1), (65540, 100), (8192, 0)):
                self.assertIsNone(bw.cache_snapshot(source.fileno(), length, offset)['pages'])

    def test_observation_brackets_dontneed_then_identical_reads(self):
        events = []
        item = dict(shard='a.gguf', absolute_offset=123, bytes=7)
        span = dict(shard='a.gguf', offset=0, length=4096)
        def event(name, value):
            events.append(name)
            return value
        def read(fd, views, offset):
            return event(('read', fd, offset, len(views[0])), len(views[0]))
        snapshot = dict(pages=1, resident_pages=0, state='all_nonresident', ranges=[])
        with patch.object(bw, 'drop_cache', side_effect=lambda *args: event(('drop', *args), None)), \
             patch.object(probe, 'observe_ranges', side_effect=lambda *args: event('cache', snapshot)), \
             patch.object(bw, 'storage_snapshot', side_effect=lambda: event('storage', dict(read_bytes=7, error=None))), \
             patch.object(probe.time, 'perf_counter', side_effect=lambda: event('clock', events.count('clock'))), \
             patch.object(os, 'preadv', side_effect=read):
            cold = probe.measured_pass({'a.gguf': 5}, [item], [span], bytearray(4), True)
            warm = probe.measured_pass({'a.gguf': 5}, [item], [span], bytearray(4), False)
        common = ['cache', 'storage', 'clock', ('read', 5, 123, 4), ('read', 5, 127, 3), 'clock', 'storage', 'cache']
        self.assertEqual(events, [('drop', 5, 0, 4096)] + common + common)
        self.assertEqual(cold['bytes'], warm['bytes'])
        self.assertEqual(cold['storage_read_bytes_delta'], 0)
        self.assertEqual(cold['seconds'], 1)
        self.assertIsNone(warm['ram_bytes_per_s'])

    def test_unknown_regressing_storage_and_advice_failure(self):
        item = dict(shard='a.gguf', absolute_offset=123, bytes=4)
        for a, b in ((None, 4), (4, None), (5, 4), (5, 9)):
            with patch.object(bw, 'drop_cache', side_effect=OSError('unsupported')), \
                 patch.object(probe, 'observe_ranges', return_value=dict(pages=None, resident_pages=None)), \
                 patch.object(bw, 'storage_snapshot', side_effect=[dict(read_bytes=a), dict(read_bytes=b)]), \
                 patch.object(os, 'preadv', return_value=4):
                result = probe.measured_pass({'a.gguf': 5}, [item], probe.page_ranges([item]), bytearray(4), True)
            self.assertTrue(result['dontneed']['errors'])
            self.assertEqual(result['storage_read_bytes_delta'], 4 if (a, b) == (5, 9) else None)
            total = probe.aggregate([result])
            self.assertIsNone(total['cache_before']['pages'])
            self.assertEqual(total['cache_before']['state'], 'unavailable')
            self.assertIsNone(total['nvme_bytes_per_s'])

    def test_actual_fixture_reads_matched_offsets_readonly_and_totals(self):
        reads, real = [], os.preadv
        def record(fd, views, offset):
            self.assertEqual(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE, os.O_RDONLY)
            count = real(fd, views, offset)
            self.assertEqual(bytes(views[0]), os.pread(fd, len(views[0]), offset))
            reads.append((os.fstat(fd).st_ino, offset, count))
            return count
        with patch.object(os, 'preadv', side_effect=record):
            result = probe.probe(self.csv, self.models, 8000)
        self.assertTrue(result['model_files_verified'])
        self.assertEqual(result['header_validation_bytes'], 24)
        cursor = 0
        for group in result['quant_results']:
            expected = [(result['model_files'][item['shard']]['ino'], item['absolute_offset'], item['bytes']) for item in group['offsets']]
            n = len(expected)
            self.assertEqual(reads[cursor:cursor + n], expected)
            self.assertEqual(reads[cursor + n:cursor + 2 * n], expected)
            cursor += 2 * n
            self.assertTrue(group['cold_requested']['dontneed']['requested'])
            self.assertFalse(group['warm_repeat']['dontneed']['requested'])
            for label in ('cold_requested', 'warm_repeat'):
                self.assertIsNone(group[label]['cache_after']['ranges'][0]['error'])
        for label in ('cold_requested', 'warm_repeat'):
            self.assertEqual(result['total'][label]['bytes'], result['coverage']['sampled_bytes'])
            self.assertEqual(result['total'][label]['seconds'], sum(g[label]['seconds'] for g in result['quant_results']))
            self.assertGreater(result['total'][label]['effective_buffered_gib_per_s'], 0)

    def test_invalid_bounds_even_excluded_phase_before_reads_and_fd_cleanup(self):
        self.rows[-1]['absolute_offset'] = 1000000
        self.write_trace()
        opened, closed, real_open, real_close = [], [], os.open, os.close
        def open_fd(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            opened.append(fd)
            return fd
        def close_fd(fd):
            closed.append(fd)
            real_close(fd)
        with patch.object(os, 'open', side_effect=open_fd), patch.object(os, 'close', side_effect=close_fd), \
             patch.object(os, 'pread', side_effect=AssertionError('read before bounds')), \
             patch.object(os, 'preadv', side_effect=AssertionError('read before bounds')):
            with self.assertRaisesRegex(ValueError, 'out of file bounds'):
                probe.probe(self.csv, self.models, 8000)
        self.assertCountEqual(opened, closed)
        self.assertLessEqual(len(opened), probe.MAX_SHARDS + 1)

    def test_symlink_fifo_unowned_magic_and_changed_file_rejected(self):
        shard = self.models / 'q20.gguf'
        original = shard.read_bytes()
        shard.unlink()
        outside = self.directory / 'outside.gguf'
        outside.write_bytes(original)
        shard.symlink_to(outside)
        with self.assertRaises(OSError):
            probe.probe(self.csv, self.models, 8000)
        shard.unlink()
        os.mkfifo(shard)
        with self.assertRaisesRegex(ValueError, 'owned regular'):
            probe.probe(self.csv, self.models, 8000)
        shard.unlink()
        shard.write_bytes(original)
        with patch.object(bw, 'filesystem_uid', return_value=os.geteuid() + 1), self.assertRaisesRegex(ValueError, 'owned regular'):
            probe.probe(self.csv, self.models, 8000)
        shard.write_bytes(b'NOPE' + original[4:])
        with self.assertRaisesRegex(ValueError, 'not a GGUF'):
            probe.probe(self.csv, self.models, 8000)
        shard.write_bytes(original)
        real_pass = probe.measured_pass
        def mutate(*args):
            result = real_pass(*args)
            with shard.open('ab') as f:
                f.write(b'x')
            return result
        with patch.object(probe, 'measured_pass', side_effect=mutate), self.assertRaisesRegex(ValueError, 'changed during probe'):
            probe.probe(self.csv, self.models, 8000)

    def test_short_read_aborts(self):
        with patch.object(os, 'preadv', return_value=1), self.assertRaisesRegex(OSError, 'short read'):
            probe.probe(self.csv, self.models, 8000)

    def test_cli_fixture_json_and_no_overwrite(self):
        output = self.directory / 'result.json'
        command = [sys.executable, str(ROOT / 'scripts/bench-trace-bandwidth.py'), '--joined', str(self.csv),
                   '--model-root', str(self.models), '--sample-mib', '1', '--json', str(output)]
        run = subprocess.run(command, text=True, capture_output=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result, json.loads(output.read_text()))
        self.assertEqual(result['phase'], 'decode')
        self.assertEqual(result['seed'], 17)
        self.assertFalse(result['full_trace_replay'])
        for extra in ([], ['--sample-mib', '1025'], ['--json', str(self.models / 'bad.json')]):
            failed = subprocess.run(command + extra, text=True, capture_output=True, timeout=20)
            self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(result, json.loads(output.read_text()))
        self.assertFalse((self.models / 'bad.json').exists())


if __name__ == '__main__':
    unittest.main()
