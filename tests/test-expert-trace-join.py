#!/usr/bin/env python3
"""CPU-only join tests. All GGUF/CSV fixtures are SYNTHETIC, not measured traces.

No model load, real model payload access, GPU, or network. A small compile-only
check (when a C++ compiler is installed) compares supported layouts to ggml.
"""
import copy
import csv
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/join-expert-trace.py"
spec = importlib.util.spec_from_file_location("trace_join", SCRIPT)
assert spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader)
joiner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joiner)

# Deliberately requests a 64 MiB prefix, like the trusted local reader. The
# utility must feed it HEADER BYTES ONLY, independent of this read length.
READER = '''import struct

def read_shard(path):
    with open(path, "rb") as source:
        data = source.read(64 * 1024 * 1024)
    pos = 0
    def number(fmt):
        nonlocal pos
        value = struct.unpack_from(fmt, data, pos)[0]
        pos += struct.calcsize(fmt)
        return value
    def string():
        nonlocal pos
        size = number("<Q")
        result = data[pos:pos+size].decode("utf-8")
        pos += size
        return result
    assert number("<I") == 0x46554747
    assert number("<I") in (2, 3)
    count, metadata = number("<Q"), number("<Q")
    meta = {}
    for _ in range(metadata):
        key = string()
        assert number("<I") == 4
        meta[key] = number("<I")
    tensors = []
    for _ in range(count):
        name = string()
        dims = [number("<Q") for _ in range(number("<I"))]
        tensors.append((name, dims, number("<I"), number("<Q")))
    assert len(data) == pos, "reader was given padding or model payload"
    alignment = meta.get("general.alignment", 32)
    return meta, tensors, (pos + alignment - 1) // alignment * alignment
'''


def gguf_string(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def synthetic_gguf(path, tensors, data_size, alignment=32):
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1))
    header += gguf_string("general.alignment") + struct.pack("<II", 4, alignment)
    for name, dims, quant, offset in tensors:
        header += gguf_string(name) + struct.pack("<I", len(dims))
        header += struct.pack("<" + "Q" * len(dims), *dims)
        header += struct.pack("<IQ", quant, offset)
    start = (len(header) + alignment - 1) // alignment * alignment
    with path.open("wb") as output:
        output.write(header)
        output.truncate(start + data_size)  # sparse synthetic payload, never read
    return start, len(header)


class JoinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-expert-join-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.model = self.home / "model"
        self.model.mkdir()
        self.inventory_path = self.home / "inventory.json"
        self.trace = self.home / "trace.csv"
        self.output = self.home / "joined.csv"
        self.reader = self.home / "trusted-reader.py"
        self.reader.write_text(READER)
        self.a = "blk.0.ffn_down_exps.weight"
        self.b = 'blk.0.ffn_"gate",\n_exps.weight'
        self.c = "blk.1.ffn_up_exps.weight"
        # Synthetic small dimensions: strides 108, 220, 246 respectively.
        self.headers = {
            "a.gguf": [("dense.weight", [8], 0, 0),
                       (self.a, [64, 3, 4], 20, 32),
                       (self.b, [256, 2, 4], 21, 480)],
            "b.gguf": [(self.c, [256, 3, 3], 22, 0),
                       ("dense_tail.weight", [8], 0, 768)],
        }
        self.sizes = {"a.gguf": 1376, "b.gguf": 800}
        self.starts, self.header_sizes = {}, {}
        self.refresh_shards()
        self.inventory = {"kind": "local_gguf_header_inventory", "model_dir": "model", "shards": 2,
                          "header_reader": "/DO/NOT/EXECUTE/from/inventory.py", "tensors": []}
        for shard, tensors in self.headers.items():
            for index, (name, dims, quant, offset) in enumerate(tensors):
                if not joiner.expert_name(name):
                    continue
                end = tensors[index + 1][3] if index + 1 < len(tensors) else self.sizes[shard]
                self.inventory["tensors"].append(dict(name=name, dims=dims, ggml_type=quant, shard=shard,
                                                       relative_offset=offset, storage_span_bytes=end - offset))
        self.save_inventory()
        self.records = [dict(seq="0", weight_entry="4", epoch="9", ids_rows="3", tensor=self.a,
                             ggml_type="20", expert="3", stride="108", relative_offset="324", bytes="108",
                             shadow_hit="0", shadow_bypass="0"),
                        dict(seq="1", weight_entry="5", epoch="9", ids_rows="3", tensor=self.b,
                             ggml_type="21", expert="2", stride="220", relative_offset="440", bytes="220",
                             shadow_hit="1", shadow_bypass="0"),
                        dict(seq="2", weight_entry="6", epoch="10", ids_rows="1", tensor=self.c,
                             ggml_type="22", expert="2", stride="246", relative_offset="492", bytes="246",
                             shadow_hit="0", shadow_bypass="1")]
        self.save_trace()

    def refresh_shards(self):
        for shard, tensors in self.headers.items():
            self.starts[shard], self.header_sizes[shard] = synthetic_gguf(self.model / shard, tensors, self.sizes[shard])

    def save_inventory(self):
        self.inventory_path.write_text(json.dumps(self.inventory), encoding="utf-8")

    def save_trace(self, records=None, columns=None, footer=None, newline="\n"):
        records = self.records if records is None else records
        columns = joiner.BASE_COLUMNS if columns is None else columns
        with self.trace.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator=newline)
            writer.writerow(columns)
            for row in records:
                writer.writerow([row.get(key, "") for key in columns])
            output.write(footer if footer is not None else f"# end written={len(records)} dropped=0 error=0{newline}")

    def run_join(self):
        return joiner.join(self.inventory_path, self.trace, self.output, self.reader)

    def assert_rejected(self, message=None):
        self.output.write_text("existing output must survive\n")
        with self.assertRaises((ValueError, OSError, csv.Error)) as caught:
            self.run_join()
        if message is not None:
            self.assertIn(message, str(caught.exception))
        self.assertEqual(self.output.read_text(), "existing output must survive\n")
        self.assertEqual(list(self.home.glob(".joined.csv.*.tmp")), [])

    def output_rows(self):
        with self.output.open(encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
        self.assertEqual(rows[-1], ["# end written=3 dropped=0 error=0"])
        return rows[0], rows[1:-1]

    def test_absolute_offsets_multiple_shards_types_and_quoted_multiline_names(self):
        self.assertEqual(self.run_join(), 3)
        columns, rows = self.output_rows()
        self.assertEqual(columns, joiner.BASE_COLUMNS + ["shard", "absolute_offset"])
        expected = [("a.gguf", self.starts["a.gguf"] + 32 + 324),
                    ("a.gguf", self.starts["a.gguf"] + 480 + 440),
                    ("b.gguf", self.starts["b.gguf"] + 492)]
        for source, actual, (shard, absolute) in zip(self.records, rows, expected):
            self.assertEqual(actual[:-2], [source[key] for key in joiner.BASE_COLUMNS])
            self.assertEqual(actual[-2:], [shard, str(absolute)])

    def test_scope_tags_preserved_without_inference(self):
        for row in self.records:
            row.update(context="main", request_id="42", seq_id="-1", phase="prefill", pos_min="0", pos_max="31")
        self.records[1].update(context="mtp", phase="decode", seq_id="2", pos_min="32", pos_max="32")
        self.records[2].update(context="unknown", request_id="0", phase="unknown", seq_id="-2", pos_min="-1", pos_max="-1")
        self.save_trace(columns=joiner.BASE_COLUMNS + joiner.TAG_COLUMNS, newline="\r\n")
        self.assertEqual(self.run_join(), 3)
        columns, rows = self.output_rows()
        self.assertEqual(columns, joiner.BASE_COLUMNS + joiner.TAG_COLUMNS + ["shard", "absolute_offset"])
        for source, actual in zip(self.records, rows):
            self.assertEqual(actual[:-2], [source[key] for key in columns[:-2]])

    def test_empty_but_complete_trace(self):
        self.save_trace(records=[])
        self.assertEqual(self.run_join(), 0)
        self.assertTrue(self.output.read_text().endswith("# end written=0 dropped=0 error=0\n"))

    def test_no_payload_or_alignment_padding_read_even_with_prefix_reader(self):
        real_open = Path.open
        reads = {}
        limits = {(self.model / shard).resolve(): length for shard, length in self.header_sizes.items()}

        class Guard:
            def __init__(guard, handle, path):
                guard.handle, guard.path = handle, path
            def __enter__(guard):
                return guard
            def __exit__(guard, *args):
                guard.handle.close()
            def read(guard, size=-1):
                self.assertGreaterEqual(size, 0, "unbounded model read")
                self.assertLessEqual(guard.handle.tell() + size, limits[guard.path], "payload read")
                data = guard.handle.read(size)
                reads[guard.path] = reads.get(guard.path, 0) + len(data)
                return data

        def guarded_open(path, mode="r", *args, **kwargs):
            handle = real_open(path, mode, *args, **kwargs)
            if path.resolve() in limits:
                self.assertIsInstance(handle, io.FileIO, "buffered model reads could read ahead into payload")
                return Guard(handle, path.resolve())
            return handle

        with patch.object(Path, "open", guarded_open):
            self.assertEqual(self.run_join(), 3)
        self.assertEqual(reads, limits)

    def test_header_scanner_metadata_arrays_and_bad_metadata(self):
        path = self.home / "metadata-only.gguf"
        values = [(0, b"\x01"), (1, b"\xff"), (2, struct.pack("<H", 2)),
                  (3, struct.pack("<h", -2)), (4, struct.pack("<I", 4)),
                  (5, struct.pack("<i", -4)), (6, struct.pack("<f", 1.25)),
                  (7, b"\x01"), (8, gguf_string("synthetic \u03bb")),
                  (9, struct.pack("<IQ", 8, 2) + gguf_string("a") + gguf_string("b")),
                  (9, struct.pack("<IQ3i", 5, 3, -1, 0, 1)),
                  (10, struct.pack("<Q", 10)), (11, struct.pack("<q", -10)),
                  (12, struct.pack("<d", 1.25))]
        header = b"GGUF" + struct.pack("<IQQ", 2, 0, len(values))
        for index, (kind, value) in enumerate(values):
            header += gguf_string(f"synthetic.{index}") + struct.pack("<I", kind) + value
        path.write_bytes(header + b"NOT HEADER: SYNTHETIC PAYLOAD")
        self.assertEqual(joiner.header_prefix(path), header)
        for kind, value in ((13, b""), (9, struct.pack("<IQ", 9, 1)),
                            (9, struct.pack("<IQ", 8, 1 << 40)),
                            (9, struct.pack("<IQ", 4, 1 << 40)),
                            (8, struct.pack("<Q", 1) + b"\xff"),
                            (8, struct.pack("<Q", 100))):
            with self.subTest(kind=kind, value=value):
                path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
                                 + gguf_string("bad") + struct.pack("<I", kind) + value)
                with self.assertRaises(ValueError):
                    joiner.header_prefix(path)
        entry = gguf_string("duplicate") + struct.pack("<II", 4, 32)
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + entry + entry)
        with self.assertRaisesRegex(ValueError, "duplicate GGUF metadata"):
            joiner.header_prefix(path)

    def test_target_model_stride_formulas(self):
        self.assertEqual(joiner.expert_stride([640, 2560, 512], 20), 921600)
        self.assertEqual(joiner.expert_stride([2560, 640, 512], 21), 704000)
        self.assertEqual(joiner.expert_stride([2560, 640, 512], 22), 524800)

    def test_layout_constants_match_actual_ggml_cpu_headers(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable for compile-only ggml layout check")
        source = self.home / "layout-check.cpp"
        source.write_text('''#define GGML_COMMON_DECL_C
#include "ggml.h"
#include "ggml-common.h"
static_assert(GGML_TYPE_IQ4_NL == 20 && GGML_TYPE_IQ3_S == 21 && GGML_TYPE_IQ2_S == 22, "type IDs");
static_assert(QK4_NL == 32 && QK_K == 256, "block elements");
static_assert(sizeof(block_iq4_nl) == 18, "IQ4_NL bytes");
static_assert(sizeof(block_iq3_s) == 110, "IQ3_S bytes");
static_assert(sizeof(block_iq2_s) == 82, "IQ2_S bytes");
''')
        result = subprocess.run([compiler, "-std=c++11", "-fsyntax-only", "-I", str(ROOT / "ggml/include"),
                                 "-I", str(ROOT / "ggml/src"), str(source)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(joiner.LAYOUTS, {20: (32, 18), 21: (256, 110), 22: (256, 82)})

    def test_invalid_runtime_numbers_layouts_and_flags(self):
        for key, value in [("seq", "1"), ("seq", "-1"), ("seq", "00"), ("seq", "0.0"),
                           ("weight_entry", " 4"), ("weight_entry", "+4"), ("epoch", str(1 << 64)),
                           ("ids_rows", "0"), ("ids_rows", "１"), ("tensor", "missing"), ("tensor", ""),
                           ("ggml_type", "21"), ("ggml_type", "60"), ("expert", "4"),
                           ("stride", "112"), ("stride", "0"), ("bytes", "107"), ("bytes", "109"),
                           ("relative_offset", "325"), ("relative_offset", "356"),
                           ("shadow_hit", "2"), ("shadow_bypass", "2")]:
            with self.subTest(key=key, value=value):
                records = copy.deepcopy(self.records)
                records[0][key] = value
                self.save_trace(records=records)
                self.assert_rejected()
        self.records[0].update(shadow_hit="1", shadow_bypass="1")
        self.save_trace()
        self.assert_rejected("shadow flags")

    def test_missing_bad_or_incomplete_footer_and_post_footer_data(self):
        for footer in ("", "# end written=3 dropped=0 error=0", "# end written=3 dropped=0 error=\n",
                       "# end written=3 dropped=1 error=0\n", "# end written=3 dropped=0 error=1\n",
                       "# end written=2 dropped=0 error=0\n", "# end written=4 dropped=0 error=0\n",
                       "# end written=03 dropped=0 error=0\n", "# end written=3 dropped=-1 error=0\n",
                       '"# end written=3 dropped=0 error=0"\n', "# end written=3 dropped=0 error=0 \n",
                       "# end written=3 dropped=0 error=0\n\n", "# end written=3 dropped=0 error=0\njunk\n",
                       "# end written=3 dropped=0 error=0\n# end written=3 dropped=0 error=0\n"):
            with self.subTest(footer=footer):
                self.save_trace(footer=footer)
                self.assert_rejected()

    def test_duplicate_or_gapped_sequence(self):
        for seq in ("0", "2"):
            with self.subTest(seq=seq):
                self.records[1]["seq"] = seq
                self.save_trace()
                self.assert_rejected("contiguous")

    def test_wrong_headers_or_partial_tags_rejected(self):
        for columns in ([], joiner.BASE_COLUMNS[:-1], joiner.BASE_COLUMNS + ["seq"],
                        joiner.BASE_COLUMNS + joiner.TAG_COLUMNS[:-1],
                        joiner.BASE_COLUMNS + ["unknown"], list(reversed(joiner.BASE_COLUMNS))):
            with self.subTest(columns=columns):
                self.save_trace(columns=columns)
                self.assert_rejected("CSV header")

    def test_malformed_csv_and_record_bounds(self):
        header = ",".join(joiner.BASE_COLUMNS) + "\n"
        for text in ("\n", "0,1\n", '0,1,1,1,"unterminated\n',
                     '0,1,1,1,"name"junk,20,0,108,0,108,0,0\n',
                     "x" * (joiner.MAX_RECORD_CHARS + 1) + "\n",
                     '0,1,1,1,"' + ("x\n" * joiner.MAX_RECORD_CHARS) + '",20,0,108,0,108,0,0\n'):
            with self.subTest(prefix=text[:40]):
                self.trace.write_text(header + text + "# end written=1 dropped=0 error=0\n")
                self.assert_rejected()

    def test_duplicate_inventory_tensor_or_json_key(self):
        self.inventory["tensors"].append(copy.deepcopy(self.inventory["tensors"][0]))
        self.save_inventory()
        self.assert_rejected("duplicate inventory tensor")
        self.inventory_path.write_text('{"kind": "a", "kind": "b"}')
        self.assert_rejected("duplicate JSON key")

    def test_inventory_schema_and_numbers_fail_closed(self):
        baseline = copy.deepcopy(self.inventory)
        for key, value in (("name", ""), ("name", None), ("dims", [64, 3]), ("dims", [64, 3, 0]),
                           ("dims", [64, 3, True]), ("dims", [33, 3, 4]), ("dims", [1 << 63, 1 << 63, 4]),
                           ("ggml_type", 0), ("ggml_type", "20"), ("shard", "../a.gguf"),
                           ("shard", str(self.model / "a.gguf")), ("relative_offset", -1),
                           ("relative_offset", True), ("storage_span_bytes", 1), ("storage_span_bytes", "448")):
            with self.subTest(key=key, value=value):
                self.inventory = copy.deepcopy(baseline)
                self.inventory["tensors"][0][key] = value
                self.save_inventory()
                self.assert_rejected()
        self.inventory = baseline
        del self.inventory["tensors"][0]["name"]
        self.save_inventory()
        self.assert_rejected("missing required fields")

    def test_declared_inventory_totals_checked(self):
        for key, value in (("shards", 3), ("shards", True), ("expert_tensors", 2),
                           ("expert_storage_span_bytes", 1), ("type_counts", {"20": 3})):
            with self.subTest(key=key):
                self.inventory[key] = value
                self.save_inventory()
                self.assert_rejected()
                del self.inventory[key]

    def test_fresh_header_mismatches_even_for_untraced_experts(self):
        self.save_trace(records=[])
        baseline = copy.deepcopy(self.inventory)
        for key, value in (("shard", "b.gguf"), ("dims", [64, 3, 3]), ("ggml_type", 21),
                           ("relative_offset", 64), ("storage_span_bytes", 480)):
            with self.subTest(key=key):
                self.inventory = copy.deepcopy(baseline)
                tensor = self.inventory["tensors"][0 if key != "ggml_type" else 2]
                tensor[key] = value
                if key == "ggml_type":
                    tensor["storage_span_bytes"] = 1024
                self.save_inventory()
                self.assert_rejected("mismatch")

    def test_missing_inventory_or_header_names(self):
        baseline = copy.deepcopy(self.inventory)
        self.inventory["tensors"].pop()
        self.save_inventory()
        self.assert_rejected("missing from inventory")
        self.inventory = baseline
        self.inventory["tensors"][0]["name"] = "blk.99.ffn_down_exps.weight"
        self.save_inventory()
        self.headers["a.gguf"][1] = ("not_an_expert", [64, 3, 4], 20, 32)
        self.refresh_shards()
        self.assert_rejected("missing from fresh headers")

    def test_duplicate_fresh_names_offsets_and_unaligned_offsets(self):
        baseline = copy.deepcopy(self.headers)
        for name, offset in ((self.a, 768), ("dense_tail.weight", 0), ("dense_tail.weight", 767)):
            with self.subTest(name=name, offset=offset):
                self.headers = copy.deepcopy(baseline)
                self.headers["b.gguf"][1] = (name, [8], 0, offset)
                self.refresh_shards()
                self.assert_rejected()

    def test_file_truncation_and_padding_span_mismatch(self):
        for size in (self.starts["a.gguf"] - 1, self.starts["a.gguf"] + 100,
                     self.starts["a.gguf"] + self.sizes["a.gguf"] - 1):
            with self.subTest(size=size):
                self.refresh_shards()
                with (self.model / "a.gguf").open("r+b") as output:
                    output.truncate(size)
                self.assert_rejected()

    def test_missing_shard(self):
        (self.model / "b.gguf").unlink()
        self.assert_rejected("shard count")

    def test_bad_header_magic_version_lengths_and_reader_start(self):
        for offset, data in ((0, b"NOPE"), (4, struct.pack("<I", 1)),
                             (24, struct.pack("<Q", 1 << 40))):
            with self.subTest(offset=offset):
                self.refresh_shards()
                with (self.model / "a.gguf").open("r+b") as output:
                    output.seek(offset)
                    output.write(data)
                self.assert_rejected()
        self.refresh_shards()
        self.reader.write_text(READER.replace("return meta, tensors, (pos", "return meta, tensors, 32 + (pos"))
        self.assert_rejected("data_start mismatch")

    def test_reader_required_and_inventory_reader_never_executed(self):
        self.assertEqual(self.run_join(), 3)  # invalid inventory reader path ignored
        self.reader.write_text("read_shard = None\n")
        self.assert_rejected("no callable read_shard")

    def test_output_must_not_alias_inputs_shards_or_hardlinks(self):
        for target in (self.trace, self.inventory_path, self.reader, self.model / "a.gguf"):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "aliases"):
                    joiner.join(self.inventory_path, self.trace, target, self.reader)
        os.link(self.trace, self.output)
        with self.assertRaisesRegex(ValueError, "aliases"):
            self.run_join()
        self.output.unlink()
        self.output.symlink_to(self.trace)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_join()

    def test_trace_or_shard_change_during_join_prevents_publication(self):
        original = joiner.stream_join
        for target in (self.trace, self.model / "a.gguf"):
            with self.subTest(target=target):
                self.refresh_shards()
                self.save_trace()
                def modify_after(*args):
                    result = original(*args)
                    with target.open("ab") as output:
                        output.write(b"x")
                    return result
                with patch.object(joiner, "stream_join", side_effect=modify_after):
                    self.assert_rejected("changed during join")

    def test_streaming_memory_does_not_scale_with_trace_rows(self):
        # Synthetic repeated logical observations; not a runtime benchmark.
        inventory, model, by_name = joiner.load_inventory(self.inventory_path)
        joiner.validate_headers(inventory, model, by_name, self.reader)
        with self.trace.open("w", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(joiner.BASE_COLUMNS)
            row = self.records[0].copy()
            for seq in range(20000):
                row["seq"] = str(seq)
                writer.writerow([row[key] for key in joiner.BASE_COLUMNS])
            output.write("# end written=20000 dropped=0 error=0\n")
        tracemalloc.start()
        try:
            with self.trace.open(newline="") as source, self.output.open("w", newline="") as output:
                self.assertEqual(joiner.stream_join(source, output, by_name), 20000)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 1024 * 1024, f"streaming allocation peak {peak} bytes")

    def test_cli_success_and_failure(self):
        command = [sys.executable, str(SCRIPT), "--inventory", str(self.inventory_path),
                   "--trace", str(self.trace), "--output", str(self.output), "--header-reader", str(self.reader)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("joined 3 logical expert slices", result.stderr)
        self.assertIn("not physical I/O", result.stderr)
        before = self.output.read_bytes()
        self.save_trace(footer="")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing completion footer", result.stderr)
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
