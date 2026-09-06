#!/usr/bin/env python3
"""Join a COMPLETE logical CPU expert trace to fresh GGUF file-layout offsets.

Usage: python3 scripts/join-expert-trace.py --inventory inventory.json \
    --trace runtime.csv --output joined.csv --header-reader /trusted/reader.py

The explicit, trusted Python reader must export read_shard(path), returning
(meta, [(name, dims, ggml_type, data_relative_offset), ...], data_start).
It is executed with runpy (NOT sandboxed). Its global open is supplied an
in-memory GGUF header, never tensor data; this supports the existing 64 MiB
prefix reader without actually reading a 64 MiB model prefix. Only GGUF v2/v3
little-endian headers, at most 64 MiB each, are accepted.

Inventory model_dir is relative to the inventory's directory if not absolute.
Every *.gguf shard and every blk.*_exps.* tensor is checked against the inventory.
Only contiguous ne[0],ne[1],ne[2] IQ4_NL/IQ3_S/IQ2_S layouts are supported;
unknown/repacked layouts fail closed. Alignment padding is NOT slice payload.

Output preserves all input columns, optionally including the complete appended
context,request_id,seq_id,phase,pos_min,pos_max tag group (opaque, not inferred),
and appends shard,absolute_offset. relative_offset remains TENSOR-relative;
absolute_offset = fresh data_start + inventory relative_offset + trace offset.
CSV records are streamed with a 64 Ki-character record cap; header/inventory
state is proportional to model metadata, not trace length. A temporary output
is atomically published only after a valid zero-drop/error footer and stat
rechecks. Existing output is left untouched on failure. No stdout output mode.

This proves logical layout compatibility only, NOT physical I/O, payload
identity, runtime/file content equivalence, cache residency, or throughput.
Stat checks detect ordinary edits, not adversarial same-metadata replacements.
"""
import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import re
import runpy
import stat
import struct
import sys
import tempfile


BASE_COLUMNS = "seq,weight_entry,epoch,ids_rows,tensor,ggml_type,expert,stride,relative_offset,bytes,shadow_hit,shadow_bypass".split(",")
TAG_COLUMNS = "context,request_id,seq_id,phase,pos_min,pos_max".split(",")
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_RECORD_CHARS = 64 * 1024
U64_MAX = (1 << 64) - 1
UINT = re.compile(r"(?:0|[1-9][0-9]*)\Z", re.ASCII)
FOOTER = re.compile(r"# end written=(0|[1-9][0-9]*) dropped=(0|[1-9][0-9]*) error=(0|[1-9][0-9]*)\Z", re.ASCII)
# ggml/include/ggml.h enum; ggml/src/ggml.c type_traits; ggml-common.h
# block_iq4_nl: half + 32/2; block_iq3_s: half + 13*(256/32) + 256/64;
# block_iq2_s: half + 256/4 + 256/16. All have row_meta_size == 0.
LAYOUTS = {20: (32, 18), 21: (256, 110), 22: (256, 82)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def uint(value, label, minimum=0):
    require(type(value) is int and minimum <= value <= U64_MAX,
            f"{label}: expected integer in [{minimum}, {U64_MAX}]")
    return value


def csv_uint(value, label, minimum=0):
    require(len(value) <= 20 and UINT.fullmatch(value), f"{label}: invalid unsigned integer {value!r}")
    return uint(int(value), label, minimum)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def signature(path):
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def expert_name(name):
    return name.startswith("blk.") and "_exps." in name


def expert_stride(dims, quant):
    require(isinstance(dims, (list, tuple)) and len(dims) == 3, "expert dims must have exactly three axes")
    for dim in dims:
        uint(dim, "expert dimension", 1)
    uint(quant, "ggml_type")
    require(quant in LAYOUTS, f"unsupported ggml_type {quant}; only 20, 21, 22 are supported")
    block, size = LAYOUTS[quant]
    require(dims[0] % block == 0, f"ne[0]={dims[0]} is not divisible by ggml block size {block}")
    stride = uint(dims[0] // block * size * dims[1], "derived expert stride", 1)
    uint(stride * dims[2], "derived tensor payload", 1)
    return stride


def header_prefix(path):
    """Read exactly the GGUF header, not alignment padding or tensor payload.

    Walk lengths/types rather than searching for a magic delimiter. Bound the
    entire header and reject bad UTF-8/types/versions before the trusted parser.
    """
    data = bytearray()
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                    7: 1, 10: 8, 11: 8, 12: 8}
    # Disable Python read-ahead: buffered small reads can otherwise consume
    # payload bytes after a short header despite requesting only header fields.
    with path.open("rb", buffering=0) as source:
        def take(size):
            require(size <= MAX_HEADER_BYTES - len(data), "GGUF header exceeds 64 MiB limit")
            chunk = source.read(size)
            require(len(chunk) == size, "truncated GGUF header")
            data.extend(chunk)
            return chunk

        def number(fmt):
            return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]

        def string():
            return take(number("<Q")).decode("utf-8", errors="strict")

        def value(kind):
            if kind in scalar_sizes:
                take(scalar_sizes[kind])
            elif kind == 8:
                string()
            elif kind == 9:
                element, count = number("<I"), number("<Q")
                require(element in scalar_sizes or element == 8, "invalid GGUF array element type")
                if element == 8:
                    require(count <= (MAX_HEADER_BYTES - len(data)) // 8, "GGUF array exceeds header limit")
                    for _ in range(count):
                        string()
                else:
                    take(count * scalar_sizes[element])
            else:
                raise ValueError(f"invalid GGUF metadata type {kind}")

        require(take(4) == b"GGUF", "not a little-endian GGUF file")
        require(number("<I") in (2, 3), "unsupported GGUF version")
        tensors, metadata = number("<Q"), number("<Q")
        require(tensors <= MAX_HEADER_BYTES // 24 and metadata <= MAX_HEADER_BYTES // 12,
                "GGUF counts exceed header limit")
        keys = set()
        for _ in range(metadata):
            key = string()
            require(key not in keys, f"duplicate GGUF metadata key: {key}")
            keys.add(key)
            value(number("<I"))
        for _ in range(tensors):
            string()
            axes = number("<I")
            require(1 <= axes <= 4, "invalid GGUF dimension count")
            take(axes * 8 + 4 + 8)
    return bytes(data)


def load_inventory(path):
    with path.open(encoding="utf-8") as source:
        inventory = json.load(source, object_pairs_hook=no_duplicate_keys)
    require(isinstance(inventory, dict), "inventory must be a JSON object")
    require(inventory.get("kind") == "local_gguf_header_inventory", "unexpected inventory kind")
    model = inventory.get("model_dir")
    require(isinstance(model, str) and model, "inventory model_dir is missing")
    model_dir = Path(model)
    if not model_dir.is_absolute():
        model_dir = path.parent / model_dir
    model_dir = model_dir.resolve(strict=True)
    require(model_dir.is_dir(), "model_dir is not a directory")
    tensors = inventory.get("tensors")
    require(isinstance(tensors, list) and tensors, "inventory tensors must be nonempty")
    by_name = {}
    for tensor in tensors:
        require(isinstance(tensor, dict), "inventory tensor must be an object")
        require(all(key in tensor for key in ("name", "dims", "ggml_type", "shard", "relative_offset", "storage_span_bytes")),
                "inventory tensor is missing required fields")
        name, shard = tensor["name"], tensor["shard"]
        require(isinstance(name, str) and expert_name(name), f"invalid expert tensor name: {name!r}")
        require(name not in by_name, f"duplicate inventory tensor: {name}")
        require(isinstance(shard, str) and Path(shard).name == shard and shard.endswith(".gguf"),
                f"shard must be a .gguf basename: {shard!r}")
        uint(tensor["relative_offset"], "inventory relative_offset")
        uint(tensor["storage_span_bytes"], "inventory storage_span_bytes", 1)
        stride = expert_stride(tensor["dims"], tensor["ggml_type"])
        require(stride * tensor["dims"][2] <= tensor["storage_span_bytes"], f"inventory span is smaller than tensor payload: {name}")
        by_name[name] = dict(tensor, stride=stride)
    for key, actual in (("expert_tensors", len(tensors)),
                        ("expert_storage_span_bytes", sum(t["storage_span_bytes"] for t in tensors))):
        if key in inventory:
            require(uint(inventory[key], key) == actual, f"inventory {key} total mismatch")
    if "type_counts" in inventory:
        counts = inventory["type_counts"]
        require(isinstance(counts, dict), "inventory type_counts must be an object")
        for count in counts.values():
            uint(count, "type_counts value")
        require(counts == dict(Counter(str(t["ggml_type"]) for t in tensors)), "inventory type_counts mismatch")
    return inventory, model_dir, by_name


def validate_headers(inventory, model_dir, by_name, reader_path):
    shards = sorted(model_dir.glob("*.gguf"))
    require(shards, "no GGUF shards in model_dir")
    if "shards" in inventory:
        require(uint(inventory["shards"], "shards", 1) == len(shards), "inventory shard count mismatch")
    # Explicit trusted code only, never the inventory's header_reader field.
    active = {}

    def header_open(path, mode="r", *args, **kwargs):
        require(mode == "rb" and not args and not kwargs, "header reader must use open(path, 'rb')")
        require(Path(path).resolve() == active.get("path"), "header reader tried to open another file")
        return io.BytesIO(active["prefix"])

    reader = runpy.run_path(str(reader_path), init_globals={"open": header_open})
    require(callable(reader.get("read_shard")), "header reader has no callable read_shard")
    snapshots, seen_names, seen_experts = {}, set(), set()
    for shard in shards:
        before = signature(shard)
        prefix = header_prefix(shard)
        active.update(path=shard.resolve(), prefix=prefix)
        meta, tensors, start = reader["read_shard"](str(shard))
        require(signature(shard) == before, f"shard changed while reading header: {shard}")
        snapshots[shard] = before
        require(isinstance(meta, dict), "header reader metadata must be an object")
        alignment = uint(meta.get("general.alignment", 32), "GGUF alignment", 1)
        require(alignment & (alignment - 1) == 0, "GGUF alignment must be a power of two")
        uint(start, "GGUF data_start")
        require(start == (len(prefix) + alignment - 1) // alignment * alignment,
                f"header reader data_start mismatch: {shard.name}")
        require(start <= before[2], f"GGUF data_start exceeds file size: {shard.name}")
        require(isinstance(tensors, (list, tuple)) and tensors, f"empty/invalid tensor header: {shard.name}")
        normalized = []
        for tensor in tensors:
            require(isinstance(tensor, (list, tuple)) and len(tensor) == 4, "invalid header tensor record")
            name, dims, quant, offset = tensor
            require(isinstance(name, str) and name and name not in seen_names, f"missing/duplicate fresh tensor name: {name!r}")
            seen_names.add(name)
            require(isinstance(dims, (list, tuple)) and 1 <= len(dims) <= 4, "invalid header dimensions")
            for dim in dims:
                uint(dim, "header dimension", 1)
            uint(quant, "header ggml_type")
            uint(offset, "header relative_offset")
            require(offset % alignment == 0 and offset < before[2] - start, f"invalid/alignment/out-of-file tensor offset: {name}")
            normalized.append((name, list(dims), quant, offset))
        normalized.sort(key=lambda tensor: tensor[3])
        for index, (name, dims, quant, offset) in enumerate(normalized):
            end = normalized[index + 1][3] if index + 1 < len(normalized) else before[2] - start
            require(end > offset, f"duplicate/overlapping tensor offset: {name}")
            if not expert_name(name):
                continue
            require(name in by_name, f"fresh expert missing from inventory: {name}")
            expected = by_name[name]
            for key, actual in (("shard", shard.name), ("dims", dims), ("ggml_type", quant),
                                ("relative_offset", offset), ("storage_span_bytes", end - offset)):
                require(expected[key] == actual, f"fresh header/inventory {key} mismatch for {name}")
            require(expert_stride(dims, quant) * dims[2] <= end - offset, f"fresh span too small: {name}")
            expected["data_start"] = start
            expected["file_size"] = before[2]
            seen_experts.add(name)
    require(seen_experts == set(by_name), f"inventory tensors missing from fresh headers: {sorted(set(by_name) - seen_experts)}")
    return snapshots


class BoundedLines:
    def __init__(self, source):
        self.source = source
        self.used = 0
        self.last = ""

    def __iter__(self):
        return self

    def __next__(self):
        line = self.source.readline(MAX_RECORD_CHARS + 1)
        if not line:
            raise StopIteration
        self.used += len(line)
        require(self.used <= MAX_RECORD_CHARS, "CSV record exceeds 64 Ki-character limit")
        self.last = line
        return line


def stream_join(source, output, by_name):
    lines = BoundedLines(source)
    rows = csv.reader(lines, strict=True)
    columns = next(rows, [])
    require(columns in (BASE_COLUMNS, BASE_COLUMNS + TAG_COLUMNS), "unsupported CSV header (expected base or base + complete scope tags)")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns + ["shard", "absolute_offset"])
    written = 0
    while True:
        lines.used = 0
        row = next(rows, None)
        if row is None:
            raise ValueError("missing completion footer (truncated trace)")
        if len(row) == 1 and row[0].startswith("# end"):
            match = FOOTER.fullmatch(row[0])
            if match is None or lines.last not in (row[0] + "\n", row[0] + "\r\n"):
                raise ValueError("malformed/truncated completion footer")
            count, dropped, error = (csv_uint(value, "footer counter") for value in match.groups())
            require(count == written, f"footer written={count} does not match {written} records")
            require(dropped == 0 and error == 0, f"incomplete trace: dropped={dropped} error={error}")
            require(source.read(1) == "", "data after completion footer")
            output.write(row[0] + "\n")
            return written
        require(len(row) == len(columns), f"record {written}: CSV field count mismatch")
        fields = dict(zip(columns, row))
        numbers = {key: csv_uint(fields[key], f"record {written} {key}", 1 if key in ("ids_rows", "stride", "bytes") else 0)
                   for key in BASE_COLUMNS if key != "tensor"}
        require(numbers["seq"] == written, f"record {written}: seq must be contiguous from zero")
        require(numbers["shadow_hit"] in (0, 1) and numbers["shadow_bypass"] in (0, 1)
                and not (numbers["shadow_hit"] and numbers["shadow_bypass"]), f"record {written}: invalid shadow flags")
        name = fields["tensor"]
        require(name in by_name, f"record {written}: tensor not in inventory: {name!r}")
        tensor = by_name[name]
        require(numbers["ggml_type"] == tensor["ggml_type"], f"record {written}: ggml_type mismatch for {name}")
        require(numbers["expert"] < tensor["dims"][2], f"record {written}: expert outside ne[2]")
        require(numbers["stride"] == tensor["stride"], f"record {written}: runtime stride mismatch for {name}")
        require(numbers["relative_offset"] == numbers["expert"] * tensor["stride"], f"record {written}: tensor-relative offset mismatch")
        require(numbers["bytes"] == tensor["stride"], f"record {written}: bytes must equal the complete expert slice")
        end = numbers["relative_offset"] + numbers["bytes"]
        require(end <= tensor["stride"] * tensor["dims"][2] and end <= tensor["storage_span_bytes"], f"record {written}: slice exceeds tensor span")
        absolute = uint(tensor["data_start"] + tensor["relative_offset"] + numbers["relative_offset"], "absolute_offset")
        require(absolute + numbers["bytes"] <= tensor["file_size"], f"record {written}: slice exceeds file")
        writer.writerow(row + [tensor["shard"], absolute])
        written += 1


def join(inventory_path, trace_path, output_path, reader_path):
    inventory_path, trace_path, output_path, reader_path = map(Path, (inventory_path, trace_path, output_path, reader_path))
    require(str(output_path) != "-", "--output must be a file path, not stdout")
    require(not output_path.is_symlink(), "output must not be a symlink")
    inventory, model_dir, by_name = load_inventory(inventory_path)
    # Refuse aliases (including hard links) before running reader or creating output.
    protected = [inventory_path, trace_path, reader_path, *model_dir.glob("*.gguf")]
    for path in protected:
        require(output_path.resolve() != path.resolve()
                and not (output_path.exists() and os.path.samefile(output_path, path)),
                f"output aliases an input/model file: {path}")
    trace_before = signature(trace_path)
    signature(reader_path)
    snapshots = validate_headers(inventory, model_dir, by_name, reader_path)
    temporary = None
    try:
        with trace_path.open(encoding="utf-8", newline="") as source, tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=output_path.parent,
                prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            count = stream_join(source, output, by_name)
            output.flush()
            os.fsync(output.fileno())
        require(signature(trace_path) == trace_before, "trace changed during join")
        for shard, before in snapshots.items():
            require(signature(shard) == before, f"shard changed during join: {shard}")
        os.replace(temporary, output_path)
        temporary = None
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header-reader", type=Path, required=True, help="explicit trusted local Python script; executes code")
    args = parser.parse_args(argv)
    try:
        count = join(args.inventory, args.trace, args.output, args.header_reader)
    except Exception as error:
        print(f"expert trace join: {error}", file=sys.stderr)
        return 1
    print(f"joined {count} logical expert slices -> {args.output}; not physical I/O or payload identity evidence", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
