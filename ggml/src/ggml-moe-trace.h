#pragma once

#include "ggml-moe-stats.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>

// Bounded logical CPU expert-access CSV. Caller serializes record/finish and
// owns FILE. Never reads a weight payload; misses are SHADOW misses, not I/O.
class ggml_moe_trace {
public:
    static constexpr uint64_t max_records = 100000;
    explicit ggml_moe_trace(FILE * file, uint64_t limit = max_records, bool request_only = false)
        : file_(file), limit_(std::min(limit, max_records)), request_only_(request_only) {
        if (!file_ || fprintf(file_, "seq,weight_entry,epoch,ids_rows,tensor,ggml_type,expert,stride,relative_offset,bytes,shadow_hit,shadow_bypass%s\n",
                request_only_ ? ",context,request_id,seq_id,phase,pos_min,pos_max" : "") < 0) error_ = true;
    }
    bool request_scoped() const { return request_only_; }
    void set_scope(ggml_moe_trace_scope scope) { scope_ = scope; }
    ~ggml_moe_trace() { finish(); }
    ggml_moe_trace(const ggml_moe_trace &) = delete;
    ggml_moe_trace & operator=(const ggml_moe_trace &) = delete;
    bool failed() const { return error_; }
    uint64_t remaining() const { return error_ || finished_ ? 0 : limit_ - written_; }
    // A skipped operation/readback is incomplete evidence, not an empty selection.
    void incomplete() { if (!finished_) { ++dropped_; error_ = true; } }

    void record(uint64_t entry, uint64_t epoch, int64_t rows, const char * name,
                int type, uint32_t expert, size_t stride, size_t offset, size_t bytes,
                bool hit, bool bypass) {
        if (finished_ || (request_only_ && scope_.phase == 0)) return;
        if (written_ >= limit_ || error_) { ++dropped_; return; }
        fprintf(file_, "%llu,%llu,%llu,%lld,\"", ull(written_), ull(entry), ull(epoch), (long long) rows);
        // RFC 4180 quoting, including names containing commas/newlines/quotes.
        for (const char * c = name; *c; ++c) {
            if (*c == '"') fputc('"', file_);
            fputc(*c, file_);
        }
        fprintf(file_, "\",%d,%u,%zu,%zu,%zu,%d,%d", type, expert, stride, offset, bytes, hit, bypass);
        if (request_only_) fprintf(file_, ",target,%lld,%d,%s,%d,%d", (long long) scope_.request_id,
                scope_.seq_id, scope_.phase == 1 ? "prefill" : scope_.phase == 2 ? "decode" : "mixed",
                scope_.pos_min, scope_.pos_max);
        fputc('\n', file_);
        ++written_;
        if (ferror(file_)) error_ = true;
    }
    void finish() {
        if (finished_) return;
        finished_ = true;
        if (file_) {
            // Detect buffered record errors BEFORE composing the footer.
            if (fflush(file_) != 0 || ferror(file_)) error_ = true;
            fprintf(file_, "# end written=%llu dropped=%llu error=%d\n", ull(written_), ull(dropped_), error_);
            if (fflush(file_) != 0 || ferror(file_)) error_ = true;
        }
        if (error_) fprintf(stderr, "expert trace: write failed; trace is incomplete\n");
    }
private:
    static unsigned long long ull(uint64_t n) { return static_cast<unsigned long long>(n); }
    FILE * file_;
    uint64_t limit_, written_ = 0, dropped_ = 0;
    bool error_ = false, finished_ = false;
    bool request_only_;
    ggml_moe_trace_scope scope_ = {-1, -1, 0, -1, -1};
};
