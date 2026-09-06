#include "ggml-moe-trace.h"
#include <cstdio>
#include <stdexcept>
#include <string>
#if defined(__linux__)
#include <unistd.h>
#endif

static void require(bool ok) { if (!ok) throw std::runtime_error("trace contract"); }
static std::string contents(FILE * f) {
    rewind(f);
    std::string s;
    char buf[1024];
    while (fgets(buf, sizeof(buf), f)) s += buf;
    return s;
}
int main() {
    try {
        FILE * f = tmpfile();
        require(f != nullptr);
        {
            ggml_moe_trace trace(f, 2); // owns neither FILE nor payload
            trace.record(4, 9, 3, "blk.1.\"up\"", 21, 7, 704000, 4928000, 704000, false, false);
            trace.record(5, 10, 1, "down\nline", 20, 2, 921600, 1843200, 921600, true, false);
            trace.record(6, 11, 1, "dropped", 20, 0, 1, 0, 1, false, true);
            trace.finish();
            trace.finish(); // idempotent footer
        }
        require(contents(f) ==
            "seq,weight_entry,epoch,ids_rows,tensor,ggml_type,expert,stride,relative_offset,bytes,shadow_hit,shadow_bypass\n"
            "0,4,9,3,\"blk.1.\"\"up\"\"\",21,7,704000,4928000,704000,0,0\n"
            "1,5,10,1,\"down\nline\",20,2,921600,1843200,921600,1,0\n"
            "# end written=2 dropped=1 error=0\n");
        fclose(f);
        f = tmpfile();
        require(f != nullptr);
        {
            ggml_moe_trace trace(f, 2, true);
            require(trace.request_scoped());
            for (int i = 0; i < 100001; ++i)
                trace.record(i, 0, 1, "warmup", 0, 0, 1, 0, 1, false, false);
            trace.set_scope({42, 0, 1, 0, 67});
            trace.record(100001, 0, 68, "prefill", 0, 0, 1, 0, 1, false, false);
            trace.set_scope({42, 0, 2, 68, 72});
            trace.record(100002, 0, 5, "verify", 0, 0, 1, 0, 1, true, false);
            trace.set_scope({-1, -1, 0, -1, -1});
            trace.record(100003, 0, 1, "draft", 0, 0, 1, 0, 1, false, false);
        }
        require(contents(f).find("warmup") == std::string::npos);
        require(contents(f).find("draft") == std::string::npos);
        require(contents(f).find(",target,42,0,prefill,0,67\n") != std::string::npos);
        require(contents(f).find(",target,42,0,decode,68,72\n") != std::string::npos);
        require(contents(f).find("# end written=2 dropped=0 error=0") != std::string::npos);
        fclose(f);
        for (uint64_t limit : {ggml_moe_trace::max_records, ggml_moe_trace::max_records + 1}) {
            f = tmpfile();
            require(f != nullptr);
            {
                ggml_moe_trace trace(f, limit);
                for (uint64_t i = 0; i <= ggml_moe_trace::max_records; ++i)
                    trace.record(i, 0, 1, "x", 0, 0, 1, 0, 1, false, false);
            }
            const std::string output = contents(f);
            require(output.find("# end written=100000 dropped=1 error=0\n") != std::string::npos);
            require(std::count(output.begin(), output.end(), '\n') == 100002);
            fclose(f);
        }
#if defined(__linux__)
        f = tmpfile();
        require(f != nullptr);
        {
            ggml_moe_trace trace(f);
            trace.record(0, 0, 1, "buffered", 0, 0, 1, 0, 1, false, false);
            close(fileno(f)); // deterministic buffered flush failure, no device dependency
            trace.finish();
            require(trace.failed());
        }
        fclose(f);
#endif
        f = tmpfile();
        require(f != nullptr);
        {
            ggml_moe_trace trace(f, 0);
            trace.record(0, 0, 1, "none", 0, 0, 1, 0, 1, false, true);
        }
        require(contents(f).find("# end written=0 dropped=1 error=0") != std::string::npos);
        fclose(f);
        return 0;
    } catch (const std::exception & e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
