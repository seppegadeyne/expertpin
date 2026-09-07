#include "../examples/server/server-verifier-trace.h"
#include <cstdio>
#include <string>
#include <limits>

int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char * name) { if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", name); } };
    float logits[] = {-4, 2, 2, 1};
    auto s = server_verifier_summarize(logits, 4, 3);
    check(s.valid && s.argmax == 1 && s.runner_up == 2 && s.margin == 0, "tie keeps first ID");
    check(s.has_proposal && s.proposal_logit == 1, "proposal captured");
    logits[2] = 3;
    s = server_verifier_summarize(logits, 4, -1);
    check(s.valid && s.argmax == 2 && s.runner_up == 1 && s.margin == 1 && !s.has_proposal, "bonus and margin");
    check(!server_verifier_summarize(nullptr, 4, -1).valid, "null");
    check(!server_verifier_summarize(logits, 1, -1).valid, "small vocab");
    check(!server_verifier_summarize(logits, 4, 4).valid, "proposal bounds");
    check(!server_verifier_summarize(logits, 4, -2).valid, "negative proposal");
    logits[0] = std::numeric_limits<float>::quiet_NaN();
    check(!server_verifier_summarize(logits, 4, -1).valid, "nan invalidates whole row");
    logits[0] = std::numeric_limits<float>::infinity();
    check(!server_verifier_summarize(logits, 4, -1).valid, "infinity invalidates whole row");
    float extreme[] = {std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    check(std::isfinite(server_verifier_summarize(extreme, 2, -1).margin), "double margin avoids overflow");
    check(std::string(server_verifier_decision(0, 1, 2, 2)) == "accepted", "accept");
    check(std::string(server_verifier_decision(0, 1, 2, 1)) == "rejected", "reject");
    check(std::string(server_verifier_decision(1, 1, -1, 2)) == "bonus", "bonus");
    check(!server_verifier_in_window(0) && server_verifier_in_window(1) && server_verifier_in_window(128) && !server_verifier_in_window(129), "bounded prefix");
    server_verifier_request_limit limit;
    for (int i = 0; i < 128; ++i) { check(limit.admit(10), "request admits first128"); }
    check(!limit.admit(10) && !limit.admit(10), "rewind cannot exceed request cap");
    check(limit.admit(11) && limit.written == 1, "new task resets limit");
    return failures ? 1 : 0;
}
