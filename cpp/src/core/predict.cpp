#include "core/predict.h"
#include "core/stats.h"
#include <algorithm>
#include <numeric>

namespace prt {

PredReport compute_prediction(uint64_t N, uint32_t L,
                              const std::vector<uint32_t>& jointL,
                              const std::vector<uint32_t>& jointL_train,
                              const TransStats& ts) {
    PredReport P;
    const uint64_t half = N / 2;

    struct Node { uint32_t r; uint64_t train, test, test_ints; };
    std::vector<Node> nodes;
    uint64_t tot_test = 0, tot_test_ints = 0, tot_train = 0;
    for (uint32_t r = 0; r < L; ++r) {
        if (gcd_u64(r, L) != 1) continue;
        Node nd;
        nd.r = r;
        nd.train = jointL_train[r];
        nd.test = jointL[r] - jointL_train[r];
        nd.test_ints = count_congruent(N, r, L) - count_congruent(half, r, L);
        tot_test += nd.test;
        tot_test_ints += nd.test_ints;
        tot_train += nd.train;
        nodes.push_back(nd);
    }
    P.train_primes = tot_train;
    P.test_primes = tot_test;
    if (nodes.empty() || tot_test_ints == 0) return P;
    P.base_precision = (double)tot_test / (double)tot_test_ints;

    // Rank by training-half density (integer counts per class are equal to
    // within one, so ranking by count is ranking by density).
    std::sort(nodes.begin(), nodes.end(),
              [](const Node& a, const Node& b) { return a.train > b.train; });

    uint64_t cum_p = 0, cum_i = 0;
    P.coverage.reserve(nodes.size());
    P.precision.reserve(nodes.size());
    P.lift.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        cum_p += nodes[i].test;
        cum_i += nodes[i].test_ints;
        P.coverage.push_back((double)(i + 1) / (double)nodes.size());
        const double prec = cum_i ? (double)cum_p / (double)cum_i : 0.0;
        P.precision.push_back(prec);
        P.lift.push_back(P.base_precision > 0 ? prec / P.base_precision : 0.0);
    }

    // ---- transition predictor ----
    const uint32_t q = ts.q;
    if (q >= 3 && !ts.classes.empty()) {
        P.trans_base_uniform = 1.0 / (double)ts.classes.size();

        // argmax of each training row (over coprime columns)
        std::vector<uint32_t> amax(q, 0);
        std::vector<uint64_t> col_train(q, 0);
        for (uint32_t i : ts.classes) {
            uint64_t best = 0;
            uint32_t bj = ts.classes[0];
            for (uint32_t j : ts.classes) {
                const uint64_t v = ts.train[(size_t)i * q + j];
                col_train[j] += v;
                if (v > best) { best = v; bj = j; }
            }
            amax[i] = bj;
        }
        uint32_t gj = ts.classes[0];
        uint64_t gbest = 0;
        for (uint32_t j : ts.classes)
            if (col_train[j] > gbest) { gbest = col_train[j]; gj = j; }

        uint64_t tot = 0, hit = 0, hit_marg = 0;
        for (uint32_t i : ts.classes)
            for (uint32_t j : ts.classes) {
                const uint64_t v = ts.test_[(size_t)i * q + j];
                tot += v;
                if (j == amax[i]) hit += v;
                if (j == gj) hit_marg += v;
            }
        if (tot > 0) {
            P.trans_acc = (double)hit / (double)tot;
            P.trans_base_marginal = (double)hit_marg / (double)tot;
        }
    }
    return P;
}

} // namespace prt
