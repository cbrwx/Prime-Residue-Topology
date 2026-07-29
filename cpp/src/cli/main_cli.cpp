// CLI for the Prime Residue Topology core: self-tests and headless runs.
//   prt_cli selftest            fast correctness checks (sieve, eig, stats)
//   prt_cli run <N>             full pipeline, prints summary (e.g. run 1e9)
#include "core/sieve.h"
#include "core/pipeline.h"
#include "core/stats.h"
#include "core/topology.h"
#include "core/report.h"
#include "core/audit.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

using namespace prt;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

static uint64_t count_primes(uint64_t N) {
    uint64_t c = 0;
    for_each_prime(N, [&](const uint64_t*, size_t n) { c += n; });
    return c;
}

static void selftest() {
    std::printf("== sieve ==\n");
    check(count_primes(100) == 25, "pi(100) == 25");
    check(count_primes(1'000'000) == 78498, "pi(10^6) == 78,498");
    check(count_primes(10'000'000) == 664579, "pi(10^7) == 664,579");
    check(count_primes(100'000'000) == 5761455, "pi(10^8) == 5,761,455");

    {
        std::vector<uint64_t> got;
        for_each_prime(50, [&](const uint64_t* p, size_t n) { got.insert(got.end(), p, p + n); });
        const std::vector<uint64_t> want = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47};
        check(got == want, "primes up to 50 exact + ordered");
    }

    std::printf("== eigensolver ==\n");
    {
        // Path graph P_n: Laplacian eigenvalues are 4*sin^2(k*pi/(2n))
        const int n = 12;
        std::vector<double> A(n * n, 0.0);
        for (int i = 0; i < n - 1; ++i) {
            A[i * n + i] += 1; A[(i + 1) * n + (i + 1)] += 1;
            A[i * n + (i + 1)] -= 1; A[(i + 1) * n + i] -= 1;
        }
        std::vector<double> Acopy = A, evals, evecs;
        const bool okc = eig_sym(Acopy, n, evals, evecs);
        bool vals_ok = okc;
        for (int k = 0; k < n && vals_ok; ++k) {
            const double want = 4.0 * std::pow(std::sin(k * 3.14159265358979323846 / (2.0 * n)), 2);
            if (std::fabs(evals[k] - want) > 1e-9) vals_ok = false;
        }
        check(vals_ok, "path-graph Laplacian spectrum matches closed form");

        // residual ||A v - lambda v|| for a few eigenpairs
        bool res_ok = okc;
        for (int k = 0; k < n && res_ok; ++k) {
            double rmax = 0;
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int j = 0; j < n; ++j) s += A[i * n + j] * evecs[j * n + k];
                rmax = std::max(rmax, std::fabs(s - evals[k] * evecs[i * n + k]));
            }
            if (rmax > 1e-8) res_ok = false;
        }
        check(res_ok, "eigenpair residuals < 1e-8");
    }

    std::printf("== stats helpers ==\n");
    {
        bool ok = true;
        for (uint32_t Lm : {7u, 10u, 30u})
            for (uint32_t r = 0; r < Lm && ok; ++r)
                for (uint64_t x : {0ull, 1ull, 5ull, 29ull, 30ull, 31ull, 997ull}) {
                    uint64_t brute = 0;
                    for (uint64_t nn = 1; nn <= x; ++nn) if (nn % Lm == r) ++brute;
                    if (count_congruent(x, r, Lm) != brute) ok = false;
                }
        check(ok, "count_congruent matches brute force");
        check(chi2_pval(0.0, 10) > 0.99, "chi2 pval(0, 10) ~ 1");
        check(chi2_pval(100.0, 10) < 1e-6, "chi2 pval(100, 10) ~ 0");
    }

    std::printf("== mini pipeline (N=10^6) ==\n");
    {
        Config cfg;
        cfg.N = 1'000'000;
        cfg.sample_target = 10000;
        Results R = run_pipeline(cfg);
        check(R.valid, "pipeline completes");
        check(R.prime_count == 78498, "pipeline pi(10^6) == 78,498");
        uint64_t sum = 0;
        for (auto c : R.jointL) sum += c;
        check(sum == R.prime_count, "joint class counts sum to pi(N)");
        // classes sharing a factor with L hold at most the primes dividing L
        bool ok = true;
        for (uint32_t r = 0; r < R.cfg.L; ++r)
            if (gcd_u64(r, R.cfg.L) != 1 && R.jointL[r] > 1) ok = false;
        check(ok, "non-coprime classes contain at most 1 prime each");
        uint64_t gsum = R.gap_overflow;
        for (auto g : R.gap_hist) gsum += g;
        check(gsum == R.prime_count - 1, "gap histogram counts pi(N)-1 gaps");
        uint64_t psum = 0;
        for (auto v : R.pair_gap) psum += v;
        check(psum + R.gap_overflow == R.prime_count - 1,
              "pair-space (residue, gap) table counts every gap");
        check(R.topo.betti0 == 1 && !R.topo.eigval.empty(), "topology connected (betti_0 == 1)");
        check(std::fabs(R.topo.eigval[0]) < 1e-8, "Laplacian lambda_1 == 0");
        check(R.patterns.rows.size() == 98, "pattern scan covers q = 3..100");
        bool cross_ok = false;
        for (const auto& row : R.patterns.rows)
            if (row.q == R.trans.q)
                cross_ok = std::fabs(row.chi2 - R.trans.chi2) <
                           1e-6 * std::max(1.0, R.trans.chi2);
        check(cross_ok, "incremental q-scan matches direct transition matrix (chi2, q=10)");
        bool gap_ok = false;
        for (const auto& row : R.patterns.rows)
            if (row.q == 10)
                gap_ok = row.acc_gapmodel > 0 && row.acc_gapmodel <= 1.0 &&
                         row.chi2_gap >= 0 && row.acc_gapmodel >= row.acc_uniform;
        check(gap_ok, "gap-model null scored (q=10: sane, beats uniform)");
        bool pat_ok = true;
        for (const auto& row : R.patterns.rows)
            if (row.q == 10 && std::fabs(row.acc_uniform - 0.25) > 1e-12) pat_ok = false;
        check(pat_ok, "pattern scan uniform baseline correct (q=10 -> 25%)");
        check(R.patterns.test_triples > 0 && R.patterns.acc_order1 > 0, "order-2 test scored");
        // Hardy-Littlewood: twin pairs below 10^6 is a known constant
        uint64_t twins = 0;
        double twin_ratio = 0, sexy_over_twin = 0;
        for (size_t i = 0; i < R.hl.gaps.size(); ++i) {
            if (R.hl.gaps[i] == 2) { twins = R.hl.pairs[i]; twin_ratio = R.hl.ratio[i]; }
            if (R.hl.gaps[i] == 6 && twins) sexy_over_twin = (double)R.hl.pairs[i] / twins;
        }
        check(twins == 8169, "HL: twin pairs below 10^6 == 8,169");
        check(std::fabs(twin_ratio - 1.0) < 0.02, "HL: twins within 2% of C(2)*Li2(N)");
        check(std::fabs(sexy_over_twin - 2.0) < 0.1, "HL: g=6 enhancement ~ 2x (singular series)");
        check(R.gapgap.pearson < -0.005, "gap memory: consecutive gaps anti-correlate");
        check(R.gapgap.r2_oos > -0.05 && R.gapgap.r2_oos < 1.0, "gap memory: OOS R^2 sane");
    }

    std::printf("== sequence auditor ==\n");
    {
        // synthetic residue-random monotone sequence: gaps from a fixed LCG
        std::vector<int64_t> rnd;
        rnd.reserve(500'000);
        uint64_t s = 0x9E3779B97F4A7C15ull;
        int64_t x = 17;
        for (int i = 0; i < 500'000; ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            x += 1 + (int64_t)((s >> 33) % 400);
            rnd.push_back(x);
        }
        prt::AuditConfig ac;
        prt::AuditResult ar = prt::run_audit(rnd, ac);
        check(ar.valid && ar.monotone_frac == 1.0, "audit: synthetic sequence accepted");
        double mx = 0;
        for (const auto& row : ar.rows) mx = std::max(mx, std::fabs(row.gain_beyond));
        check(mx < 0.015, "audit: random sequence shows no memory beyond diffs (< 1.5 pp)");

        // primes through the general auditor must show the mod-3 signal
        std::vector<int64_t> pv;
        for_each_prime(1'000'000, [&](const uint64_t* p, size_t cnt) {
            for (size_t i = 0; i < cnt; ++i) pv.push_back((int64_t)p[i]);
        });
        prt::AuditResult ap = prt::run_audit(pv, ac);
        bool q3 = false;
        for (const auto& row : ap.rows)
            if (row.q == 3 && row.gain_beyond > 0.05) q3 = true;
        check(ap.valid && q3, "audit: primes show mod-3 memory beyond diffs (> 5 pp)");
    }

    std::printf("\n%s (%d failure%s)\n", g_fail ? "SELFTEST FAILED" : "SELFTEST OK",
                g_fail, g_fail == 1 ? "" : "s");
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "selftest") == 0) {
        selftest();
        return g_fail ? 1 : 0;
    }
    if (argc >= 3 && std::strcmp(argv[1], "run") == 0) {
        Config cfg;
        cfg.N = (uint64_t)std::strtod(argv[2], nullptr);
        Results R = run_pipeline(cfg);
        if (!R.valid) { std::printf("error: %s\n", R.error.c_str()); return 1; }
        std::printf("\n%s", make_summary(R).c_str());
        const std::string dir = make_run_dir("results", R);
        if (!dir.empty() && export_data(R, dir))
            std::printf("\ndata exported to %s\n", dir.c_str());
        return 0;
    }
    if (argc >= 3 && std::strcmp(argv[1], "audit") == 0) {
        prt::AuditConfig ac;
        if (argc >= 4) ac.q = (uint32_t)std::atoi(argv[3]);
        prt::AuditResult A = prt::audit_file(argv[2], ac);
        std::printf("%s", prt::audit_summary(A).c_str());
        if (!A.valid) return 1;
        const std::string dir = prt::export_audit(A);
        if (!dir.empty()) std::printf("\ndata exported to %s\n", dir.c_str());
        return 0;
    }
    std::printf("usage: prt_cli selftest | prt_cli run <N> | prt_cli audit <file> [q]\n");
    return 2;
}
