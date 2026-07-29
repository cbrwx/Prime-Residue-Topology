#include "core/audit.h"
#include "core/stats.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cmath>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <filesystem>

namespace prt {

static uint32_t mod_i64(int64_t x, uint32_t m) {
    int64_t r = x % (int64_t)m;
    if (r < 0) r += m;
    return (uint32_t)r;
}

static bool parse_ints(const std::string& path, std::vector<int64_t>& out, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open file: " + path; return false; }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (long)1u << 30) {
        std::fclose(f);
        err = "file empty or larger than 1 GB";
        return false;
    }
    std::string buf((size_t)sz, '\0');
    const size_t got = std::fread(buf.data(), 1, (size_t)sz, f);
    std::fclose(f);
    buf.resize(got);

    const char* p = buf.c_str();
    const char* end = p + buf.size();
    while (p < end) {
        // token starts at a digit, or a '-' immediately followed by a digit
        if ((*p >= '0' && *p <= '9') ||
            (*p == '-' && p + 1 < end && p[1] >= '0' && p[1] <= '9')) {
            char* stop = nullptr;
            const long long v = std::strtoll(p, &stop, 10);
            out.push_back((int64_t)v);
            p = stop;
        } else {
            ++p;
        }
    }
    if (out.empty()) { err = "no integers found in file"; return false; }
    return true;
}

// order-2 vs order-1 on the same test triples, over occupied classes
static void order2_general(uint32_t q,
                           const std::vector<uint64_t>& o2_obs,
                           const std::vector<uint64_t>& o2_train,
                           const std::vector<uint64_t>& t1_obs,
                           const std::vector<uint64_t>& t1_train,
                           AuditResult& A) {
    A.q2 = q;
    const std::vector<uint32_t> classes = occupied_classes(q, t1_obs.data());
    const int k = (int)classes.size();
    if (k < 2) return;
    A.acc_uniform = 1.0 / (double)k;

    std::vector<uint32_t> amax1(q, classes[0]);
    for (uint32_t j : classes) {
        uint64_t best = 0;
        for (uint32_t kk : classes) {
            const uint64_t v = t1_train[(size_t)j * q + kk];
            if (v > best) { best = v; amax1[j] = kk; }
        }
    }
    std::vector<uint32_t> amax2((size_t)q * q, 0);
    for (uint32_t i : classes)
        for (uint32_t j : classes) {
            uint64_t best = 0;
            uint32_t bk = amax1[j];
            for (uint32_t kk : classes) {
                const uint64_t v = o2_train[((size_t)i * q + j) * q + kk];
                if (v > best) { best = v; bk = kk; }
            }
            amax2[(size_t)i * q + j] = bk;
        }

    uint64_t tot = 0, hit1 = 0, hit2 = 0;
    for (uint32_t i : classes)
        for (uint32_t j : classes)
            for (uint32_t kk : classes) {
                const size_t idx = ((size_t)i * q + j) * q + kk;
                const uint64_t v = o2_obs[idx] - o2_train[idx];
                tot += v;
                if (kk == amax1[j]) hit1 += v;
                if (kk == amax2[(size_t)i * q + j]) hit2 += v;
            }
    A.test_triples = tot;
    if (tot > 0) {
        A.acc_order1 = (double)hit1 / (double)tot;
        A.acc_order2 = (double)hit2 / (double)tot;
    }
}

AuditResult run_audit(const std::vector<int64_t>& seq, const AuditConfig& cfg) {
    AuditResult A;
    const size_t n = seq.size();
    if (n < 100) { A.error = "need at least 100 terms (got " + std::to_string(n) + ")"; return A; }
    A.count = n;
    A.vmin = *std::min_element(seq.begin(), seq.end());
    A.vmax = *std::max_element(seq.begin(), seq.end());

    const uint32_t PQ_MIN = 3, PQ_MAX = 100;
    std::vector<size_t> off(PQ_MAX + 2, 0);
    for (uint32_t qq = PQ_MIN; qq <= PQ_MAX; ++qq)
        off[qq + 1] = off[qq] + (size_t)qq * qq;
    std::vector<uint64_t> obs(off[PQ_MAX + 1], 0), train(off[PQ_MAX + 1], 0);

    const uint32_t q = std::clamp(cfg.q, 3u, 50u);
    std::vector<uint64_t> o2_obs((size_t)q * q * q, 0), o2_train((size_t)q * q * q, 0);

    static const uint32_t STAT_M[] = {3, 4, 5, 7, 8, 9, 11, 12, 15, 16, 25, 30, 105, 210};
    A.mods.resize(std::size(STAT_M));
    for (size_t i = 0; i < std::size(STAT_M); ++i) {
        A.mods[i].m = STAT_M[i];
        A.mods[i].counts.assign(STAT_M[i], 0);
    }

    std::vector<uint64_t> dhist(2047, 0);

    std::vector<uint32_t> pr(PQ_MAX + 1, 0);
    for (uint32_t qq = PQ_MIN; qq <= PQ_MAX; ++qq) pr[qq] = mod_i64(seq[0], qq);
    for (auto& ms : A.mods) ++ms.counts[mod_i64(seq[0], ms.m)];
    uint32_t p1 = mod_i64(seq[0], q), p2 = 0;

    const size_t half = n / 2;
    uint64_t mono = 0;

    for (size_t i = 1; i < n; ++i) {
        const int64_t x = seq[i];
        const int64_t d = x - seq[i - 1];
        if (d > 0) ++mono;
        if (d >= -1023 && d <= 1023) ++dhist[(size_t)(d + 1023)];
        else ++A.diff_overflow;

        for (auto& ms : A.mods) ++ms.counts[mod_i64(x, ms.m)];

        const bool tr = (i <= half);
        for (uint32_t qq = PQ_MIN; qq <= PQ_MAX; ++qq) {
            const uint32_t r = mod_i64(x, qq);
            const size_t o = off[qq] + (size_t)pr[qq] * qq + r;
            ++obs[o];
            if (tr) ++train[o];
            pr[qq] = r;
        }

        const uint32_t rq = mod_i64(x, q);
        if (i >= 2) {
            const size_t o2 = ((size_t)p2 * q + p1) * q + rq;
            ++o2_obs[o2];
            if (tr) ++o2_train[o2];
        }
        p2 = p1;
        p1 = rq;
    }
    A.monotone_frac = (double)mono / (double)(n - 1);

    for (size_t d = 0; d < dhist.size(); ++d)
        if (dhist[d]) {
            A.diff_x.push_back((int32_t)d - 1023);
            A.diff_count.push_back(dhist[d]);
        }

    for (auto& ms : A.mods) finalize_mod_stats_uniform(ms);

    for (uint32_t qq = PQ_MIN; qq <= PQ_MAX; ++qq)
        A.rows.push_back(analyze_transition_general(qq, obs.data() + off[qq],
                                                    train.data() + off[qq]));

    // order-1 matrix at q for the order-2 comparison
    std::vector<uint64_t> t1o((size_t)q * q, 0), t1t((size_t)q * q, 0);
    if (q >= PQ_MIN && q <= PQ_MAX) {
        std::copy(obs.begin() + off[q], obs.begin() + off[q] + (size_t)q * q, t1o.begin());
        std::copy(train.begin() + off[q], train.begin() + off[q] + (size_t)q * q, t1t.begin());
    }
    order2_general(q, o2_obs, o2_train, t1o, t1t, A);

    A.valid = true;
    return A;
}

AuditResult audit_file(const std::string& path, const AuditConfig& cfg) {
    std::vector<int64_t> seq;
    std::string err;
    if (!parse_ints(path, seq, err)) {
        AuditResult A;
        A.error = err;
        A.source = path;
        return A;
    }
    AuditResult A = run_audit(seq, cfg);
    A.source = path;
    return A;
}

namespace {
void appendf(std::string& s, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    s += buf;
}
} // namespace

std::string audit_summary(const AuditResult& A) {
    std::string s;
    appendf(s, "=== Sequence audit ===\n");
    appendf(s, "source: %s\n", A.source.c_str());
    if (!A.valid) {
        appendf(s, "ERROR: %s\n", A.error.c_str());
        return s;
    }
    appendf(s, "terms: %llu   range: [%lld, %lld]   strictly-increasing steps: %.1f%%\n",
            (unsigned long long)A.count, (long long)A.vmin, (long long)A.vmax,
            100.0 * A.monotone_frac);

    appendf(s, "\n-- marginal occupancy (uniform null) --\n");
    double worst_p = 1.0;
    uint32_t worst_m = 0;
    for (const auto& ms : A.mods) {
        appendf(s, "  mod %4u: chi2 = %12.2f (dof %3d, p = %.3g)   max |z| = %7.2f at class %u\n",
                ms.m, ms.chi2, ms.dof, ms.pval, ms.max_abs_z, ms.argmax_class);
        if (ms.pval < worst_p) { worst_p = ms.pval; worst_m = ms.m; }
    }
    if (worst_p < 1e-3)
        appendf(s, "  -> occupancy structure detected (strongest at mod %u): the sequence "
                   "does not fill residue classes uniformly (wheel-like or biased values).\n",
                worst_m);
    else
        appendf(s, "  -> residue occupancy is consistent with uniform.\n");

    std::vector<PatternRow> rows = A.rows;
    std::sort(rows.begin(), rows.end(),
              [](const PatternRow& x, const PatternRow& y) { return x.gain_beyond > y.gain_beyond; });
    appendf(s, "\n-- successive-term structure (out-of-sample; difference-model null) --\n");
    appendf(s, "  strongest structure BEYOND difference statistics:\n");
    for (int i = 0; i < 5 && i < (int)rows.size(); ++i)
        appendf(s, "    q = %2u: beyond-gain %+.2f pp (acc %.2f%%, diff model %.2f%%, uniform %.2f%%), "
                   "chi2_diffnull = %.0f (p = %.3g)\n",
                rows[i].q, 100.0 * rows[i].gain_beyond, 100.0 * rows[i].acc,
                100.0 * rows[i].acc_gapmodel, 100.0 * rows[i].acc_uniform,
                rows[i].chi2_gap, rows[i].pval_gap);
    const double max_beyond = rows.empty() ? 0.0 : rows[0].gain_beyond;
    double min_pg = 1.0;
    for (const auto& row : A.rows) min_pg = std::min(min_pg, row.pval_gap);
    if (max_beyond < 0.005 && min_pg > 1e-3)
        appendf(s, "  -> VERDICT: no residue memory beyond difference statistics. "
                   "The sequence looks residue-random given its difference distribution.\n");
    else if (max_beyond < 0.005)
        appendf(s, "  -> VERDICT: distributional correlation detected (chi2 significant) but "
                   "too diffuse to improve prediction. Weak residue-difference coupling.\n");
    else
        appendf(s, "  -> VERDICT: genuine residue memory detected (+%.2f pp beyond difference "
                   "statistics). Successive terms carry information about each other beyond "
                   "their difference distribution.\n", 100.0 * max_beyond);

    appendf(s, "  order-2 memory (mod %u, %llu test triples): order-1 %.2f%%, order-2 %.2f%% "
               "(extra %+.2f pp)\n",
            A.q2, (unsigned long long)A.test_triples,
            100.0 * A.acc_order1, 100.0 * A.acc_order2,
            100.0 * (A.acc_order2 - A.acc_order1));
    return s;
}

std::string export_audit(const AuditResult& A) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_s(&tm, &tt);
    std::string base = std::filesystem::path(A.source).stem().string();
    if (base.empty()) base = "sequence";
    for (char& c : base)
        if (!isalnum((unsigned char)c) && c != '-' && c != '_') c = '_';
    char name[192];
    std::snprintf(name, sizeof(name), "audit_%04d%02d%02d_%02d%02d%02d_%s",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec, base.c_str());
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::path("results") / name;
    std::filesystem::create_directories(dir, ec);
    if (ec) return "";
    const std::string d = dir.string();

    if (FILE* f = std::fopen((d + "/summary.txt").c_str(), "wb")) {
        const std::string s = audit_summary(A);
        std::fwrite(s.data(), 1, s.size(), f);
        std::fclose(f);
    }
    if (FILE* f = std::fopen((d + "/patterns.csv").c_str(), "wb")) {
        std::fprintf(f, "q,chi2,pval,mean_diag_bias,acc_argmax,acc_uniform,gain,"
                        "acc_diffmodel,gain_beyond,chi2_diffnull,pval_diffnull,resid_diag\n");
        for (const auto& row : A.rows)
            std::fprintf(f, "%u,%.6f,%.6g,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f,%.6g,%.6f\n",
                         row.q, row.chi2, row.pval, row.diag_bias,
                         row.acc, row.acc_uniform, row.gain,
                         row.acc_gapmodel, row.gain_beyond,
                         row.chi2_gap, row.pval_gap, row.resid_diag);
        std::fclose(f);
    }
    if (FILE* f = std::fopen((d + "/diffs.csv").c_str(), "wb")) {
        std::fprintf(f, "difference,count\n");
        for (size_t i = 0; i < A.diff_x.size(); ++i)
            std::fprintf(f, "%d,%llu\n", A.diff_x[i], (unsigned long long)A.diff_count[i]);
        if (A.diff_overflow)
            std::fprintf(f, "overflow_abs_gt_1023,%llu\n", (unsigned long long)A.diff_overflow);
        std::fclose(f);
    }
    if (FILE* f = std::fopen((d + "/marginals.csv").c_str(), "wb")) {
        std::fprintf(f, "modulus,class,count,z\n");
        for (const auto& ms : A.mods)
            for (uint32_t r = 0; r < ms.m; ++r)
                std::fprintf(f, "%u,%u,%llu,%.6f\n", ms.m, r,
                             (unsigned long long)ms.counts[r], ms.z[r]);
        std::fclose(f);
    }
    return d;
}

} // namespace prt
