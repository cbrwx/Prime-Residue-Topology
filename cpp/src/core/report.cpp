#include "core/report.h"
#include "core/stats.h"
#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <vector>

namespace prt {

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

std::string make_summary(const Results& R) {
    const Config& cfg = R.cfg;
    std::string s;

    appendf(s, "=== Prime Residue Topology -- run summary ===\n");
    appendf(s, "config: N = %llu, L = %u, q = %u, kNN = %d, sample target = %u "
               "(stride %llu), threads = %u\n",
            (unsigned long long)cfg.N, cfg.L, cfg.q, cfg.knn, cfg.sample_target,
            (unsigned long long)R.sample_stride, cfg.threads);
    appendf(s, "stat moduli:");
    for (uint32_t m : cfg.stat_moduli) appendf(s, " %u", m);
    appendf(s, "\n\n");

    appendf(s, "pi(N) = %llu   last prime = %llu   max gap = %llu\n",
            (unsigned long long)R.prime_count, (unsigned long long)R.last_prime,
            (unsigned long long)R.max_gap);
    appendf(s, "sieve+accumulate %.2fs, analysis %.2fs\n", R.t_sieve, R.t_analysis);

    appendf(s, "\n-- residue class equidistribution (Dirichlet null) --\n");
    for (const auto& ms : R.mods)
        appendf(s, "  mod %4u: chi2 = %10.2f (dof %3d, p = %.3g)   max |z| = %6.2f at class %u\n",
                ms.m, ms.chi2, ms.dof, ms.pval, ms.max_abs_z, ms.argmax_class);

    appendf(s, "\n-- consecutive-prime transitions mod %u --\n", R.trans.q);
    appendf(s, "  chi2 vs independence = %.1f (dof %d, p = %.3g)\n",
            R.trans.chi2, R.trans.dof, R.trans.pval);
    appendf(s, "  mean same-residue (diagonal) bias = %+.1f%%  (Lemke Oliver-Soundararajan repulsion)\n",
            100.0 * R.trans.mean_diag_bias);
    appendf(s, "  bias matrix (rows: residue of p_n, cols: residue of p_n+1):\n        ");
    for (uint32_t j : R.trans.classes) appendf(s, "%7u", j);
    appendf(s, "\n");
    for (uint32_t i : R.trans.classes) {
        appendf(s, "  %4u: ", i);
        for (uint32_t j : R.trans.classes)
            appendf(s, "%+6.1f%%", 100.0 * R.trans.bias[(size_t)i * R.trans.q + j]);
        appendf(s, "\n");
    }

    appendf(s, "\n-- Chebyshev races --\n");
    int pos4 = 0, pos3 = 0;
    for (size_t i = 0; i < R.cp_x.size(); ++i) {
        if (R.race4[i] > 0) ++pos4;
        if (R.race3[i] > 0) ++pos3;
    }
    appendf(s, "  pi(x;4,3)-pi(x;4,1) > 0 at %d/%zu checkpoints, final = %+.0f\n",
            pos4, R.cp_x.size(), R.race4.empty() ? 0.0 : R.race4.back());
    appendf(s, "  pi(x;3,2)-pi(x;3,1) > 0 at %d/%zu checkpoints, final = %+.0f\n",
            pos3, R.cp_x.size(), R.race3.empty() ? 0.0 : R.race3.back());

    appendf(s, "\n-- residue torus topology (L = %u, %zu classes) --\n",
            R.topo.L, R.topo.nodes.size());
    appendf(s, "  betti_0 = %d   spectral gap = %.4f   cheeger ~ %.4f   Moran's I = %.4f\n",
            R.topo.betti0, R.topo.spectral_gap, R.topo.cheeger, R.topo.moran);
    std::vector<std::pair<double, int>> pw;
    for (size_t k = 1; k < R.topo.power.size(); ++k)
        pw.emplace_back(R.topo.power[k], (int)k);
    std::sort(pw.rbegin(), pw.rend());
    appendf(s, "  density-field power: top modes ");
    for (int i = 0; i < 5 && i < (int)pw.size(); ++i)
        appendf(s, "#%d (%.1f%%) ", pw[i].second, 100.0 * pw[i].first);
    appendf(s, "\n");

    appendf(s, "\n-- pattern classification (transition-structure scan, q = 3..100) --\n");
    {
        std::vector<PatternRow> rows = R.patterns.rows;
        std::sort(rows.begin(), rows.end(),
                  [](const PatternRow& x, const PatternRow& y) { return x.gain > y.gain; });
        appendf(s, "  strongest RAW structure (out-of-sample gain over uniform):\n");
        for (int i = 0; i < 3 && i < (int)rows.size(); ++i)
            appendf(s, "    q = %2u: acc %.2f%% vs uniform %.2f%% (gain %+.2f pp), "
                       "diag bias %+.1f%%\n",
                    rows[i].q, 100.0 * rows[i].acc, 100.0 * rows[i].acc_uniform,
                    100.0 * rows[i].gain, 100.0 * rows[i].diag_bias);

        std::sort(rows.begin(), rows.end(),
                  [](const PatternRow& x, const PatternRow& y) { return x.gain_beyond > y.gain_beyond; });
        appendf(s, "  strongest structure BEYOND the gap-model null (the genuine correlation):\n");
        for (int i = 0; i < 5 && i < (int)rows.size(); ++i)
            appendf(s, "    q = %2u: beyond-gap gain %+.2f pp (raw acc %.2f%%, gap model %.2f%%), "
                       "residual diag %+.1f%%, chi2_gapnull = %.0f (p = %.3g)\n",
                    rows[i].q, 100.0 * rows[i].gain_beyond, 100.0 * rows[i].acc,
                    100.0 * rows[i].acc_gapmodel, 100.0 * rows[i].resid_diag,
                    rows[i].chi2_gap, rows[i].pval_gap);

        if (!rows.empty()) {
            const PatternRow* q0 = nullptr;
            for (const auto& row : R.patterns.rows)
                if (row.q == R.trans.q) q0 = &row;
            const PatternRow& b = rows[0];
            appendf(s, "  verdict: ");
            if (q0)
                appendf(s, "at q = %u the raw gain of %+.2f pp splits into %+.2f pp from gap "
                           "frequencies + wheel and %+.2f pp genuine residue-gap correlation. ",
                        q0->q, 100.0 * q0->gain, 100.0 * (q0->acc_gapmodel - q0->acc_uniform),
                        100.0 * q0->gain_beyond);
            appendf(s, "Strongest genuine correlation: q = %u (%+.2f pp beyond the gap model). "
                       "Names: the raw effect is the Lemke Oliver-Soundararajan bias; the part "
                       "beyond the gap model is the residue-gap correlation predicted by the "
                       "Hardy-Littlewood k-tuple conjecture.\n",
                    b.q, 100.0 * b.gain_beyond);
        }
        appendf(s, "  order-2 memory test (mod %u, %llu test triples): "
                   "order-1 %.2f%%, order-2 %.2f%% (extra memory %+.2f pp)\n",
                R.patterns.q2, (unsigned long long)R.patterns.test_triples,
                100.0 * R.patterns.acc_order1, 100.0 * R.patterns.acc_order2,
                100.0 * (R.patterns.acc_order2 - R.patterns.acc_order1));
    }

    appendf(s, "\n-- gap memory (consecutive gaps g_n, g_n+1) --\n");
    appendf(s, "  pearson r = %+.4f   chi2 vs independence = %.0f (dof %d, p = %.3g)\n",
            R.gapgap.pearson, R.gapgap.chi2, R.gapgap.dof, R.gapgap.pval);
    appendf(s, "  out-of-sample R^2 of predicting g_n+1 from g_n: %.4f "
               "(conditional mean vs global mean; mean gap %.2f)\n",
            R.gapgap.r2_oos, R.gapgap.mean_gap);

    appendf(s, "\n-- Hardy-Littlewood singular series test (pairs (p, p+g), even g <= 120) --\n");
    appendf(s, "  Li2(N) = %.1f\n", R.hl.li2);
    appendf(s, "  mean |measured/predicted - 1| = %.3f%%   worst at g = %u (%.3f%%)\n",
            100.0 * R.hl.mean_abs_dev, R.hl.argmax_g, 100.0 * R.hl.max_abs_dev);
    for (size_t i = 0; i < R.hl.gaps.size(); ++i) {
        if (R.hl.gaps[i] == 2 || R.hl.gaps[i] == 6 || R.hl.gaps[i] == 30)
            appendf(s, "  g = %3u: measured %llu, predicted %.0f, ratio %.5f (C(g) = %.6f)\n",
                    R.hl.gaps[i], (unsigned long long)R.hl.pairs[i],
                    R.hl.predicted[i], R.hl.ratio[i], R.hl.singular[i]);
    }

    appendf(s, "\n-- out-of-sample prediction (train (0,N/2], test (N/2,N]) --\n");
    appendf(s, "  wheel baseline precision on coprime classes: %.4f%%\n",
            100.0 * R.pred.base_precision);
    auto lift_at = [&](double cov) {
        for (size_t i = 0; i < R.pred.coverage.size(); ++i)
            if (R.pred.coverage[i] >= cov) return R.pred.lift[i];
        return 1.0;
    };
    appendf(s, "  density-ranking lift: @10%% coverage %.4fx, @25%% %.4fx, @50%% %.4fx\n",
            lift_at(0.10), lift_at(0.25), lift_at(0.50));
    appendf(s, "  next-prime residue mod %u: predicted %.2f%% (train-argmax) vs %.2f%% "
               "(marginal) vs %.2f%% (uniform)\n",
            R.trans.q, 100.0 * R.pred.trans_acc, 100.0 * R.pred.trans_base_marginal,
            100.0 * R.pred.trans_base_uniform);
    return s;
}

std::string make_run_dir(const std::string& base_dir, const Results& R) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_s(&tm, &tt);
    char name[128];
    std::snprintf(name, sizeof(name), "run_%04d%02d%02d_%02d%02d%02d_N%llu",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec,
                  (unsigned long long)R.cfg.N);
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::path(base_dir) / name;
    std::filesystem::create_directories(dir, ec);
    if (ec) return "";
    return dir.string();
}

namespace {
FILE* open_csv(const std::string& dir, const char* name, bool& ok) {
    const std::string path = dir + "/" + name;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) ok = false;
    return f;
}
} // namespace

bool export_data(const Results& R, const std::string& dir) {
    bool ok = true;

    if (FILE* f = std::fopen((dir + "/summary.txt").c_str(), "wb")) {
        const std::string s = make_summary(R);
        std::fwrite(s.data(), 1, s.size(), f);
        std::fclose(f);
    } else ok = false;

    if (FILE* f = open_csv(dir, "mod_chi2.csv", ok)) {
        std::fprintf(f, "modulus,chi2,dof,pval,max_abs_z,argmax_class\n");
        for (const auto& ms : R.mods)
            std::fprintf(f, "%u,%.6f,%d,%.6g,%.6f,%u\n",
                         ms.m, ms.chi2, ms.dof, ms.pval, ms.max_abs_z, ms.argmax_class);
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "mod_counts.csv", ok)) {
        std::fprintf(f, "modulus,class,count,coprime,z\n");
        for (const auto& ms : R.mods)
            for (uint32_t r = 0; r < ms.m; ++r)
                std::fprintf(f, "%u,%u,%llu,%d,%.6f\n", ms.m, r,
                             (unsigned long long)ms.counts[r],
                             (int)ms.coprime[r], ms.z[r]);
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "transitions.csv", ok)) {
        std::fprintf(f, "q,from,to,obs,train,test,bias\n");
        const uint32_t q = R.trans.q;
        for (uint32_t i = 0; i < q; ++i)
            for (uint32_t j = 0; j < q; ++j) {
                const size_t idx = (size_t)i * q + j;
                std::fprintf(f, "%u,%u,%u,%llu,%llu,%llu,%.6f\n", q, i, j,
                             (unsigned long long)R.trans.obs[idx],
                             (unsigned long long)R.trans.train[idx],
                             (unsigned long long)R.trans.test_[idx],
                             R.trans.bias[idx]);
            }
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "joint_classes.csv", ok)) {
        std::fprintf(f, "L,class,coprime,count,count_train\n");
        for (uint32_t r = 0; r < R.cfg.L; ++r)
            std::fprintf(f, "%u,%u,%d,%u,%u\n", R.cfg.L, r,
                         gcd_u64(r, R.cfg.L) == 1 ? 1 : 0,
                         R.jointL[r], R.jointL_train[r]);
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "gaps.csv", ok)) {
        std::fprintf(f, "gap,count,count_train\n");
        for (size_t g = 1; g < R.gap_hist.size(); ++g)
            if (R.gap_hist[g])
                std::fprintf(f, "%zu,%llu,%llu\n", g, (unsigned long long)R.gap_hist[g],
                             (unsigned long long)(g < R.gap_hist_train.size() ? R.gap_hist_train[g] : 0));
        if (R.gap_overflow)
            std::fprintf(f, "overflow_gt_1023,%llu\n", (unsigned long long)R.gap_overflow);
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "pairspace.csv", ok)) {
        std::fprintf(f, "residue,gap,count\n");
        const uint32_t q = R.cfg.q;
        if (R.pair_gap.size() >= (size_t)q * 1024)
            for (uint32_t i = 0; i < q; ++i)
                for (uint32_t g = 1; g < 1024; ++g) {
                    const uint64_t v = R.pair_gap[(size_t)i * 1024 + g];
                    if (v) std::fprintf(f, "%u,%u,%llu\n", i, g, (unsigned long long)v);
                }
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "races.csv", ok)) {
        std::fprintf(f, "x,race_mod4_3_minus_1,race_mod3_2_minus_1\n");
        for (size_t i = 0; i < R.cp_x.size(); ++i)
            std::fprintf(f, "%.0f,%.0f,%.0f\n", R.cp_x[i], R.race4[i], R.race3[i]);
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "topology_nodes.csv", ok)) {
        std::fprintf(f, "class,phi_zscore,embed_x,embed_y,embed_z\n");
        for (size_t i = 0; i < R.topo.nodes.size(); ++i) {
            const bool has_e = R.topo.embed3.size() >= (i + 1) * 3;
            const bool has_p = R.topo.phi.size() > i;
            std::fprintf(f, "%u,%.6f,%.8f,%.8f,%.8f\n", R.topo.nodes[i],
                         has_p ? R.topo.phi[i] : 0.0,
                         has_e ? R.topo.embed3[i * 3 + 0] : 0.0,
                         has_e ? R.topo.embed3[i * 3 + 1] : 0.0,
                         has_e ? R.topo.embed3[i * 3 + 2] : 0.0);
        }
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "topology_spectrum.csv", ok)) {
        std::fprintf(f, "mode,eigenvalue,field_power\n");
        for (size_t k = 0; k < R.topo.eigval.size(); ++k)
            std::fprintf(f, "%zu,%.8f,%.8f\n", k, R.topo.eigval[k],
                         k < R.topo.power.size() ? R.topo.power[k] : 0.0);
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "patterns.csv", ok)) {
        std::fprintf(f, "q,chi2,pval,mean_diag_bias,acc_argmax,acc_uniform,gain,"
                        "acc_gapmodel,gain_beyond,chi2_gapnull,pval_gapnull,resid_diag\n");
        for (const auto& row : R.patterns.rows)
            std::fprintf(f, "%u,%.6f,%.6g,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f,%.6g,%.6f\n",
                         row.q, row.chi2, row.pval, row.diag_bias,
                         row.acc, row.acc_uniform, row.gain,
                         row.acc_gapmodel, row.gain_beyond,
                         row.chi2_gap, row.pval_gap, row.resid_diag);
        std::fprintf(f, "# order2 q=%u test_triples=%llu acc_order1=%.8f acc_order2=%.8f\n",
                     R.patterns.q2, (unsigned long long)R.patterns.test_triples,
                     R.patterns.acc_order1, R.patterns.acc_order2);
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "hardy_littlewood.csv", ok)) {
        std::fprintf(f, "g,singular_series,pairs_measured,pairs_predicted,ratio\n");
        for (size_t i = 0; i < R.hl.gaps.size(); ++i)
            std::fprintf(f, "%u,%.8f,%llu,%.2f,%.6f\n", R.hl.gaps[i], R.hl.singular[i],
                         (unsigned long long)R.hl.pairs[i], R.hl.predicted[i], R.hl.ratio[i]);
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "gap_memory.csv", ok)) {
        std::fprintf(f, "gap,cond_mean_next_gap,count\n");
        for (size_t i = 0; i < R.gapgap.cond_x.size(); ++i)
            std::fprintf(f, "%.0f,%.4f,%.0f\n", R.gapgap.cond_x[i],
                         R.gapgap.cond_mean[i], R.gapgap.cond_n[i]);
        std::fclose(f);
    }

    if (FILE* f = open_csv(dir, "prediction.csv", ok)) {
        std::fprintf(f, "coverage,precision,lift\n");
        for (size_t i = 0; i < R.pred.coverage.size(); ++i)
            std::fprintf(f, "%.6f,%.8f,%.6f\n", R.pred.coverage[i],
                         R.pred.precision[i], R.pred.lift[i]);
        std::fclose(f);
    }

    return ok;
}

} // namespace prt
