#include "core/stats.h"
#include <cmath>

namespace prt {

uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b) { uint64_t t = a % b; a = b; b = t; }
    return a;
}

double chi2_pval(double chi2, int dof) {
    if (dof <= 0) return 1.0;
    if (chi2 <= 0) return 1.0;
    // Wilson–Hilferty: (chi2/dof)^(1/3) is ~normal
    const double k = (double)dof;
    const double t = std::cbrt(chi2 / k);
    const double mu = 1.0 - 2.0 / (9.0 * k);
    const double sd = std::sqrt(2.0 / (9.0 * k));
    const double zval = (t - mu) / sd;
    return 0.5 * std::erfc(zval / std::sqrt(2.0));
}

uint64_t count_congruent(uint64_t x, uint32_t r, uint32_t L) {
    if (L == 0) return 0;
    if (r == 0) return x / L;
    if (x < r) return 0;
    return (x - r) / L + 1;
}

void finalize_mod_stats(ModulusStats& ms) {
    const uint32_t m = ms.m;
    ms.coprime.assign(m, 0);
    ms.z.assign(m, 0.0);
    uint32_t phi = 0;
    uint64_t total_coprime = 0;
    for (uint32_t r = 0; r < m; ++r) {
        if (gcd_u64(r, m) == 1) {
            ms.coprime[r] = 1;
            ++phi;
            total_coprime += ms.counts[r];
        }
    }
    if (phi < 2 || total_coprime == 0) { ms.chi2 = 0; ms.dof = 0; ms.pval = 1; return; }

    const double E = (double)total_coprime / (double)phi;
    const double sd = std::sqrt(E);
    double chi2 = 0;
    ms.max_abs_z = 0;
    for (uint32_t r = 0; r < m; ++r) {
        if (!ms.coprime[r]) continue;
        const double d = (double)ms.counts[r] - E;
        ms.z[r] = d / sd;
        chi2 += d * d / E;
        if (std::fabs(ms.z[r]) > ms.max_abs_z) {
            ms.max_abs_z = std::fabs(ms.z[r]);
            ms.argmax_class = r;
        }
    }
    ms.chi2 = chi2;
    ms.dof = (int)phi - 1;
    ms.pval = chi2_pval(chi2, ms.dof);
}

void finalize_trans_stats(TransStats& ts) {
    const uint32_t q = ts.q;
    ts.classes.clear();
    for (uint32_t r = 0; r < q; ++r)
        if (gcd_u64(r, q) == 1) ts.classes.push_back(r);

    ts.bias.assign((size_t)q * q, 0.0);
    ts.test_.assign((size_t)q * q, 0);
    for (size_t i = 0; i < ts.obs.size(); ++i)
        ts.test_[i] = ts.obs[i] - ts.train[i];

    // Row/column sums over coprime cells
    uint64_t total = 0;
    std::vector<uint64_t> row(q, 0), col(q, 0);
    for (uint32_t i : ts.classes)
        for (uint32_t j : ts.classes) {
            const uint64_t v = ts.obs[(size_t)i * q + j];
            row[i] += v; col[j] += v; total += v;
        }
    if (total == 0) { ts.chi2 = 0; ts.dof = 0; ts.pval = 1; return; }

    double chi2 = 0, diag_sum = 0;
    int diag_n = 0;
    for (uint32_t i : ts.classes)
        for (uint32_t j : ts.classes) {
            const double E = (double)row[i] * (double)col[j] / (double)total;
            if (E <= 0) continue;
            const double o = (double)ts.obs[(size_t)i * q + j];
            ts.bias[(size_t)i * q + j] = o / E - 1.0;
            chi2 += (o - E) * (o - E) / E;
            if (i == j) { diag_sum += o / E - 1.0; ++diag_n; }
        }
    const int k = (int)ts.classes.size();
    ts.chi2 = chi2;
    ts.dof = (k - 1) * (k - 1);
    ts.pval = chi2_pval(chi2, ts.dof);
    ts.mean_diag_bias = diag_n ? diag_sum / diag_n : 0.0;
}

std::vector<uint32_t> occupied_classes(uint32_t q, const uint64_t* obs) {
    std::vector<uint64_t> marg(q, 0);
    uint64_t total = 0;
    for (uint32_t i = 0; i < q; ++i)
        for (uint32_t j = 0; j < q; ++j) {
            const uint64_t v = obs[(size_t)i * q + j];
            marg[i] += v;
            marg[j] += v;
            total += v;
        }
    const uint64_t thresh = std::max<uint64_t>(8, total / (1000ull * q));
    std::vector<uint32_t> classes;
    for (uint32_t r = 0; r < q; ++r)
        if (marg[r] >= thresh) classes.push_back(r);
    return classes;
}

void finalize_mod_stats_uniform(ModulusStats& ms) {
    const uint32_t m = ms.m;
    ms.coprime.assign(m, 1);
    ms.z.assign(m, 0.0);
    uint64_t total = 0;
    for (uint32_t r = 0; r < m; ++r) total += ms.counts[r];
    if (m < 2 || total == 0) { ms.chi2 = 0; ms.dof = 0; ms.pval = 1; return; }
    const double E = (double)total / (double)m;
    const double sd = std::sqrt(E);
    double chi2 = 0;
    ms.max_abs_z = 0;
    for (uint32_t r = 0; r < m; ++r) {
        const double d = (double)ms.counts[r] - E;
        ms.z[r] = d / sd;
        chi2 += d * d / E;
        if (std::fabs(ms.z[r]) > ms.max_abs_z) {
            ms.max_abs_z = std::fabs(ms.z[r]);
            ms.argmax_class = r;
        }
    }
    ms.chi2 = chi2;
    ms.dof = (int)m - 1;
    ms.pval = chi2_pval(chi2, ms.dof);
}

PatternRow analyze_transition_general(uint32_t q, const uint64_t* obs, const uint64_t* train) {
    PatternRow row;
    row.q = q;
    const std::vector<uint32_t> classes = occupied_classes(q, obs);
    const int k = (int)classes.size();
    if (k < 2) return row;
    row.acc_uniform = 1.0 / (double)k;

    uint64_t total = 0;
    std::vector<uint64_t> rs(q, 0), cs(q, 0);
    for (uint32_t i : classes)
        for (uint32_t j : classes) {
            const uint64_t v = obs[(size_t)i * q + j];
            rs[i] += v; cs[j] += v; total += v;
        }
    if (total == 0) return row;

    double chi2 = 0, diag = 0;
    int dn = 0;
    for (uint32_t i : classes)
        for (uint32_t j : classes) {
            const double E = (double)rs[i] * (double)cs[j] / (double)total;
            if (E <= 0) continue;
            const double o = (double)obs[(size_t)i * q + j];
            chi2 += (o - E) * (o - E) / E;
            if (i == j) { diag += o / E - 1.0; ++dn; }
        }
    row.chi2 = chi2;
    row.pval = chi2_pval(chi2, (k - 1) * (k - 1));
    row.diag_bias = dn ? diag / dn : 0.0;

    uint64_t tot = 0, hit = 0;
    for (uint32_t i : classes) {
        uint64_t best = 0;
        uint32_t bj = classes[0];
        for (uint32_t j : classes) {
            const uint64_t v = train[(size_t)i * q + j];
            if (v > best) { best = v; bj = j; }
        }
        for (uint32_t j : classes) {
            const uint64_t v = obs[(size_t)i * q + j] - train[(size_t)i * q + j];
            tot += v;
            if (j == bj) hit += v;
        }
    }
    if (tot > 0) row.acc = (double)hit / (double)tot;
    row.gain = row.acc - row.acc_uniform;

    // difference-model null by folding along diagonals: gf[d] = #transitions
    // with (j - i) = d (mod q)
    std::vector<double> gf(q, 0.0), gft(q, 0.0);
    for (uint32_t i : classes)
        for (uint32_t j : classes) {
            const uint32_t d = (j + q - i) % q;
            gf[d]  += (double)obs[(size_t)i * q + j];
            gft[d] += (double)train[(size_t)i * q + j];
        }

    double chi2g = 0, diag_resid = 0;
    int dn2 = 0;
    for (uint32_t i : classes) {
        double Si = 0;
        for (uint32_t j : classes) Si += gf[(j + q - i) % q];
        if (Si <= 0) continue;
        for (uint32_t j : classes) {
            const double E = (double)rs[i] * gf[(j + q - i) % q] / Si;
            if (E <= 0) continue;
            const double o = (double)obs[(size_t)i * q + j];
            chi2g += (o - E) * (o - E) / E;
            if (i == j) { diag_resid += o / E - 1.0; ++dn2; }
        }
    }
    row.chi2_gap = chi2g;
    row.pval_gap = chi2_pval(chi2g, (k - 1) * (k - 1));
    row.resid_diag = dn2 ? diag_resid / dn2 : 0.0;

    uint64_t gtot = 0, ghit = 0;
    for (uint32_t i : classes) {
        double best = -1.0;
        uint32_t bj = classes[0];
        for (uint32_t j : classes) {
            const double v = gft[(j + q - i) % q];
            if (v > best) { best = v; bj = j; }
        }
        for (uint32_t j : classes) {
            const uint64_t v = obs[(size_t)i * q + j] - train[(size_t)i * q + j];
            gtot += v;
            if (j == bj) ghit += v;
        }
    }
    if (gtot > 0) row.acc_gapmodel = (double)ghit / (double)gtot;
    row.gain_beyond = row.acc - row.acc_gapmodel;
    return row;
}

PatternRow analyze_transition_q(uint32_t q, const uint64_t* obs, const uint64_t* train,
                                const uint64_t* gap_hist, const uint64_t* gap_hist_train,
                                size_t gap_n) {
    PatternRow row;
    row.q = q;
    std::vector<uint32_t> classes;
    for (uint32_t r = 0; r < q; ++r)
        if (gcd_u64(r, q) == 1) classes.push_back(r);
    const int k = (int)classes.size();
    if (k < 2) return row;
    row.acc_uniform = 1.0 / (double)k;

    uint64_t total = 0;
    std::vector<uint64_t> rs(q, 0), cs(q, 0);
    for (uint32_t i : classes)
        for (uint32_t j : classes) {
            const uint64_t v = obs[(size_t)i * q + j];
            rs[i] += v; cs[j] += v; total += v;
        }
    if (total == 0) return row;

    double chi2 = 0, diag = 0;
    int dn = 0;
    for (uint32_t i : classes)
        for (uint32_t j : classes) {
            const double E = (double)rs[i] * (double)cs[j] / (double)total;
            if (E <= 0) continue;
            const double o = (double)obs[(size_t)i * q + j];
            chi2 += (o - E) * (o - E) / E;
            if (i == j) { diag += o / E - 1.0; ++dn; }
        }
    row.chi2 = chi2;
    row.pval = chi2_pval(chi2, (k - 1) * (k - 1));
    row.diag_bias = dn ? diag / dn : 0.0;

    // out-of-sample: argmax per row from train, scored on obs - train
    uint64_t tot = 0, hit = 0;
    for (uint32_t i : classes) {
        uint64_t best = 0;
        uint32_t bj = classes[0];
        for (uint32_t j : classes) {
            const uint64_t v = train[(size_t)i * q + j];
            if (v > best) { best = v; bj = j; }
        }
        for (uint32_t j : classes) {
            const uint64_t v = obs[(size_t)i * q + j] - train[(size_t)i * q + j];
            tot += v;
            if (j == bj) hit += v;
        }
    }
    if (tot > 0) row.acc = (double)hit / (double)tot;
    row.gain = row.acc - row.acc_uniform;

    // ---- gap-model null ----
    // Fold the measured gap histograms mod q: gf[d] = #gaps with g = d (mod q).
    std::vector<double> gf(q, 0.0), gft(q, 0.0);
    for (size_t g = 1; g < gap_n; ++g) {
        if (gap_hist[g])       gf[g % q]  += (double)gap_hist[g];
        if (gap_hist_train[g]) gft[g % q] += (double)gap_hist_train[g];
    }

    // chi2 of the observed matrix against E[i][j] = R_i * gf[d] / S_i, where
    // d = (j - i) mod q and S_i renormalizes over the wheel-allowed targets.
    double chi2g = 0, diag_resid = 0;
    int dn2 = 0;
    for (uint32_t i : classes) {
        double Si = 0;
        for (uint32_t j : classes) Si += gf[(j + q - i % q) % q];
        if (Si <= 0) continue;
        for (uint32_t j : classes) {
            const double E = (double)rs[i] * gf[(j + q - i % q) % q] / Si;
            if (E <= 0) continue;
            const double o = (double)obs[(size_t)i * q + j];
            chi2g += (o - E) * (o - E) / E;
            if (i == j) { diag_resid += o / E - 1.0; ++dn2; }
        }
    }
    row.chi2_gap = chi2g;
    row.pval_gap = chi2_pval(chi2g, (k - 1) * (k - 1));
    row.resid_diag = dn2 ? diag_resid / dn2 : 0.0;

    // gap-model predictor: from residue i predict the wheel-allowed target
    // whose gap class is most frequent in the TRAINING half.
    uint64_t gtot = 0, ghit = 0;
    for (uint32_t i : classes) {
        double best = -1.0;
        uint32_t bj = classes[0];
        for (uint32_t j : classes) {
            const double v = gft[(j + q - i % q) % q];
            if (v > best) { best = v; bj = j; }
        }
        for (uint32_t j : classes) {
            const uint64_t v = obs[(size_t)i * q + j] - train[(size_t)i * q + j];
            gtot += v;
            if (j == bj) ghit += v;
        }
    }
    if (gtot > 0) row.acc_gapmodel = (double)ghit / (double)gtot;
    row.gain_beyond = row.acc - row.acc_gapmodel;
    return row;
}

void analyze_order2(PatternScan& ps, uint32_t q,
                    const std::vector<uint64_t>& o2_obs,
                    const std::vector<uint64_t>& o2_train,
                    const std::vector<uint64_t>& t1_train) {
    ps.q2 = q;
    std::vector<uint32_t> classes;
    for (uint32_t r = 0; r < q; ++r)
        if (gcd_u64(r, q) == 1) classes.push_back(r);
    const int k = (int)classes.size();
    if (k < 2) return;
    ps.acc_uniform = 1.0 / (double)k;

    // order-1 predictor: argmax_k t1_train[j][k]
    std::vector<uint32_t> amax1(q, classes[0]);
    for (uint32_t j : classes) {
        uint64_t best = 0;
        for (uint32_t kk : classes) {
            const uint64_t v = t1_train[(size_t)j * q + kk];
            if (v > best) { best = v; amax1[j] = kk; }
        }
    }
    // order-2 predictor: argmax_k o2_train[(i,j)][k], falling back to order-1
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
    ps.test_triples = tot;
    if (tot > 0) {
        ps.acc_order1 = (double)hit1 / (double)tot;
        ps.acc_order2 = (double)hit2 / (double)tot;
    }
}

double li2_integral(double x) {
    if (x <= 2.0) return 0.0;
    const double a = std::log(2.0), b = std::log(x);
    const int n = 200000;                        // Simpson panels
    const double h = (b - a) / n;
    auto f = [](double u) { return std::exp(u) / (u * u); };
    double s = f(a) + f(b);
    for (int i = 1; i < n; ++i) s += f(a + h * i) * ((i & 1) ? 4.0 : 2.0);
    return s * h / 3.0;
}

double hl_singular_series(uint32_t g) {
    if (g < 2 || (g & 1)) return 0.0;
    const double C2 = 0.6601618158468696;        // twin prime constant
    double c = 2.0 * C2;
    uint32_t m = g;
    while ((m & 1) == 0) m >>= 1;
    for (uint32_t p = 3; (uint64_t)p * p <= m; p += 2)
        if (m % p == 0) {
            c *= (double)(p - 1) / (double)(p - 2);
            while (m % p == 0) m /= p;
        }
    if (m > 1) c *= (double)(m - 1) / (double)(m - 2);
    return c;
}

void finalize_gapgap(GapGapReport& G, const uint64_t* obs, const uint64_t* train, uint32_t maxg) {
    // moments for the Pearson correlation
    double n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    std::vector<double> row(maxg, 0.0), col(maxg, 0.0);
    for (uint32_t g1 = 1; g1 < maxg; ++g1)
        for (uint32_t g2 = 1; g2 < maxg; ++g2) {
            const double v = (double)obs[(size_t)g1 * maxg + g2];
            if (v <= 0) continue;
            n += v;
            sx += v * g1; sy += v * g2;
            sxx += v * (double)g1 * g1;
            syy += v * (double)g2 * g2;
            sxy += v * (double)g1 * g2;
            row[g1] += v; col[g2] += v;
        }
    if (n < 100) return;
    const double mx = sx / n, my = sy / n;
    const double vx = sxx / n - mx * mx, vy = syy / n - my * my;
    const double cov = sxy / n - mx * my;
    if (vx > 0 && vy > 0) G.pearson = cov / std::sqrt(vx * vy);
    G.mean_gap = mx;

    // chi2 vs independence over cells with a stable expectation
    double chi2 = 0;
    int rows_used = 0, cols_used = 0;
    std::vector<uint8_t> rused(maxg, 0), cused(maxg, 0);
    for (uint32_t g1 = 1; g1 < maxg; ++g1) {
        if (row[g1] <= 0) continue;
        for (uint32_t g2 = 1; g2 < maxg; ++g2) {
            const double E = row[g1] * col[g2] / n;
            if (E < 5.0) continue;
            const double o = (double)obs[(size_t)g1 * maxg + g2];
            chi2 += (o - E) * (o - E) / E;
            if (!rused[g1]) { rused[g1] = 1; ++rows_used; }
            if (!cused[g2]) { cused[g2] = 1; ++cols_used; }
        }
    }
    G.chi2 = chi2;
    G.dof = std::max(1, (rows_used - 1) * (cols_used - 1));
    G.pval = chi2_pval(chi2, G.dof);

    // conditional mean curve E[g_{n+1} | g_n]
    for (uint32_t g1 = 1; g1 < maxg; ++g1) {
        if (row[g1] < 1000) continue;
        double s = 0;
        for (uint32_t g2 = 1; g2 < maxg; ++g2)
            s += (double)obs[(size_t)g1 * maxg + g2] * g2;
        G.cond_x.push_back(g1);
        G.cond_mean.push_back(s / row[g1]);
        G.cond_n.push_back(row[g1]);
    }

    // out-of-sample R^2: conditional-mean predictor (train) vs global train mean
    std::vector<double> trow(maxg, 0.0), tmean(maxg, 0.0);
    double tn = 0, tsum = 0;
    for (uint32_t g1 = 1; g1 < maxg; ++g1)
        for (uint32_t g2 = 1; g2 < maxg; ++g2) {
            const double v = (double)train[(size_t)g1 * maxg + g2];
            if (v <= 0) continue;
            trow[g1] += v;
            tmean[g1] += v * g2;
            tn += v;
            tsum += v * g2;
        }
    if (tn < 100) return;
    const double tglobal = tsum / tn;
    for (uint32_t g1 = 1; g1 < maxg; ++g1)
        tmean[g1] = (trow[g1] >= 100) ? tmean[g1] / trow[g1] : tglobal;
    double sse_c = 0, sse_b = 0;
    for (uint32_t g1 = 1; g1 < maxg; ++g1)
        for (uint32_t g2 = 1; g2 < maxg; ++g2) {
            const double v = (double)(obs[(size_t)g1 * maxg + g2] - train[(size_t)g1 * maxg + g2]);
            if (v <= 0) continue;
            const double dc = (double)g2 - tmean[g1];
            const double db = (double)g2 - tglobal;
            sse_c += v * dc * dc;
            sse_b += v * db * db;
        }
    if (sse_b > 0) G.r2_oos = 1.0 - sse_c / sse_b;
}

void finalize_hl(HLReport& H, const std::vector<uint64_t>& pair_counts, uint64_t N) {
    H.li2 = li2_integral((double)N);
    double devsum = 0;
    int cnt = 0;
    for (uint32_t g = 2; g < pair_counts.size() && g <= 120; g += 2) {
        const double c = hl_singular_series(g);
        const double pred = c * H.li2;
        if (pred <= 0) continue;
        const double r = (double)pair_counts[g] / pred;
        H.gaps.push_back(g);
        H.pairs.push_back(pair_counts[g]);
        H.singular.push_back(c);
        H.predicted.push_back(pred);
        H.ratio.push_back(r);
        const double dev = std::fabs(r - 1.0);
        devsum += dev;
        ++cnt;
        if (dev > H.max_abs_dev) { H.max_abs_dev = dev; H.argmax_g = g; }
    }
    if (cnt) H.mean_abs_dev = devsum / cnt;
}

} // namespace prt
