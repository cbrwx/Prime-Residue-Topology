// All ImGui/ImPlot panels.
#include "app/app_state.h"
#include <commdlg.h>
#include "app/clouds.h"
#include "app/run_export.h"
#include "core/stats.h"
#include "imgui.h"
#include "implot.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

using prt::Results;

// ---------------------------------------------------------------- helpers

static std::string fmt_u64(uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)v);
    std::string s(buf);
    for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert(i, "'");
    return s;
}

static std::vector<int> divisors_of(uint32_t L, int lo, int hi) {
    std::vector<int> d;
    for (uint32_t i = (uint32_t)lo; i <= (uint32_t)hi && i <= L; ++i)
        if (L % i == 0) d.push_back((int)i);
    return d;
}

static bool divisor_combo(const char* label, int* value, uint32_t L, int lo, int hi) {
    std::vector<int> divs = divisors_of(L, lo, hi);
    if (divs.empty()) return false;
    if (std::find(divs.begin(), divs.end(), *value) == divs.end())
        *value = divs[divs.size() / 2];
    bool changed = false;
    char cur[16];
    std::snprintf(cur, sizeof(cur), "%d", *value);
    if (ImGui::BeginCombo(label, cur)) {
        for (int d : divs) {
            char t[16];
            std::snprintf(t, sizeof(t), "%d", d);
            if (ImGui::Selectable(t, d == *value)) { *value = d; changed = true; }
        }
        ImGui::EndCombo();
    }
    return changed;
}

// ---------------------------------------------------------------- controls

static const double N_PRESETS[] = {1e6, 1e7, 1e8, 1e9, 2e9, 5e9, 1e10, 1e11};
static const char* N_LABELS[] = {"10^6", "10^7", "10^8", "10^9", "2*10^9", "5*10^9", "10^10", "10^11"};

static void draw_controls(AppState& st) {
    ImGui::SeparatorText("Run configuration");

    for (int i = 0; i < IM_ARRAYSIZE(N_PRESETS); ++i)
        if (st.cfg.N == (uint64_t)N_PRESETS[i] && st.n_choice != i) st.n_choice = i;
    if (ImGui::Combo("N (range)", &st.n_choice, N_LABELS, IM_ARRAYSIZE(N_LABELS)))
        st.cfg.N = (uint64_t)N_PRESETS[st.n_choice];
    if (st.cfg.N >= 100'000'000'000ull)
        ImGui::TextDisabled("10^11: ~4.1B primes, expect ~10 min per run");

    const char* L_LABELS[] = {"210 (2*3*5*7)", "2310 (2*3*5*7*11)"};
    if (ImGui::Combo("L (torus modulus)", &st.L_choice, L_LABELS, 2))
        st.cfg.L = (st.L_choice == 0) ? 210u : 2310u;

    int q = (int)st.cfg.q;
    if (ImGui::InputInt("q (transitions)", &q)) st.cfg.q = (uint32_t)std::clamp(q, 3, 50);
    ImGui::SetItemTooltip("Modulus for the consecutive-prime transition matrix.\nq=10 shows the last-digit pattern.");

    int knn = st.cfg.knn;
    if (ImGui::SliderInt("kNN (torus graph)", &knn, 4, 24)) st.cfg.knn = knn;

    static const char* SAMPLE_LABELS[] = {"500k", "1M", "2M", "5M", "10M", "20M", "50M"};
    static const uint32_t SAMPLE_VALUES[] = {500'000, 1'000'000, 2'000'000, 5'000'000,
                                             10'000'000, 20'000'000, 50'000'000};
    static int sample_sel = 6;
    if (ImGui::Combo("prime sample (3D)", &sample_sel, SAMPLE_LABELS, 7))
        st.cfg.sample_target = SAMPLE_VALUES[sample_sel];
    ImGui::SetItemTooltip("How many primes are kept in RAM for the raw 3D views.\n"
                          "50M keeps every prime at N = 10^9 (~400 MB RAM).");

    ImGui::Spacing();
    const bool running = st.running.load();
    if (!running) {
        if (ImGui::Button("Run pipeline", ImVec2(-1, 0))) st.start_run();
    } else {
        if (ImGui::Button("Cancel", ImVec2(-1, 0))) st.cancel_run();
    }

    if (running) {
        const float frac = (float)st.progress.frac.load();
        char overlay[96];
        std::snprintf(overlay, sizeof(overlay), "%s  %.0f%%",
                      prt::stage_name(st.progress.stage.load()), 100.0 * frac);
        ImGui::ProgressBar(frac, ImVec2(-1, 0), overlay);
    }

    auto R = st.get_results();
    ImGui::Spacing();
    ImGui::SeparatorText("Last run");
    if (R) {
        ImGui::Text("N = %s", fmt_u64(R->cfg.N).c_str());
        ImGui::Text("pi(N) = %s", fmt_u64(R->prime_count).c_str());
        ImGui::Text("last prime = %s", fmt_u64(R->last_prime).c_str());
        ImGui::Text("max gap = %s", fmt_u64(R->max_gap).c_str());
        ImGui::Text("sieve %.2fs, analysis %.2fs", R->t_sieve, R->t_analysis);
        if (!st.last_export_path.empty()) {
            ImGui::Spacing();
            ImGui::TextWrapped("logged to %s", st.last_export_path.c_str());
        }
    } else {
        ImGui::TextDisabled("no data yet - hit Run");
    }

    ImGui::Spacing();
    ImGui::TextDisabled("See the Info tab for how the pipeline works.");
}

// ---------------------------------------------------------------- overview

static void draw_overview(const Results& R) {
    ImGui::SeparatorText("What this run found");

    ImGui::TextWrapped(
        "Full sieve of [2, %s]: %s primes, ending at %s.",
        fmt_u64(R.cfg.N).c_str(), fmt_u64(R.prime_count).c_str(), fmt_u64(R.last_prime).c_str());
    ImGui::Spacing();

    double worst_p = 1.0;
    uint32_t worst_m = 0;
    for (const auto& ms : R.mods)
        if (ms.pval < worst_p) { worst_p = ms.pval; worst_m = ms.m; }

    ImGui::BulletText("Equidistribution: across the statistics moduli the smallest");
    ImGui::TextWrapped("   chi-square p-value is %.3g (mod %u). Values near or above ~0.01 mean prime "
                       "counts per residue class match the Dirichlet prediction - no static pattern "
                       "in single-residue space.", worst_p, worst_m);
    ImGui::Spacing();

    ImGui::BulletText("Consecutive primes are NOT independent (the real higher-dim pattern):");
    ImGui::TextWrapped("   transition chi-square mod %u = %.0f (p = %.3g). Same-residue repetition is "
                       "suppressed by %.1f%% on average - the Lemke Oliver-Soundararajan repulsion.",
                       R.trans.q, R.trans.chi2, R.trans.pval, -100.0 * R.trans.mean_diag_bias);
    ImGui::Spacing();

    int pos4 = 0;
    for (double v : R.race4) if (v > 0) ++pos4;
    ImGui::BulletText("Chebyshev race: pi(x;4,3) leads pi(x;4,1) at %d of %d checkpoints.",
                      pos4, (int)R.race4.size());
    ImGui::Spacing();

    ImGui::BulletText("Residue torus (L = %u, %d classes): Betti_0 = %d, spectral gap = %.4f,",
                      R.topo.L, (int)R.topo.nodes.size(), R.topo.betti0, R.topo.spectral_gap);
    ImGui::TextWrapped("   Moran's I of the measured density field = %+.4f "
                       "(negative = neighbouring classes anti-correlate slightly).", R.topo.moran);
    ImGui::Spacing();

    auto lift_at = [&](double cov) {
        for (size_t i = 0; i < R.pred.coverage.size(); ++i)
            if (R.pred.coverage[i] >= cov) return R.pred.lift[i];
        return 1.0;
    };
    const prt::PatternRow* pq0 = nullptr;
    const prt::PatternRow* pbest = nullptr;
    for (const auto& row : R.patterns.rows) {
        if (row.q == R.trans.q) pq0 = &row;
        if (!pbest || row.gain_beyond > pbest->gain_beyond) pbest = &row;
    }
    if (pq0 && pbest) {
        ImGui::BulletText("Pattern identity (gap-model decomposition):");
        ImGui::TextWrapped(
            "   At q = %u: raw gain %+.2f pp = %+.2f pp gap frequencies + wheel, plus "
            "%+.2f pp genuine residue-gap correlation. Strongest genuine signal: q = %u "
            "(%+.2f pp). First name: the Lemke Oliver-Soundararajan bias; family name: "
            "Hardy-Littlewood k-tuple correlations. Details in the Patterns tab.",
            pq0->q, 100.0 * pq0->gain, 100.0 * (pq0->acc_gapmodel - pq0->acc_uniform),
            100.0 * pq0->gain_beyond, pbest->q, 100.0 * pbest->gain_beyond);
        ImGui::Spacing();
    }

    ImGui::BulletText("Out-of-sample prediction (train first half, test second half):");
    ImGui::TextWrapped("   Static class density gives lift %.4fx over the wheel at 25%% coverage - "
                       "residue density alone adds ~nothing beyond divisibility, exactly as theory "
                       "predicts. But the transition structure predicts the NEXT prime's residue at "
                       "%.2f%% vs %.2f%% uniform baseline - real out-of-sample information living in "
                       "the correlations, not the densities.",
                       lift_at(0.25), 100.0 * R.pred.trans_acc, 100.0 * R.pred.trans_base_uniform);
}

// ---------------------------------------------------------------- residue classes

static void draw_classes(AppState& st, const Results& R) {
    // modulus selector
    std::vector<std::string> labels;
    for (const auto& ms : R.mods) labels.push_back("mod " + std::to_string(ms.m));
    std::vector<const char*> litems;
    for (auto& s : labels) litems.push_back(s.c_str());
    if (st.mod_sel >= (int)R.mods.size()) st.mod_sel = 0;
    ImGui::SetNextItemWidth(160);
    ImGui::Combo("##modsel", &st.mod_sel, litems.data(), (int)litems.size());
    const auto& ms = R.mods[st.mod_sel];

    uint32_t phi = 0;
    uint64_t tot = 0;
    for (uint32_t r = 0; r < ms.m; ++r)
        if (ms.coprime[r]) { ++phi; tot += ms.counts[r]; }
    const double E = phi ? (double)tot / phi : 0.0;
    const double sd = std::sqrt(std::max(1.0, E));

    ImGui::SameLine();
    ImGui::Text("chi2 = %.2f (dof %d, p = %.3g), max |z| = %.2f at class %u",
                ms.chi2, ms.dof, ms.pval, ms.max_abs_z, ms.argmax_class);

    if (ImPlot::BeginPlot("prime counts per residue class", ImVec2(-1, 320))) {
        ImPlot::SetupAxes("residue class r", "primes with p = r (mod m)",
                          ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        static std::vector<double> xs, ys;
        xs.clear(); ys.clear();
        for (uint32_t r = 0; r < ms.m; ++r) {
            xs.push_back(r);
            ys.push_back((double)ms.counts[r]);
        }
        const double bx[2] = {-0.5, (double)ms.m - 0.5};
        const double be1[2] = {E - 2 * sd, E - 2 * sd}, be2[2] = {E + 2 * sd, E + 2 * sd};
        const double bE[2] = {E, E};
        ImPlot::SetNextFillStyle(ImVec4(0.9f, 0.7f, 0.2f, 0.25f));
        ImPlot::PlotShaded("Dirichlet +/-2 sigma", bx, be1, be2, 2);
        ImPlot::SetNextLineStyle(ImVec4(0.9f, 0.7f, 0.2f, 1.0f));
        ImPlot::PlotLine("Dirichlet expectation", bx, bE, 2);
        ImPlot::PlotBars("measured", xs.data(), ys.data(), (int)xs.size(), 0.75);
        ImPlot::EndPlot();
    }

    if (ImPlot::BeginPlot("deviation z-scores (coprime classes)", ImVec2(-1, 220))) {
        ImPlot::SetupAxes("residue class r", "z", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        static std::vector<double> xs, zs;
        xs.clear(); zs.clear();
        for (uint32_t r = 0; r < ms.m; ++r)
            if (ms.coprime[r]) { xs.push_back(r); zs.push_back(ms.z[r]); }
        ImPlot::PlotStems("z", xs.data(), zs.data(), (int)xs.size());
        ImPlot::EndPlot();
    }

    ImGui::SeparatorText("All moduli");
    if (ImGui::BeginTable("chitab", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("modulus");
        ImGui::TableSetupColumn("chi2");
        ImGui::TableSetupColumn("dof");
        ImGui::TableSetupColumn("p-value");
        ImGui::TableSetupColumn("max |z| (class)");
        ImGui::TableHeadersRow();
        for (const auto& m2 : R.mods) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%u", m2.m);
            ImGui::TableNextColumn(); ImGui::Text("%.2f", m2.chi2);
            ImGui::TableNextColumn(); ImGui::Text("%d", m2.dof);
            ImGui::TableNextColumn(); ImGui::Text("%.3g", m2.pval);
            ImGui::TableNextColumn(); ImGui::Text("%.2f (%u)", m2.max_abs_z, m2.argmax_class);
        }
        ImGui::EndTable();
    }
}

// ---------------------------------------------------------------- gaps

static void draw_gaps(AppState& st, const Results& R) {
    ImGui::BeginChild("gapsscroll");
    ImGui::Checkbox("log scale", &st.gaps_log);
    ImGui::SameLine();
    ImGui::Text("max gap = %s%s", fmt_u64(R.max_gap).c_str(),
                R.gap_overflow ? " (some gaps > 1023 uncounted!)" : "");
    if (ImPlot::BeginPlot("prime gap histogram", ImVec2(-1, 340))) {
        ImPlot::SetupAxes("gap g = p_{n+1} - p_n", "count",
                          ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        if (st.gaps_log) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        static std::vector<double> xs, ys;
        xs.clear(); ys.clear();
        for (size_t g = 1; g <= R.max_gap && g < R.gap_hist.size(); ++g)
            if (R.gap_hist[g]) { xs.push_back((double)g); ys.push_back((double)R.gap_hist[g]); }
        ImPlot::PlotBars("gaps", xs.data(), ys.data(), (int)xs.size(), 1.6);
        ImPlot::EndPlot();
    }

    ImGui::SeparatorText("Gap memory");
    const auto& G = R.gapgap;
    ImGui::TextWrapped(
        "Do consecutive gaps remember each other? Pearson r = %+.4f, chi2 vs "
        "independence = %.0f (p = %.3g). Predicting the next gap from the current "
        "one (conditional mean trained on the first half, tested on the second) "
        "explains R^2 = %.4f of the variance beyond the global mean.",
        G.pearson, G.chi2, G.pval, G.r2_oos);
    if (!G.cond_x.empty() && ImPlot::BeginPlot("E[ g_{n+1} | g_n ]", ImVec2(-1, 320))) {
        ImPlot::SetupAxes("current gap g_n", "mean next gap",
                          ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::PlotLine("conditional mean", G.cond_x.data(), G.cond_mean.data(),
                         (int)G.cond_x.size());
        const double bx[2] = {G.cond_x.front(), G.cond_x.back()};
        const double by[2] = {G.mean_gap, G.mean_gap};
        ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.8f, 0.8f, 0.7f));
        ImPlot::PlotLine("global mean", bx, by, 2);
        ImPlot::EndPlot();
    }
    ImGui::TextWrapped(
        "A downward-sloping curve means anti-correlation: large gaps tend to be "
        "followed by smaller ones and vice versa.");
    ImGui::EndChild();
}

// ---------------------------------------------------------------- Hardy-Littlewood

static void draw_hl(const Results& R) {
    const auto& H = R.hl;
    if (H.gaps.empty()) { ImGui::TextDisabled("no data"); return; }
    ImGui::BeginChild("hlscroll");
    ImGui::TextWrapped(
        "The Hardy-Littlewood k-tuple conjecture predicts how many prime pairs "
        "(p, p+g) exist up to N: approximately C(g) * Li2(N), where C(g) is the "
        "singular series built from the twin prime constant. The pipeline counts "
        "every such pair for even g <= 120 (a sliding window over the sieve "
        "output) and compares. g = 2 is the twin primes.");
    ImGui::Text("Li2(N) = %.1f   mean |measured/predicted - 1| = %.3f%%   worst: g = %u (%.3f%%)",
                H.li2, 100.0 * H.mean_abs_dev, H.argmax_g, 100.0 * H.max_abs_dev);

    static std::vector<double> xs, ms, pr, rt;
    xs.clear(); ms.clear(); pr.clear(); rt.clear();
    for (size_t i = 0; i < H.gaps.size(); ++i) {
        xs.push_back(H.gaps[i]);
        ms.push_back((double)H.pairs[i]);
        pr.push_back(H.predicted[i]);
        rt.push_back(H.ratio[i]);
    }
    if (ImPlot::BeginPlot("prime pairs (p, p+g): measured vs predicted", ImVec2(-1, 340))) {
        ImPlot::SetupAxes("g", "pairs", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::PlotBars("measured", xs.data(), ms.data(), (int)xs.size(), 1.2);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3.0f);
        ImPlot::PlotScatter("C(g) * Li2(N)", xs.data(), pr.data(), (int)xs.size());
        ImPlot::EndPlot();
    }
    ImGui::TextWrapped(
        "The sawtooth is the singular series: gaps divisible by 3 get a factor 2, "
        "by 5 a factor 4/3, and so on. The prediction (dots) traces the measured "
        "counts (bars) across the whole range.");
    if (ImPlot::BeginPlot("measured / predicted", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("g", "ratio", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::PlotLine("ratio", xs.data(), rt.data(), (int)xs.size());
        const double bx[2] = {xs.front(), xs.back()}, by[2] = {1.0, 1.0};
        ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.8f, 0.8f, 0.7f));
        ImPlot::PlotLine("1.0", bx, by, 2);
        ImPlot::EndPlot();
    }
    ImGui::EndChild();
}

// ---------------------------------------------------------------- transitions

static void draw_transitions(const Results& R) {
    const auto& T = R.trans;
    const int k = (int)T.classes.size();
    if (k == 0) return;

    ImGui::TextWrapped(
        "Bias of consecutive-prime residue transitions mod %u vs independence "
        "(0%% = independent). chi2 = %.0f, dof %d, p = %.3g. Mean diagonal bias "
        "%+.1f%%: primes avoid repeating their own residue class.",
        T.q, T.chi2, T.dof, T.pval, 100.0 * T.mean_diag_bias);

    static std::vector<double> vals;
    vals.assign((size_t)k * k, 0.0);
    double vmax = 0;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            const double b = 100.0 * T.bias[(size_t)T.classes[i] * T.q + T.classes[j]];
            vals[(size_t)i * k + j] = b;
            vmax = std::max(vmax, std::fabs(b));
        }

    static std::vector<std::string> tickstr;
    static std::vector<const char*> ticklab;
    tickstr.clear(); ticklab.clear();
    for (int i = 0; i < k; ++i) tickstr.push_back(std::to_string(T.classes[i]));
    for (auto& s : tickstr) ticklab.push_back(s.c_str());
    static std::vector<double> xt, yt;
    xt.clear(); yt.clear();
    for (int i = 0; i < k; ++i) { xt.push_back(i + 0.5); yt.push_back(k - i - 0.5); }

    ImPlot::PushColormap(ImPlotColormap_RdBu);
    const float side = std::min(ImGui::GetContentRegionAvail().x - 120.0f,
                                ImGui::GetContentRegionAvail().y);
    if (ImPlot::BeginPlot("##transheat", ImVec2(side, side),
                          ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
        ImPlot::SetupAxes("residue of p_{n+1}", "residue of p_n",
                          ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines,
                          ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines);
        ImPlot::SetupAxesLimits(0, k, 0, k, ImPlotCond_Always);
        ImPlot::SetupAxisTicks(ImAxis_X1, xt.data(), k, ticklab.data());
        ImPlot::SetupAxisTicks(ImAxis_Y1, yt.data(), k, ticklab.data());
        ImPlot::PlotHeatmap("bias", vals.data(), k, k, -vmax, vmax,
                            k <= 12 ? "%.1f" : nullptr,
                            ImPlotPoint(0, 0), ImPlotPoint(k, k));
        ImPlot::EndPlot();
    }
    ImGui::SameLine();
    ImPlot::ColormapScale("bias %", -vmax, vmax, ImVec2(90, side));
    ImPlot::PopColormap();
}

// ---------------------------------------------------------------- races

static void draw_races(AppState& st, const Results& R) {
    ImGui::Checkbox("log x", &st.races_logx);
    ImGui::SameLine();
    ImGui::TextWrapped("Chebyshev races: cumulative lead of the 'non-square' team.");
    if (ImPlot::BeginPlot("prime races", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("x", "lead", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        if (st.races_logx) ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        const int n = (int)R.cp_x.size();
        if (n > 0) {
            ImPlot::PlotLine("pi(x;4,3) - pi(x;4,1)", R.cp_x.data(), R.race4.data(), n);
            ImPlot::PlotLine("pi(x;3,2) - pi(x;3,1)", R.cp_x.data(), R.race3.data(), n);
        }
        ImPlot::EndPlot();
    }
}

// ---------------------------------------------------------------- topology

static void draw_topology(const Results& R) {
    const auto& T = R.topo;
    if (T.eigval.empty()) {
        ImGui::TextDisabled("no topology computed");
        return;
    }
    ImGui::TextWrapped(
        "Nodes are the %d residue classes coprime to L = %u placed on a %d-torus "
        "(one angular dimension per prime factor: 2, 3, 5, 7%s). Edges: canonical "
        "torus-lattice steps in every dimension plus k nearest metric neighbours. "
        "Geometry uses residue arithmetic only; the scalar field phi is the "
        "MEASURED prime density z-score per class.",
        (int)T.nodes.size(), T.L, (int)T.torus_dims.size(), T.L == 2310 ? ", 11" : "");
    ImGui::Text("Betti_0 = %d   spectral gap (lambda_2) = %.5f   Cheeger ~ %.5f   Moran's I = %+.4f",
                T.betti0, T.spectral_gap, T.cheeger, T.moran);

    const int show = std::min((int)T.eigval.size(), 80);
    if (ImPlot::BeginPlot("Laplacian spectrum (first modes)", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("mode k", "lambda_k", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        static std::vector<double> xs, ys;
        xs.clear(); ys.clear();
        for (int i = 0; i < show; ++i) { xs.push_back(i); ys.push_back(T.eigval[i]); }
        ImPlot::PlotStems("lambda", xs.data(), ys.data(), show);
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("graph-Fourier power of the density field", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("mode k", "fraction of field energy",
                          ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        static std::vector<double> xs, ys;
        xs.clear(); ys.clear();
        for (int i = 1; i < (int)T.power.size() && i < 200; ++i) {
            xs.push_back(i);
            ys.push_back(T.power[i]);
        }
        ImPlot::PlotBars("power", xs.data(), ys.data(), (int)xs.size(), 0.8);
        ImPlot::EndPlot();
    }
    ImGui::TextWrapped(
        "A flat power spectrum means the prime-density field looks like noise on "
        "the torus (no smooth low-frequency structure); sharp peaks would mean "
        "the bias concentrates in specific harmonics.");

    ImGui::SeparatorText("Character spectrum (canonical basis)");
    if (!T.char_power.empty()) {
        ImGui::TextWrapped(
            "The group (Z/L)* has an exact harmonic basis: its %zu Dirichlet "
            "characters. This is the same decomposition, done in the canonical "
            "basis instead of graph eigenvectors. Top character carries %.2f%% of "
            "the field energy (order %d); real characters (order <= 2, the "
            "quadratic-residue patterns) carry %.2f%%. A flat spectrum here is "
            "equidistribution restated character by character.",
            T.char_power.size(), 100.0 * T.char_power[0], T.char_order[0],
            100.0 * T.char_low_frac);
        static std::vector<double> cxs, cys;
        cxs.clear(); cys.clear();
        for (size_t i = 0; i < T.char_power.size() && i < 60; ++i) {
            cxs.push_back((double)(i + 1));
            cys.push_back(100.0 * T.char_power[i]);
        }
        if (ImPlot::BeginPlot("character energy (top 60, sorted)", ImVec2(-1, 240))) {
            ImPlot::SetupAxes("rank", "energy (%)",
                              ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotBars("energy", cxs.data(), cys.data(), (int)cxs.size(), 0.8);
            ImPlot::EndPlot();
        }
    }
}

// ---------------------------------------------------------------- prediction

static void draw_prediction(const Results& R) {
    const auto& P = R.pred;
    ImGui::TextWrapped(
        "Calibrated on (0, N/2], evaluated on (N/2, N]. Baseline: the wheel "
        "(all classes coprime to L = %u), precision %.4f%%.",
        R.cfg.L, 100.0 * P.base_precision);

    if (ImPlot::BeginPlot("density-ranking lift vs coverage", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("fraction of coprime classes included (best first)",
                          "precision / wheel precision",
                          ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        const int n = (int)P.coverage.size();
        if (n > 0) {
            ImPlot::PlotLine("lift", P.coverage.data(), P.lift.data(), n);
            const double bx[2] = {0, 1}, by[2] = {1, 1};
            ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.8f, 0.8f, 0.7f));
            ImPlot::PlotLine("wheel baseline", bx, by, 2);
        }
        ImPlot::EndPlot();
    }
    ImGui::TextWrapped(
        "Lift ~ 1.0 means that within coprime classes, measured density carries "
        "essentially no extra primality information - Dirichlet wins.");

    ImGui::SeparatorText("Next-prime residue prediction (mod q)");
    if (ImGui::BeginTable("predtab", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        auto row = [](const char* a, double v) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(a);
            ImGui::TableNextColumn(); ImGui::Text("%.2f%%", 100.0 * v);
        };
        row("train-argmax transition predictor (test half)", P.trans_acc);
        row("marginal baseline (always most common residue)", P.trans_base_marginal);
        row("uniform baseline (1 / phi(q))", P.trans_base_uniform);
        ImGui::EndTable();
    }
    ImGui::TextWrapped(
        "The gain over baseline is real out-of-sample structure: consecutive primes "
        "carry information about each other's residues even though single-prime "
        "densities are flat.");
}

// ---------------------------------------------------------------- patterns

static void draw_patterns(const Results& R) {
    const auto& P = R.patterns;
    if (P.rows.empty()) { ImGui::TextDisabled("no pattern scan"); return; }

    ImGui::BeginChild("patscroll");

    ImGui::TextWrapped(
        "Pattern classification: WHERE does the consecutive-prime structure live, HOW "
        "strong is it, WHAT is it made of, and how deep is its memory? For every "
        "modulus q = 3..100 the transition matrix is measured and scored out-of-sample "
        "(trained on the first half, tested on the second). The gap-model null asks: "
        "what if the next gap were independent of the current residue (given the wheel)? "
        "Whatever survives that null is genuine residue-gap correlation.");

    static std::vector<double> xs, gain, beyond, diag;
    xs.clear(); gain.clear(); beyond.clear(); diag.clear();
    for (const auto& row : P.rows) {
        xs.push_back(row.q);
        gain.push_back(100.0 * row.gain);
        beyond.push_back(100.0 * row.gain_beyond);
        diag.push_back(100.0 * row.diag_bias);
    }
    if (ImPlot::BeginPlot("the genuine pattern: gain BEYOND the gap model vs q", ImVec2(-1, 280))) {
        ImPlot::SetupAxes("q", "beyond-gap gain (pp)",
                          ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::PlotBars("beyond gap model", xs.data(), beyond.data(), (int)xs.size(), 0.7);
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("raw gain over uniform vs q (gap statistics included)", ImVec2(-1, 220))) {
        ImPlot::SetupAxes("q", "raw gain (pp)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::PlotBars("raw gain", xs.data(), gain.data(), (int)xs.size(), 0.7);
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("same-residue repulsion vs q", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("q", "mean diagonal bias (%)",
                          ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::PlotBars("diag bias", xs.data(), diag.data(), (int)xs.size(), 0.7);
        ImPlot::EndPlot();
    }
    static std::vector<double> l2;
    l2.clear();
    for (const auto& row : P.rows) l2.push_back(std::fabs(row.lambda2));
    if (ImPlot::BeginPlot("transition operator |lambda2| vs q", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("q", "|second eigenvalue|",
                          ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::PlotBars("|lambda2|", xs.data(), l2.data(), (int)xs.size(), 0.7);
        ImPlot::EndPlot();
    }
    ImGui::TextWrapped(
        "The transition matrices are Markov operators; the magnitude of the second "
        "eigenvalue is the operator form of the one-step memory, and 1 - |lambda2| "
        "is the operator's spectral gap. At q = 3 (two classes) the value is exact "
        "and signed: negative means repulsion.");

    ImGui::SeparatorText("What the pattern is");
    const prt::PatternRow* q0 = nullptr;
    const prt::PatternRow* best = nullptr;
    for (const auto& row : P.rows) {
        if (row.q == R.trans.q) q0 = &row;
        if (!best || row.gain_beyond > best->gain_beyond) best = &row;
    }
    if (q0 && best)
        ImGui::TextWrapped(
            "At q = %u the raw gain of %+.2f pp decomposes into %+.2f pp explained by gap "
            "frequencies + the wheel, and %+.2f pp of genuine residue-gap correlation. The "
            "strongest genuine signal in the scan is at q = %u (%+.2f pp). Names: the raw "
            "effect is the Lemke Oliver-Soundararajan bias (2016); the surviving correlation "
            "is the second-order term predicted by the Hardy-Littlewood k-tuple conjecture. "
            "The Chebyshev races live in the Races tab; all three are faces of how primes "
            "repel their own residue classes.",
            q0->q, 100.0 * q0->gain, 100.0 * (q0->acc_gapmodel - q0->acc_uniform),
            100.0 * q0->gain_beyond, best->q, 100.0 * best->gain_beyond);

    ImGui::SeparatorText("Memory depth (order-2 Markov test)");
    ImGui::TextWrapped(
        "Predicting the residue of the NEXT prime mod %u on %s test triples: "
        "uniform guess %.2f%%, knowing 1 previous prime %.2f%%, knowing 2 previous "
        "primes %.2f%%. The order-2 surplus (%+.2f pp) is how much extra pattern "
        "lives beyond one step of memory.",
        P.q2, fmt_u64(P.test_triples).c_str(), 100.0 * P.acc_uniform,
        100.0 * P.acc_order1, 100.0 * P.acc_order2,
        100.0 * (P.acc_order2 - P.acc_order1));

    ImGui::SeparatorText("Full scan");
    if (ImGui::BeginTable("pattab", 11, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                        ImGuiTableFlags_ScrollY, ImVec2(0, 340))) {
        ImGui::TableSetupColumn("q");
        ImGui::TableSetupColumn("chi2 indep");
        ImGui::TableSetupColumn("diag bias");
        ImGui::TableSetupColumn("acc");
        ImGui::TableSetupColumn("uniform");
        ImGui::TableSetupColumn("raw gain");
        ImGui::TableSetupColumn("gap model");
        ImGui::TableSetupColumn("beyond");
        ImGui::TableSetupColumn("chi2 gapnull");
        ImGui::TableSetupColumn("resid diag");
        ImGui::TableSetupColumn("lambda2");
        ImGui::TableHeadersRow();
        for (const auto& row : P.rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%u", row.q);
            ImGui::TableNextColumn(); ImGui::Text("%.0f", row.chi2);
            ImGui::TableNextColumn(); ImGui::Text("%+.1f%%", 100.0 * row.diag_bias);
            ImGui::TableNextColumn(); ImGui::Text("%.2f%%", 100.0 * row.acc);
            ImGui::TableNextColumn(); ImGui::Text("%.2f%%", 100.0 * row.acc_uniform);
            ImGui::TableNextColumn(); ImGui::Text("%+.2f pp", 100.0 * row.gain);
            ImGui::TableNextColumn(); ImGui::Text("%.2f%%", 100.0 * row.acc_gapmodel);
            ImGui::TableNextColumn(); ImGui::Text("%+.2f pp", 100.0 * row.gain_beyond);
            ImGui::TableNextColumn(); ImGui::Text("%.0f", row.chi2_gap);
            ImGui::TableNextColumn(); ImGui::Text("%+.1f%%", 100.0 * row.resid_diag);
            ImGui::TableNextColumn(); ImGui::Text("%+.4f", row.lambda2);
        }
        ImGui::EndTable();
    }

    ImGui::EndChild();
}

// ---------------------------------------------------------------- 3D explorer

static void draw_explorer(AppState& st, const std::shared_ptr<Results>& Rp) {
    const Results& R = *Rp;
    ImGui::BeginChild("ctrl3d", ImVec2(270, 0));
    ImGui::SeparatorText("View");
    const char* MODES[] = {"Residue strands (raw primes)", "Lattice cube (all primes)",
                           "Density torus (all primes)", "Spectral embedding (classes)",
                           "Pair space (the deep pattern)"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##mode", &st.view_mode, MODES, 5)) {
        static const float DEF_SIZE[] = {3.0f, 22.0f, 18.0f, 26.0f, 16.0f};
        static const bool DEF_GLOW[] = {true, false, false, false, false};
        st.point_size = DEF_SIZE[st.view_mode];
        st.glow = DEF_GLOW[st.view_mode];
        st.cam = prt3d::Camera{};
    }

    const uint32_t L = R.cfg.L;
    switch (st.view_mode) {
        case 0:
            ImGui::TextWrapped("Sampled primes on a cylinder: angle = p mod m, height = p. "
                               "Strands are the allowed residue classes; brightness of a strand "
                               "is its real density.");
            ImGui::SliderInt("m", &st.strands_m, 3, 210);
            ImGui::SliderInt("points", &st.strands_points, 10'000, (int)R.sample.size());
            break;
        case 1:
            ImGui::TextWrapped("All %s primes aggregated on the CRT lattice "
                               "(p mod m1, p mod m2, p mod m3). Structure = divisibility; "
                               "brightness = density.", fmt_u64(R.prime_count).c_str());
            divisor_combo("m1", &st.lat_m1, L, 2, 40);
            divisor_combo("m2", &st.lat_m2, L, 2, 40);
            divisor_combo("m3", &st.lat_m3, L, 2, 40);
            break;
        case 2:
            ImGui::TextWrapped("All primes aggregated on the torus (p mod m1) x (p mod m2) - "
                               "the 'view from above' of the residue space.");
            divisor_combo("m1 (big circle)", &st.torus_m1, L, 2, 240);
            divisor_combo("m2 (small circle)", &st.torus_m2, L, 2, 40);
            break;
        case 3:
            ImGui::TextWrapped("Spectral (Laplacian eigenmap) embedding of the %d coprime classes; "
                               "color = measured density z-score (blue low, yellow high).",
                               (int)R.topo.nodes.size());
            break;
        case 4:
            ImGui::TextWrapped("THE DEEP PATTERN: x = residue of p_n (mod %u), z = residue of "
                               "p_n+1, y = gap. Color = deviation of the exact (residue, gap) "
                               "count from the gap-model null (purple = suppressed, yellow = "
                               "enhanced, green = as the null predicts). The single-prime views "
                               "show the wheel; THIS is where the Lemke Oliver-Soundararajan "
                               "correlation actually lives. Set q = 3 and re-run for the "
                               "strongest bands.", R.cfg.q);
            ImGui::SliderInt("gap axis max", &st.pair_gmax, 30, 400);
            break;
    }

    ImGui::Spacing();
    ImGui::SliderFloat("point size", &st.point_size, 1.0f, 32.0f);
    ImGui::Checkbox("glow blend", &st.glow);
    if (ImGui::Button("reset camera", ImVec2(-1, 0))) st.cam = prt3d::Camera{};
    ImGui::TextDisabled("drag: orbit\nright-drag: pan\nwheel: zoom");
    ImGui::Text("points: %s", fmt_u64(st.scene.point_count()).c_str());
    ImGui::EndChild();

    ImGui::SameLine();

    // rebuild cloud if inputs changed
    uint64_t key = st.generation.load();
    key = key * 1315423911ull + (uint64_t)st.view_mode;
    key = key * 31 + (uint64_t)st.strands_m;
    key = key * 31 + (uint64_t)st.strands_points;
    key = key * 31 + (uint64_t)st.lat_m1;
    key = key * 31 + (uint64_t)st.lat_m2;
    key = key * 31 + (uint64_t)st.lat_m3;
    key = key * 31 + (uint64_t)st.torus_m1;
    key = key * 31 + (uint64_t)st.torus_m2;
    key = key * 31 + (uint64_t)st.pair_gmax;
    if (key != st.cloud_key && st.scene_ok) {
        static std::vector<float> pts;
        switch (st.view_mode) {
            case 0: prtclouds::build_strands(R, st.strands_m, st.strands_points, pts); break;
            case 1: prtclouds::build_lattice(R, st.lat_m1, st.lat_m2, st.lat_m3, pts); break;
            case 2: prtclouds::build_torus(R, st.torus_m1, st.torus_m2, pts); break;
            case 3: prtclouds::build_spectral(R, pts); break;
            case 4: prtclouds::build_pairspace(R, st.pair_gmax, pts); break;
        }
        st.scene.set_points(pts);
        st.cloud_key = key;
    }

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 16 || avail.y < 16 || !st.scene_ok) {
        if (!st.scene_ok) ImGui::TextDisabled("3D unavailable (shader init failed)");
        return;
    }
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("view3d", avail,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsItemActive()) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            st.cam.yaw += io.MouseDelta.x * 0.008f;
            st.cam.pitch = std::clamp(st.cam.pitch + io.MouseDelta.y * 0.008f, -1.5f, 1.5f);
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
            st.cam.pan_x -= io.MouseDelta.x * 0.0022f * st.cam.dist;
            st.cam.pan_y += io.MouseDelta.y * 0.0022f * st.cam.dist;
        }
    }
    if (ImGui::IsItemHovered() && io.MouseWheel != 0.0f)
        st.cam.dist = std::clamp(st.cam.dist * std::exp(-io.MouseWheel * 0.12f), 0.4f, 30.0f);

    const unsigned tex = st.scene.render(st.cam, (int)avail.x, (int)avail.y,
                                         st.point_size, st.glow);
    if (tex)
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)tex, pos,
                                             ImVec2(pos.x + avail.x, pos.y + avail.y),
                                             ImVec2(0, 1), ImVec2(1, 0));
}

// ---------------------------------------------------------------- auditor

static void draw_auditor(AppState& st) {
    ImGui::BeginChild("auditscroll");
    ImGui::TextWrapped(
        "Sequence auditor: run the analysis pipeline on ANY integer sequence "
        "from a file (any text with numbers in it - newline, space, or comma "
        "separated; order matters). Detects occupancy structure, residue memory "
        "between successive terms beyond their difference statistics, and order-2 "
        "memory. Scoring is out-of-sample: first half trains, second half tests.");
    ImGui::Spacing();

    ImGui::SetNextItemWidth(-220);
    ImGui::InputText("##apath", st.audit_path, sizeof(st.audit_path));
    ImGui::SameLine();
    if (ImGui::Button("Browse...")) {
        OPENFILENAMEA ofn = {};
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = GetActiveWindow();
        ofn.lpstrFile = st.audit_path;
        ofn.nMaxFile = sizeof(st.audit_path);
        ofn.lpstrFilter = "All files\0*.*\0Text\0*.txt;*.csv\0";
        ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
        GetOpenFileNameA(&ofn);
    }
    ImGui::SameLine();
    const bool running = st.audit_running.load();
    ImGui::BeginDisabled(running || st.audit_path[0] == '\0');
    if (ImGui::Button("Audit")) st.start_audit();
    ImGui::EndDisabled();
    if (running) ImGui::Text("auditing...");

    auto A = st.get_audit();
    if (!A) { ImGui::EndChild(); return; }
    if (!A->valid) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.4f, 1.0f), "error: %s", A->error.c_str());
        ImGui::EndChild();
        return;
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Result");
    ImGui::Text("%s", A->source.c_str());
    ImGui::Text("terms: %s   range: [%lld, %lld]   increasing steps: %.1f%%",
                fmt_u64(A->count).c_str(), (long long)A->vmin, (long long)A->vmax,
                100.0 * A->monotone_frac);
    if (!st.audit_export_path.empty())
        ImGui::TextWrapped("logged to %s", st.audit_export_path.c_str());

    // verdict
    const prt::PatternRow* best = nullptr;
    double min_pg = 1.0;
    for (const auto& row : A->rows) {
        if (!best || row.gain_beyond > best->gain_beyond) best = &row;
        min_pg = std::min(min_pg, row.pval_gap);
    }
    ImGui::Spacing();
    if (best && best->gain_beyond >= 0.005)
        ImGui::TextWrapped(
            "VERDICT: genuine residue memory detected - strongest at q = %u, %+.2f pp "
            "beyond difference statistics (acc %.2f%% vs diff-model %.2f%%).",
            best->q, 100.0 * best->gain_beyond, 100.0 * best->acc, 100.0 * best->acc_gapmodel);
    else if (min_pg < 1e-3)
        ImGui::TextWrapped(
            "VERDICT: distributional correlation detected (chi2 vs the difference-model "
            "null is significant) but too diffuse to improve prediction.");
    else
        ImGui::TextWrapped(
            "VERDICT: no residue memory beyond difference statistics - the sequence "
            "looks residue-random given its difference distribution.");

    static std::vector<double> xs, beyond;
    xs.clear(); beyond.clear();
    for (const auto& row : A->rows) {
        xs.push_back(row.q);
        beyond.push_back(100.0 * row.gain_beyond);
    }
    if (ImPlot::BeginPlot("memory beyond difference statistics vs q", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("q", "beyond-gain (pp)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::PlotBars("beyond", xs.data(), beyond.data(), (int)xs.size(), 0.7);
        ImPlot::EndPlot();
    }

    static std::vector<double> dx, dy;
    dx.clear(); dy.clear();
    for (size_t i = 0; i < A->diff_x.size(); ++i) {
        dx.push_back(A->diff_x[i]);
        dy.push_back((double)A->diff_count[i]);
    }
    if (!dx.empty() && ImPlot::BeginPlot("difference histogram", ImVec2(-1, 220))) {
        ImPlot::SetupAxes("x_{n+1} - x_n", "count", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::PlotBars("diffs", dx.data(), dy.data(), (int)dx.size(), 1.0);
        ImPlot::EndPlot();
    }

    ImGui::Text("order-2 memory (mod %u): uniform %.2f%%, order-1 %.2f%%, order-2 %.2f%% (extra %+.2f pp)",
                A->q2, 100.0 * A->acc_uniform, 100.0 * A->acc_order1, 100.0 * A->acc_order2,
                100.0 * (A->acc_order2 - A->acc_order1));

    ImGui::SeparatorText("Marginal occupancy (uniform null)");
    if (ImGui::BeginTable("audmarg", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                        ImGuiTableFlags_ScrollY, ImVec2(0, 220))) {
        ImGui::TableSetupColumn("modulus");
        ImGui::TableSetupColumn("chi2");
        ImGui::TableSetupColumn("dof");
        ImGui::TableSetupColumn("p-value");
        ImGui::TableSetupColumn("max |z| (class)");
        ImGui::TableHeadersRow();
        for (const auto& ms : A->mods) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%u", ms.m);
            ImGui::TableNextColumn(); ImGui::Text("%.1f", ms.chi2);
            ImGui::TableNextColumn(); ImGui::Text("%d", ms.dof);
            ImGui::TableNextColumn(); ImGui::Text("%.3g", ms.pval);
            ImGui::TableNextColumn(); ImGui::Text("%.2f (%u)", ms.max_abs_z, ms.argmax_class);
        }
        ImGui::EndTable();
    }

    ImGui::SeparatorText("Full scan");
    if (ImGui::BeginTable("audtab", 8, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                       ImGuiTableFlags_ScrollY, ImVec2(0, 320))) {
        ImGui::TableSetupColumn("q");
        ImGui::TableSetupColumn("chi2 indep");
        ImGui::TableSetupColumn("acc");
        ImGui::TableSetupColumn("uniform");
        ImGui::TableSetupColumn("diff model");
        ImGui::TableSetupColumn("beyond");
        ImGui::TableSetupColumn("chi2 diffnull");
        ImGui::TableSetupColumn("p diffnull");
        ImGui::TableHeadersRow();
        for (const auto& row : A->rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%u", row.q);
            ImGui::TableNextColumn(); ImGui::Text("%.0f", row.chi2);
            ImGui::TableNextColumn(); ImGui::Text("%.2f%%", 100.0 * row.acc);
            ImGui::TableNextColumn(); ImGui::Text("%.2f%%", 100.0 * row.acc_uniform);
            ImGui::TableNextColumn(); ImGui::Text("%.2f%%", 100.0 * row.acc_gapmodel);
            ImGui::TableNextColumn(); ImGui::Text("%+.2f pp", 100.0 * row.gain_beyond);
            ImGui::TableNextColumn(); ImGui::Text("%.0f", row.chi2_gap);
            ImGui::TableNextColumn(); ImGui::Text("%.3g", row.pval_gap);
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();
}

// ---------------------------------------------------------------- info

static void draw_info() {
    ImGui::BeginChild("infoscroll");

    ImGui::SeparatorText("What this program does");
    ImGui::TextWrapped(
        "Prime Residue Topology maps the primes into residue space - the vector of "
        "remainders (n mod m1, n mod m2, ...) - and measures the structure that "
        "appears there. A multithreaded segmented sieve of Eratosthenes enumerates "
        "every prime up to N in order; a single streaming pass aggregates them into "
        "the tables behind every chart. Nothing is sampled or estimated unless a "
        "view says so (the raw 3D strands use a strided sample; everything else "
        "uses full counts).");

    ImGui::SeparatorText("The key idea: aggregation on the residue torus");
    ImGui::TextWrapped(
        "By the Chinese Remainder Theorem, a number's residue fingerprint depends "
        "only on n mod L (L = 2310 = 2*3*5*7*11 by default). The residue space is "
        "therefore a finite torus with L points, and any range [2, N] - even "
        "billions of primes - collapses onto it by counting. That is what makes "
        "analysis at N = 10^11 cheap: the statistics run on a few thousand "
        "aggregate cells, never on N numbers.");

    ImGui::SeparatorText("Null models: what counts as a pattern");
    ImGui::TextWrapped(
        "Every claim of structure is a comparison against the model that says "
        "there is none:\n\n"
        "- Residue classes: Dirichlet equidistribution (primes split evenly among "
        "classes coprime to m). Chi-square and per-class z-scores quantify any "
        "deviation.\n"
        "- Transitions: independence of consecutive primes' residues (expected = "
        "row x column marginals).\n"
        "- The gap-model null: the stricter question - could the transition "
        "structure be explained by gap frequencies alone, given the wheel? "
        "Expected(i -> j) = (transitions from i) x (frequency of gaps with the "
        "right size mod q), renormalized over reachable classes. Structure that "
        "survives THIS null is genuine correlation between a prime's residue and "
        "its next gap.\n"
        "- Prime pairs: the Hardy-Littlewood prediction C(g) * Li2(N).");

    ImGui::SeparatorText("Out-of-sample protocol");
    ImGui::TextWrapped(
        "Every predictor is calibrated on the first half of the range (or "
        "sequence) and scored only on the second half, against stated baselines "
        "(uniform, marginal, wheel, gap model, global mean). This separates "
        "structure that generalizes from structure that was memorized.");

    ImGui::SeparatorText("The tabs");
    ImGui::BulletText("Overview - headline numbers from the last run.");
    ImGui::BulletText("Residue classes - per-class counts vs the Dirichlet prediction.");
    ImGui::BulletText("Gaps - gap histogram; gap-to-gap memory (conditional means, R^2).");
    ImGui::BulletText("Transitions - consecutive-prime residue transition bias heatmap.");
    ImGui::BulletText("Races - Chebyshev races pi(x;4,3)-pi(x;4,1) and mod 3.");
    ImGui::BulletText("Topology - graph Laplacian of the residue torus: spectrum,");
    ImGui::Text("   Betti_0, spectral gap, Moran's I, graph-Fourier power of the density field.");
    ImGui::BulletText("Patterns - transition scan q = 3..100, gap-model decomposition,");
    ImGui::Text("   order-2 memory test. This is where the deep structure is classified.");
    ImGui::BulletText("Hardy-Littlewood - measured prime pairs (p, p+g) vs C(g)*Li2(N).");
    ImGui::BulletText("Prediction - class-density ranking vs the wheel; next-residue accuracy.");
    ImGui::BulletText("3D Explorer - strands, CRT lattice, density torus, spectral embedding,");
    ImGui::Text("   and pair space. The single-prime views show divisibility structure (the");
    ImGui::Text("   wheel); the pair-space view shows deviations from the gap-model null.");
    ImGui::BulletText("Auditor - the same analysis applied to any integer sequence from a file.");

    ImGui::SeparatorText("Run logging");
    ImGui::TextWrapped(
        "Every completed run writes results/run_<timestamp>_N<range>/ with a text "
        "summary, CSVs of all aggregate tables, and 3D snapshots from several "
        "angles. Audits write results/audit_<timestamp>_<name>/. The CLI "
        "(prt_cli) offers selftest, run <N>, and audit <file> [q].");

    ImGui::SeparatorText("Limits");
    ImGui::TextWrapped(
        "Precision grows like sqrt(N) while cost grows like N, so a desktop tops "
        "out around 10^12-10^13. Residue structure predicts residue classes, not "
        "the primality of individual numbers. The Hardy-Littlewood comparisons "
        "test a conjecture numerically; measurement is not proof.");

    ImGui::EndChild();
}

// ---------------------------------------------------------------- root

void draw_ui(AppState& st) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("Prime Residue Topology", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                 ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImGui::BeginChild("controls", ImVec2(300, 0), ImGuiChildFlags_Border);
    draw_controls(st);
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginChild("main", ImVec2(0, 0));
    auto R = st.get_results();

    // capture every completed run: summary + CSVs + 3D snapshots
    if (R && R->valid && R.get() != st.last_export_ptr) {
        st.last_export_path = export_run(st, *R);
        st.last_export_ptr = R.get();
    }
    // capture every completed audit: summary + CSVs
    if (auto A = st.get_audit(); A && A->valid && A.get() != st.audit_export_ptr) {
        st.audit_export_path = prt::export_audit(*A);
        st.audit_export_ptr = A.get();
    }

    auto no_data = []() {
        ImGui::Spacing();
        ImGui::TextWrapped(
            "No prime data yet - configure a run on the left and press Run. "
            "The Info tab explains what the pipeline computes.");
    };
    if (ImGui::BeginTabBar("tabs")) {
        if (ImGui::BeginTabItem("Overview"))      { R ? draw_overview(*R)      : no_data(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Residue classes")) { R ? draw_classes(st, *R) : no_data(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Gaps"))          { R ? draw_gaps(st, *R)      : no_data(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Transitions"))   { R ? draw_transitions(*R)   : no_data(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Races"))         { R ? draw_races(st, *R)     : no_data(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Topology"))      { R ? draw_topology(*R)      : no_data(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Patterns"))      { R ? draw_patterns(*R)      : no_data(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Hardy-Littlewood")) { R ? draw_hl(*R)         : no_data(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Prediction"))    { R ? draw_prediction(*R)    : no_data(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("3D Explorer"))   { if (R) draw_explorer(st, R); else no_data(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Auditor"))       { draw_auditor(st);          ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Info"))          { draw_info();               ImGui::EndTabItem(); }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();
    ImGui::End();
}
