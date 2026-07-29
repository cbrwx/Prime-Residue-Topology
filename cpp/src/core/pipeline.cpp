#include "core/pipeline.h"
#include "core/sieve.h"
#include "core/stats.h"
#include "core/topology.h"
#include "core/predict.h"
#include <cmath>
#include <chrono>
#include <thread>

namespace prt {

const char* stage_name(int stage) {
    switch (stage) {
        case 0: return "idle";
        case 1: return "sieving + accumulating";
        case 2: return "statistics";
        case 3: return "topology";
        case 4: return "prediction";
        case 5: return "done";
        default: return "?";
    }
}

Results run_pipeline(const Config& cfg, Progress* prog) {
    using clock = std::chrono::steady_clock;
    Results R;
    R.cfg = cfg;
    const uint64_t N = cfg.N;
    const uint32_t q = cfg.q;
    const uint32_t L = cfg.L;
    if (N < 100 || q < 3 || L < 2) { R.error = "invalid config"; return R; }

    R.jointL.assign(L, 0);
    R.jointL_train.assign(L, 0);
    R.gap_hist.assign(1024, 0);
    R.gap_hist_train.assign(1024, 0);
    R.pair_gap.assign((size_t)q * 1024, 0);
    R.trans.q = q;
    R.trans.obs.assign((size_t)q * q, 0);
    R.trans.train.assign((size_t)q * q, 0);

    R.mods.resize(cfg.stat_moduli.size());
    for (size_t i = 0; i < cfg.stat_moduli.size(); ++i) {
        R.mods[i].m = cfg.stat_moduli[i];
        R.mods[i].counts.assign(cfg.stat_moduli[i], 0);
    }

    // Fast residue plan: two joint moduli cover all standard choices.
    const uint32_t J1 = 2310;   // 2*3*5*7*11
    const uint32_t J2 = 7200;   // 2^5 * 3^2 * 5^2
    struct Plan { uint64_t* counts; uint32_t m; int src; }; // src: 0=J1, 1=J2, 2=direct
    std::vector<Plan> plan(R.mods.size());
    for (size_t i = 0; i < R.mods.size(); ++i) {
        const uint32_t m = R.mods[i].m;
        plan[i] = { R.mods[i].counts.data(), m, (J1 % m == 0) ? 0 : (J2 % m == 0) ? 1 : 2 };
    }
    const int q_src = (J1 % q == 0) ? 0 : (J2 % q == 0) ? 1 : 2;
    const int L_src = (J1 % L == 0) ? 0 : (J2 % L == 0) ? 1 : 2;

    const double lnN = std::log((double)N);
    const uint64_t est_pi = (uint64_t)((double)N / std::max(2.0, lnN));
    R.sample_stride = std::max<uint64_t>(1, est_pi / std::max<uint32_t>(1, cfg.sample_target));
    // pi(N) exceeds N/ln(N); reserve with headroom to avoid a doubling realloc
    R.sample.reserve((size_t)(est_pi * 1.15 / (double)R.sample_stride) + 16);
    const uint64_t half = N / 2;

    // pattern scan: transition matrices for every modulus in [3, 100].
    // The consumer only records the gap sequence (1 byte/prime); the scan runs
    // as a parallel post-pass, one modulus at a time per worker thread.
    const uint32_t PQ_MIN = 3, PQ_MAX = 100;
    std::vector<size_t> pq_off(PQ_MAX + 2, 0);
    for (uint32_t qq = PQ_MIN; qq <= PQ_MAX; ++qq)
        pq_off[qq + 1] = pq_off[qq] + (size_t)qq * qq;
    std::vector<uint64_t> pq_obs(pq_off[PQ_MAX + 1], 0), pq_train(pq_off[PQ_MAX + 1], 0);
    const uint32_t GS = 128;                        // padded row stride
    std::vector<uint8_t> gmod((size_t)1024 * GS, 0);
    for (uint32_t g = 0; g < 1024; ++g)
        for (uint32_t qq = PQ_MIN; qq <= PQ_MAX; ++qq)
            gmod[(size_t)g * GS + qq] = (uint8_t)(g % qq);
    // gap codes: 0 -> gap 1 (2 to 3), 255 -> overflow list, else gap = 2*code.
    // The buffer is scanned and flushed in chunks so memory stays bounded even
    // at N = 10^11 (4.1e9 gaps); per-q residue state carries across chunks.
    const size_t GAP_CHUNK = 1ull << 26;            // ~67M gaps per flush
    std::vector<uint8_t> gap_code;
    gap_code.reserve(std::min((size_t)(est_pi * 1.15), GAP_CHUNK + (1 << 19)));
    std::vector<uint32_t> gap_over;
    std::vector<uint32_t> pq_r(PQ_MAX + 1, 0);
    for (uint32_t qq = PQ_MIN; qq <= PQ_MAX; ++qq) pq_r[qq] = 2 % qq;
    uint64_t chunk_prev = 2;                        // prime preceding the buffered gaps

    auto scan_chunk = [&](uint64_t start_prev) {
        if (gap_code.empty()) return;
        const unsigned T = std::max(1u, cfg.threads ? cfg.threads
                                                    : std::thread::hardware_concurrency());
        std::atomic<uint32_t> next_q{PQ_MIN};
        auto worker = [&]() {
            for (;;) {
                const uint32_t qq = next_q.fetch_add(1);
                if (qq > PQ_MAX) break;
                uint64_t* obs = pq_obs.data() + pq_off[qq];
                uint64_t* trn = pq_train.data() + pq_off[qq];
                uint32_t r = pq_r[qq];
                uint64_t pp = start_prev;
                size_t ov = 0;
                for (const uint8_t c : gap_code) {
                    uint32_t g;
                    uint32_t rn;
                    if (c == 255) {
                        g = gap_over[ov++];
                        rn = (r + g % qq) % qq;
                    } else {
                        g = (c == 0) ? 1u : (uint32_t)c << 1;
                        rn = r + gmod[(size_t)g * GS + qq];
                        if (rn >= qq) rn -= qq;
                    }
                    pp += g;
                    const size_t o = (size_t)r * qq + rn;
                    ++obs[o];
                    if (pp <= half) ++trn[o];
                    r = rn;
                }
                pq_r[qq] = r;
            }
        };
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < T; ++t) workers.emplace_back(worker);
        for (auto& w : workers) w.join();
        gap_code.clear();
        gap_over.clear();
    };
    // order-2 transitions at cfg.q
    std::vector<uint64_t> o2_obs((size_t)q * q * q, 0), o2_train((size_t)q * q * q, 0);
    uint32_t prev2_q = 0;

    // consecutive-gap pairs (g_n, g_{n+1})
    std::vector<uint64_t> gg_obs((size_t)1024 * 1024, 0), gg_train((size_t)1024 * 1024, 0);
    uint32_t prev_gap = 0;

    // prime pairs (p, p+g) for g <= HLG via a sliding window of recent primes
    const uint64_t HLG = 120;
    std::vector<uint64_t> hl_pairs(HLG + 1, 0);
    uint64_t ring[64];
    int rhead = 0, rcount = 0;

    const int CP = std::max(16, cfg.checkpoints);
    const uint64_t cp_step = std::max<uint64_t>(1, N / (uint64_t)CP);
    uint64_t next_cp = cp_step;
    int64_t race4 = 0, race3 = 0;

    uint64_t prev = 0, idx = 0;
    uint32_t prev_q = 0;

    SieveProgress sprog;
    if (prog) { sprog.cancel = &prog->cancel; prog->stage.store(1); }

    const auto t0 = clock::now();
    const bool ok = for_each_prime(N, [&](const uint64_t* pr, size_t cnt) {
        for (size_t kk = 0; kk < cnt; ++kk) {
            const uint64_t p = pr[kk];

            while (p > next_cp && next_cp < N) {
                R.cp_x.push_back((double)next_cp);
                R.race4.push_back((double)race4);
                R.race3.push_back((double)race3);
                next_cp += cp_step;
            }

            const uint32_t a = (uint32_t)(p % J1);
            const uint32_t b = (uint32_t)(p % J2);

            for (const Plan& pl : plan) {
                const uint32_t r = (pl.src == 0) ? a % pl.m
                                 : (pl.src == 1) ? b % pl.m
                                 : (uint32_t)(p % pl.m);
                ++pl.counts[r];
            }

            const uint32_t rL = (L_src == 0) ? a % L
                              : (L_src == 1) ? b % L
                              : (uint32_t)(p % L);
            ++R.jointL[rL];
            if (p <= half) ++R.jointL_train[rL];

            if (p > 2) { const uint32_t r4 = b & 3; if (r4 == 3) ++race4; else --race4; }
            if (p > 3) { const uint32_t r3 = a % 3; if (r3 == 2) ++race3; else --race3; }

            const uint32_t rq = (q_src == 0) ? a % q
                              : (q_src == 1) ? b % q
                              : (uint32_t)(p % q);
            if (prev) {
                const uint64_t g = p - prev;
                if (g < 1024) {
                    ++R.gap_hist[g];
                    if (p <= half) ++R.gap_hist_train[g];
                    ++R.pair_gap[(size_t)prev_q * 1024 + g];
                    if (prev_gap) {
                        const size_t gi = (size_t)prev_gap * 1024 + g;
                        ++gg_obs[gi];
                        if (p <= half) ++gg_train[gi];
                    }
                    prev_gap = (uint32_t)g;
                } else { ++R.gap_overflow; prev_gap = 0; }
                if (g > R.max_gap) R.max_gap = g;
                ++R.trans.obs[(size_t)prev_q * q + rq];
                if (p <= half) ++R.trans.train[(size_t)prev_q * q + rq];
                if (idx >= 2) {
                    const size_t i2 = ((size_t)prev2_q * q + prev_q) * q + rq;
                    ++o2_obs[i2];
                    if (p <= half) ++o2_train[i2];
                }
            }

            if (prev) {
                const uint64_t g = p - prev;
                if (g == 1) gap_code.push_back(0);
                else if (g >= 510) { gap_code.push_back(255); gap_over.push_back((uint32_t)g); }
                else gap_code.push_back((uint8_t)(g >> 1));
            }

            // prime pairs within HLG: count (w, p) for every recent prime w
            for (int t = 0; t < rcount; ++t) {
                const uint64_t w = ring[(rhead - 1 - t) & 63];
                const uint64_t d = p - w;
                if (d > HLG) { rcount = t; break; }
                ++hl_pairs[d];
            }
            ring[rhead & 63] = p;
            rhead = (rhead + 1) & 63;
            if (rcount < 64) ++rcount;

            if (idx % R.sample_stride == 0) R.sample.push_back(p);

            prev = p;
            prev2_q = prev_q;
            prev_q = rq;
            ++idx;
        }
        if (gap_code.size() >= GAP_CHUNK) {
            scan_chunk(chunk_prev);
            chunk_prev = prev;
        }
        if (prog) prog->frac.store(0.85 * (double)prev / (double)N);
    }, &sprog, cfg.threads);
    R.t_sieve = std::chrono::duration<double>(clock::now() - t0).count();

    if (!ok) { R.error = "cancelled"; return R; }

    while (next_cp <= N) {
        R.cp_x.push_back((double)next_cp);
        R.race4.push_back((double)race4);
        R.race3.push_back((double)race3);
        next_cp += cp_step;
    }

    R.prime_count = idx;
    R.last_prime = prev;

    const auto t1 = clock::now();
    if (prog) { prog->stage.store(2); prog->frac.store(0.87); }
    for (auto& ms : R.mods) finalize_mod_stats(ms);
    finalize_trans_stats(R.trans);

    scan_chunk(chunk_prev);                          // flush the final partial chunk

    for (uint32_t qq = PQ_MIN; qq <= PQ_MAX; ++qq)
        R.patterns.rows.push_back(analyze_transition_q(qq, pq_obs.data() + pq_off[qq],
                                                       pq_train.data() + pq_off[qq],
                                                       R.gap_hist.data(), R.gap_hist_train.data(),
                                                       R.gap_hist.size()));
    analyze_order2(R.patterns, q, o2_obs, o2_train, R.trans.train);
    finalize_gapgap(R.gapgap, gg_obs.data(), gg_train.data(), 1024);
    finalize_hl(R.hl, hl_pairs, N);

    if (prog) { prog->stage.store(3); prog->frac.store(0.90); }
    R.topo = compute_topology(L, R.jointL, cfg.knn);

    if (prog) { prog->stage.store(4); prog->frac.store(0.97); }
    R.pred = compute_prediction(N, L, R.jointL, R.jointL_train, R.trans);

    R.t_analysis = std::chrono::duration<double>(clock::now() - t1).count();
    if (prog) { prog->stage.store(5); prog->frac.store(1.0); }
    R.valid = true;
    return R;
}

} // namespace prt
