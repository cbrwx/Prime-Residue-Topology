#include "app/clouds.h"
#include <cmath>
#include <algorithm>

using prt::Results;

namespace prtclouds {

void build_strands(const Results& R, int m, int max_pts, std::vector<float>& out) {
    out.clear();
    if (R.sample.empty() || m < 2) return;
    const size_t step = std::max<size_t>(1, R.sample.size() / (size_t)std::max(1, max_pts));
    const double N = (double)R.cfg.N;
    out.reserve((R.sample.size() / step + 1) * 4);
    for (size_t i = 0; i < R.sample.size(); i += step) {
        const uint64_t p = R.sample[i];
        const double r = (double)(p % (uint64_t)m);
        const double ang = 2.0 * 3.14159265358979323846 * r / (double)m;
        out.push_back((float)std::cos(ang));
        out.push_back((float)((p / N) * 3.2 - 1.6));
        out.push_back((float)std::sin(ang));
        out.push_back((float)(r / (double)m));
    }
}

void build_lattice(const Results& R, int m1, int m2, int m3, std::vector<float>& out) {
    out.clear();
    const uint32_t L = R.cfg.L;
    if (m1 < 2 || m2 < 2 || m3 < 2 || (uint64_t)m1 * m2 * m3 > 300000) return;
    std::vector<uint64_t> cell((size_t)m1 * m2 * m3, 0);
    for (uint32_t r = 0; r < L; ++r) {
        if (!R.jointL[r]) continue;
        const size_t idx = ((size_t)(r % m1) * m2 + (r % m2)) * m3 + (r % m3);
        cell[idx] += R.jointL[r];
    }
    uint64_t mx = 0;
    for (uint64_t c : cell) mx = std::max(mx, c);
    if (!mx) return;
    out.reserve(cell.size() * 4);
    for (int a = 0; a < m1; ++a)
        for (int b = 0; b < m2; ++b)
            for (int c = 0; c < m3; ++c) {
                const uint64_t v = cell[((size_t)a * m2 + b) * m3 + c];
                if (!v) continue;
                out.push_back(((a + 0.5f) / m1) * 2.0f - 1.0f);
                out.push_back(((b + 0.5f) / m2) * 2.0f - 1.0f);
                out.push_back(((c + 0.5f) / m3) * 2.0f - 1.0f);
                out.push_back((float)((double)v / (double)mx));
            }
}

void build_torus(const Results& R, int m1, int m2, std::vector<float>& out) {
    out.clear();
    const uint32_t L = R.cfg.L;
    if (m1 < 2 || m2 < 2 || (uint64_t)m1 * m2 > 300000) return;
    std::vector<uint64_t> cell((size_t)m1 * m2, 0);
    for (uint32_t r = 0; r < L; ++r) {
        if (!R.jointL[r]) continue;
        cell[(size_t)(r % m1) * m2 + (r % m2)] += R.jointL[r];
    }
    uint64_t mx = 0;
    for (uint64_t c : cell) mx = std::max(mx, c);
    if (!mx) return;
    const double TAU = 2.0 * 3.14159265358979323846;
    const float R0 = 1.05f, r0 = 0.45f;
    out.reserve(cell.size() * 4);
    for (int a = 0; a < m1; ++a)
        for (int b = 0; b < m2; ++b) {
            const uint64_t v = cell[(size_t)a * m2 + b];
            if (!v) continue;
            const double u = TAU * (a + 0.5) / m1;
            const double w = TAU * (b + 0.5) / m2;
            const float ring = R0 + r0 * (float)std::cos(w);
            out.push_back(ring * (float)std::cos(u));
            out.push_back(r0 * (float)std::sin(w));
            out.push_back(ring * (float)std::sin(u));
            out.push_back((float)((double)v / (double)mx));
        }
}

void build_spectral(const Results& R, std::vector<float>& out) {
    out.clear();
    const auto& T = R.topo;
    const int n = (int)T.nodes.size();
    if (n == 0 || (int)T.embed3.size() < n * 3) return;
    double mx = 1e-12;
    for (double v : T.embed3) mx = std::max(mx, std::fabs(v));
    out.reserve((size_t)n * 4);
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c)
            out.push_back((float)(T.embed3[(size_t)i * 3 + c] / mx) * 1.4f);
        const double z = T.phi.empty() ? 0.0 : T.phi[i];
        out.push_back((float)std::clamp(0.5 + z / 6.0, 0.0, 1.0));
    }
}

void build_pairspace(const Results& R, int gmax, std::vector<float>& out) {
    out.clear();
    const uint32_t q = R.cfg.q;
    if (R.pair_gap.size() < (size_t)q * 1024 || q < 3) return;
    gmax = std::clamp(gmax, 10, 1023);

    // G(g) marginal and per-row sums over gaps allowed by the wheel
    std::vector<double> G(1024, 0.0);
    std::vector<double> Ri(q, 0.0), Si(q, 0.0);
    for (uint32_t i = 0; i < q; ++i)
        for (uint32_t g = 1; g < 1024; ++g) {
            const double v = (double)R.pair_gap[(size_t)i * 1024 + g];
            G[g] += v;
            Ri[i] += v;
        }
    auto gcd_u = [](uint32_t a, uint32_t b) { while (b) { uint32_t t = a % b; a = b; b = t; } return a; };
    for (uint32_t i = 0; i < q; ++i)
        for (uint32_t g = 1; g < 1024; ++g)
            if (gcd_u((i + g) % q, q) == 1) Si[i] += G[g];

    // deviations for cells with a stable expectation
    struct Cell { float x, y, z; double dev; };
    std::vector<Cell> cells;
    std::vector<double> absdev;
    for (uint32_t i = 0; i < q; ++i) {
        if (gcd_u(i, q) != 1 || Si[i] <= 0) continue;
        for (int g = 1; g <= gmax; ++g) {
            const double E = Ri[i] * G[g] / Si[i];
            if (E < 30.0) continue;
            const double o = (double)R.pair_gap[(size_t)i * 1024 + g];
            const double dev = o / E - 1.0;
            const uint32_t j = (i + (uint32_t)g) % q;
            Cell c;
            c.x = ((i + 0.5f) / q) * 2.0f - 1.0f;
            c.y = ((float)g / (float)gmax) * 2.4f - 1.2f;
            c.z = ((j + 0.5f) / q) * 2.0f - 1.0f;
            c.dev = dev;
            cells.push_back(c);
            absdev.push_back(std::fabs(dev));
        }
    }
    if (cells.empty()) return;

    // adaptive color scale: 95th percentile of |deviation|
    std::sort(absdev.begin(), absdev.end());
    const double dscale = std::max(1e-6, absdev[(size_t)((absdev.size() - 1) * 95) / 100]);

    out.reserve(cells.size() * 4);
    for (const Cell& c : cells) {
        out.push_back(c.x);
        out.push_back(c.y);
        out.push_back(c.z);
        out.push_back((float)std::clamp(0.5 + c.dev / (2.0 * dscale), 0.0, 1.0));
    }
}

} // namespace prtclouds
