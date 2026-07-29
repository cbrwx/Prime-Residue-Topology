#include "core/topology.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace prt {

// ---- Householder reduction to tridiagonal form (Numerical-Recipes style) ----
static void tred2(double* a, int n, double* d, double* e) {
    for (int i = n - 1; i > 0; --i) {
        const int l = i - 1;
        double h = 0.0, scale = 0.0;
        if (l > 0) {
            for (int k = 0; k <= l; ++k) scale += std::fabs(a[i * n + k]);
            if (scale == 0.0) {
                e[i] = a[i * n + l];
            } else {
                for (int k = 0; k <= l; ++k) {
                    a[i * n + k] /= scale;
                    h += a[i * n + k] * a[i * n + k];
                }
                double f = a[i * n + l];
                double g = (f >= 0.0) ? -std::sqrt(h) : std::sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                a[i * n + l] = f - g;
                f = 0.0;
                for (int j = 0; j <= l; ++j) {
                    a[j * n + i] = a[i * n + j] / h;
                    g = 0.0;
                    for (int k = 0; k <= j; ++k) g += a[j * n + k] * a[i * n + k];
                    for (int k = j + 1; k <= l; ++k) g += a[k * n + j] * a[i * n + k];
                    e[j] = g / h;
                    f += e[j] * a[i * n + j];
                }
                const double hh = f / (h + h);
                for (int j = 0; j <= l; ++j) {
                    f = a[i * n + j];
                    e[j] = g = e[j] - hh * f;
                    for (int k = 0; k <= j; ++k)
                        a[j * n + k] -= (f * e[k] + g * a[i * n + k]);
                }
            }
        } else {
            e[i] = a[i * n + l];
        }
        d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    for (int i = 0; i < n; ++i) {
        if (d[i] != 0.0) {
            for (int j = 0; j < i; ++j) {
                double g = 0.0;
                for (int k = 0; k < i; ++k) g += a[i * n + k] * a[k * n + j];
                for (int k = 0; k < i; ++k) a[k * n + j] -= g * a[k * n + i];
            }
        }
        d[i] = a[i * n + i];
        a[i * n + i] = 1.0;
        for (int j = 0; j < i; ++j) a[j * n + i] = a[i * n + j] = 0.0;
    }
}

// ---- Implicit QL with shifts on the tridiagonal form ----
static bool tqli(double* d, double* e, int n, double* z) {
    for (int i = 1; i < n; ++i) e[i - 1] = e[i];
    e[n - 1] = 0.0;
    for (int l = 0; l < n; ++l) {
        int iter = 0, m;
        do {
            for (m = l; m < n - 1; ++m) {
                const double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
                if (std::fabs(e[m]) <= 1e-15 * dd || dd == 0.0) break;
            }
            if (m != l) {
                if (iter++ == 60) return false;
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = std::hypot(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + ((g >= 0.0) ? std::fabs(r) : -std::fabs(r)));
                double s = 1.0, c = 1.0, p = 0.0;
                int i;
                for (i = m - 1; i >= l; --i) {
                    double f = s * e[i];
                    const double b = c * e[i];
                    e[i + 1] = (r = std::hypot(f, g));
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (p = s * r);
                    g = c * r - b;
                    for (int k = 0; k < n; ++k) {
                        f = z[k * n + (i + 1)];
                        z[k * n + (i + 1)] = s * z[k * n + i] + c * f;
                        z[k * n + i] = c * z[k * n + i] - s * f;
                    }
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
    return true;
}

bool eig_sym(std::vector<double>& A, int n,
             std::vector<double>& evals, std::vector<double>& evecs) {
    if (n <= 0 || (int)A.size() < n * n) return false;
    std::vector<double> d(n), e(n);
    tred2(A.data(), n, d.data(), e.data());
    if (!tqli(d.data(), e.data(), n, A.data())) return false;

    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int x, int y) { return d[x] < d[y]; });

    evals.resize(n);
    evecs.assign((size_t)n * n, 0.0);
    for (int k = 0; k < n; ++k) {
        evals[k] = d[perm[k]];
        for (int i = 0; i < n; ++i)
            evecs[(size_t)i * n + k] = A[(size_t)i * n + perm[k]];
    }
    return true;
}

// ---- union-find for Betti-0 ----
namespace {
struct DSU {
    std::vector<int> p;
    explicit DSU(int n) : p(n) { std::iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[x] != x) x = p[x] = p[p[x]]; return x; }
    void unite(int a, int b) { a = find(a); b = find(b); if (a != b) p[a] = b; }
};
} // namespace

TopoReport compute_topology(uint32_t L, const std::vector<uint32_t>& jointL, int knn) {
    TopoReport T;
    T.L = L;

    // distinct prime factors of L
    uint32_t x = L;
    for (uint32_t dvs = 2; (uint64_t)dvs * dvs <= x; ++dvs)
        if (x % dvs == 0) {
            T.torus_dims.push_back(dvs);
            while (x % dvs == 0) x /= dvs;
        }
    if (x > 1) T.torus_dims.push_back(x);

    for (uint32_t r = 0; r < L; ++r)
        if (gcd_u64(r, L) == 1) T.nodes.push_back(r);
    const int n = (int)T.nodes.size();
    if (n < 8) return T;

    // Empirical field: z-score of measured prime count per class
    double total = 0;
    for (uint32_t r : T.nodes) total += (double)jointL[r];
    const double E = total / n;
    if (E <= 0) return T;
    T.phi.resize(n);
    for (int i = 0; i < n; ++i)
        T.phi[i] = ((double)jointL[T.nodes[i]] - E) / std::sqrt(E);

    // Torus coordinates: (cos, sin) of 2*pi*(r mod p)/p per prime factor.
    // Euclidean distance in this embedding respects the wrap-around metric.
    const int D = 2 * (int)T.torus_dims.size();
    std::vector<double> coord((size_t)n * D);
    for (int i = 0; i < n; ++i) {
        const uint32_t r = T.nodes[i];
        for (size_t dgt = 0; dgt < T.torus_dims.size(); ++dgt) {
            const double ang = 2.0 * 3.14159265358979323846 *
                               (double)(r % T.torus_dims[dgt]) / (double)T.torus_dims[dgt];
            coord[(size_t)i * D + 2 * dgt]     = std::cos(ang);
            coord[(size_t)i * D + 2 * dgt + 1] = std::sin(ang);
        }
    }

    // Brute-force kNN (n <= ~500 for L <= 2310)
    const int k = std::max(2, std::min(knn, n - 1));
    std::vector<std::vector<std::pair<double, int>>> nn(n);
    std::vector<double> d2row(n);
    double sigma2_acc = 0;
    int sigma2_cnt = 0;
    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<double, int>> cand;
        cand.reserve(n - 1);
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            double s = 0;
            for (int t = 0; t < D; ++t) {
                const double dd = coord[(size_t)i * D + t] - coord[(size_t)j * D + t];
                s += dd * dd;
            }
            cand.emplace_back(s, j);
        }
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end());
        cand.resize(k);
        for (auto& c : cand) { sigma2_acc += c.first; ++sigma2_cnt; }
        nn[i] = std::move(cand);
    }
    const double sigma2 = std::max(1e-12, sigma2_acc / std::max(1, sigma2_cnt));

    // Symmetric weight matrix from the kNN union
    std::vector<double> W((size_t)n * n, 0.0);
    DSU dsu(n);
    for (int i = 0; i < n; ++i)
        for (auto& [d2, j] : nn[i]) {
            const double w = std::exp(-d2 / sigma2);
            if (w > W[(size_t)i * n + j]) {
                W[(size_t)i * n + j] = W[(size_t)j * n + i] = w;
            }
            dsu.unite(i, j);
        }

    // Canonical torus-lattice edges: step to the adjacent coprime residue in
    // each prime dimension (wrapping). kNN alone cannot connect the torus --
    // the coarsest dimension is metrically far wider than the finest -- so
    // these edges supply the true product-of-cycles topology.
    {
        std::vector<int> idx_of(L, -1);
        for (int i = 0; i < n; ++i) idx_of[T.nodes[i]] = i;
        for (uint32_t p : T.torus_dims) {
            if (p < 3) continue;                      // mod 2 has a single coprime class
            // CRT unit u: u = 1 (mod p), u = 0 (mod L/p)
            const uint32_t M = L / p;
            uint32_t inv = 0;
            for (uint32_t t = 1; t < p; ++t)
                if ((uint64_t)t * (M % p) % p == 1) { inv = t; break; }
            const uint64_t u = (uint64_t)M * inv % L;
            for (int i = 0; i < n; ++i) {
                const uint32_t r = T.nodes[i];
                const uint32_t rp = r % p;
                const uint32_t next = (rp == p - 1) ? 1 : rp + 1;   // skip residue 0
                const uint32_t r2 = (uint32_t)((r + (uint64_t)(next + p - rp) % p * u) % L);
                const int j = idx_of[r2];
                if (j < 0 || j == i) continue;
                double d2 = 0;
                for (int t = 0; t < D; ++t) {
                    const double dd = coord[(size_t)i * D + t] - coord[(size_t)j * D + t];
                    d2 += dd * dd;
                }
                const double w = std::exp(-d2 / sigma2);
                if (w > W[(size_t)i * n + j])
                    W[(size_t)i * n + j] = W[(size_t)j * n + i] = w;
                dsu.unite(i, j);
            }
        }
    }
    int comps = 0;
    for (int i = 0; i < n; ++i) if (dsu.find(i) == i) ++comps;
    T.betti0 = comps;

    // Laplacian L = D - W
    std::vector<double> A((size_t)n * n, 0.0);
    double S0 = 0;
    for (int i = 0; i < n; ++i) {
        double deg = 0;
        for (int j = 0; j < n; ++j) deg += W[(size_t)i * n + j];
        S0 += deg;
        for (int j = 0; j < n; ++j) A[(size_t)i * n + j] = -W[(size_t)i * n + j];
        A[(size_t)i * n + i] = deg;
    }

    std::vector<double> evals, evecs;
    if (!eig_sym(A, n, evals, evecs)) return T;
    for (double& v : evals)
        if (std::fabs(v) < 1e-9) v = 0.0;   // Laplacian eigenvalues are >= 0; kill noise
    T.eigval = evals;
    T.spectral_gap = (n > 1) ? evals[1] : 0.0;
    T.cheeger = T.spectral_gap / 2.0;

    // Spectral embedding: eigenvectors 1..3 (skipping the constant mode)
    T.embed3.assign((size_t)n * 3, 0.0);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c)
            if (c + 1 < n) T.embed3[(size_t)i * 3 + c] = evecs[(size_t)i * n + (c + 1)];

    // Graph Fourier power spectrum of the density field
    double mean = 0;
    for (double v : T.phi) mean += v;
    mean /= n;
    std::vector<double> xc(n);
    double xnorm2 = 0;
    for (int i = 0; i < n; ++i) { xc[i] = T.phi[i] - mean; xnorm2 += xc[i] * xc[i]; }
    T.power.assign(n, 0.0);
    if (xnorm2 > 0) {
        for (int kk = 0; kk < n; ++kk) {
            double dot = 0;
            for (int i = 0; i < n; ++i) dot += evecs[(size_t)i * n + kk] * xc[i];
            T.power[kk] = dot * dot / xnorm2;
        }
    }

    // Moran's I with the kNN weights
    double num = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            num += W[(size_t)i * n + j] * xc[i] * xc[j];
    if (S0 > 0 && xnorm2 > 0) T.moran = ((double)n / S0) * (num / xnorm2);

    return T;
}

} // namespace prt
