#pragma once
// Prime Residue Topology — honest core pipeline.
//
// One streaming pass over all primes <= N produces exact aggregate data:
//   - prime counts per residue class for a set of statistics moduli
//   - joint counts modulo L (the residue torus; fingerprints mod any divisor
//     of L follow by CRT)
//   - prime gap histogram
//   - consecutive-prime residue transition matrix mod q
//   - Chebyshev race checkpoints (mod 4 and mod 3)
//   - an evenly strided sample of primes for raw 3D visualization
//
// No feature anywhere is computed FROM primality; primality only enters as the
// thing being counted. All statistics are compared against the Dirichlet
// equidistribution null model.
#include <cstdint>
#include <vector>
#include <string>
#include <atomic>

namespace prt {

struct Config {
    uint64_t N = 1'000'000'000ull;
    // Every value here must divide 2310 (=2*3*5*7*11) or 7200 (=2^5*3^2*5^2)
    // for the fast residue path; others fall back to a direct 64-bit modulo.
    std::vector<uint32_t> stat_moduli = {3, 4, 5, 7, 8, 9, 11, 12, 15, 16, 25, 30, 105, 210};
    uint32_t q = 10;                 // transition-matrix modulus
    uint32_t L = 2310;               // torus modulus (topology + prediction)
    uint32_t sample_target = 50'000'000;
    int      checkpoints = 512;
    int      knn = 12;               // k for the torus kNN graph
    unsigned threads = 0;            // 0 = hardware concurrency
};

struct ModulusStats {
    uint32_t m = 0;
    std::vector<uint64_t> counts;    // size m, counts[r] = #primes p<=N with p%m==r
    std::vector<uint8_t>  coprime;   // size m, gcd(r,m)==1
    std::vector<double>   z;         // size m, deviation z-score (coprime classes only)
    double chi2 = 0; int dof = 0; double pval = 1;
    double max_abs_z = 0; uint32_t argmax_class = 0;
};

struct TransStats {
    uint32_t q = 0;
    std::vector<uint64_t> obs;       // q*q, obs[i*q+j] = #(p_n%q==i, p_{n+1}%q==j)
    std::vector<uint64_t> train;     // transitions with p_{n+1} <= N/2
    std::vector<uint64_t> test_;     // obs - train
    std::vector<double>   bias;      // q*q, obs/expected - 1 under independence (coprime cells)
    std::vector<uint32_t> classes;   // coprime residues of q, ascending
    double chi2 = 0; int dof = 0; double pval = 1;
    double mean_diag_bias = 0;       // mean of bias[i][i] over coprime i (LO-S repulsion)
};

struct TopoReport {
    uint32_t L = 0;
    std::vector<uint32_t> nodes;     // residues coprime to L, ascending (n of them)
    std::vector<uint32_t> torus_dims;// distinct prime factors of L
    std::vector<double> phi;         // n, empirical density field (z-score per class)
    std::vector<double> eigval;      // n, Laplacian eigenvalues ascending
    std::vector<double> embed3;      // n*3, spectral embedding (eigvecs 2..4)
    std::vector<double> power;       // n, graph-Fourier power of phi per mode (normalized)
    double spectral_gap = 0;         // lambda_2
    double cheeger = 0;              // lambda_2 / 2 (approximation)
    int    betti0 = 0;               // connected components of the kNN graph
    double moran = 0;                // Moran's I spatial autocorrelation of phi
    // group Fourier (Dirichlet character) spectrum of the density field:
    // energy fraction per character of (Z/L)*, sorted descending
    std::vector<double> char_power;
    std::vector<int>    char_order;  // order of each character, matching char_power
    double char_low_frac = 0;        // energy in real characters (order <= 2)
};

struct PatternRow {
    uint32_t q = 0;
    double chi2 = 0, pval = 1;       // vs independence
    double diag_bias = 0;            // mean same-residue bias (repulsion if < 0)
    double acc = 0;                  // out-of-sample argmax accuracy
    double acc_uniform = 0;          // 1/phi(q)
    double gain = 0;                 // acc - acc_uniform
    // gap-model null: next gap independent of current residue, subject to the
    // wheel constraint (target class must be coprime to q)
    double chi2_gap = 0, pval_gap = 1;
    double resid_diag = 0;           // diagonal bias REMAINING under the gap model
    double acc_gapmodel = 0;         // out-of-sample acc of the gap-model predictor
    double gain_beyond = 0;          // acc - acc_gapmodel: the genuine correlation
    // second eigenvalue of the empirical transition operator P(i -> j):
    // signed (trace - 1) when there are two classes, |lambda_2| estimate otherwise.
    // 1 - |lambda_2| is the operator's spectral gap; |lambda_2| is the memory.
    double lambda2 = 0;
};

struct PatternScan {
    std::vector<PatternRow> rows;    // q = 3..30
    uint32_t q2 = 0;                 // modulus of the order-2 test (= cfg.q)
    double acc_order1 = 0;           // predict next residue from 1 previous prime
    double acc_order2 = 0;           // ... from 2 previous primes (same test triples)
    double acc_uniform = 0;
    uint64_t test_triples = 0;
};

struct GapGapReport {
    double pearson = 0;              // correlation of consecutive gaps
    double chi2 = 0; int dof = 0; double pval = 1;   // vs independence
    double r2_oos = 0;               // out-of-sample R^2: conditional mean vs global mean
    double mean_gap = 0;
    std::vector<double> cond_x;      // g_n values with enough data
    std::vector<double> cond_mean;   // E[g_{n+1} | g_n]
    std::vector<double> cond_n;      // sample count per point
};

struct HLReport {
    double li2 = 0;                  // integral_2^N dt/ln^2 t
    std::vector<uint32_t> gaps;      // even g <= 120
    std::vector<uint64_t> pairs;     // exact #(p, p+g both prime <= N)
    std::vector<double> singular;    // C(g)
    std::vector<double> predicted;   // C(g) * Li2(N)
    std::vector<double> ratio;       // measured / predicted
    double mean_abs_dev = 0;         // mean |ratio - 1|
    double max_abs_dev = 0;
    uint32_t argmax_g = 0;
};

struct PredReport {
    // Density predictor: rank classes mod L by training-half prime density,
    // sweep coverage, measure test-half precision. Baseline = wheel (all
    // coprime classes).
    std::vector<double> coverage;    // fraction of coprime classes included
    std::vector<double> precision;   // test primes / test integers in included classes
    std::vector<double> lift;        // precision / base_precision
    double base_precision = 0;
    uint64_t train_primes = 0, test_primes = 0;
    // Transition predictor: predict residue of next prime from current one.
    double trans_acc = 0;            // argmax-row accuracy on test half
    double trans_base_marginal = 0;  // predict globally most common next residue
    double trans_base_uniform = 0;   // 1/phi(q)
};

struct Results {
    Config cfg;
    bool valid = false;
    std::string error;

    uint64_t prime_count = 0;
    uint64_t last_prime = 0;
    uint64_t max_gap = 0;

    std::vector<ModulusStats> mods;
    TransStats trans;
    std::vector<uint32_t> jointL;        // size L, exact counts of all primes <= N
    std::vector<uint32_t> jointL_train;  // primes <= N/2
    std::vector<uint64_t> gap_hist;      // index = gap, size 1024
    std::vector<uint64_t> gap_hist_train;// gaps with p_{n+1} <= N/2
    uint64_t gap_overflow = 0;
    std::vector<uint64_t> pair_gap;      // q*1024: [i*1024+g] = #(p_n%q==i, gap==g)

    std::vector<uint64_t> sample;        // strided sample of primes
    uint64_t sample_stride = 1;

    std::vector<double> cp_x;            // checkpoint x values
    std::vector<double> race4;           // pi(x;4,3) - pi(x;4,1)
    std::vector<double> race3;           // pi(x;3,2) - pi(x;3,1)

    TopoReport topo;
    PredReport pred;
    PatternScan patterns;
    GapGapReport gapgap;
    HLReport hl;

    double t_sieve = 0, t_analysis = 0;
};

struct Progress {
    std::atomic<double> frac{0};
    std::atomic<int>    stage{0};        // index into stage_name()
    std::atomic<bool>   cancel{false};
};

const char* stage_name(int stage);

Results run_pipeline(const Config& cfg, Progress* prog = nullptr);

uint64_t gcd_u64(uint64_t a, uint64_t b);

} // namespace prt
