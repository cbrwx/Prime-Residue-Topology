#pragma once
#include "core/pipeline.h"

namespace prt {

// Approximate upper-tail p-value of the chi-square distribution
// (Wilson–Hilferty normal approximation; adequate for reporting).
double chi2_pval(double chi2, int dof);

// #{ n : 1 <= n <= x, n ≡ r (mod L) }
uint64_t count_congruent(uint64_t x, uint32_t r, uint32_t L);

// Fill z-scores, chi2, pval for one modulus (Dirichlet equidistribution null
// over the coprime classes).
void finalize_mod_stats(ModulusStats& ms);

// Fill bias matrix, chi2 (vs independence of consecutive residues), pval,
// coprime class list, mean diagonal bias.
void finalize_trans_stats(TransStats& ts);

// Analyze one transition matrix mod q: chi2 vs independence, mean diagonal
// bias, out-of-sample argmax accuracy (train matrix -> test = obs-train), and
// the gap-model null (expected transitions if the next gap were independent of
// the current residue, renormalized over wheel-allowed target classes).
PatternRow analyze_transition_q(uint32_t q, const uint64_t* obs, const uint64_t* train,
                                const uint64_t* gap_hist, const uint64_t* gap_hist_train,
                                size_t gap_n);

// Classes with enough occupancy in a q x q transition matrix to analyze
// (marginal count >= max(8, total/(1000*q))). For primes this recovers the
// coprime classes; for arbitrary sequences it adapts to whatever occurs.
std::vector<uint32_t> occupied_classes(uint32_t q, const uint64_t* obs);

// finalize_mod_stats against a UNIFORM null over all m classes (for arbitrary
// sequences, where no wheel is assumed).
void finalize_mod_stats_uniform(ModulusStats& ms);

// analyze_transition_q for arbitrary sequences: classes = occupied, and the
// difference-model null is derived by folding the matrix along its diagonals
// (equivalent to the gap histogram for monotone sequences).
PatternRow analyze_transition_general(uint32_t q, const uint64_t* obs, const uint64_t* train);

// Order-2 vs order-1 Markov comparison at modulus q, scored on the SAME test
// triples: predict residue k of p_{n+2} from (i, j) vs from j alone.
// o2 arrays are q^3 (index (i*q + j)*q + k); t1_train is the q^2 order-1
// training matrix.
void analyze_order2(PatternScan& ps, uint32_t q,
                    const std::vector<uint64_t>& o2_obs,
                    const std::vector<uint64_t>& o2_train,
                    const std::vector<uint64_t>& t1_train);

// integral_2^x dt / ln^2(t)  (Simpson; the Hardy-Littlewood pair-count scale)
double li2_integral(double x);

// Hardy-Littlewood singular series C(g) = 2*C2 * prod_{odd p | g} (p-1)/(p-2);
// 0 for odd g.
double hl_singular_series(uint32_t g);

// Consecutive-gap correlation analysis from the joint (g_n, g_{n+1}) table
// (maxg x maxg, row = g_n). Train counts are the pairs from the first half.
void finalize_gapgap(GapGapReport& G, const uint64_t* obs, const uint64_t* train, uint32_t maxg);

// Compare measured pair counts (index = gap, up to 120) with C(g)*Li2(N).
void finalize_hl(HLReport& H, const std::vector<uint64_t>& pair_counts, uint64_t N);

} // namespace prt
