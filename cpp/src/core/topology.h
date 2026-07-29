#pragma once
#include "core/pipeline.h"

namespace prt {

// Dense symmetric eigensolver (Householder tridiagonalization + implicit QL).
// A: n*n row-major, destroyed. On return evals ascending, evecs n*n row-major
// with evecs[i*n+k] = component i of eigenvector k. Returns false on
// non-convergence.
bool eig_sym(std::vector<double>& A, int n,
             std::vector<double>& evals, std::vector<double>& evecs);

// Build the honest residue-torus topology: nodes are classes coprime to L,
// geometry comes from wrap-around residue coordinates only, and the scalar
// field phi is the MEASURED prime density deviation per class.
TopoReport compute_topology(uint32_t L, const std::vector<uint32_t>& jointL, int knn);

} // namespace prt
