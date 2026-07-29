#pragma once
// Point-cloud builders for the 3D explorer and run snapshots.
// Output: interleaved x,y,z,value (value in [0,1] -> colormap).
#include <vector>
#include "core/pipeline.h"

namespace prtclouds {

void build_strands(const prt::Results& R, int m, int max_pts, std::vector<float>& out);
void build_lattice(const prt::Results& R, int m1, int m2, int m3, std::vector<float>& out);
void build_torus(const prt::Results& R, int m1, int m2, std::vector<float>& out);
void build_spectral(const prt::Results& R, std::vector<float>& out);
// Pair space: x = residue of p_n, z = residue of p_{n+1}, y = gap;
// color = deviation of the exact (residue, gap) count from the gap-model null.
void build_pairspace(const prt::Results& R, int gmax, std::vector<float>& out);

} // namespace prtclouds
