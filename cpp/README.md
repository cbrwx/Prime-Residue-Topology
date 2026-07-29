# Prime Residue Topology (C++)

A reimplementation of the Prime Residue Topology idea: map primes into
residue ("fingerprint") space and measure the structure that appears there,
with every statistic tested against an explicit null model.

## The two ideas that make it work

1. No feature is computed from primality. Primality is only the thing being
   counted. Every claim of structure is compared against the model that says
   there is none: Dirichlet equidistribution for densities, independence and
   a gap frequency model for transitions, and out of sample scoring (first
   half trains, second half tests) for every predictor.

2. Scale through aggregation. By the Chinese Remainder Theorem, a number's
   fingerprint depends only on n mod L (L = lcm of the moduli). The residue
   space is a finite torus with at most L points, so a multithreaded
   segmented sieve (10^9 in about 2 seconds) can aggregate every prime up to
   N exactly, and analysis cost is independent of N.

## What it measures (single pass over all primes up to N)

* Prime counts per residue class for a set of statistics moduli, with chi
  square tests and z scores against the Dirichlet prediction
* Joint counts mod L (the torus; any divisor triple follows by CRT)
* Prime gap histogram and gap to gap memory (conditional means, correlation,
  out of sample R squared)
* Consecutive prime transition matrices for every modulus q = 3 to 100, each
  decomposed against a gap model null; the surviving correlation is strongest
  at mod 3 (the bias described by Lemke Oliver and Soundararajan)
* An order 2 Markov test measuring memory depth beyond one step
* Prime pair counts (p, p+g) for even g up to 120, compared against the
  Hardy-Littlewood prediction C(g) times Li2(N)
* Chebyshev races (mod 4 and mod 3 checkpoints)
* Torus topology: lattice plus nearest neighbour graph, Laplacian spectrum,
  Betti 0, spectral gap, Moran's I, graph Fourier power of the density field,
  3D spectral embedding
* Out of sample prediction: class density ranking versus the wheel baseline,
  and next residue prediction versus uniform and marginal baselines

## Building (Windows, MSVC 2022 + CMake + Ninja)

```
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

Targets:

* `build\prime_topology.exe`: the GUI (Dear ImGui + ImPlot + OpenGL, vendored
  in `external/`). Configure a run on the left, press Run pipeline, explore
  the tabs. The 3D Explorer has five views: residue strands (raw sampled
  primes), CRT lattice cube and density torus (aggregates of all primes),
  the spectral embedding of the torus classes, and a pair space view showing
  deviations from the gap model null. The Auditor tab runs the same analysis
  on any integer sequence read from a file. The Info tab explains the method.
  Every completed run is logged to `results/` with a summary, CSV tables and
  3D snapshots. Optional flags: `--autorun` starts a run at launch, `--n=`
  sets the range.
* `build\prt_cli.exe`: headless. `prt_cli selftest` (sieve, eigensolver,
  statistics and auditor checks), `prt_cli run 1e9` (full pipeline plus data
  export), `prt_cli audit <file> [q]` (sequence audit plus data export).

## Layout

```
src/core/     sieve, streaming pipeline, statistics, topology, prediction,
              pattern scan, Hardy-Littlewood comparison, sequence auditor
src/cli/      self tests and headless runs
src/app/      Win32/WGL shell, GL loader, 3D point renderer, ImGui panels
external/     Dear ImGui v1.90.9, ImPlot v0.16, stb_image_write (vendored)
```

Verified against known values: pi(10^6) through pi(10^11), maximal prime
gaps at each scale, twin pair counts (8,169 below 10^6; 224,376,048 below
10^11), path graph Laplacian spectra, and the Bays-Hudson region of the
mod 4 race.
