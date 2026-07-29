# Prime Residue Topology — the honest rebuild (C++)

A real, exact-numbers reimplementation of the Prime-Residue-Topology idea:
map primes into residue ("fingerprint") space and look for higher-dimensional
structure — but with every leap of faith removed.

## The two ideas that make it work

1. **Honesty.** No feature is ever computed *from* primality (the original
   Python used φ(n)/(n−1) — a primality test in disguise — and boosted graph
   edges between primes). Here primality is only the thing being *counted*.
   Every statistic is tested against the Dirichlet equidistribution null, and
   every predictor is scored out-of-sample against the trivial wheel baseline.

2. **Scale.** By CRT, the fingerprint of n depends only on n mod L
   (L = lcm of the moduli). The residue space is a finite torus with at most
   L points — so we sieve *every* prime up to N with a multithreaded segmented
   sieve (10⁹ in ~2 s) and aggregate exact counts onto the torus. Analysis cost
   is independent of N.

## What it measures (all exact, single pass)

- Prime counts per residue class for a set of statistics moduli (+ chi², z-scores)
- Joint counts mod L (the torus; any divisor triple follows by CRT)
- Prime gap histogram
- Consecutive-prime transition matrix mod q → the Lemke Oliver–Soundararajan
  repulsion (~−28% diagonal bias at N = 10⁹, matching the literature)
- Chebyshev races (mod 4 and mod 3 checkpoints)
- Torus topology: kNN graph on wrap-around residue coordinates, Laplacian
  spectrum, Betti₀, spectral gap, Moran's I, graph-Fourier power of the
  *measured* density field, 3D spectral embedding
- Honest prediction: train on (0, N/2], test on (N/2, N] — density ranking
  vs wheel baseline (lift ≈ 1.0: the honest result) and next-prime residue
  prediction (~30% vs 25% uniform: real out-of-sample structure)
- Pattern classification: transition matrices for every q = 3..100, each scored
  out-of-sample AND against a **gap-model null** (next gap independent of the
  current residue, wheel-constrained). What survives the null is genuine
  residue–gap correlation: strongest at q = 3 (~+10.6 pp beyond the gap model
  at N = 10⁹, residual diagonal −28%) — the Lemke Oliver–Soundararajan bias,
  whose mechanism is the Hardy–Littlewood k-tuple correlations. An order-2
  Markov test measures memory depth beyond one step (~+0.3 pp).

## Building (Windows, MSVC 2022 + CMake + Ninja)

```
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

Targets:
- `build\prime_topology.exe` — the GUI (Dear ImGui + ImPlot + OpenGL, vendored
  in `external/`). Configure a run on the left, press **Run pipeline**, explore
  the tabs. The 3D Explorer has four views: residue strands (raw sampled
  primes), CRT lattice cube and density torus (exact aggregates of *all*
  primes), and the spectral embedding of the torus classes.
- `build\prt_cli.exe` — headless: `prt_cli selftest` (sieve/eigensolver/stats
  checks) and `prt_cli run 1e9` (full pipeline, text summary).

## Layout

```
src/core/     sieve, streaming pipeline, statistics, torus topology, prediction
src/cli/      self-tests + headless runs
src/app/      Win32/WGL shell, GL loader, 3D point renderer, ImGui panels
external/     Dear ImGui v1.90.9, ImPlot v0.16 (vendored)
```

Verified against: π(10⁶)=78,498; π(10⁷)=664,579; π(10⁸)=5,761,455;
π(10⁹)=50,847,534; max gap below 10⁹ = 282; path-graph Laplacian spectra.
