# Prime Residue Topology

An exploration of prime number structure in residue space — in two acts.

## Act 1: the hypothesis (primer.py)

The original idea: map integers into a multi-dimensional residue space
("fingerprints" mod a set of moduli) and look for topological structure that
separates primes from composites — potential fields, tensor gradients,
Laplacian spectra. The full theory writeup is in [THEORY.md](THEORY.md) and the
original Python implementation is [primer.py](primer.py), both kept as-is.

## Act 2: the rebuild (cpp/)

The hypothesis was later rebuilt from scratch in C++ ([cpp/](cpp/)) with two
rules: no feature may be computed from primality itself, and every claim of
structure must beat the null model that says there is none — Dirichlet
equidistribution for densities, independence and a gap-frequency model for
transitions, out-of-sample scoring for every predictor. A multithreaded
segmented sieve measures every prime up to N (10^9 in ~2 s, 10^11 in minutes)
and aggregates them exactly onto the residue torus via CRT.

### What it found

- **Single-prime residue space is flat.** Prime counts per residue class match
  the Dirichlet prediction to fractions of a standard deviation at every
  modulus tested, through N = 10^11. The dramatic geometry in residue-space
  visualizations is the wheel (divisibility), not hidden order.
- **Pair space is where the structure lives.** Consecutive primes repel their
  own residue class (the Lemke Oliver–Soundararajan bias, 2016). A gap-model
  null splits it: most is gap statistics, but an irreducible core at **mod 3**
  survives everything — ~10 pp of out-of-sample predictive gain and ~27%
  same-class suppression beyond what gap frequencies explain.
- **The decay law.** Measured across N = 10^9 → 10^11, the genuine correlation
  decays like C/log N (gain × ln N ≈ 220–227), matching the Hardy–Littlewood
  prediction. Order-2 memory exists too (+0.3 pp) and fades the same way.
- **Hardy–Littlewood, tested.** Counting all prime pairs (p, p+g) for even
  g ≤ 120 against C(g)·Li₂(N): mean deviation **0.003%** at N = 10^11
  (twins: 224,376,048 measured vs 224,368,865 predicted). Consecutive gaps
  also anti-correlate (r ≈ −0.02, fading with N).
- **Classical phenomena reproduced along the way:** Chebyshev races, the
  Bays–Hudson lead-change region near 6.35×10^9, exact π(N) and maximal-gap
  values at every scale tested.

### The instrument

A Dear ImGui desktop app (Windows) with live charts and five 3D views
(residue strands, CRT lattice, density torus, spectral embedding of the torus
Laplacian, and a pair-space view of deviations from the gap-model null), a
pattern-classification scan over transition moduli q = 3..100, and a
**sequence auditor** that runs the same analysis on any integer sequence from
a file. Every run is logged to a results folder (summary + CSVs + snapshots).
A CLI (`prt_cli`) provides self-tests and headless runs. Build instructions in
[cpp/README.md](cpp/README.md).

### The moral

The hypothesis was half right, one dimension up from where it started: there
is no static pattern in residue space, but the correlations *between*
consecutive primes are real, strongest at mod 3, governed by the
Hardy–Littlewood k-tuple machinery, and slowly dissolving as numbers grow.
Getting there required deleting every feature that secretly knew the answer —
which turned out to be the most instructive part.
