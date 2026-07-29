# Prime Residue Topology

An exploration of prime number structure in residue space, in two acts.

## Act 1: the hypothesis (primer.py)

The original idea: map integers into a multidimensional residue space (their
"fingerprints" modulo a set of moduli) and look for topological structure that
separates primes from composites, using potential fields, tensor gradients and
Laplacian spectra. The full theory writeup is in [THEORY.md](THEORY.md) and the
original Python implementation is [primer.py](primer.py). Both are kept as is.

## Act 2: the rebuild (cpp/)

The hypothesis was later rebuilt from scratch in C++ ([cpp/](cpp/)) with two
rules. First, no feature may be computed from primality itself. Second, every
claim of structure must beat the null model that says there is none: Dirichlet
equidistribution for densities, independence and a gap frequency model for
transitions, and out of sample scoring for every predictor. A multithreaded
segmented sieve measures every prime up to N (10^9 in about 2 seconds, 10^11
in minutes) and aggregates them exactly onto the residue torus via the Chinese
Remainder Theorem.

### What it found

* Single prime residue space is flat. Prime counts per residue class match
  the Dirichlet prediction to fractions of a standard deviation at every
  modulus tested, through N = 10^11. The dramatic geometry seen in residue
  space visualizations is the wheel (divisibility), not hidden order.
* Pair space is where the structure lives. Consecutive primes avoid repeating
  their own residue class (the bias found by Lemke Oliver and Soundararajan
  in 2016). A gap model null splits the effect: most of it is gap statistics,
  but an irreducible core at mod 3 survives everything, worth about 10
  percentage points of out of sample predictive gain and about 27 percent
  same class suppression beyond what gap frequencies explain.
* The decay law. Measured from N = 10^9 to 10^11, the genuine correlation
  decays like C divided by log N (gain times ln N stays near 220 to 227),
  matching the Hardy-Littlewood prediction. Second order memory exists too
  (about 0.3 percentage points) and fades the same way.
* Hardy-Littlewood, tested. Counting all prime pairs (p, p+g) for even
  g up to 120 against the conjectured C(g) times Li2(N): mean deviation
  0.003 percent at N = 10^11 (twins: 224,376,048 measured versus 224,368,865
  predicted). Consecutive gaps also anticorrelate (r near minus 0.02, fading
  as N grows).
* Classical phenomena reproduced along the way: the Chebyshev races, the
  Bays-Hudson lead change region near 6.35 times 10^9, and exact values of
  pi(N) and the maximal gaps at every scale tested.

### The instrument

A Dear ImGui desktop app (Windows) with live charts and five 3D views
(residue strands, CRT lattice, density torus, spectral embedding of the torus
Laplacian, and a pair space view of deviations from the gap model null), a
pattern classification scan over transition moduli q = 3 to 100, and a
sequence auditor that runs the same analysis on any integer sequence read
from a file. Every run is logged to a results folder with a text summary,
CSV tables and 3D snapshots. A command line tool (prt_cli) provides self
tests and headless runs. Build instructions are in
[cpp/README.md](cpp/README.md).

### The moral

The hypothesis was half right, one dimension up from where it started. There
is no static pattern in residue space, but the correlations between
consecutive primes are real, strongest at mod 3, governed by the
Hardy-Littlewood k tuple machinery, and slowly dissolving as numbers grow.
Getting there required deleting every feature that secretly knew the answer,
which turned out to be the most instructive part.
