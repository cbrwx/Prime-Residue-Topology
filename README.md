# Prime Residue Topology

An exploration of prime number structure in residue space: a hypothesis, a
rigorous rebuild of it, and the measurements that came out. The repository has
two parts. `primer.py` and [THEORY.md](THEORY.md) are the original hypothesis
and implementation, kept unchanged. [cpp/](cpp/) is the measurement instrument
that was built to test it properly.

## 1. The hypothesis

Every integer n has a residue fingerprint: the vector of its remainders
(n mod m1, n mod m2, ...) across a chosen set of moduli. The fingerprints live
in a bounded multidimensional space, and the hypothesis was that primes occupy
that space in a structured way. Viewed from a high enough dimension, primes
would appear as critical points of a potential field, connected by a tensor
flow, with topological invariants (Laplacian spectra, Betti numbers, spectral
gaps) that distinguish where primes live from where composites live, and
perhaps even allow prediction. THEORY.md develops this in detail and primer.py
implements it with potential fields, gradients, graph Laplacians and low
dimensional embeddings.

The original implementation had two flaws that the rebuild removed. Its
"primeness" potential included the factor phi(n)/(n-1), which equals 1 exactly
when n is prime, so the field partly contained the answer it was looking for.
And its similarity graph strengthened edges between numbers that were both
prime, painting structure around primes by hand. Any pattern found that way
cannot count as evidence, which motivated rule one below.

## 2. What the program actually does

The C++ instrument in cpp/ runs under two rules:

1. No feature is computed from primality. Primality is only the thing being
   counted.
2. Structure only counts if it beats the null model that says there is none,
   and predictions only count out of sample.

### The pipeline

* A multithreaded segmented sieve of Eratosthenes enumerates every prime up
  to N in increasing order (bit packed, odd only, segments sieved in parallel
  and delivered in sequence). 10^9 takes about 2 seconds on a desktop, 10^11
  takes minutes.
* A single streaming pass over the primes fills exact aggregate tables:
  counts per residue class for a set of statistics moduli, joint counts
  modulo L = 2310 (the residue torus; by the Chinese Remainder Theorem a
  fingerprint depends only on n mod L, so the whole range collapses onto a
  few thousand cells and analysis cost is independent of N), the gap
  histogram, a (residue, gap) table, a (gap, next gap) table, consecutive
  prime transition matrices for every modulus q from 3 to 100, an order 2
  transition tensor, prime pair counts (p, p+g) for even g up to 120 via a
  sliding window, and Chebyshev race checkpoints.
* Statistics are then computed against explicit null models:
  * Densities against Dirichlet equidistribution (primes split evenly among
    the classes coprime to m), with chi square tests and per class z scores.
  * Transitions against independence of consecutive residues.
  * Transitions against a stricter gap model null: what the matrix would look
    like if the next gap were independent of the current residue, given the
    gap frequencies and the wheel. Structure that survives this null is
    genuine correlation between a prime's residue and its next gap.
  * Prime pair counts against the Hardy-Littlewood prediction C(g) Li2(N),
    where C(g) is the singular series built from the twin prime constant.
* Every predictor is trained on the first half of the range and scored only
  on the second half, against stated baselines (uniform, marginal, wheel,
  gap model, global mean).
* The torus topology of the original hypothesis is computed on the aggregate:
  a lattice plus nearest neighbour graph on the classes coprime to L,
  Laplacian spectrum, Betti 0, spectral gap, Moran's I of the measured
  density field, and its graph Fourier power spectrum.
* Every completed run is logged to results/ with a text summary, CSV tables
  of every aggregate, and 3D snapshots. A sequence auditor applies the same
  machinery to any integer sequence read from a file, so the method
  generalizes beyond primes.

## 3. Results

All numbers below are from full runs of the instrument at N = 10^9, 10^10
and 10^11 (about 51 million, 455 million and 4.1 billion primes).

### Densities are flat

Prime counts per residue class match the Dirichlet prediction at every
modulus tested (3 through 210), at every scale. Typical chi square values sit
far below their degrees of freedom; the largest single class deviation seen
through 10^11 was under one standard deviation. There is no static pattern in
single prime residue space. The striking geometry visible in residue space
visualizations is the wheel (divisibility by small primes), not hidden order.

### Consecutive primes are correlated

The transition matrix between residues of consecutive primes is far from
independent (chi square about 1.9 million at 10^9 for q = 10). Same residue
repetition is suppressed, for example at 10^9 mod 10:

| from \ to | 1 | 3 | 7 | 9 |
|---|---|---|---|---|
| 1 | -26.7% | +19.4% | +20.9% | -13.6% |
| 3 | -4.1% | -29.8% | +13.1% | +20.8% |
| 7 | +2.1% | +8.4% | -29.9% | +19.5% |
| 9 | +28.8% | +2.1% | -4.1% | -26.7% |

This is the bias discovered by Lemke Oliver and Soundararajan (2016).

### The gap model decomposition finds the core

Most of the raw transition structure is explained by gap frequencies plus the
wheel. What survives the gap model null concentrates at mod 3 and its
multiples: a prime's residue mod 3 genuinely correlates with its next gap.

| | 10^9 | 10^10 | 10^11 |
|---|---|---|---|
| beyond gap gain at q = 3 (test half) | +10.61 pp | +9.72 pp | +8.96 pp |
| gain times ln N | 220 | 224 | 227 |
| residual same class suppression | -27.7% | -27.5% | -27.3% |

The gain decays like a constant over log N, which is the leading order decay
predicted by the Hardy-Littlewood correlation machinery. The slow rise of the
product is consistent with the known second order correction. By contrast,
the famous last digit view (q = 10) retains almost nothing beyond gap
statistics at the predictor level (+0.01 pp at 10^11), even though its
distributional deviation remains enormous.

### Memory is shallow but real

Knowing two previous primes instead of one improves next residue prediction
by +0.33 pp at 10^9, +0.27 pp at 10^10 and +0.24 pp at 10^11 (mod 10, scored
on roughly 2 billion test triples at the largest scale). Consecutive gaps
anticorrelate: Pearson r = -0.0275 at 10^9 and -0.0219 at 10^11, with the
conditional mean of the next gap sloping down as the current gap grows.
Out of sample, gap memory explains about 0.1 percent of next gap variance.
The structure is statistically overwhelming and practically thin.

### Hardy-Littlewood passes a precision test

Counting every prime pair (p, p+g) for even g up to 120 and comparing with
C(g) Li2(N):

| | 10^9 | 10^11 |
|---|---|---|
| mean deviation over all g | 0.022% | 0.003% |
| worst single g | 0.063% | 0.009% |
| twin pairs measured | 3,424,506 | 224,376,048 |
| twin pairs predicted | 3,425,308 | 224,368,865 |

The singular series structure (a factor 2 for gaps divisible by 3, 4/3 for 5,
and so on) is traced by the data across the whole range, and the agreement
tightens with N exactly as the conjectured error term suggests.

### Classical phenomena reproduced

The instrument re-finds known results from raw sieve output: exact values of
pi(N) at every scale (50,847,534 at 10^9 up to 4,118,054,813 at 10^11), the
maximal prime gaps at each scale (282, 354, 464), the Chebyshev races with
team 3 leading, and the Bays-Hudson region near 6.35 times 10^9 where
pi(x;4,1) briefly overtakes pi(x;4,3).

## 4. What it means

Primes are globally fair, locally correlated, and asymptotically forgetful.
Every allowed residue class gets its fair share (Dirichlet), yet each prime
carries real information about its successor, concentrated in a mod 3
repulsion that no amount of gap statistics explains away. All of these
correlations fade like one over log N, so the primes drift toward perfect
randomness without ever quite arriving. The measurements agree with the
Hardy-Littlewood k tuple picture everywhere they touch it.

Two honest boundaries. Residue structure predicts residue classes, never the
primality of individual numbers: the class density lift against the plain
wheel baseline is 1.0000, so nothing here helps factoring or cryptography.
And measurement is not proof: the Hardy-Littlewood comparisons test an open
conjecture numerically; they cannot settle it.

The original hypothesis, in retrospect, was half right and located one
dimension too low. Single primes in residue space are featureless, and that
part of the space is provably flat. The structure the hypothesis reached for
exists in the space of consecutive pairs, and it took removing every feature
that secretly knew the answer to see it clearly.

## 5. The instrument

A Dear ImGui desktop application (Windows) with live charts across eleven
tabs (overview, residue classes, gaps and gap memory, transitions, races,
topology, pattern classification, Hardy-Littlewood, prediction, 3D explorer,
sequence auditor, plus an info page describing the method), five 3D views
(residue strands, CRT lattice cube, density torus, spectral embedding, and a
pair space view that renders deviations from the gap model null), and full
run logging. A command line tool provides self tests (verified against known
values of pi, twin counts, closed form spectra and published phenomena) and
headless runs. Build instructions: [cpp/README.md](cpp/README.md).

## 6. References

* R. J. Lemke Oliver, K. Soundararajan, Unexpected biases in the distribution
  of consecutive primes, PNAS 113 (2016).
* G. H. Hardy, J. E. Littlewood, Some problems of Partitio Numerorum III: on
  the expression of a number as a sum of primes, Acta Mathematica 44 (1923).
* C. Bays, R. H. Hudson, details of the first region where pi(x;4,1) exceeds
  pi(x;4,3), Mathematics of Computation (1978).
* M. Rubinstein, P. Sarnak, Chebyshev's bias, Experimental Mathematics 3
  (1994).
