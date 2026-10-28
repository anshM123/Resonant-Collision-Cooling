# Finite-resource cooling of qudits beyond thermodynamic length

Ansh Mishra and Aryan Senthilkumar

This repository contains the manuscript and the code for our paper on finite-resource cooling in Markovian collision
models. The paper addresses a question left open by P. Taranto, P. Lipka-Bartosik, N. A. Rodríguez-Briones,
M. Perarnau-Llobet, N. Friis, M. Huber and P. Bakhshinezhad, *Efficiently cooling quantum systems with finite
resources: insights from thermodynamic geometry*, Phys. Rev. Lett. **134**, 070401 (2025)
([arXiv:2404.06649](https://arxiv.org/abs/2404.06649)).

## Background

In the collision model of Taranto et al., a d-level system starts in the Gibbs state τ_S(β) and is cooled to τ_S(λβ)
in N steps. In each step it interacts unitarily with a fresh machine, which starts in a thermal state at inverse
temperature β and whose Hamiltonian can be chosen freely. With swaps, the heat dissipated beyond Landauer's bound is
(L\*)²/2N + O(N⁻²), where L\* is the thermodynamic length between the initial and final Gibbs states. Taranto et al.
proved that this is optimal for qubits. Their numerics suggested that it might be optimal for every d, and they left
the general case open.

## Results

**Swaps are not optimal for any d ≥ 3.** Permutations of energy eigenstates that move population through several
resonant system–machine channels at once dissipate less than every swap protocol with the same number of machines, and
the gain is already in the leading coefficient in 1/N. The reason is that the dissipation is quadratic in the population
flow, so splitting the flow between channels lowers it, much as a current shared between parallel resistors dissipates
less heat. Near infinite temperature and near the ground state this works for every spectrum, and for commensurate
spectra (for example equally spaced levels) it also works at most intermediate temperatures. Near the ground state the
excess cost is reduced by a factor approaching d − 1.

**Theorem 1: coherence does not help, to second order.** Around a dissipationless collision, every collision costs at
least as much as a mixture of resonant permutations with the same first-order change of the spectrum. The proof uses
monotonicity of the Kubo–Mori metric under the partial trace for the coherent part, and Birkhoff's theorem with a
convexity argument for the rest. For qubits this recovers the optimality of swaps among all unitaries. For the equally
spaced qutrit at infinite temperature, a dual certificate over all 9! permutations and all machine states (checked in
floating point and in exact rational arithmetic) shows that half the thermodynamic-length rate is optimal for qutrit
machines.

**Theorem 2: maximal gain.** At the maximally mixed state, a collision with d_M-level machines that is close to
dissipationless costs at least 1/(d_M − 1) of the swap cost of the same move, and transfers between two levels attain
this bound. For qubit machines the bound holds at every state of the system. We conjecture that it holds at every state
for all d_M, which would give Σ_N ≥ (L\*)²/2(d_M − 1)N in the quasistatic regime. The evidence is exhaustive enumeration
for (d, d_M) = (2,3), (2,4) and (3,3); about 95,000 energy-conserving permutations for (3,4) and (4,4); a 60-digit
check along the qutrit Gibbs curve up to βω = 40; and an adversarial search over 9,687 random resonant permutations, in
which the bound is never exceeded. A weaker version of the conjecture, which keeps only the machine-marginal condition,
is true for d_M = 2 (proved) and, numerically, for d_M = 3, but fails for d_M = 4 (ratio up to 0.773 > 3/4, checked in
50 digits), so a general proof has to use the full resonance structure.

## Exact protocols

All protocols use 41 machines. The swap values are global optima of a convex problem with exact endpoints, and the
resonant protocols were re-evaluated in 40-digit arithmetic with explicit machine Hamiltonians. Dissipations are in
units of 10⁻⁶.

| d | βω → λβω | best swap protocol | resonant protocol | reduction |
|---|---|---|---|---|
| 3 | 0.01 → 0.05 | 13.001 | 6.911 | 46.8% |
| 3 | 0.1 → 0.5 | 1235.9 | 967.5 | 21.7% |
| 3 | 3 → 9 | 2288.5 | 1794.1 | 21.6% |
| 3 | 8 → 16 | 16.288 | 8.193 | 49.7% |
| 4 | 0.05 → 0.25 | 596.37 | 451.72 | 24.3% |
| 4 | 3 → 6 | 1533.6 | 872.2 | 43.1% |
| 4 | 0.01 → 0.05 | 24.368 | 16.335 | 33.0% |
| 5 | 0.01 → 0.05 | 38.971 | 28.161 | 27.7% |
| 6 | 0.01 → 0.05 | 56.799 | 43.666 | 23.1% |
| 4 | 8 → 16 | 16.288 | 6.060 | 62.8% |
| 5 | 8 → 16 | 16.288 | 4.613 | 71.7% |
| 6 | 8 → 16 | 16.288 | 3.730 | 77.1% |
| 7 | 8 → 16 | 16.288 | 3.137 | 80.7% |
| 8 | 8 → 16 | 16.288 | 2.714 | 83.3% |
| 3, E = (0, 1, 2.3) | β: 8 → 16 | 16.282 | 8.791 | 46.0% |
| 4, E = (0, 1, 1.7, 3.1) | β: 8 → 16 | 16.346 | 6.662 | 59.2% |
| 3, E = (0, 0.3, 1) | β: 0.01 → 0.05 | 3.4059 | 2.5479 | 25.2% |
| 3, E = (0, 0.8, 1) | β: 0.01 → 0.05 | 3.6697 | 3.0665 | 16.4% |
| 3, E = (0, 1, 2.3) | β: 0.01 → 0.05 | 17.212 | 11.845 | 31.2% |

Runs with N = 40, 80 and 160 show that the gain is in the leading coefficient. For d = 3 and βω = 0.01 → 0.05, using
π_A in nine of every ten steps and a swap in the tenth, N·Σ_N = 2.82·10⁻⁴ for all three values of N, compared with
(L\*)²/2 = 5.33·10⁻⁴ for swaps.

## Limitations

- Both theorems are second-order statements around dissipationless collisions. The N-step bound assumes the
  quasistatic regime, in which each collision is an O(1/N) perturbation of one that is dissipationless for its incoming
  state.
- Collisions applied O(1/N) away from a resonance manifold are still dominated by permutations, but the offset could
  open additional moves. In an optimistic relaxation where each mechanism picks its own offset
  (`c23_offset_test.py`), the qutrit hull ratio drops by 0.14% at βω = 0.02, 0.6% at 0.1, 0 at 1.6, 3.6% at 4 and
  1.5% at 8. Whether real protocols can use any of this is open.
- At intermediate temperatures the gain needs nearly commensurate levels. For H_S = ω·diag(0, 1, 2 + δ) and
  βω = 0.1 → 0.5, the pattern π_B, π_A saves 23% at δ = 0.02 and 15% at δ = 0.1 but nothing at δ = 0.3. At δ = 0.3
  the extreme-level exchange still saves 31% closer to infinite temperature.
- Near infinite temperature and near the ground state the gains are an instance of the larger-machine effect already
  noted by Taranto et al. Here it works with machines of the same dimension as the system.
- The resonant machines use level spacings up to about twice those of the swap protocol. This is allowed in the
  setting, where machine Hamiltonians are free.
- Theorem 1 of Taranto et al., which is about swap protocols, is not affected.

## Reproducing the results

Use Python 3 with numpy, scipy, torch, cvxpy and mpmath, and run the scripts from `code/`. For example:

```
python c2_local_metric.py 3 0                         # local cost ratios of all resonant qutrit permutations
python c7_protocol.py 0.1 5 40 hot                    # optimize a resonant qutrit protocol (writes logits_*.npy)
python c8_independent_check.py 0.1 5.0 40 hot         # 40-digit re-evaluation with explicit Hamiltonians
python c9_swap_convex.py 0.1 5 40 41                  # global optimum over swap protocols
python c11_infinite_T.py 3 20000 4                    # infinite-temperature optimum (1/2 for d = 3)
python c14_protocol_d.py 4 0.01 5 40 ext_d4.json 0S   # d-level protocol with the extreme-level exchange
python c15_check_d.py 4 0.01 5.0 40 ext_d4.json 0S    # its 40-digit re-evaluation
python c19_local_domination.py 30                     # Theorem 1: decomposition identity on random coherent collisions
python c19b_mp_check.py 5                             # Theorem 1: second-order formula vs exact dissipation, 60 digits
python c20_infT_certificate.py                        # infinite-temperature dual certificate (all 9! permutations)
```

The optimized machine populations of every protocol in the table are included as `logits_*.npy`, so the checks can be
run directly. The figures are produced by `paper/make_fig1.py` and `paper/make_fig2.py` (these need matplotlib).

## Where each result comes from

Every number quoted in the paper has a saved log (`*.log`) next to the script that produced it.

| Scripts | What they do |
|---|---|
| `c2`, `c4`, `c10` | local quadratic forms and convex hulls |
| `c7`, `c14` | protocol optimization, for qutrits and for general d |
| `c8`, `c15` | independent high-precision checks of the protocols |
| `c9`, `c17` | swap baselines (convex program; exact endpoints) |
| `c11` | infinite temperature |
| `c12`, `c13` | cone-program gauge and active mechanisms; ququart ratios (`c12_ququart.log`) |
| `c18`, `c19`, `c19b` | Theorem 1: coherent collisions vs permutation mixtures, and the 60-digit expansion check |
| `c19c` | Theorem 1 for near-resonant collisions, in 50 digits |
| `c20`, `c20b` | infinite-temperature dual certificate over all 9! qutrit permutations (floating point; exact rationals) |
| `c21`, `c22`, `c22b` | exact hull gauge along the Gibbs curve (primal–dual bounds; Fig. 1a) |
| `c23` | near-resonant offset relaxation |
| `c24`, `c24b`, `c25` | Theorem 2 tests (exhaustive enumeration; 60-digit Gibbs-curve check) |
| `c26`, `c27b`, `c27c` | the marginal-only relaxation (random sampling; ratio LP; 50-digit vertex check); `c27` is superseded |
| `c28`, `c28b`, `c28c` | Theorem 2 tests for (3,4) and (4,4), with a 50-digit recheck and random spectra |
| `c29` | exact hull on the interior of the Fisher–Rao geodesic (swaps locally optimal for βω from 1 to 3) |
| `c30` | adversarial search over resonance sets |

Data files: `ratio_scan*.csv`, mechanism files `*.json`, and protocol data `logits_*.npy`.

## License

MIT (see LICENSE).
