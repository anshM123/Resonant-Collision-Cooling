"""Independent re-evaluation of saved protocols with explicit Hamiltonians, density matrices and energy bookkeeping.

No use of the dissipation identity: for each step build H_M = -(1/beta) log q (machine thermal at beta),
the 9x9 permutation unitary P, rho' = P (rho_S x tau_M) P^T, and accumulate beta * tr[H_M (rho'_M - tau_M)].
Dissipation = beta*DeltaE_M - (S(rho_0) - S(rho_N)).  Also report the final state's distance to the Gibbs target.
High precision via mpmath (40 digits).
usage: c8_independent_check.py beta lam N pattern
"""
import sys
import numpy as np
import mpmath as mp
mp.mp.dps = 40

SWAP = [3 * (c % 3) + c // 3 for c in range(9)]
PI576 = [5, 0, 2, 8, 3, 1, 6, 7, 4]
PI711 = [8, 5, 2, 7, 0, 1, 6, 3, 4]
PI636 = [5, 3, 2, 8, 4, 0, 6, 7, 1]
PI713 = [8, 6, 2, 3, 4, 5, 1, 7, 0]


LETTERS = {'S': [3 * (c % 3) + c // 3 for c in range(9)], 'A': [8, 6, 2, 3, 4, 5, 1, 7, 0], 'B': [5, 3, 2, 8, 4, 0, 6, 7, 1],
           'C': [5, 0, 2, 8, 3, 1, 6, 7, 4], 'D': [8, 5, 2, 7, 0, 1, 6, 3, 4], 'F': [8, 0, 2, 3, 4, 5, 6, 7, 1]}
def seq_pattern(pattern, N):
    word = pattern.split(':', 1)[1]
    return [LETTERS[word[n % len(word)]] for n in range(N)]

beta, lam, N, pattern = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
lg = np.load(f"logits_{pattern.replace(':', '')}_b{beta}_l{lam}_N{N}.npy")
if pattern.startswith('seq:'): pis = seq_pattern(pattern, N)
elif pattern == 'swap': pis = [SWAP] * N
elif pattern == 'alt': pis = [PI576 if n % 2 == 0 else PI711 for n in range(N)]
elif pattern == 'hot': pis = [PI636 if n % 2 == 0 else PI713 for n in range(N)]
elif pattern == 'p713': pis = [PI713] * (N - 1) + [SWAP]
elif pattern == 'p713b': pis = [PI713 if n % 4 else SWAP for n in range(N)]
elif pattern == 'p713c': pis = [SWAP if n % 10 == 9 else PI713 for n in range(N)]
elif pattern == 'alt_swap': pis = [PI576 if n % 2 == 0 else PI711 for n in range(N - 2)] + [SWAP, SWAP]

B = mp.mpf(beta)
ES = [mp.mpf(0), mp.mpf(1), mp.mpf(2)]
def gibbs(b):
    w = [mp.e ** (-b * e) for e in ES]; Z = sum(w); return [x / Z for x in w]
def ent(v):
    return -sum(x * mp.log(x) for x in v if x > 0)

rho = mp.matrix(3, 3)
g0 = gibbs(B)
for i in range(3): rho[i, i] = g0[i]
S0 = ent(g0)
betaEM = mp.mpf(0)
for n in range(N):
    # machine: q = softmax(logits), Hamiltonian H_M = -(1/beta) log q  => tau_M(beta) = q exactly
    l = [mp.mpf(float(x)) for x in lg[n]]
    m = max(l); w = [mp.e ** (x - m) for x in l]; Z = sum(w); q = [x / Z for x in w]
    HM = [-mp.log(x) / B for x in q]
    tauM = mp.diag(q)
    joint = mp.matrix(9, 9)
    for a in range(3):
        for b_ in range(3):
            for c in range(3):
                for d_ in range(3):
                    joint[3 * a + c, 3 * b_ + d_] = rho[a, b_] * tauM[c, d_]
    P = mp.matrix(9, 9)
    for c in range(9): P[pis[n][c], c] = 1
    joint2 = P * joint * P.T
    rhoS = mp.matrix(3, 3); rhoM = mp.matrix(3, 3)
    for a in range(3):
        for b_ in range(3):
            rhoS[a, b_] = sum(joint2[3 * a + c, 3 * b_ + c] for c in range(3))
            rhoM[a, b_] = sum(joint2[3 * c + a, 3 * c + b_] for c in range(3))
    betaEM += B * sum(HM[k] * (rhoM[k, k] - q[k]) for k in range(3))
    offS = max(abs(rhoS[a, b_]) for a in range(3) for b_ in range(3) if a != b_)
    assert offS < mp.mpf('1e-30'), 'system acquired coherence?'
    rho = rhoS
pN = [rho[i, i] for i in range(3)]
target = gibbs(B * lam)
SN = ent(pN)
diss = betaEM - (S0 - SN)
err = max(abs(pN[i] - target[i]) for i in range(3))
print(f'{pattern} N={N}: beta*DeltaE_M = {mp.nstr(betaEM, 15)}   e*DeltaS_S = {mp.nstr(S0 - SN, 15)}   '
      f'dissipation = {mp.nstr(diss, 15)}   final |p_N - target| = {mp.nstr(err, 5)}')
# exact completion: one extra swap step to the exact target (cost D(p_N||target))
corr = sum(pN[i] * mp.log(pN[i] / target[i]) for i in range(3))
print(f'   + exact final swap to the Gibbs target: extra dissipation {mp.nstr(corr, 5)}  => total {mp.nstr(diss + corr, 15)} with N+1 steps')
