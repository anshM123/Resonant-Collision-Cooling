"""Which energy-conserving permutations carry the optimal mixture (SOCP) at a given beta; saves them for protocols.
usage: c13_active_mechs.py d beta kmax max_perms top
"""
import sys, itertools, json
import numpy as np
import cvxpy as cp
from c12_resonant_d import mech_forms, shells_perms

d, beta, kmax, maxp, top = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rng = np.random.default_rng(0)
E = np.arange(d, dtype=float)
p = np.exp(-beta * E); p /= p.sum()
w = -p * (E - p @ E)
Dm = 1 / np.sqrt(p); gF = np.sqrt(0.5 * np.sum(w ** 2 / p)); target = Dm * w / gF
blocks, meta = [], []
for e in itertools.combinations_with_replacement(range(kmax + 1), d - 1):
    e = (0,) + e
    q = np.exp(-beta * np.array(e, float)); q /= q.sum()
    for pi in shells_perms(d, e, maxp, rng):
        if np.all(pi == np.arange(d * d)): continue
        M, C = mech_forms(p, q, pi, d)
        if np.linalg.norm(M) < 1e-14: continue
        lam, V = np.linalg.eigh(C)
        if lam.min() < -1e-9 * max(1, abs(lam).max()): continue
        keep = lam > 1e-12 * lam.max()
        T = V[:, keep] / np.sqrt(lam[keep])
        blocks.append((Dm[:, None] * M) @ T); meta.append((e, pi.tolist()))
us = [cp.Variable(B.shape[1]) for B in blocks]
prob = cp.Problem(cp.Minimize(sum(cp.norm(u, 2) for u in us)), [sum(B @ u for B, u in zip(blocks, us)) == target])
prob.solve(solver=cp.CLARABEL)
norms = np.array([np.linalg.norm(u.value) for u in us])
order = np.argsort(-norms)
print(f'd={d} beta={beta}: gauge ratio {prob.value:.6f}; total weight {norms.sum():.4f}')
out = []
swap = [(c % d) * d + c // d for c in range(d * d)]
for k in order[:top]:
    e, pi = meta[k]
    tag = ' (SWAP)' if pi == swap else ''
    print(f'  weight {norms[k]:.4f}  machine spectrum e={e}  pi={pi}{tag}')
    out.append({'weight': float(norms[k]), 'e': list(e), 'pi': pi})
json.dump(out, open(f'active_d{d}_b{beta}.json', 'w'))
