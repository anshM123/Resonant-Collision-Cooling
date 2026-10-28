"""Adversarial test of the PHYSICAL Theorem 2 conjecture: lambda(pi, p, q*) = max_Theta ||delta||_F^2 / 2Q <= d_M - 1
for every permutation pi of the d*d_M configurations and every (p, q*) at which pi is resonant.
For a fixed pi, resonance (p_x q_y = p_x' q_y' along every moved pair) is LINEAR in (log p, log q): we parametrise its
solution space and maximise lambda over it (Nelder-Mead, several starts).  Permutations are random products of a few
disjoint cycles (sparse permutations have large resonance sets; dense ones only resonate at the maximally mixed state,
where Theorem 2(a) is proved).
usage: c30_adversarial_physical.py d dM n_perms max_cycles seed
"""
import sys
import numpy as np
from scipy.optimize import minimize
d, dm, nper, maxc, seed = (int(v) for v in sys.argv[1:6])
rng = np.random.default_rng(seed)
n = d * dm
cx, cy = np.arange(n) // dm, np.arange(n) % dm

def lam(pi, p, q):
    inv = np.argsort(pi)
    sig = np.array([[inv[x * dm + y] % dm for y in range(dm)] for x in range(d)])
    Bq = np.eye(dm)[:, :dm - 1] - np.outer(np.ones(dm), q[:dm - 1])
    A = np.zeros((dm - 1, dm - 1)); F = np.zeros((dm - 1, dm - 1))
    for x in range(d):
        L = Bq[sig[x]] - Bq; m = q @ L
        A += p[x] * (L.T @ (q[:, None] * L) - np.outer(m, m)); F += p[x] * np.outer(m, m)
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    keep = w > 1e-12 * max(np.abs(w).max(), 1e-300)
    if not keep.any(): return 0.0
    W = V[:, keep] / np.sqrt(w[keep])
    return float(np.linalg.eigvalsh(W.T @ F @ W).max())

def random_sparse_perm():
    k = rng.integers(1, maxc + 1)
    perm = np.arange(n); free = list(rng.permutation(n))
    for _ in range(k):
        L = int(rng.integers(2, 5))
        if len(free) < L: break
        cyc = [free.pop() for _ in range(L)]
        for i in range(L): perm[cyc[i]] = cyc[(i + 1) % L]
    return perm

best = (0.0, None); tested = 0
for it in range(nper):
    pi = random_sparse_perm()
    moved = np.where(pi != np.arange(n))[0]
    if np.all(cx[pi[moved]] == cx[moved]): continue             # does not move the system
    rows = []
    for c in moved:
        r = np.zeros(d + dm); r[cx[c]] += 1; r[d + cy[c]] += 1; r[cx[pi[c]]] -= 1; r[d + cy[pi[c]]] -= 1
        rows.append(r)
    M = np.array(rows)
    _, s, vt = np.linalg.svd(M); N = vt[int(np.sum(s > 1e-9)):].T  # null space basis (d+dm) x k
    # remove the two normalisation directions (constant shifts of log p and of log q)
    e1 = np.concatenate([np.ones(d), np.zeros(dm)]); e2 = np.concatenate([np.zeros(d), np.ones(dm)])
    E = np.stack([e1, e2], 1); Pp = np.eye(d + dm) - E @ np.linalg.pinv(E)
    Nr = Pp @ N; u, sv, _ = np.linalg.svd(Nr, full_matrices=False); Nr = u[:, sv > 1e-9]
    if Nr.shape[1] == 0: continue                                # resonant only at the maximally mixed state
    tested += 1
    def neg(z):
        v = Nr @ z; a, b = v[:d], v[d:]
        p = np.exp(a - a.max()); p /= p.sum(); q = np.exp(b - b.max()); q /= q.sum()
        if (p[:, None] * q[None, :]).min() < 1e-9: return 0.0
        return -lam(pi, p, q)
    for start in range(3):
        z0 = rng.normal(0, 2.0, Nr.shape[1])
        res = minimize(neg, z0, method='Nelder-Mead', options=dict(maxiter=400 * Nr.shape[1], xatol=1e-8, fatol=1e-12))
        if -res.fun > best[0]:
            best = (-res.fun, (list(pi), res.x))
    if it % max(1, nper // 10) == 0:
        print(f'  {it}/{nper}: tested {tested}, running max lambda = {best[0]:.6f}', flush=True)
print(f'd={d} dM={dm}: {tested} sparse resonant permutations optimised over their resonance sets; '
      f'max lambda = {best[0]:.8f} (bound {dm - 1})')
if best[1] is not None: print('  argmax pi =', best[1][0])
