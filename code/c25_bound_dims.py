"""Universal-bound test for other dimension pairs: lambda_max = max_Theta ||delta||_F^2/2Q <= d_M - 1 ?
Exhaustive over all permutations of the d*d_M configurations (feasible for d*d_M <= 9), at several system states,
resonance families sampled.  Ill-conditioned draws (min weight < 1e-9) are skipped.
usage: c25_bound_dims.py d dM [nrandom] [seed]
"""
import itertools, sys
import numpy as np

d, dm = int(sys.argv[1]), int(sys.argv[2])
nr = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 3)
n = d * dm
cx, cy = np.arange(n) // dm, np.arange(n) % dm
B = np.zeros((dm, dm - 1)); B[:dm - 1] = np.eye(dm - 1); B[dm - 1] = -1
PERMS = np.array(list(itertools.permutations(range(n))), dtype=np.int64)

def lam_max(pi, p, q):
    r = p[cx] * q[cy]
    if r.min() < 1e-9: return None
    D = p[cx][:, None] * B[cy]
    X = np.empty_like(D); X[pi] = D; X = X - D
    M = np.zeros((d, dm - 1))
    for c in range(n): M[cx[c]] += X[c]
    A = X.T @ (X / r[:, None]) - M.T @ (M / p[:, None]); A = 0.5 * (A + A.T)
    F = M.T @ (M / p[:, None]); F = 0.5 * (F + F.T)
    wa, Va = np.linalg.eigh(A)
    keep = wa > 1e-10 * max(1.0, np.abs(wa).max())
    if (~keep).any() and np.linalg.norm(M @ Va[:, ~keep]) > 1e-7 * max(1.0, np.abs(M).max()):
        return np.inf
    if not keep.any(): return 0.0
    W = Va[:, keep] / np.sqrt(wa[keep])
    return float(np.linalg.eigvalsh(W.T @ F @ W).max())

def worst_at(p):
    lp = np.log(p); best = 0.0; nf = 0
    for pi in PERMS:
        moved = np.where(pi != np.arange(n))[0]
        if len(moved) == 0: continue
        x, y, x2, y2 = cx[moved], cy[moved], cx[pi[moved]], cy[pi[moved]]
        if np.any((y == y2) & (x != x2) & (np.abs(lp[x] - lp[x2]) > 1e-12)): continue
        A = np.zeros((len(moved), dm)); A[np.arange(len(moved)), y] += 1; A[np.arange(len(moved)), y2] -= 1
        b = lp[x2] - lp[x]
        u0, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.max(np.abs(A @ u0 - b)) > 1e-9: continue
        _, s, vt = np.linalg.svd(A); ker = vt[int(np.sum(s > 1e-9)):]
        nf += 1
        ts = [np.zeros(len(ker))] + [rng.normal(0, 3, len(ker)) for _ in range(12 if len(ker) > 1 else 0)]
        for t in ts:
            lq = u0 + (t @ ker if len(ker) else 0.0)
            q = np.exp(lq - lq.max()); q /= q.sum()
            v = lam_max(pi, p, q)
            if v is not None and v > best: best = v
    return best, nf

E = np.arange(d, dtype=float)
tests = [('uniform', np.ones(d) / d)]
for b in [0.3, 1.0, 3.0, 6.0]:
    v = np.exp(-b * E); tests.append((f'gibbs b={b}', v / v.sum()))
for k in range(nr):
    tests.append(('random', rng.dirichlet(np.ones(d))))
ov = 0.0
for name, p in tests:
    v, nf = worst_at(p); ov = max(ov, v)
    print(f'd={d} dM={dm} {name:12s} p={np.round(p, 4)}  resonant={nf:6d}  max lambda={v:.6f}  (d_M-1={dm - 1})', flush=True)
print(f'OVERALL d={d} dM={dm}: max lambda = {ov:.6f}')
