"""Exact hull gauge at generic points on the Fisher-Rao geodesic between qutrit Gibbs states (main text: 'along the interior
of the Fisher-Rao geodesic the hull equals the Fisher ellipse in the direction of motion' for beta*omega in [1, 3]).
Resonant permutations at a generic p: solve the resonance equations for every permutation of the 9 configurations;
families (positive-dimensional resonance sets) are sampled densely.  Primal SOCP upper bound in the stable variance form.
usage: c29_geodesic_check.py beta0 beta1 s1 [s2 ...]     (s in (0,1): position along the geodesic)
"""
import itertools, sys
import numpy as np
import cvxpy as cp
import importlib.util
spec = importlib.util.spec_from_file_location('c22b', 'c22b_hull_bounds_stable.py'); c22b = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ['c22b']; spec.loader.exec_module(c22b); sys.argv = _argv
d = dm = 3; n = 9
cx, cy = np.arange(n) // dm, np.arange(n) % dm
PERMS = np.array(list(itertools.permutations(range(n))), dtype=np.int64)

def resonant(p):
    lp = np.log(p); out = []
    for pi in PERMS:
        moved = np.where(pi != np.arange(n))[0]
        if len(moved) == 0: continue
        x, y, x2, y2 = cx[moved], cy[moved], cx[pi[moved]], cy[pi[moved]]
        if np.any((y == y2) & (x != x2)): continue
        A = np.zeros((len(moved), dm)); A[np.arange(len(moved)), y] += 1; A[np.arange(len(moved)), y2] -= 1
        b = lp[x2] - lp[x]
        u0, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.max(np.abs(A @ u0 - b)) > 1e-9: continue
        _, s, vt = np.linalg.svd(A); ker = vt[int(np.sum(s > 1e-9)):]
        if len(ker):
            one = np.ones(dm) / np.sqrt(dm); ker = ker - np.outer(ker @ one, one)
            _, s2, vt2 = np.linalg.svd(ker); ker = vt2[:int(np.sum(s2 > 1e-9))]
        out.append((pi, u0, ker))
    return out

E = np.arange(d, dtype=float)
b0, b1 = float(sys.argv[1]), float(sys.argv[2])
g0 = np.exp(-b0 * E); g0 /= g0.sum(); g1 = np.exp(-b1 * E); g1 /= g1.sum()
a, bb = np.sqrt(g0), np.sqrt(g1); th = np.arccos(np.clip(a @ bb, -1, 1))
for s in [float(v) for v in sys.argv[3:]]:
    x = (np.sin((1 - s) * th) * a + np.sin(s * th) * bb) / np.sin(th)
    dx = (-th * np.cos((1 - s) * th) * a + th * np.cos(s * th) * bb) / np.sin(th)
    p = x ** 2; w = 2 * x * dx
    fam = resonant(p)
    wn = w / np.sqrt(p); wn /= np.linalg.norm(wn)
    Pz = np.eye(d) - np.outer(np.sqrt(p), np.sqrt(p)); Uz, _, _ = np.linalg.svd(Pz); Tz = Uz[:, :d - 1]
    cols, sizes = [Tz], [d - 1]
    for pi, u0, K in fam:
        ts = [np.zeros(0)] if len(K) == 0 else [np.array([t]) for t in np.arange(-14, 14.001, 0.25)] if len(K) == 1 else \
             [np.array([t1, t2]) for t1 in np.arange(-12, 12.01, 1.0) for t2 in np.arange(-12, 12.01, 1.0)]
        for t in ts:
            lq = u0 + (t @ K if len(K) else 0.0); q = np.exp(lq - lq.max()); q /= q.sum()
            if (p[:, None] * q[None, :]).min() < 1e-12: continue
            A, Mn = c22b.forms(pi, p, q)
            if np.linalg.norm(Mn) < 1e-14: continue
            wv, V = np.linalg.eigh(A); keep = wv > 1e-12 * max(np.abs(wv).max(), 1e-300)
            if not keep.any(): continue
            cols.append(Mn @ (V[:, keep] / np.sqrt(wv[keep]))); sizes.append(int(keep.sum()))
    Mall = np.hstack(cols); v = cp.Variable(Mall.shape[1]); idx = np.cumsum([0] + sizes)
    obj = cp.sum(cp.hstack([cp.norm(v[idx[i]:idx[i + 1]], 2) for i in range(len(sizes))]))
    con = [Mall @ v == wn]; cp.Problem(cp.Minimize(obj), con).solve(solver=cp.CLARABEL)
    print(f'beta {b0}->{b1}, s={s:.2f}, p={np.round(p, 4)}: resonant={len(fam)}, hull/Fisher ratio along the geodesic <= {obj.value:.6f}',
          flush=True)
