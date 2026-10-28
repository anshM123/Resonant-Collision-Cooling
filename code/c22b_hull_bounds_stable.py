"""Exact hull gauge along the qutrit Gibbs curve, numerically stable version of c22 (works for tiny populations).

For a permutation resonant at (p, q*), write psi = Theta/q* (q*-centred) and, for each system level x,
Delta_x(y) = psi(sigma_x(y)) - psi(y), with sigma_x(y) the machine level of the configuration flowing into (x, y).
Then (SM, Sec. S11)
    2Q = sum_x p_x Var_q*(Delta_x)            (a sum of variances: no cancellation),
    delta_x = p_x E_q*(Delta_x),   Fisher-normalised move  (delta_x / sqrt(p_x)) = sqrt(p_x) E_q*(Delta_x).
Upper bound: primal SOCP over all rigid mechanisms + one-parameter families sampled with step 0.25.
Lower bound: dual direction u of that SOCP; true support H(u) with families maximised over t in [-30, 30].
usage: c22b_hull_bounds_stable.py beta1 [beta2 ...]
"""
import sys
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar
import importlib.util
spec = importlib.util.spec_from_file_location('c21', 'c21_exact_hull.py'); c21 = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ['c21']; spec.loader.exec_module(c21); sys.argv = _argv
d = dm = 3
n = d * dm

def q_of(beta, u0, K, t):
    lq = beta * u0 + (t @ K if K.shape[0] else 0.0)
    q = np.exp(lq - lq.max()); return q / q.sum()

def forms(pi, p, q):
    """A (2x2, = 2Q in psi-coordinates) and Mn (3x2, Fisher-normalised move) for psi = Bq c"""
    inv = np.argsort(pi)
    sig = np.array([[inv[x * dm + y] % dm for y in range(dm)] for x in range(d)])   # sigma_x(y)
    # basis of q-centred psi: columns e_j - q-weighted mean ... orthonormalise in the q-inner product
    Bq = np.eye(dm)[:, :dm - 1] - np.outer(np.ones(dm), q[:dm - 1])               # psi_j = e_j - q_j 1  (centred)
    A = np.zeros((dm - 1, dm - 1)); Mn = np.zeros((d, dm - 1))
    for x in range(d):
        L = Bq[sig[x]] - Bq                                                        # Delta_x = L c  (dm x 2)
        m = q @ L
        A += p[x] * (L.T @ (q[:, None] * L) - np.outer(m, m))
        Mn[x] = np.sqrt(p[x]) * m
    return 0.5 * (A + A.T), Mn

def h2(A, Mn, un):
    w, V = np.linalg.eigh(A)
    b = V.T @ (Mn.T @ un)
    tot = 0.0
    scaleA = max(np.abs(w).max(), 1e-300)
    for i in range(len(w)):
        mv = np.linalg.norm(Mn @ V[:, i])
        if w[i] > 1e-14 * scaleA:
            tot += b[i] ** 2 / w[i]
        elif mv > 1e-9 * max(np.linalg.norm(Mn), 1e-300):
            return np.inf
    return tot

def bounds(beta, fam, tgrid=np.arange(-14, 14.001, 0.25)):
    E = np.arange(d, dtype=float)
    p = np.exp(-beta * E); p /= p.sum()
    w = -p * (E - p @ E)
    wn = w / np.sqrt(p); wscale = np.linalg.norm(wn); wn = wn / wscale        # Fisher-normalised direction
    cols, sizes = [], []
    # swap: Fisher unit ball itself: move = any vector in the sum-zero plane (normalised coords) with cost |v|
    Pz = np.eye(d) - np.outer(np.sqrt(p), np.sqrt(p))                          # tangent plane in normalised coords
    Uz, sz, _ = np.linalg.svd(Pz); Tz = Uz[:, :d - 1]
    cols.append(Tz); sizes.append(d - 1)
    for pi, u0, K in fam:
        ts = [np.zeros(0)] if K.shape[0] == 0 else [np.array([t]) for t in tgrid]
        for t in ts:
            A, Mn = forms(pi, p, q_of(beta, u0, K, t))
            if np.linalg.norm(Mn) < 1e-14: continue
            wv, V = np.linalg.eigh(A)
            keep = wv > 1e-14 * max(np.abs(wv).max(), 1e-300)
            if not keep.any(): continue
            cols.append(Mn @ (V[:, keep] / np.sqrt(wv[keep]))); sizes.append(int(keep.sum()))
    # drop blocks whose moves are negligible (they cannot shape the hull) to keep the cone program well conditioned
    big = max(np.abs(cblk).max() for cblk in cols)
    keepb = [i for i, cblk in enumerate(cols) if np.abs(cblk).max() > 1e-9 * big]
    cols = [cols[i] for i in keepb]; sizes = [sizes[i] for i in keepb]
    Mall = np.hstack(cols)
    v = cp.Variable(Mall.shape[1]); idx = np.cumsum([0] + sizes)
    obj = cp.sum(cp.hstack([cp.norm(v[idx[i]:idx[i + 1]], 2) for i in range(len(sizes))]))
    con = [Mall @ v == wn]
    prob = cp.Problem(cp.Minimize(obj), con)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        print('  CLARABEL failed; retrying with SCS', flush=True)
        prob.solve(solver=cp.SCS, eps=1e-10, max_iters=200000)
    g_up = obj.value
    un = np.array(con[0].dual_value).ravel()
    un = un - (un @ np.sqrt(p)) * np.sqrt(p)                                    # project onto the tangent plane
    if un @ wn < 0: un = -un
    # exact support at un
    H2 = float(np.sum((Tz.T @ un) ** 2)); arg = 'swap'
    tfine = np.linspace(-30, 30, 2401)
    for pi, u0, K in fam:
        if K.shape[0] == 0:
            vv = h2(*forms(pi, p, q_of(beta, u0, K, np.zeros(0))), un)
            if vv > H2: H2, arg = vv, (tuple(int(c) for c in pi), None)
        else:
            f = lambda t: h2(*forms(pi, p, q_of(beta, u0, K, np.array([t]))), un)
            vals = np.array([f(t) for t in tfine])
            for i in np.where((vals[1:-1] >= vals[:-2]) & (vals[1:-1] >= vals[2:]))[0] + 1:
                res = minimize_scalar(lambda t: -f(t), bounds=(tfine[i - 1], tfine[i + 1]), method='bounded',
                                      options=dict(xatol=1e-10))
                vv = max(-res.fun, vals[i])
                if vv > H2: H2, arg = vv, (tuple(int(c) for c in pi), float(res.x))
            if vals[0] > H2 or vals[-1] > H2:
                print(f'  note: support still increasing at |t|=30 for {list(pi)}', flush=True)
    g_low = (un @ wn) / np.sqrt(H2)
    return g_up, g_low, arg

if __name__ == '__main__':
    fam = c21.resonant_family()
    for b in [float(v) for v in sys.argv[1:]]:
        up, lo, arg = bounds(b, fam)
        print(f'beta={b:7.3f}   exact hull ratio in [{lo:.6f}, {up:.6f}]   (gap {up - lo:.1e}; support attained by {arg})',
              flush=True)
