"""Two-sided bounds on the exact hull gauge along the qutrit Gibbs curve (optimal local cost per unit length, Thm 1).

Upper bound: primal SOCP over a finite set of mechanisms (all rigid resonant permutations + one-parameter families
sampled densely), g <= min sum_k ||L_k theta_k|| s.t. sum_k M_k theta_k = w   (an attainable decomposition).
Lower bound: for the dual direction u* of that SOCP, g >= (u*.w) / H(u*), where H(u*) is the TRUE hull support:
max over all mechanisms, with the one-parameter families maximised over their whole parameter line
(dense grid on [-30, 30] + bounded scalar refinement around every local maximum).
usage: c22_hull_bounds.py beta1 [beta2 ...]
"""
import sys
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar
import importlib.util
spec = importlib.util.spec_from_file_location('c21', 'c21_exact_hull.py'); c21 = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ['c21']; spec.loader.exec_module(c21); sys.argv = _argv

d = 3

def q_of(beta, u0, K, t):
    lq = beta * u0 + (t @ K if K.shape[0] else 0.0)
    q = np.exp(lq - lq.max()); return q / q.sum()

def mech(pi, p, q):
    Mdr, Md = c21.mech_matrices(pi, p)
    r = p[c21.cx] * q[c21.cy]
    C2 = Mdr.T @ (Mdr / r[:, None]) - Md.T @ (Md / p[:, None])
    return Md, 0.5 * (C2 + C2.T)

def bounds(beta, fam, tgrid=np.arange(-14, 14.001, 0.25)):
    E = np.arange(d, dtype=float)
    p = np.exp(-beta * E); p /= p.sum()
    w = -p * (E - p @ E)
    gF = np.sqrt(np.sum(w ** 2 / p))
    mechs = []
    # swap (Fisher): move M theta with theta in the sum-zero plane directly: use M = B, 2C = B^T diag(1/p) B
    Bm = c21.B
    mechs.append((Bm, Bm.T @ (Bm / p[:, None])))
    for pi, u0, K in fam:
        ts = [np.zeros(0)] if K.shape[0] == 0 else [np.array([t]) for t in tgrid]
        for t in ts:
            Md, C2 = mech(pi, p, q_of(beta, u0, K, t))
            if np.linalg.norm(Md) < 1e-14: continue
            mechs.append((Md, C2))
    # primal SOCP in Fisher-normalised, whitened coordinates (well scaled even for tiny populations):
    # theta_k = S_k v_k with S_k = C2^{-1/2} on its range, so the cost is ||v_k||; constraint Dm sum_k M_k S_k v_k = Dm w
    Dm = 1 / np.sqrt(p)
    wn = Dm * w; scale = np.linalg.norm(wn); wn = wn / scale
    cols, sizes = [], []
    for Md, C2 in mechs:
        wv, V = np.linalg.eigh(C2)
        keep = wv > 1e-13 * max(1.0, np.abs(wv).max())
        S = V[:, keep] / np.sqrt(wv[keep])
        cols.append((Dm[:, None] * Md) @ S); sizes.append(int(keep.sum()))
    Mall = np.hstack(cols)
    v = cp.Variable(Mall.shape[1])
    idx = np.cumsum([0] + sizes)
    obj = cp.sum(cp.hstack([cp.norm(v[idx[i]:idx[i + 1]], 2) for i in range(len(sizes)) if sizes[i] > 0]))
    con = [Mall @ v == wn]
    prob = cp.Problem(cp.Minimize(obj), con)
    prob.solve(solver=cp.CLARABEL)
    g_up = prob.value * scale
    u = Dm * np.array(con[0].dual_value).ravel()
    u = -u if (u @ w) < 0 else u
    # true hull support at u: rigid mechanisms exactly; one-parameter families maximised over t in R
    def h2_of(Md, C2):
        wv, V = np.linalg.eigh(C2)
        b = V.T @ (Md.T @ u)
        tot = 0.0
        for i in range(2):
            if wv[i] > 1e-13 * max(1.0, np.abs(wv).max()):
                tot += b[i] ** 2 / wv[i]
            elif abs(b[i]) > 1e-10:
                return np.inf
        return tot
    H2 = np.var(u, ddof=0) if False else (u ** 2 @ p - (u @ p) ** 2)
    arg = 'swap'
    tfine = np.linspace(-30, 30, 2401)
    for pi, u0, K in fam:
        if K.shape[0] == 0:
            Md, C2 = mech(pi, p, q_of(beta, u0, K, np.zeros(0)))
            v = h2_of(Md, C2)
            if v > H2: H2, arg = v, (tuple(pi), None)
        else:
            f = lambda t: h2_of(*mech(pi, p, q_of(beta, u0, K, np.array([t]))))
            vals = np.array([f(t) for t in tfine])
            for i in np.where((vals[1:-1] >= vals[:-2]) & (vals[1:-1] >= vals[2:]))[0] + 1:
                res = minimize_scalar(lambda t: -f(t), bounds=(tfine[i - 1], tfine[i + 1]), method='bounded',
                                      options=dict(xatol=1e-10))
                v = max(-res.fun, vals[i])
                if v > H2: H2, arg = v, (tuple(pi), res.x)
            if vals[0] > H2 or vals[-1] > H2:
                print(f'  note: support still increasing at |t|=30 for {list(pi)}', flush=True)
    g_low = (u @ w) / np.sqrt(H2)
    return g_up / gF, g_low / gF, arg

if __name__ == '__main__':
    fam = c21.resonant_family()
    for b in [float(v) for v in sys.argv[1:]]:
        up, lo, arg = bounds(b, fam)
        print(f'beta={b:7.3f}   exact hull ratio in [{lo:.6f}, {up:.6f}]   (gap {up - lo:.1e}; support attained by {arg})',
              flush=True)
