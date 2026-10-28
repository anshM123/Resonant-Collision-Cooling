"""Exact convex-hull gauge along the Gibbs curve of the equally spaced qutrit (d = d_M = 3).

By Theorem 1 the hull of the unit sets of ALL exactly resonant permutations (every resonant machine state q*) gives the
optimal leading-order cost per unit length over all collisions.  Earlier scans (c4/c10, Fig. 1a) sampled six machine
states on resonance sets of positive dimension, which is an inner approximation of the hull (upper bound on the gauge).
Here the support function of every mechanism is maximised over its whole resonance set:
    unit set of (pi, q*):  {delta = M theta : theta^T (2C) theta <= 1},   support h(u)^2 = u^T M (2C)^+ M^T u,
    hull support H(u) = max over pi, q* (and swaps),  gauge g(w) = max_u (u.w) / H(u).
On the Gibbs curve log p_x = -beta x - log Z, so the resonant set of permutations does not depend on beta > 0, and
log q* = beta * u0 + kernel.
usage: c21_exact_hull.py beta1 [beta2 ...]
"""
import itertools, sys
import numpy as np
from scipy.optimize import minimize

d = 3
n = d * d
B = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
cx, cy = np.arange(n) // d, np.arange(n) % d

def resonant_family():
    """permutations resonant on the Gibbs curve (for beta>0) that move the system, with log q* = beta*u0 + span(ker)"""
    out = []
    for pi in itertools.permutations(range(n)):
        pi = np.array(pi)
        rows, rhs, ok = [], [], True
        for c in range(n):
            c2 = pi[c]
            if c2 == c: continue
            x, y, x2, y2 = cx[c], cy[c], cx[c2], cy[c2]
            if y == y2:                       # needs p_x = p_x2: impossible on the Gibbs curve
                ok = False; break
            row = np.zeros(d); row[y] += 1; row[y2] -= 1
            rows.append(row); rhs.append(-(x2 - x))     # log p_x2 - log p_x = -beta (x2 - x)   (per unit beta)
        if not ok or not rows: continue
        A = np.array(rows); b = np.array(rhs, float)
        u0, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.max(np.abs(A @ u0 - b)) > 1e-9: continue
        _, s, vt = np.linalg.svd(A); ker = vt[int(np.sum(s > 1e-9)):]
        # drop the normalisation direction (all ones) from the kernel
        ker = [k for k in ker]
        if ker:
            K = np.array(ker); one = np.ones(d) / np.sqrt(d)
            K = K - np.outer(K @ one, one)
            u_, s_, vt_ = np.linalg.svd(K); K = vt_[:int(np.sum(s_ > 1e-9))]
        else:
            K = np.zeros((0, d))
        # system move map (independent of q*): delta r = P(p x Theta) - p x Theta, Theta = B v
        out.append((pi, u0, K))
    return out

def mech_matrices(pi, p):
    Mdr = np.zeros((n, 2))
    for j in range(2):
        pd = (p[cx] * B[cy, j])
        drp = np.empty(n); drp[pi] = pd
        Mdr[:, j] = drp - pd
    Md = np.zeros((d, 2))
    for c in range(n): Md[cx[c]] += Mdr[c]
    return Mdr, Md

def G_of(Mdr, Md, p, q):
    r = p[cx] * q[cy]
    C2 = Mdr.T @ (Mdr / r[:, None]) - Md.T @ (Md / p[:, None])        # = 2C
    C2 = 0.5 * (C2 + C2.T)
    w, V = np.linalg.eigh(C2)
    inv = np.zeros_like(C2)
    for i in range(2):
        if w[i] > 1e-13 * max(1.0, abs(w).max()):
            inv += np.outer(V[:, i], V[:, i]) / w[i]
        else:
            # null direction: infinite support if it moves the system
            if np.linalg.norm(Md @ V[:, i]) > 1e-9:
                return None
    return Md @ inv @ Md.T

def gauge_ratio(beta, fam, ngrid=25, nphi=360, refine=True):
    E = np.arange(d, dtype=float)
    p = np.exp(-beta * E); p /= p.sum()
    w = -p * (E - p @ E)
    e1 = np.array([1, -1, 0]) / np.sqrt(2); e2 = np.array([1, 1, -2]) / np.sqrt(6)
    phis = np.linspace(0, np.pi, nphi, endpoint=False)          # u and -u give the same gauge contributions
    U = np.stack([np.cos(phis)[:, None] * e1 + np.sin(phis)[:, None] * e2])[0]   # (nphi, 3)
    # swap (Fisher) support: var_p(u)
    HF2 = (U ** 2) @ p - (U @ p) ** 2
    H2 = HF2.copy()
    ts1 = np.linspace(-14, 14, 4 * ngrid + 1)
    ts2 = np.linspace(-14, 14, ngrid)
    best_src = [None] * nphi
    cache = []
    for idx, (pi, u0, K) in enumerate(fam):
        Mdr, Md = mech_matrices(pi, p)
        k = K.shape[0]
        grid = [np.zeros(0)] if k == 0 else ([np.array([t]) for t in ts1] if k == 1 else
                                              [np.array([a, b]) for a in ts2 for b in ts2])
        for t in grid:
            lq = beta * u0 + (t @ K if k else 0.0)
            q = np.exp(lq - lq.max()); q /= q.sum()
            G = G_of(Mdr, Md, p, q)
            if G is None:
                print('INFINITE support (free move) for', list(pi), 't', t); continue
            h2 = np.einsum('ij,jk,ik->i', U, G, U)
            upd = h2 > H2
            if upd.any():
                H2[upd] = h2[upd]
                for i in np.where(upd)[0]: best_src[i] = (idx, t.copy())
        cache.append((Mdr, Md, k))
    def gauge(H2v):
        vals = (U @ w) / np.sqrt(H2v)
        return np.max(np.abs(vals)), int(np.argmax(np.abs(vals)))
    g, i = gauge(H2)
    if refine:
        # refine the active mechanisms near the optimal dual direction by continuous optimisation over t
        for sweep in range(3):
            g, i = gauge(H2)
            win = [(i + j) % nphi for j in range(-6, 7)]
            cands = set(best_src[j][0] for j in win if best_src[j] is not None)
            for idx in cands:
                pi, u0, K = fam[idx]; Mdr, Md, k = cache[idx]
                if k == 0: continue
                for j in win:
                    u = U[j]
                    def neg(t):
                        lq = beta * u0 + t @ K
                        q = np.exp(lq - lq.max()); q /= q.sum()
                        G = G_of(Mdr, Md, p, q)
                        return -1e300 if G is None else -(u @ G @ u)
                    t0 = best_src[j][1] if (best_src[j] is not None and best_src[j][0] == idx) else np.zeros(k)
                    res = minimize(neg, t0, method='Nelder-Mead', options=dict(xatol=1e-8, fatol=1e-14, maxiter=2000))
                    if -res.fun > H2[j]:
                        H2[j] = -res.fun; best_src[j] = (idx, res.x)
        g, i = gauge(H2)
    gF = np.sqrt(np.sum(w ** 2 / p))
    src = best_src[i]
    return g / gF, (list(fam[src[0]][0]) if src else 'swap'), (src[1] if src else None)

if __name__ == '__main__':
    fam = resonant_family()
    ks = [K.shape[0] for _, _, K in fam]
    print(f'resonant moving permutations on the Gibbs curve: {len(fam)}  (free machine parameters: '
          f'{ {k: ks.count(k) for k in sorted(set(ks))} })', flush=True)
    for b in [float(v) for v in sys.argv[1:]]:
        r, pi, t = gauge_ratio(b, fam)
        print(f'beta={b:7.3f}  exact hull ratio = {r:.6f}   (active: {pi}, t={t})', flush=True)
