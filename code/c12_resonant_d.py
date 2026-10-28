"""Effective collision metric along the Gibbs curve of an equally spaced d-level system (d_M = d machines).

On the Gibbs curve, a permutation is resonant iff it conserves H_S + H_M for some machine spectrum e (machine thermal
state q = Gibbs(beta, e)).  We enumerate integer machine spectra e (e_0 = 0, nondecreasing, max <= kmax) and the
energy-conserving permutations (products of permutations inside each energy shell; sampled if too many).
Each permutation k gives: system move delta = M_k dq, cost dq^T C_k dq (second order).  Alternating mechanisms over many
steps, the asymptotic cost per unit "length" along w is the gauge of the convex hull of the unit sets:
    g(w) = min sum_k sqrt(dq_k^T C_k dq_k)   s.t.  sum_k M_k dq_k = w        (an SOCP)
Swaps alone give g_F(w) = sqrt(1/2 sum w^2/p).  ratio = g(w)/g_F(w)  (cost ratio = ratio^2).
usage: c12_resonant_d.py d beta kmax max_perms_per_e
"""
import sys, itertools, math
import numpy as np
import cvxpy as cp

def mech_forms(p, q, pi, d):
    B = np.zeros((d, d - 1)); B[:d - 1] = np.eye(d - 1); B[d - 1] = -1
    rstar = np.outer(p, q).ravel()
    Mdr = np.zeros((d * d, d - 1))
    for j in range(d - 1):
        pd = np.outer(p, B[:, j]).ravel()
        drp = np.empty(d * d); drp[pi] = pd
        Mdr[:, j] = drp - pd
    Md = Mdr.reshape(d, d, d - 1).sum(1)
    C = 0.5 * (Mdr.T @ (Mdr / rstar[:, None]) - Md.T @ (Md / p[:, None]))
    return Md, (C + C.T) / 2

def shells_perms(d, e, max_perms, rng):
    tot = {}
    for x in range(d):
        for y in range(d):
            tot.setdefault(x + e[y], []).append(x * d + y)
    groups = [g for g in tot.values() if len(g) > 1]
    count = math.prod(math.factorial(len(g)) for g in groups) if groups else 1
    perms = []
    if count <= max_perms:
        for combo in itertools.product(*[itertools.permutations(g) for g in groups]):
            pi = np.arange(d * d)
            for g, img in zip(groups, combo):
                pi[list(g)] = list(img)
            perms.append(pi)
    else:
        seen = set()
        for _ in range(max_perms):
            pi = np.arange(d * d)
            for g in groups:
                pi[list(g)] = rng.permutation(g)
            t = tuple(pi)
            if t not in seen:
                seen.add(t); perms.append(pi)
    return perms

def gauge_ratio(d, beta, kmax, max_perms, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    E = np.arange(d, dtype=float)
    p = np.exp(-beta * E); p /= p.sum()
    w = -p * (E - p @ E)                     # Gibbs tangent dp/dbeta (cooling direction)
    forms = []
    for e in itertools.combinations_with_replacement(range(kmax + 1), d - 1):
        e = (0,) + e
        q = np.exp(-beta * np.array(e, float)); q /= q.sum()
        for pi in shells_perms(d, e, max_perms, rng):
            if np.all(pi == np.arange(d * d)):
                continue
            M, C = mech_forms(p, q, pi, d)
            if np.linalg.norm(M) < 1e-14:
                continue
            ev = np.linalg.eigvalsh(C)
            if ev.min() < -1e-9 * max(1, abs(ev).max()):
                continue                          # numerical junk
            forms.append((M, C))
    # SOCP in Fisher-normalised coordinates: dq_k = T_k u_k with cost |u_k|^2, constraint on D*delta, D = diag(1/sqrt p)
    Dm = 1.0 / np.sqrt(p)
    gF = np.sqrt(0.5 * np.sum(w ** 2 / p))
    target = Dm * w / gF                       # unit Fisher-gauge target
    blocks = []
    for M, C in forms:
        lam, V = np.linalg.eigh(C)
        keep = lam > 1e-12 * max(lam.max(), 1e-300)
        if not np.any(keep):
            continue
        T = V[:, keep] / np.sqrt(lam[keep])
        # moves along dropped (zero-cost) directions must be negligible, else skip (should not happen)
        if np.any(~keep):
            leak = np.linalg.norm((Dm[:, None] * M) @ V[:, ~keep]) / max(np.linalg.norm((Dm[:, None] * M) @ T), 1e-300)
            if leak > 1e-6:
                continue
        blocks.append((Dm[:, None] * M) @ T)
    K = len(blocks)
    us = [cp.Variable(Bk.shape[1]) for Bk in blocks]
    cons = [sum(Bk @ u for Bk, u in zip(blocks, us)) == target]
    prob = cp.Problem(cp.Minimize(sum(cp.norm(u, 2) for u in us)), cons)
    prob.solve(solver=cp.CLARABEL)
    ratio = prob.value                          # gauge relative to the swap (Fisher) gauge
    if verbose:
        print(f'd={d} beta={beta}: mechanisms={K}  gauge ratio={ratio:.6f}  cost ratio={ratio**2:.6f}  status={prob.status}', flush=True)
    return ratio, K

if __name__ == '__main__':
    d, kmax, maxp = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4])
    for b in [float(x) for x in sys.argv[2].split(',')]:
        gauge_ratio(d, b, kmax, maxp)
