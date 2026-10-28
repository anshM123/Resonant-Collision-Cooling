"""Infinite-temperature local collision metric for a d-level system with d-level machines.

At p = q = uniform every configuration weighs 1/d^2, so every permutation pi of [d]x[d] is resonant.  Perturbing the
machine by dq (sum 0):  Delta r(c') = (1/d) (dq[y(pi^{-1} c')] - dq[y(c')]),  dp_x = sum_y Delta r(x, y),
dissipation  Sigma = 1/2 [ d^2 sum_c Delta r_c^2 - d sum_x dp_x^2 ]   (second order, exact identity expanded).
Swap cost for the same move: 1/2 * d * sum_x dp_x^2.  Ratio R = Sigma / swap cost (swap: R = 1).
For a target direction g (the Gibbs tangent ~ linear profile for equally spaced levels) minimise R over pi
(simulated annealing over permutations) and dq (closed form: least-norm under the linear constraint dp = g).
"""
import sys
import numpy as np

def cost_for_pi(pi_inv, d, g):
    """min over dq of Sigma/(swap cost) subject to dp = t*g for some t>0 (scale-free)"""
    # linear maps dq -> Delta r (d^2) and dq -> dp (d); dq lives in sum-zero space (basis B)
    B = np.zeros((d, d - 1)); B[:d - 1] = np.eye(d - 1); B[d - 1] = -1
    src = np.array([pi_inv[c] % d for c in range(d * d)])      # y of the preimage
    own = np.arange(d * d) % d
    Mr = (B[src] - B[own]) / d                                  # (d^2, d-1)
    Mp = Mr.reshape(d, d, d - 1).sum(1)                         # (d, d-1)
    # minimise d^2 |Mr a|^2 - d |Mp a|^2  subject to Mp a = g  (then divide by (d/2)|g|^2 * ... ) ; Sigma has the 1/2
    # feasibility
    a0, res, rk, _ = np.linalg.lstsq(Mp, g, rcond=None)
    if np.linalg.norm(Mp @ a0 - g) > 1e-9 * np.linalg.norm(g):
        return np.inf
    # null space of Mp
    u, s, vt = np.linalg.svd(Mp)
    rank = int(np.sum(s > 1e-12))
    Nn = vt[rank:].T                                            # (d-1, k)
    Q = d * d * Mr.T @ Mr                                       # quadratic in a (the -d|Mp a|^2 term is constant = -d|g|^2)
    if Nn.shape[1] > 0:
        # minimise (a0 + N z)^T Q (a0 + N z); drop numerically-null directions with an ABSOLUTE cutoff
        M = Nn.T @ Q @ Nn
        w, V = np.linalg.eigh((M + M.T) / 2)
        keep = w > 1e-10 * max(1.0, np.abs(Q).max())
        rhs = V.T @ (Nn.T @ Q @ a0)
        zc = np.zeros_like(w); zc[keep] = -rhs[keep] / w[keep]
        a = a0 + Nn @ (V @ zc)
    else:
        a = a0
    sigma = 0.5 * (a @ Q @ a - d * g @ g)
    swap = 0.5 * d * g @ g
    return sigma / swap

def anneal(d, g, iters, seed):
    rng = np.random.default_rng(seed)
    n = d * d
    pi = rng.permutation(n)
    inv = np.argsort(pi)
    cur = cost_for_pi(inv, d, g)
    best = (cur, pi.copy())
    T = 1.0
    for it in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        pi2 = pi.copy(); pi2[i], pi2[j] = pi2[j], pi2[i]
        inv2 = np.argsort(pi2)
        c2 = cost_for_pi(inv2, d, g)
        if c2 < cur or (np.isfinite(c2) and rng.random() < np.exp(-(c2 - cur) / max(T, 1e-9))):
            pi, cur = pi2, c2
            if cur < best[0]:
                best = (cur, pi.copy())
        T *= 0.9995
    return best

if __name__ == '__main__':
    d = int(sys.argv[1]); iters = int(sys.argv[2]); nseeds = int(sys.argv[3])
    E = np.arange(d, dtype=float)
    g = (E.mean() - E) / d               # Gibbs tangent at beta=0 for equally spaced levels (cooling direction)
    swap_pi = np.array([(c % d) * d + c // d for c in range(d * d)])
    print(f'd={d}: swap check R = {cost_for_pi(np.argsort(swap_pi), d, g):.6f}')
    best = (np.inf, None)
    for s in range(nseeds):
        b = anneal(d, g, iters, s)
        if b[0] < best[0]:
            best = b
        print(f'  seed {s}: R = {b[0]:.6f}   best = {best[0]:.6f}', flush=True)
    print(f'RESULT d={d}: min R along Gibbs direction at infinite T = {best[0]:.6f}  (sqrt = {np.sqrt(max(best[0],0)):.6f})')
    print('  best pi =', list(best[1]))
