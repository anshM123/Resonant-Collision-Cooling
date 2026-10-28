"""Classical small-step analysis of collision-model cooling, d_S = d_M = d.

For a permutation pi of the configurations (x, y) (system level x, machine level y) we need a resonant machine
distribution q* with p_x q*_y = p_x' q*_y' whenever pi(x, y) = (x', y'); then pi does nothing at q*.
Perturbing q = q* + dq moves the system by delta (linear in dq) and dissipates
    Sigma = D(r'||r) - D(p'||p) ~ 1/2 [ sum_c dr_c^2 / r*_c - sum_x delta_x^2 / p_x ].
The swap protocol costs 1/2 sum_x delta_x^2/p_x (Fisher).  R = Sigma / (1/2 Fisher(delta)) ; swap gives R = 1.
min R < 1 for some pi would mean swaps are not locally optimal.
"""
import sys, itertools
import numpy as np

def analyse(p, d, pi):
    """pi: array of length d*d, configuration index c = x*d + y maps to pi[c]. Returns min R or None if infeasible."""
    rows, rhs = [], []
    for c in range(d * d):
        c2 = pi[c]
        if c2 == c:
            continue
        x, y = divmod(c, d); x2, y2 = divmod(c2, d)
        if y == y2:
            if x != x2:
                return None
            continue
        row = np.zeros(d); row[y] += 1; row[y2] -= 1     # u_y - u_y2 = log p_x2 - log p_x
        rows.append(row); rhs.append(np.log(p[x2]) - np.log(p[x]))
    if not rows:
        return None
    A = np.array(rows); b = np.array(rhs)
    u, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.max(np.abs(A @ u - b)) > 1e-9:
        return None
    # family of resonant q*: u + kernel(A) (mod constants); use the min-norm solution and also sample the kernel
    _, s, vt = np.linalg.svd(A)
    rank = int(np.sum(s > 1e-9))
    ker = vt[rank:]
    best = np.inf
    for trial in range(1 if len(ker) <= 1 else 25):
        uu = u.copy()
        if trial > 0:
            uu = uu + np.random.default_rng(trial).normal(0, 1.5, len(ker)) @ ker
        q = np.exp(uu); q /= q.sum()
        rstar = np.outer(p, q).ravel()
        # linear maps from dq (tangent: sum dq = 0) to dr and delta
        B = np.zeros((d, d - 1)); B[:d - 1, :] = np.eye(d - 1); B[d - 1, :] = -1
        M_dr = np.zeros((d * d, d - 1))
        for k in range(d - 1):
            dq = B[:, k]
            pd = np.outer(p, dq).ravel()
            dr = np.empty(d * d)
            # r' = r o pi^{-1}: r'(pi(c)) = r(c)  ->  dr'(pi(c)) = pd(c)
            drp = np.empty(d * d); drp[pi] = pd
            M_dr[:, k] = drp - pd
        M_delta = M_dr.reshape(d, d, d - 1).sum(axis=1)
        Q1 = M_dr.T @ (M_dr / rstar[:, None])            # sum dr^2 / r*
        F = M_delta.T @ (M_delta / p[:, None])           # Fisher of delta
        if np.linalg.matrix_rank(F, tol=1e-12) == 0:
            continue
        # min over dq of (Q1 - F)/F  restricted to range of F
        w, V = np.linalg.eigh(F)
        keep = w > 1e-12 * max(w.max(), 1)
        T = V[:, keep] / np.sqrt(w[keep])
        Qr = T.T @ (Q1 - F) @ T
        # Q1 may be nonzero on ker F (moves of the machine without moving S); restrict to directions with delta != 0
        ev = np.linalg.eigvalsh((Qr + Qr.T) / 2)
        best = min(best, ev.min())
    return best

if __name__ == '__main__':
    d = int(sys.argv[1]); seed = int(sys.argv[2])
    rng = np.random.default_rng(seed)
    p = np.sort(rng.dirichlet(np.ones(d)))[::-1]
    print('p =', np.round(p, 4))
    res = []
    nfeas = 0
    for pi in itertools.permutations(range(d * d)):
        pi = np.array(pi)
        R = analyse(p, d, pi)
        if R is None:
            continue
        nfeas += 1
        res.append((R, tuple(pi)))
    res.sort()
    print('feasible permutations:', nfeas)
    for R, pi in res[:12]:
        print(f'R = {R:.6f}  pi = {pi}')
