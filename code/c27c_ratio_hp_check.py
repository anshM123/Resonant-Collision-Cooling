"""High-precision check of the ratio-LP maximisers of c27b (relaxed Theorem 2 bound, SM S11d).
For each restart: maximise R(q, psi) = max_P sum P a^2 / sum P G (marginal condition) by Nelder-Mead on the LP value,
then re-solve the LP at the maximiser with tight tolerances, take its support maps, recompute the exact mixture weights
in 50-digit arithmetic (they must be >= 0 and satisfy the marginal condition), and report the exact ratio - (1-1/m).
usage: c27c_ratio_hp_check.py m restarts seed
"""
import itertools, sys
import numpy as np, mpmath as mp
from scipy.optimize import linprog, minimize
mp.mp.dps = 50
m, restarts, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
maps = np.array(list(itertools.product(range(m), repeat=m)))
rng = np.random.default_rng(seed)
bound = 1 - 1 / m

def setup(z):
    q = np.exp(z[:m]); q /= q.sum()
    psi = z[m:]; psi = psi - q @ psi; psi = psi / np.sqrt(q @ psi ** 2)
    return q, psi

def lp(z, tol=None):
    q, psi = setup(z)
    a = (q[None, :] * psi[maps]).sum(1); G = (q[None, :] * (psi[maps] - psi[None, :]) ** 2).sum(1)
    pu = np.zeros((len(maps), m))
    for i, s in enumerate(maps): np.add.at(pu[i], s, q)
    nv = len(maps) + 1
    Aeq = np.zeros((m + 2, nv)); beq = np.zeros(m + 2)
    Aeq[0, :-1] = G; beq[0] = 1; Aeq[1:m + 1, :-1] = pu.T; Aeq[1:m + 1, -1] = -q; Aeq[m + 1, :-1] = 1; Aeq[m + 1, -1] = -1
    opts = {} if tol is None else dict(primal_feasibility_tolerance=tol, dual_feasibility_tolerance=tol)
    res = linprog(np.concatenate([-(a ** 2), [0.0]]), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * nv, method='highs',
                  options=opts)
    return res, q, psi

def Rval(z):
    res = lp(z)[0]
    return -res.fun if res.status == 0 else 0.0

worst = mp.mpf(-1)
for r in range(restarts):
    z0 = np.concatenate([rng.normal(0, 1.5, m), rng.normal(0, 1, m)])
    opt = minimize(lambda z: -Rval(z), z0, method='Nelder-Mead', options=dict(maxiter=600 * m, xatol=1e-10, fatol=1e-13))
    res, q, psi = lp(opt.x, tol=1e-10)
    if res.status != 0: continue
    Y = res.x[:-1]; t = res.x[-1]
    S = [i for i in range(len(maps)) if Y[i] > 1e-12 * max(Y.max(), 1)]
    qm = [mp.mpf(float(v)) for v in q]; qs = sum(qm); qm = [v / qs for v in qm]
    pm = [mp.mpf(float(v)) for v in psi]; c0 = sum(qm[i] * pm[i] for i in range(m)); pm = [v - c0 for v in pm]
    nr = mp.sqrt(sum(qm[i] * pm[i] ** 2 for i in range(m))); pm = [v / nr for v in pm]
    rows = []
    for i in S:
        pu = [mp.mpf(0)] * m
        for y in range(m): pu[maps[i][y]] += qm[y]
        rows.append(pu + [mp.mpf(1)])
    Amat = mp.matrix([[rows[j][k] for j in range(len(S))] for k in range(m + 1)])
    bvec = mp.matrix(qm + [mp.mpf(1)])
    try:
        P = mp.lu_solve(Amat.T * Amat, Amat.T * bvec)
    except ZeroDivisionError:
        print(f'restart {r}: singular support system, skipped'); continue
    resid = mp.norm(Amat * P - bvec)
    num = sum(P[j] * sum(qm[y] * pm[maps[S[j]][y]] for y in range(m)) ** 2 for j in range(len(S)))
    den = sum(P[j] * sum(qm[y] * (pm[maps[S[j]][y]] - pm[y]) ** 2 for y in range(m)) for j in range(len(S)))
    ex = num / den - mp.mpf(m - 1) / m
    ok = min(P) >= 0 and resid < mp.mpf(10) ** -30
    worst = max(worst, ex) if ok else worst
    if ok and ex > 0:
        print('   VIOLATION support maps and weights:', [(tuple(int(v) for v in maps[S[j]]), mp.nstr(P[j], 8)) for j in range(len(S))])
        print('   q =', [mp.nstr(v, 10) for v in qm], ' psi =', [mp.nstr(v, 10) for v in pm])
    print(f'restart {r}: LP R - bound = {-res.fun - bound:+.3e};  50-digit exact mixture: excess = {mp.nstr(ex, 6)}, '
          f'min P = {mp.nstr(min(P), 4)}, residual = {mp.nstr(resid, 3)}, q = {np.round(q, 5)}', flush=True)
print(f'm={m}: largest exact excess over 1-1/m among valid mixtures = {mp.nstr(worst, 6)}')
