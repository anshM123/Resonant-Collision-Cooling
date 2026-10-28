"""60-digit recheck of the second-order formula of c19 (L2) with EXACT resonances and exactly unitary U0.
Draws include tiny machine populations (where double precision fails)."""
import sys, numpy as np, mpmath as mp
mp.mp.dps = 60
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 5)

def mlog(A):
    E, Q = mp.eigh(A)
    L = mp.diag([mp.log(e) for e in E])
    return Q * L * Q.H

def relent(a, b):
    M = a * (mlog(a) - mlog(b))
    return mp.re(sum(M[i, i] for i in range(a.rows)))

def ptr(X, d, dm):
    R = mp.matrix(d, d)
    for x in range(d):
        for x2 in range(d):
            R[x, x2] = sum(X[x * dm + y, x2 * dm + y] for y in range(dm))
    return R

def kmc(a, b):
    return 1 / a if abs(a - b) < mp.mpf(10) ** (-40) * max(a, b) else (mp.log(a) - mp.log(b)) / (a - b)

def km2(A, diag):
    n = len(diag)
    return mp.re(sum(abs(A[i, j]) ** 2 * kmc(diag[i], diag[j]) for i in range(n) for j in range(n)))

def rand_unitary(k):
    Z = mp.matrix([[mp.mpc(rng.normal(), rng.normal()) for _ in range(k)] for _ in range(k)])
    Q, R = mp.qr(Z)
    return Q

def setup(kind, d, dm):
    if kind == 'ext':
        p = sorted([mp.mpf(v) for v in rng.dirichlet(np.ones(d) * 0.6)], reverse=True)
        s = sum(p); p = [v / s for v in p]
        r = p[0] / p[-1]; q = [r ** y for y in range(dm)]
    elif kind == 'partial':
        p = [mp.mpf(v) for v in rng.dirichlet(np.ones(d))]; s = sum(p); p = [v / s for v in p]
        q = [mp.mpf(v) for v in rng.dirichlet(np.ones(dm))]; q[1] = q[0] * p[1] / p[0]
    elif kind == 'swap':
        p = [mp.mpf(v) for v in rng.dirichlet(np.ones(d) * 0.5)]; s = sum(p); p = [v / s for v in p]; q = list(p)
    elif kind == 'degen_ext':
        a = mp.mpf(rng.uniform(0.1, 0.3)); rest = sorted([mp.mpf(v) for v in rng.dirichlet(np.ones(d - 2))], reverse=True)
        rest = [v * (1 - 2 * a) for v in rest]; p = [a, a] + rest
        if p[0] < p[2]: p = [p[2] + mp.mpf('0.01')] * 2 + rest
        s = sum(p); p = sorted([v / s for v in p], reverse=True)
        r = p[0] / p[-1]; q = [r ** y for y in range(dm)]
    s = sum(q); q = [v / s for v in q]
    return p, q

def trial(kind, d, dm, use_K):
    p, qs = setup(kind, d, dm); n = d * dm
    rstar = [p[x] * qs[y] for x in range(d) for y in range(dm)]
    used = [False] * n; blocks = []
    for i in range(n):
        if used[i]: continue
        idx = [j for j in range(n) if abs(rstar[j] - rstar[i]) < mp.mpf(10) ** (-40) * rstar[i]]
        for j in idx: used[j] = True
        blocks.append(idx)
    U0 = mp.zeros(n, n)
    for idx in blocks:
        Ub = rand_unitary(len(idx))
        for a, i in enumerate(idx):
            for b, j in enumerate(idx): U0[i, j] = Ub[a, b]
    K = mp.matrix([[mp.mpc(rng.normal(), rng.normal()) for _ in range(n)] for _ in range(n)])
    K = (K + K.H) / 2
    if not use_K: K = mp.zeros(n, n)
    th = [mp.mpf(v) for v in rng.normal(size=dm)]; m = sum(th) / dm; th = [v - m for v in th]
    S = mp.diag(rstar); Dm = mp.diag([p[x] * th[y] for x in range(d) for y in range(dm)])
    X = U0 * Dm * U0.H - Dm + U0 * (1j * (K * S - S * K)) * U0.H
    Qf = km2(X, rstar) / 2 - km2(ptr(X, d, dm), p) / 2
    pos = km2(X, rstar) / 2
    vals = []
    e0 = mp.mpf(10) ** (-14) * min(qs)          # expansion needs eps << min q
    for e in (e0, e0 / 2):
        Ue = U0 * mp.expm(1j * e * K)
        s = mp.diag([p[x] * (qs[y] + e * th[y]) for x in range(d) for y in range(dm)])
        s2 = Ue * s * Ue.H
        Sg = relent(s2, s) - relent(ptr(s2, d, dm), mp.diag(p))
        vals.append(Sg / e ** 2)
    Qn = 2 * vals[1] - vals[0]
    return Qf, Qn, pos, min(qs), [len(b) for b in blocks if len(b) > 1]

worst = 0
for kind, d, dm in [('ext', 4, 4), ('degen_ext', 4, 4), ('ext', 3, 4), ('partial', 4, 3), ('partial', 3, 3), ('swap', 3, 3)]:
    for t in range(3):
        Qf, Qn, pos, mq, bl = trial(kind, d, dm, use_K=(t != 0))
        err = abs(Qf - Qn) / pos; worst = max(worst, err)
        print(f'{kind:9s} d={d} dm={dm} K={t != 0}: Q_formula={mp.nstr(Qf, 15)}  Q_exact={mp.nstr(Qn, 15)}  rel.err={mp.nstr(err, 3)}  '
              f'min q={mp.nstr(mq, 3)} blocks={bl}', flush=True)
print('worst relative error', mp.nstr(worst, 3))
