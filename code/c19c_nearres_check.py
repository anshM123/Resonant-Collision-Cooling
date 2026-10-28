"""Near-resonant (R != 0, nondegenerate p*) check of S10: second-order formula with D = R x tau* + rho* x Theta,
first-order spectral move = Birkhoff mixture move, and Q >= sum lam_k Q_k.  50-digit exact dissipation."""
import numpy as np, mpmath as mp
from scipy.optimize import linear_sum_assignment
mp.mp.dps = 50
rng = np.random.default_rng(3)

def mlog(A):
    E, Q = mp.eigh(A)
    return Q * mp.diag([mp.log(e) for e in E]) * Q.H

def relent(a, b):
    M = a * (mlog(a) - mlog(b)); return mp.re(sum(M[i, i] for i in range(a.rows)))

def ptr(X, d, dm):
    R = mp.matrix(d, d)
    for x in range(d):
        for x2 in range(d):
            R[x, x2] = sum(X[x * dm + y, x2 * dm + y] for y in range(dm))
    return R

def kmc(a, b):
    return 1 / a if abs(a - b) < mp.mpf(10) ** (-35) * max(a, b) else (mp.log(a) - mp.log(b)) / (a - b)

def km2(A, diag):
    n = len(diag); return mp.re(sum(abs(A[i, j]) ** 2 * kmc(diag[i], diag[j]) for i in range(n) for j in range(n)))

def birkhoff(B, tol=1e-13):
    B = B.copy(); out = []
    while B.max() > tol:
        cost = np.where(B > tol, -np.log(np.maximum(B, 1e-300)), 1e6)
        r, c = linear_sum_assignment(cost); lam = B[r, c].min()
        if lam <= tol: break
        P = np.zeros_like(B); P[r, c] = 1; out.append((lam, P)); B = B - lam * P
    return out

def rand_unitary(k):
    Z = mp.matrix([[mp.mpc(rng.normal(), rng.normal()) for _ in range(k)] for _ in range(k)])
    Q, R = mp.qr(Z); return Q

def trial(kind, d=3, dm=3, useK=True):
    p = sorted([mp.mpf(v) for v in rng.dirichlet(np.ones(d))], reverse=True); s = sum(p); p = [v / s for v in p]
    if kind == 'ext':
        rr = p[0] / p[-1]; q = [rr ** y for y in range(dm)]
    elif kind == 'swap':
        q = list(p)
    else:  # partial
        q = [mp.mpf(v) for v in rng.dirichlet(np.ones(dm))]; q[1] = q[0] * p[1] / p[0]
    s = sum(q); q = [v / s for v in q]
    n = d * dm; rstar = [p[x] * q[y] for x in range(d) for y in range(dm)]
    used = [False] * n; blocks = []
    for i in range(n):
        if used[i]: continue
        idx = [j for j in range(n) if abs(rstar[j] - rstar[i]) < mp.mpf(10) ** (-35) * rstar[i]]
        for j in idx: used[j] = True
        blocks.append(idx)
    U0 = mp.zeros(n, n)
    for idx in blocks:
        Ub = rand_unitary(len(idx))
        for a, i in enumerate(idx):
            for b, j in enumerate(idx): U0[i, j] = Ub[a, b]
    K = mp.matrix([[mp.mpc(rng.normal(), rng.normal()) for _ in range(n)] for _ in range(n)]); K = (K + K.H) / 2
    if not useK: K = mp.zeros(n, n)
    th = [mp.mpf(v) for v in rng.normal(size=dm)]; m = sum(th) / dm; th = [v - m for v in th]
    Rv = [mp.mpf(v) for v in rng.normal(size=d)]; m = sum(Rv) / d; Rv = [v - m for v in Rv]
    S = mp.diag(rstar)
    Dm = mp.diag([Rv[x] * q[y] + p[x] * th[y] for x in range(d) for y in range(dm)])
    Xpar = U0 * Dm * U0.H - Dm
    Xperp = U0 * (1j * (K * S - S * K)) * U0.H
    X = Xpar + Xperp
    Qf = km2(X, rstar) / 2 - km2(ptr(X, d, dm), p) / 2
    vals = []; moves = []
    e0 = mp.mpf(10) ** (-12) * min(min(q), min(p))
    for e in (e0, e0 / 2):
        Ue = U0 * mp.expm(1j * e * K)
        rho = [p[x] + e * Rv[x] for x in range(d)]
        sig = mp.diag([rho[x] * (q[y] + e * th[y]) for x in range(d) for y in range(dm)])
        s2 = Ue * sig * Ue.H
        rS = ptr(s2, d, dm)
        Sg = relent(s2, sig) - relent(rS, mp.diag(rho))
        vals.append(Sg / e ** 2)
        ev = sorted(mp.eigh(rS)[0], reverse=True)
        moves.append([(ev[x] - rho[x]) / e for x in range(d)])
    Qn = 2 * vals[1] - vals[0]
    mv = [2 * moves[1][x] - moves[0][x] for x in range(d)]
    # Birkhoff
    Bn = np.array([[float(abs(U0[i, j]) ** 2) for j in range(n)] for i in range(n)])
    Dn = np.array([float(Dm[i, i]) for i in range(n)]); rn = np.array([float(v) for v in rstar]); pn = np.array([float(v) for v in p])
    mixQ = 0.0; mixd = np.zeros(d)
    for lam, P in birkhoff(Bn):
        dr = P @ Dn - Dn
        dk = dr.reshape(d, dm).sum(1)
        Qk = 0.5 * np.sum(dr ** 2 / rn) - 0.5 * np.sum(dk ** 2 / pn)
        mixQ += lam * Qk; mixd += lam * dk
    return float(Qf), float(Qn), float(abs(Qf - Qn) / abs(Qf)), mixQ, np.array([float(v) for v in mv]), mixd

for kind in ['ext', 'swap', 'partial']:
    for t in range(3):
        Qf, Qn, err, mixQ, mv, mixd = trial(kind, useK=(t != 0))
        print(f'{kind:7s} K={t != 0}: Qf={Qf:.12g} Qexact={Qn:.12g} relerr={err:.1e}  Q-mixQ={Qf - mixQ:+.3e}  '
              f'move err={np.abs(mv - mixd).max():.1e} (|move|={np.abs(mixd).max():.2e})', flush=True)
