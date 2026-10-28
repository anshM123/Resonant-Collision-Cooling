"""Verify the local-domination theorem: no collision beats a mixture of resonant permutations at second order.

Dissipationless collision: U0 commuting with s* = rho x tau*.  Perturb: tau = tau* + eps*Theta, U = U0 exp(i eps K).
Claims checked (exact arithmetic up to O(eps^3), Richardson-extrapolated):
  L2  Sigma = eps^2 Q + O(eps^3),  Q = 1/2 ||X||^2_{s*} - 1/2 ||Tr_M X||^2_rho   (Kubo-Mori norms),
      X = U0 D U0^+ - D + U0 i[K, s*] U0^+,  D = rho x Theta.
  T   Q = sum_k lam_k Q(P_k) + Jensen + Q_perp,  Jensen >= 0,  Q_perp >= 0,
      where B = |U0'|^2 = sum_k lam_k P_k (Birkhoff, in an eigenbasis of rho diagonalising Tr_M X_par),
      and the first-order spectral move equals sum_k lam_k delta_k.
usage: c19_local_domination.py [trials]
"""
import sys
import numpy as np
from scipy.stats import unitary_group
from scipy.optimize import linear_sum_assignment
rng = np.random.default_rng(7)

def logm_h(a):
    w, v = np.linalg.eigh(a)
    return v @ np.diag(np.log(w)) @ v.conj().T

def relent(a, b):
    return float(np.real(np.trace(a @ (logm_h(a) - logm_h(b)))))

def km_c(a, b):
    return 1.0 / a if abs(a - b) < 1e-14 * max(a, b) else (np.log(a) - np.log(b)) / (a - b)

def km_norm2(A, diag):
    """Kubo-Mori norm^2 of Hermitian A at the diagonal state diag"""
    n = len(diag)
    C = np.array([[km_c(diag[i], diag[j]) for j in range(n)] for i in range(n)])
    return float(np.real(np.sum(np.abs(A) ** 2 * C)))

def ptr_M(X, d, dm):
    return np.trace(X.reshape(d, dm, d, dm), axis1=1, axis2=3)

def birkhoff(B, tol=1e-13):
    B = B.copy(); out = []
    while B.max() > tol:
        cost = np.where(B > tol, -np.log(np.maximum(B, 1e-300)), 1e6)
        r, c = linear_sum_assignment(cost)
        lam = B[r, c].min()
        if lam <= tol: break
        P = np.zeros_like(B); P[r, c] = 1
        out.append((lam, P)); B = B - lam * P
    return out

def blocks_of(vals, tol=1e-11):
    used = np.zeros(len(vals), bool); out = []
    for i in range(len(vals)):
        if used[i]: continue
        idx = [j for j in range(len(vals)) if abs(vals[j] - vals[i]) < tol * vals[i]]
        used[idx] = True; out.append(idx)
    return out

def expm_h(K):
    w, v = np.linalg.eigh(K)
    return v @ np.diag(np.exp(1j * w)) @ v.conj().T

def sigma_exact(U, p, q, d, dm):
    s = np.kron(np.diag(p), np.diag(q)).astype(complex)
    s2 = U @ s @ U.conj().T
    rS = ptr_M(s2, d, dm)
    return relent(s2, s) - relent(rS, np.diag(p).astype(complex)), rS

def setup(kind, d, dm):
    """return p, q* (both normalised) with designed degeneracies"""
    if kind == 'swap':
        p = rng.dirichlet(np.ones(d)); return p, p.copy()
    if kind == 'ext':           # q_{y+1}/q_y = p_0/p_{d-1}
        p = np.sort(rng.dirichlet(np.ones(d)))[::-1]
        r = p[0] / p[-1]; q = r ** np.arange(dm); return p, q / q.sum()
    if kind == 'partial':       # only q1/q0 = p1/p0 imposed
        p = rng.dirichlet(np.ones(d)); q = rng.dirichlet(np.ones(dm)); q[1] = q[0] * p[1] / p[0]
        return p, q / q.sum()
    if kind == 'uniform':
        return rng.dirichlet(np.ones(d)), np.ones(dm) / dm
    if kind == 'degen':         # degenerate system spectrum, swap-resonant machine
        a = rng.uniform(0.05, 0.3); p = np.array([a, a] + list(rng.dirichlet(np.ones(d - 2)) * (1 - 2 * a)))
        return p, p.copy() if dm == d else np.ones(dm) / dm
    if kind == 'degen_ext':
        a = rng.uniform(0.05, 0.3); rest = np.sort(rng.dirichlet(np.ones(d - 2)) * (1 - 2 * a))[::-1]
        p = np.array([max(a, rest[0] + 0.01)] * 2 + list(rest)); p /= p.sum()
        r = p[0] / p[-1]; q = r ** np.arange(dm); return p, q / q.sum()
    raise ValueError(kind)

def one_trial(kind, d, dm, use_K=True):
    p, qs = setup(kind, d, dm)
    rstar = np.outer(p, qs).ravel()
    blk = blocks_of(rstar)
    n = d * dm
    U0 = np.zeros((n, n), complex)
    for idx in blk:
        U0[np.ix_(idx, idx)] = unitary_group.rvs(len(idx), random_state=rng) if len(idx) > 1 else np.exp(1j * rng.uniform(0, 6.3))
    K = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); K = (K + K.conj().T) / 2
    if not use_K: K[:] = 0
    Th = rng.normal(size=dm); Th -= Th.mean()
    sstar = np.diag(rstar).astype(complex)
    D = np.kron(np.diag(p), np.diag(Th)).astype(complex)
    Xpar = U0 @ D @ U0.conj().T - D
    Xperp = U0 @ (1j * (K @ sstar - sstar @ K)) @ U0.conj().T
    X = Xpar + Xperp
    Q_formula = 0.5 * km_norm2(X, rstar) - 0.5 * km_norm2(ptr_M(X, d, dm), p)
    # exact Sigma at two step sizes, Richardson
    eps = 2e-4 * min(qs.min(), 1.0) / (1 + np.abs(Th).max())
    vals = []
    for e in (eps, eps / 2):
        U = U0 @ expm_h(e * K)
        S, rS = sigma_exact(U, p, qs + e * Th, d, dm)
        vals.append(S / e ** 2)
    Q_num = 2 * vals[1] - vals[0]
    # decomposition
    Qpar = 0.5 * km_norm2(Xpar, rstar) - 0.5 * km_norm2(ptr_M(Xpar, d, dm), p)
    Qperp = 0.5 * km_norm2(Xperp, rstar) - 0.5 * km_norm2(ptr_M(Xperp, d, dm), p)
    # rotate within rho-eigenspaces so that Tr_M Xpar is diagonal
    TS = ptr_M(Xpar, d, dm)
    W = np.eye(d, dtype=complex)
    for idx in blocks_of(p):
        if len(idx) > 1:
            w, v = np.linalg.eigh(TS[np.ix_(idx, idx)]); W[np.ix_(idx, idx)] = v
    WW = np.kron(W, np.eye(dm))
    U0r = WW.conj().T @ U0 @ WW
    assert np.allclose(U0r @ sstar, sstar @ U0r, atol=1e-10)
    B = np.abs(U0r) ** 2
    mixQ, mixd, sqF = 0.0, np.zeros(d), 0.0
    for lam, P in birkhoff(B):
        Xk = P @ D @ P.T - D
        dk = np.real(np.diag(ptr_M(Xk, d, dm)))
        Qk = 0.5 * km_norm2(Xk, rstar) - 0.5 * np.sum(dk ** 2 / p)
        mixQ += lam * Qk; mixd += lam * dk; sqF += lam * np.sum(dk ** 2 / p)
    jensen = 0.5 * (sqF - np.sum(mixd ** 2 / p))
    # first-order spectral move of the true collision: eigenvalue shifts within rho-eigenspaces
    TSr = W.conj().T @ ptr_M(X, d, dm) @ W
    move_first = np.real(np.diag(TSr))
    off_in_eig = max([np.abs(TSr[np.ix_(idx, idx)] - np.diag(np.diag(TSr[np.ix_(idx, idx)]))).max()
                      for idx in blocks_of(p)] + [0.0])
    pos = 0.5 * km_norm2(X, rstar)                       # natural scale of Q
    mscale = np.abs(ptr_M(X, d, dm)).max() + np.abs(Xpar).max()
    return dict(Q_num=Q_num, Q_formula=Q_formula, Qpar=Qpar, Qperp=Qperp, mixQ=mixQ, jensen=jensen, pos=pos,
                move_err=np.abs(move_first - mixd).max() / max(mscale, 1e-30), off=off_in_eig / max(mscale, 1e-30))

if __name__ == '__main__':
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    worst = dict(L2=0.0, T=0.0, jensen=np.inf, perp=np.inf, gap=np.inf, move=0.0, off=0.0)
    for d, dm in [(3, 3), (4, 4), (3, 4), (4, 3), (3, 2)]:
        for kind in ['swap', 'ext', 'partial', 'uniform', 'degen', 'degen_ext']:
            if kind in ('swap', 'degen') and dm != d and kind == 'swap': continue
            for t in range(trials):
                r = one_trial(kind, d, dm, use_K=(t % 3 != 0))
                scale = max(r['pos'], 1e-300)
                worst['L2'] = max(worst['L2'], abs(r['Q_num'] - r['Q_formula']) / scale)
                worst['T'] = max(worst['T'], abs(r['Qpar'] - r['mixQ'] - r['jensen']) / scale)
                worst['jensen'] = min(worst['jensen'], r['jensen'] / scale)
                worst['perp'] = min(worst['perp'], r['Qperp'] / scale)
                worst['gap'] = min(worst['gap'], (r['Q_formula'] - r['mixQ']) / scale)
                worst['move'] = max(worst['move'], r['move_err'])
                worst['off'] = max(worst['off'], r['off'])
            print(f'd={d} dm={dm} {kind:10s} done; running worst: ' +
                  ', '.join(f'{k}={v:+.2e}' for k, v in worst.items()), flush=True)
    print('\nL2 = max rel |Q_num - Q_formula|;  T = max rel |Q_par - mix - Jensen|;  jensen, perp, gap = min rel values (must be >= 0)')
    print('move = max rel error of first-order spectral move vs Birkhoff mixture move')
