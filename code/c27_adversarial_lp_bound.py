"""SM S11d: adversarial LP test of the relaxed d_M-1 bound over ALL maps sigma (usage: c27_adversarial_lp_bound.py m seed restarts)."""
# Adversarial test of the relaxed universal bound via LP over ALL maps sigma:[m]->[m]:
#   max_P sum_s P_s [ a_s^2 - (1-1/m) G_s ]  s.t. sum_s P_s * (s#q) = q, P >= 0, sum P = 1
# (a_s = E_q psi(s(Y)), G_s = E_q (psi(s(Y))-psi(Y))^2).  Claim: max <= 0 for all q, centred psi.
import numpy as np, itertools, sys
from scipy.optimize import linprog, minimize
m = int(sys.argv[1]); rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
maps = np.array(list(itertools.product(range(m), repeat=m)))
def lpval(z):
    q = np.exp(z[:m]); q /= q.sum()
    psi = z[m:2 * m]; psi = psi - q @ psi
    nrm = np.sqrt(q @ psi ** 2); psi = psi / max(nrm, 1e-12)
    a = (q[None, :] * psi[maps]).sum(1)
    G = (q[None, :] * (psi[maps] - psi[None, :]) ** 2).sum(1)
    c = a ** 2 - (1 - 1 / m) * G
    push = np.zeros((len(maps), m))
    for i, s in enumerate(maps):
        np.add.at(push[i], s, q)
    res = linprog(-c, A_eq=np.vstack([push.T, np.ones(len(maps))]), b_eq=np.concatenate([q, [1.0]]),
                  bounds=[(0, None)] * len(maps), method='highs')
    return -res.fun if res.status == 0 else -np.inf
best = -np.inf
for r in range(int(sys.argv[3]) if len(sys.argv) > 3 else 30):
    z0 = rng.normal(0, 1.5, 2 * m)
    res = minimize(lambda z: -lpval(z), z0, method='Nelder-Mead', options=dict(maxiter=1500, xatol=1e-9, fatol=1e-12))
    v = -res.fun
    if v > best:
        best = v; print(f'restart {r}: max LP value = {v:.3e}', flush=True)
print(f'm={m}: worst LP value over restarts = {best:.3e}  (claim: <= 0)')
