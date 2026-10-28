"""Check: a general (coherent) collision is never locally cheaper than a mixture of resonant permutations.

At a resonant point (system p, machine q*), configurations of equal weight form degenerate blocks of r* = p x q*.
Take a random block unitary U0 (commutes with r*), perturb the machine q = q* + eps*dq, and compute EXACTLY
    Sigma(U0) = D(U0 s U0^+ || s) - D(rho'_S || rho),   s = rho x tau,
and the system move dp.  Compare with the cost of the permutation mixture with the same average move: by Birkhoff,
B = |U0|^2 = sum_k lam_k P_k (permutations within blocks, each resonant), and the mixture costs sum_k lam_k Sigma(P_k).
Claim: Sigma(U0) >= sum_k lam_k Sigma(P_k) - O(eps^3), and the moves agree to first order.
"""
import numpy as np, itertools
from scipy.linalg import logm
from scipy.stats import unitary_group
from scipy.optimize import linear_sum_assignment
rng = np.random.default_rng(3)

def rel(a, b):
    wa, va = np.linalg.eigh(a); wb, vb = np.linalg.eigh(b)
    la = va @ np.diag(np.log(np.clip(wa, 1e-300, None))) @ va.conj().T
    lb = vb @ np.diag(np.log(np.clip(wb, 1e-300, None))) @ vb.conj().T
    return float(np.real(np.trace(a @ (la - lb))))

def ptrace_M(X, d):
    return np.trace(X.reshape(d, d, d, d), axis1=1, axis2=3)

def sigma_of(Umat, p, q, d):
    s = np.kron(np.diag(p), np.diag(q))
    s2 = Umat @ s @ Umat.conj().T
    rS = ptrace_M(s2, d)
    return rel(s2, s) - rel(rS, np.diag(p)), np.real(np.diag(rS)) - p

def birkhoff(B, tol=1e-12):
    """greedy Birkhoff-von Neumann decomposition of a doubly stochastic matrix"""
    B = B.copy(); out = []
    while B.max() > tol:
        cost = np.where(B > tol, -np.log(np.maximum(B, 1e-300)), 1e6)
        r, c = linear_sum_assignment(cost)
        lam = B[r, c].min()
        if lam <= tol: break
        P = np.zeros_like(B); P[r, c] = 1
        out.append((lam, P)); B = B - lam * P
    return out

d = 3
worst = np.inf
for trial in range(300):
    p = rng.dirichlet(np.ones(d))
    # resonant machine making a chosen pattern of equal weights: pick q* so that r* has some degenerate blocks
    # (use the qutrit pi_ext resonance q_{y+1}/q_y = p0/p2 and the swap resonance q = p alternately)
    if trial % 2 == 0:
        rr = p[0] / p[2]; qs = np.array([rr ** k for k in range(d)]); qs /= qs.sum()
    else:
        qs = p.copy()
    rstar = np.outer(p, qs).ravel()
    # degenerate blocks of r*
    blocks = []
    used = np.zeros(d * d, bool)
    for i in range(d * d):
        if used[i]: continue
        idx = [j for j in range(d * d) if abs(rstar[j] - rstar[i]) < 1e-12 * rstar[i]]
        for j in idx: used[j] = True
        blocks.append(idx)
    U0 = np.zeros((d * d, d * d), complex)
    for idx in blocks:
        U0[np.ix_(idx, idx)] = unitary_group.rvs(len(idx), random_state=rng) if len(idx) > 1 else 1.0
    dq = rng.normal(size=d); dq -= dq.mean()
    eps = 1e-3 * min(qs)
    q = qs + eps * dq
    S_U, dp_U = sigma_of(U0, p, q, d)
    B = np.abs(U0) ** 2
    mix_S, mix_dp = 0.0, np.zeros(d)
    for lam, P in birkhoff(B):
        S_P, dp_P = sigma_of(P.astype(complex), p, q, d)
        mix_S += lam * S_P; mix_dp += lam * dp_P
    gap = (S_U - mix_S) / max(abs(S_U), 1e-30)
    worst = min(worst, gap)
    if trial < 6 or gap < -1e-3:
        print(f'trial {trial}: Sigma(U)={S_U:.4e}  mixture={mix_S:.4e}  rel.gap={gap:+.3e}  |dp_U - dp_mix|/|dp_U| = '
              f'{np.linalg.norm(dp_U - mix_dp) / max(np.linalg.norm(dp_U), 1e-30):.1e}')
print('worst relative gap (Sigma(U) - mixture)/Sigma(U) over trials:', worst)
