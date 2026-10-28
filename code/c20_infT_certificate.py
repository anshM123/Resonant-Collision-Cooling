"""Dual certificate: at infinite temperature the qutrit optimum (over the convex hull of ALL resonant permutations,
hence by Theorem 1 over all collisions) along the cooling direction g = (1,0,-1) is exactly 1/2 of the swap cost.

At p = q* = uniform every permutation pi of the 9 configurations is resonant.  For machine perturbation theta (sum 0):
  D_c = p_x theta_y,  X_c = D_{pi^-1 c} - D_c,  delta_x = sum_y X_(x,y),  2Q = sum_c X_c^2/r_c - sum_x delta_x^2/p_x.
Support of mechanism pi's unit set {delta : Q <= 1/2} along the Fisher-dual of g:
  h_pi = sup_theta <g, delta>_F / sqrt(2Q),   <g, delta>_F = sum_x g_x delta_x / p_x.
If h_pi <= h* for all pi, the functional <g,.>_F / h* is <= 1 on the hull, so gauge_hull(g) >= ||g||_F^2 / h*.
pi_A attains gauge ||g||_F/sqrt(2); the certificate closes iff max_pi h_pi^2 = 2 ||g||_F^2.
Machine states q* != uniform at uniform p only add mechanisms with support h^2 < ||g||_F^2 (shown in the SM).
"""
import itertools, sys
import numpy as np

d = 3
n = d * d
p = np.full(d, 1 / d); r = np.full(n, 1 / n)
g = np.array([1.0, 0.0, -1.0])
B = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])          # theta = B u, sum-zero
Y = np.zeros((n, d)); Y[np.arange(n), np.arange(n) % d] = 1.0
Ymat = (np.repeat(p, d)[:, None] * Y) @ B                     # D = Ymat u   (9 x 2)
S = np.zeros((d, n)); S[np.arange(n) // d, np.arange(n)] = 1.0  # delta = S X
gF = g / p                                                     # <g, delta>_F = gF . delta
gnorm2 = float(g @ (g / p))

best = (-1.0, None); count_inf = 0; count_move = 0
perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)   # 362880 x 9, perms[k][c] = pi(c)
inv = np.argsort(perms, axis=1)                                            # pi^{-1}
# X = D[inv] - D  for all permutations at once: (P, 9, 2)
X = Ymat[inv] - Ymat[None, :, :]
Dl = np.einsum('xc,kcj->kxj', S, X)                                         # (P, 3, 2)
A = np.einsum('kcj,kcl->kjl', X / r[None, :, None], X) - np.einsum('kxj,kxl->kjl', Dl / p[None, :, None], Dl)
b = np.einsum('x,kxj->kj', gF, Dl)                                          # (P, 2)
A = 0.5 * (A + np.transpose(A, (0, 2, 1)))
w, V = np.linalg.eigh(A)
bt = np.einsum('kji,kj->ki', V, b)                                          # b in eigenbasis
tol = 1e-12
h2 = np.zeros(len(perms))
for i in range(2):
    pos = w[:, i] > tol
    h2[pos] += bt[pos, i] ** 2 / w[pos, i]
    bad = (~pos) & (np.abs(bt[:, i]) > 1e-10)
    h2[bad] = np.inf
k = int(np.argmax(h2))
print(f'permutations: {len(perms)}; with nonzero support along g: {int(np.sum(h2 > 1e-12))}; '
      f'infinite (free move along g): {int(np.sum(np.isinf(h2)))}')
print(f'min eigenvalue of 2Q over all perms (>= -tol): {w.min():.3e}')
print(f'||g||_F^2 = {gnorm2:.6f};  max_pi h_pi^2 = {h2[k]:.12f}  ->  ratio max h^2 / ||g||_F^2 = {h2[k] / gnorm2:.12f}')
print('maximiser pi =', list(perms[k]))
print(f'certified lower bound on gauge_hull(g)/||g||_F = {np.sqrt(gnorm2 / h2[k]):.12f}  (1/sqrt2 = {1/np.sqrt(2):.12f})')
top = np.argsort(-h2)[:5]
for t in top:
    print('  h^2/||g||^2 =', round(h2[t] / gnorm2, 12), list(perms[t]))
