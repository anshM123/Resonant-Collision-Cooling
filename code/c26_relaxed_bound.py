"""Relaxed universal-bound test (SM S11d): arbitrary maps sigma_x subject only to the marginal condition sigma_X(Y) ~ q*."""
# Relaxed claim: X~p indep Y~q, Y'=s_X(Y) with Y'~q (pushforward constraint only), centered psi:
#   E[(E[psi(Y')|X])^2] <= (1-1/m) E[(psi(Y')-psi(Y))^2],  m = number of machine levels.
import numpy as np, itertools
from scipy.optimize import linprog
rng = np.random.default_rng(0)
worst = 0
for trial in range(4000):
    m = rng.integers(2, 6); k = rng.integers(2, 7)
    q = rng.dirichlet(np.ones(m) * rng.uniform(0.2, 3))
    S = [rng.integers(0, m, m) for _ in range(k)]
    # pushforwards
    mu = np.array([[q[s == yp].sum() for yp in range(m)] for s in S])   # k x m
    c = rng.normal(size=k)
    res = linprog(c, A_eq=np.vstack([mu.T, np.ones(k)]), b_eq=np.concatenate([q, [1.0]]), bounds=[(0, None)] * k, method='highs')
    if res.status != 0: continue
    p = res.x
    # quadratic forms in psi (restricted to q-centered psi)
    Fm = np.zeros((m, m)); Gm = np.zeros((m, m))
    for px, s in zip(p, S):
        a = np.zeros(m)                       # m_x = sum_y q_y psi(s(y)) = a . psi
        for y in range(m): a[s[y]] += q[y]
        Fm += px * np.outer(a, a)
        for y in range(m):
            e = np.zeros(m); e[s[y]] += 1; e[y] -= 1
            Gm += px * q[y] * np.outer(e, e)
    # centered subspace basis
    C = np.eye(m) - np.outer(np.ones(m), q)        # psi -> psi - E_q psi  (columns span centered functions)
    U, sv, _ = np.linalg.svd(C); Bc = U[:, :m - 1]  # basis of range (dimension m-1)
    Fr = Bc.T @ Fm @ Bc; Gr = Bc.T @ Gm @ Bc
    w, V = np.linalg.eigh(Gr)
    keep = w > 1e-12
    if (~keep).any():
        # null directions of G must not carry F
        if np.abs(V[:, ~keep].T @ Fr @ V[:, ~keep]).max() > 1e-10: print('null-G with F>0!', trial); continue
    W = V[:, keep] / np.sqrt(w[keep])
    ratio = np.linalg.eigvalsh(W.T @ Fr @ W).max() if keep.any() else 0
    val = ratio / (1 - 1 / m)
    if val > worst:
        worst = val; print(f'trial {trial}: m={m} k={k} F/G max = {ratio:.6f}, (1-1/m) = {1-1/m:.4f}, normalised = {val:.6f}', flush=True)
print('worst normalised (<=1 means claim holds):', worst)
