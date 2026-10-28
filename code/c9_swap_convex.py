"""Global optimum of swap-only protocols: min sum_n D(p_{n-1}||p_n), p_0 = Gibbs(beta), p_N = Gibbs(lam*beta).
Jointly convex in the intermediate distributions -> exponential-cone program (global optimum + duality gap)."""
import sys, numpy as np, cvxpy as cp
beta, lam = float(sys.argv[1]), float(sys.argv[2])
E = np.array([0.0, 1.0, 2.0])
g = lambda b: (lambda v: v / v.sum())(np.exp(-b * E))
p0, pT = g(beta), g(lam * beta)
for N in [int(x) for x in sys.argv[3:]]:
    P = cp.Variable((N + 1, 3))
    cons = [P[0] == p0, P[N] == pT, P >= 1e-12, cp.sum(P, axis=1) == 1]
    obj = sum(cp.sum(cp.rel_entr(P[n], P[n + 1])) for n in range(N))
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-13, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
    print(f'N={N}: swap-protocol global optimum = {prob.value:.12f}  status={prob.status}  N*opt={N*prob.value:.6f}', flush=True)
