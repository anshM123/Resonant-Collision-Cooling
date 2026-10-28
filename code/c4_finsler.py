"""Effective (Finsler) metric for classical collision-model steps, d_S = d_M = 3.

Per mechanism (permutation pi with resonant machine q*), small steps cost a quadratic form in the system move.
Over N steps one may alternate mechanisms, so the asymptotic cost is (1/N) * gauge(Delta)^2, where the unit ball is
the convex hull of the mechanisms' unit sets {delta : cost(delta) <= 1}.  Swaps alone give the ellipse
cost = 1/2 * sum delta^2/p.  ratio(w) = gauge_hull(w) / gauge_F(w)  < 1  means alternating mechanisms beats swaps
along direction w.
"""
import itertools, sys
import numpy as np
from scipy.spatial import ConvexHull

def mechanisms(p, d=3, ksamples=6):
    """yield (M, C): system move delta = M dq (3x2), cost = dq^T C dq (2x2), for every feasible permutation"""
    B = np.array([[1, 0], [0, 1], [-1, -1]], float)
    out = []
    for pi in itertools.permutations(range(d * d)):
        pi = np.array(pi)
        rows, rhs, ok = [], [], True
        for c in range(d * d):
            c2 = pi[c]
            if c2 == c: continue
            x, y = divmod(c, d); x2, y2 = divmod(c2, d)
            if y == y2:
                ok = False; break
            row = np.zeros(d); row[y] += 1; row[y2] -= 1
            rows.append(row); rhs.append(np.log(p[x2]) - np.log(p[x]))
        if not ok or not rows: continue
        A = np.array(rows); b = np.array(rhs)
        u, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.max(np.abs(A @ u - b)) > 1e-9: continue
        _, s, vt = np.linalg.svd(A); ker = vt[int(np.sum(s > 1e-9)):]
        rng = np.random.default_rng(0)
        for k in range(1 if len(ker) <= 1 else ksamples):
            uu = u if k == 0 else u + rng.normal(0, 1.0, len(ker)) @ ker
            q = np.exp(uu); q /= q.sum()
            rstar = np.outer(p, q).ravel()
            Mdr = np.zeros((d * d, 2))
            for j in range(2):
                pd = np.outer(p, B[:, j]).ravel()
                drp = np.empty(d * d); drp[pi] = pd
                Mdr[:, j] = drp - pd
            Md = Mdr.reshape(d, d, 2).sum(1)
            C = 0.5 * (Mdr.T @ (Mdr / rstar[:, None]) - Md.T @ (Md / p[:, None]))
            if np.linalg.norm(Md) < 1e-12: continue
            out.append((Md, C, tuple(pi)))
    return out

def tangent_basis():
    e = np.array([[1, -1, 0], [1, 1, -2]], float)
    return (e / np.linalg.norm(e, axis=1, keepdims=True)).T      # 3x2 orthonormal basis of sum=0

def unit_points(M, C, T, n=180):
    """points delta (in tangent coords) with cost 1, for system move delta = M dq, cost dq^T C dq"""
    Mt = T.T @ M                      # 2x2 map dq -> tangent coords
    pts = []
    for ang in np.linspace(0, 2 * np.pi, n, endpoint=False):
        dq = np.array([np.cos(ang), np.sin(ang)])
        c = dq @ C @ dq
        if c <= 1e-14:
            continue   # free direction in dq: only matters if it moves the system (then cost 0 => infinitely cheap)
        pts.append(Mt @ dq / np.sqrt(c))
    return np.array(pts)

def ray_extent(points, w):
    hull = ConvexHull(points)
    best = 0.0
    P = points[hull.vertices]
    m = len(P)
    for i in range(m):
        a, b = P[i], P[(i + 1) % m]
        # solve t w = a + s (b - a)
        Mx = np.array([w, a - b]).T
        if abs(np.linalg.det(Mx)) < 1e-14: continue
        t, s = np.linalg.solve(Mx, a)
        if t > 0 and -1e-9 <= s <= 1 + 1e-9:
            best = max(best, t)
    return best

def analyse_point(p, w):
    T = tangent_basis()
    mech = mechanisms(p)
    allpts = []
    Fpts = []
    for ang in np.linspace(0, 2 * np.pi, 720, endpoint=False):
        v = T @ np.array([np.cos(ang), np.sin(ang)])
        Fpts.append(np.array([np.cos(ang), np.sin(ang)]) / np.sqrt(0.5 * np.sum(v ** 2 / p)))
    Fpts = np.array(Fpts)
    zero_cost = False
    for M, C, pi in mech:
        ev = np.linalg.eigvalsh(C)
        if ev.min() < -1e-10:
            print('WARNING negative cost eigenvalue', ev, pi)
        pts = unit_points(M, C, T)
        if len(pts):
            allpts.append(pts)
    allpts = np.vstack(allpts + [Fpts])
    wt = T.T @ w
    tF = ray_extent(Fpts, wt)
    tH = ray_extent(allpts, wt)
    return tF / tH, len(mech)

if __name__ == '__main__':
    E = np.array([0.0, float(sys.argv[1]), float(sys.argv[2])])
    beta, lam = float(sys.argv[3]), float(sys.argv[4])
    p0 = np.exp(-beta * E); p0 /= p0.sum()
    p1 = np.exp(-lam * beta * E); p1 /= p1.sum()
    # Fisher-Rao geodesic on the sphere sqrt(p)
    a, b = np.sqrt(p0), np.sqrt(p1)
    th = np.arccos(np.clip(a @ b, -1, 1))
    print(f'E={E} beta={beta} lam={lam}  L*={2*th:.6f}')
    for s in np.linspace(0.0, 1.0, 6):
        x = (np.sin((1 - s) * th) * a + np.sin(s * th) * b) / np.sin(th)
        dx = (-th * np.cos((1 - s) * th) * a + th * np.cos(s * th) * b) / np.sin(th)
        p = x ** 2; w = 2 * x * dx
        ratio, nm = analyse_point(p, w)
        print(f's={s:.2f} p={np.round(p,4)}  gauge_hull/gauge_F along geodesic = {ratio:.6f}  (mechanisms {nm})', flush=True)
