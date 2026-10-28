"""Independent exact-rational re-derivation of the infinite-temperature certificate (qutrit, uniform p, uniform q*).
Config c = 3x + y.  r'_c = r_{pi^{-1} c}.  theta = (u0, u1, -u0-u1).
e_c := theta_{y(pi^{-1}c)} - theta_{y(c)}  (delta r = e/3),  f_x := sum_y e_(x,y)  (delta p = f/3).
2Q = 9 sum (e/3)^2 - 3 sum (f/3)^2 = sum e^2 - sum f^2 / 3.
<g, delta>_F = sum_x g_x (f_x/3)/(1/3) = f_0 - f_2.
h^2 = sup (b.u)^2 / (u^T A0 u),  A0 = sum E E^T - (1/3) sum F F^T.  Use A' = 3 A0 (integer): h^2 = 3 b^T A'^+ b.
"""
import itertools
from fractions import Fraction as Fr
from collections import Counter

TH = [(1, 0), (0, 1), (-1, -1)]      # theta_y as integer coefficient vector in (u0, u1)

def h2_of(pi):
    inv = [0] * 9
    for c, c2 in enumerate(pi):
        inv[c2] = c
    E = []
    for c in range(9):
        ys = inv[c] % 3; y = c % 3
        E.append((TH[ys][0] - TH[y][0], TH[ys][1] - TH[y][1]))
    F = []
    for x in range(3):
        F.append((sum(E[3 * x + y][0] for y in range(3)), sum(E[3 * x + y][1] for y in range(3))))
    a11 = 3 * sum(e[0] * e[0] for e in E) - sum(f[0] * f[0] for f in F)
    a12 = 3 * sum(e[0] * e[1] for e in E) - sum(f[0] * f[1] for f in F)
    a22 = 3 * sum(e[1] * e[1] for e in E) - sum(f[1] * f[1] for f in F)
    b = (F[0][0] - F[2][0], F[0][1] - F[2][1])
    # exact-g move possible?  g=(1,0,-1) in column span of F (3x2) with nonzero coefficient
    # f = F u proportional to (1,0,-1): f_1 = 0 and f_0 = -f_2 and f != 0
    # constraints: F1.u = 0, (F0+F2).u = 0 ; need nontrivial solution with F0.u != 0
    M = [F[1], (F[0][0] + F[2][0], F[0][1] + F[2][1])]
    # sum over x of f = 0 always (perm conserves), so F0+F1+F2 = 0 -> second row = -F1; single constraint F1.u = 0
    if F[1] == (0, 0):
        exact_g = F[0] != (0, 0)
    else:
        u = (-F[1][1], F[1][0])     # kernel of F1
        exact_g = (F[0][0] * u[0] + F[0][1] * u[1]) != 0
    det = a11 * a22 - a12 * a12
    assert det >= 0 and a11 >= 0 and a22 >= 0, (pi, a11, a12, a22)
    if det > 0:
        # b^T A'^{-1} b = (a22 b0^2 - 2 a12 b0 b1 + a11 b1^2)/det
        val = Fr(a22 * b[0] ** 2 - 2 * a12 * b[0] * b[1] + a11 * b[1] ** 2, det)
        return 3 * val, exact_g
    tr = a11 + a22
    if tr == 0:
        return (Fr(0) if b == (0, 0) else None), exact_g
    # rank 1: range spanned by column (a11, a12) or (a12, a22)
    col = (a11, a12) if (a11, a12) != (0, 0) else (a12, a22)
    if col[0] * b[1] - col[1] * b[0] != 0:
        return None, exact_g     # infinite
    # A'^+ = A'/tr^2
    val = Fr(a11 * b[0] ** 2 + 2 * a12 * b[0] * b[1] + a22 * b[1] ** 2, tr * tr)
    return 3 * val, exact_g

cnt = Counter(); n_inf = 0; n_supp = 0; n_exact = 0
for pi in itertools.permutations(range(9)):
    v, ex = h2_of(pi)
    if v is None:
        n_inf += 1; continue
    cnt[v] += 1
    if v > 0: n_supp += 1
    if ex: n_exact += 1
vals = sorted(cnt.items(), key=lambda t: -t[0])
print('infinite:', n_inf, ' nonzero support:', n_supp, ' exact-g movers:', n_exact)
print('top distinct h^2 values (value, value/6, count):')
for v, k in vals[:6]:
    print('  ', v, float(v / 6), k)
piA = (8, 6, 2, 3, 4, 5, 1, 7, 0)
print('pi_A h^2 =', h2_of(piA))
swap = tuple(3 * (c % 3) + c // 3 for c in range(9))
print('full swap h^2 =', h2_of(swap))
