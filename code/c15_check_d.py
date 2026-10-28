"""Independent 40-digit check for d-level protocols from c14 (explicit H_M, permutation unitaries, energy bookkeeping).
usage: c15_check_d.py d beta lam N mechfile pattern
"""
import sys, json
import numpy as np
import mpmath as mp
mp.mp.dps = 40
d, beta, lam, N = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
mechfile, pattern = sys.argv[5], sys.argv[6]
spec = sys.argv[7] if len(sys.argv) > 7 else None
lg = np.load(f'logits_d{d}_b{beta}_l{lam}_N{N}_{pattern}' + ('_E' + spec.replace(',', '_') if spec else '') + '.npy')
mechs = json.load(open(mechfile)) if mechfile != 'swap' else []
SWAP = [(c % d) * d + c // d for c in range(d * d)]
pis = [SWAP if pattern[n % len(pattern)] == 'S' else mechs[int(pattern[n % len(pattern)])]['pi'] for n in range(N)]
B = mp.mpf(beta); ES = [mp.mpf(v) for v in spec.split(',')] if spec else [mp.mpf(k) for k in range(d)]
def gibbs(b):
    w = [mp.e ** (-b * e) for e in ES]; Z = sum(w); return [x / Z for x in w]
ent = lambda v: -sum(x * mp.log(x) for x in v if x > 0)
p = gibbs(B); S0 = ent(p); betaEM = mp.mpf(0)
for n in range(N):
    l = [mp.mpf(float(x)) for x in lg[n]]; m = max(l); w = [mp.e ** (x - m) for x in l]; Z = sum(w); q = [x / Z for x in w]
    HM = [-mp.log(x) / B for x in q]
    r = [p[c // d] * q[c % d] for c in range(d * d)]
    r2 = [mp.mpf(0)] * (d * d)
    for c in range(d * d): r2[pis[n][c]] = r[c]          # permutation unitary on the product basis (diagonal states)
    p2 = [sum(r2[x * d + y] for y in range(d)) for x in range(d)]
    q2 = [sum(r2[x * d + y] for x in range(d)) for y in range(d)]
    betaEM += B * sum(HM[k] * (q2[k] - q[k]) for k in range(d))
    p = p2
T = gibbs(B * lam)
diss = betaEM - (S0 - ent(p))
comp = sum(p[i] * mp.log(p[i] / T[i]) for i in range(d))
print(f'd={d} {pattern} N={N}: dissipation {mp.nstr(diss, 12)}  + exact final swap {mp.nstr(comp, 3)}  => {N+1} machines: '
      f'{mp.nstr(diss + comp, 12)}   max |p_N - target| = {mp.nstr(max(abs(p[i] - T[i]) for i in range(d)), 3)}')
