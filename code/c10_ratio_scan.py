import numpy as np, sys, warnings
from c4_finsler import analyse_point
warnings.filterwarnings('ignore')
E = np.array([0., 1., 2.])
betas = [float(b) for b in sys.argv[1:]]
for b in betas:
    p = np.exp(-b * E); p /= p.sum(); w = -p * (E - p @ E)
    try:
        r, nm = analyse_point(p, w)
    except Exception as ex:
        r, nm = float('nan'), -1
    print(f'{b:.4f},{r:.6f},{nm}', flush=True)
