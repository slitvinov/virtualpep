import numpy as np
from scipy.linalg import expm

eta = 10
out = {3: [], 4: []}
for line in open("latt3.raw"):
    h, lh, k, m21, m22 = map(float, line.split())
    M = np.array([[0, 1], [m21, m22]])
    dt = min(0.01, 0.05 / (eta * k * k))
    t = np.arange(int(2.5 / (eta * k * k) / dt) + 1) * dt
    V = np.array([expm(M * s)[1, 1] for s in t])
    m = V > V[0] * np.exp(-2)
    out[int(h)].append((lh, -np.polyfit(t[m], np.log(V[m]), 1)[0] / (eta * k * k)))
for h in out:
    np.savetxt("latt3.%d" % h, out[h])
