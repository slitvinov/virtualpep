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

c0 = np.sqrt(5 / 3)
out = {3: [], 4: []}
for line in open("latt3s.raw"):
    h, lh, k, a21, a22, a23, a31, a33 = map(float, line.split())
    M = np.array([[0, 1, 0], [a21, a22, a23], [a31, 0, a33]])
    t = np.arange(int(4 * 2 * np.pi / (0.5 * c0 * k) / 0.01) + 1) * 0.01
    V = np.array([expm(M * s)[1, 1] for s in t])
    z = np.where(np.sign(V[1:]) != np.sign(V[:-1]))[0]
    tz = t[z] - V[z] * (t[z + 1] - t[z]) / (V[z + 1] - V[z])
    out[int(h)].append((lh, np.pi * (len(tz) - 1) / (tz[-1] - tz[0]) / k / c0))
for h in out:
    np.savetxt("latt3s.%d" % h, out[h])
