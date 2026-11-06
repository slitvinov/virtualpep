import glob
import sys
import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
})
blue, gray = "#2a78d6", "#52514e"
eta, kappa, kB, T, cv = 1.0, 1.0, 0.01, 1.0, 1.5
f = 1 - kB / 3
d, h, ncell = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
L, N = ncell * 4**(1 / 3), 4 * ncell**3
th = np.loadtxt("modes3")

boxes = [np.loadtxt(x) for x in sorted(glob.glob(d + "/*.v"))]
t = boxes[0][:, 0]
R = len(boxes)
cs = np.array([b[:, 1] for b in boxes])
full = cs.mean(0)
loo = (cs.sum(0) - cs) / (R - 1)
err = np.sqrt((R - 1) / R * ((loo - loo.mean(0))**2).sum(0))

m = int(np.ceil((3 * N / (4 * np.pi))**(1 / 3))) + 1
g = np.arange(-m, m + 1)
n = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
n2 = (n * n).sum(1)
order = np.argsort(n2)
ks = 2 * np.pi * np.sqrt(n2[order][1:N]) / L
pred = np.zeros(len(t))
cache = {}
for k in ks:
    key = round(k, 12)
    if key not in cache:
        lh = 2 * np.pi / (k * h)
        gt, gl, gT, c = (np.interp(lh, th[:, 0], th[:, q]) for q in (1, 2, 3, 4))
        Wk = c * c
        H = np.array([[2 * Wk * k * k - Wk * Wk * k * k / 3, 0, 2 * Wk * k / 3], [0, 1, 0],
                      [2 * Wk * k / 3, 0, 2 / 3]])
        J = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        M = (J - np.diag([0, f * 4 / 3 * eta * k * k * gl, kappa * k * k * gT])) @ H
        C0 = np.linalg.inv(H)
        lon = np.array([(expm(M * s) @ C0)[1, 1] / C0[1, 1] for s in t])
        cache[key] = 2 * np.exp(-f * eta * k * k * gt * t) + lon
    pred += cache[key]
pred *= kB * T / N

z = (full - pred) / err
print("t     C sim          C theory       z")
for i in range(0, len(t), max(1, len(t) // 16)):
    print("%5.1f %.6e %.6e %6.2f" % (t[i], full[i], pred[i], z[i]))
fig, axs = plt.subplots(2, 1, figsize=(6.4, 6.4))
axs[0].plot(t, pred / pred[0], color=gray, lw=1.5, label="theory")
axs[0].errorbar(t, full / full[0], 1.96 * err / full[0], fmt="o", ms=2.5, color=blue,
                lw=0.8, label="simulation")
axs[0].set_ylabel(r"$C(t)/C(0)$")
axs[0].legend(frameon=False)
w = t >= 2
axs[1].loglog(t[w], pred[w] / pred[0], color=gray, lw=1.5)
pos = w & (full > 0)
axs[1].errorbar(t[pos], full[pos] / full[0], 1.96 * err[pos] / full[0], fmt="o", ms=2.5,
                color=blue, lw=0.8)
axs[1].set_xlabel("$t$")
axs[1].set_ylabel(r"$C(t)/C(0)$")
fig.tight_layout()
fig.savefig("vacf.svg")
