import glob
import os
import sys
import matplotlib
import matplotlib.ticker

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
eta, kappa, cv = 1.0, 1.0, 1.5
d, h, ncell, kB = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
f = 1 - kB / 3
L, N = ncell * 4**(1 / 3), 4 * ncell**3
th = np.loadtxt("modes3")

boxes = [np.loadtxt(x) for x in sorted(glob.glob(d + "/*.v"))]
outs = [np.loadtxt(x[:-2]) for x in sorted(glob.glob(d + "/*.v")) if os.path.exists(x[:-2])][:8]
T = np.mean([o[len(o) // 5:, 5].mean() for o in outs]) / kB
t = boxes[0][:, 0]
R = len(boxes)
cs = np.array([b[:, 1] for b in boxes])
full = cs.mean(0)
loo = (cs.sum(0) - cs) / (R - 1)
err = np.sqrt((R - 1) / R * ((loo - loo.mean(0))**2).sum(0))
D = np.trapezoid(full, t) / 3
print("T %.4f D %.4e" % (T, D))

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
        H = np.array([[T * (2 * Wk * k * k - Wk * Wk * k * k / 3), 0, T * 2 * Wk * k / 3],
                      [0, 1, 0], [T * 2 * Wk * k / 3, 0, T * 2 / 3]])
        J = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        M = (J - np.diag([0, f * 4 / 3 * eta * k * k * gl, kappa * k * k * gT / T])) @ H
        C0 = np.linalg.inv(H)
        lon = np.array([(expm(M * s) @ C0)[1, 1] / C0[1, 1] for s in t])
        cache[key] = (2 * np.exp(-f * eta * k * k * gt * t) + lon) * np.exp(-D * k * k * t)
    pred += cache[key]
pred *= kB * T / N

z = (full - pred) / err
print("t     C sim          C theory       z")
for i in range(0, len(t), max(1, len(t) // 16)):
    print("%5.1f %.6e %.6e %6.2f" % (t[i], full[i], pred[i], z[i]))
fig, ax = plt.subplots(figsize=(6.4, 6.4))
w = t >= 2
ax.loglog(t[w], pred[w] / pred[0], color=gray, lw=1.5, label="mode sum")
pos = w & (full > 0)
ax.errorbar(t[pos], full[pos] / full[0], 1.96 * err[pos] / full[0], fmt="o", ms=2.5,
            color=blue, lw=0.8, label="simulation")
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([2, 5, 10, 20, 50])
ax.set_xticklabels(["2", "5", "10", "20", "50"])
ax.set_ylim(3e-6, 2e-3)
ax.set_xlabel("$t$")
ax.set_ylabel(r"$C(t)/C(0)$")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("vacf.svg")
