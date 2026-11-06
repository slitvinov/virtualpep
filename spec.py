import glob
import sys
import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
})
blue, gray = "#2a78d6", "#52514e"
kB, T, cv, NM, dt, maxlag = 0.01, 1.0, 1.5, 4, 0.5, 160
f = 1 - kB / 3

eta, kappa, h, L = 1.0, 1.0, 3.0, 15 * 4**(1 / 3)
th = np.loadtxt("modes3")
pred = {}
t = np.arange(maxlag) * dt
for m in range(1, NM + 1):
    k = 2 * np.pi * m / L
    lh = 2 * np.pi / (k * h)
    gt, gl, gT, cs = (np.interp(lh, th[:, 0], th[:, q]) for q in (1, 2, 3, 4))
    Wk = cs * cs
    Gv = f * 4 / 3 * eta * k * k * gl
    GS = kappa * k * k * gT
    H = np.array([[2 * Wk * k * k - Wk * Wk * k * k / 3, 0, 2 * Wk * k / 3], [0, 1, 0],
                  [2 * Wk * k / 3, 0, 2 / 3]])
    J = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    M = (J - np.diag([0, Gv, GS])) @ H
    C0 = kB * T * np.linalg.inv(H)
    C = np.array([expm(M * s) @ C0 for s in t])
    pred[m] = (np.exp(-f * eta * k * k * gt * t), C[:, 2, 2] / C0[2, 2], C[:, 0, 0] / C0[0, 0])
    print("m %d lh %.2f: adiabatic sound speed %.4f" % (m, lh, np.sqrt(H[0, 0]) / k))


def col(a, m, q, sin):
    return 1 + 10 * (NM * a + m - 1) + 2 * q + sin


def acf(x):
    n = len(x)
    g = np.fft.rfft(x, 2 * n)
    return np.fft.irfft(g * np.conj(g))[:maxlag], n - np.arange(maxlag)


boxes = []
for name in sorted(glob.glob(sys.argv[1] + "/f.*.k")):
    o = np.loadtxt(name)
    o = o[len(o) // 5:]
    box = {}
    for m in range(1, NM + 1):
        for key, qs in (("trans", (1, 2)), ("S", (4,)), ("rho", (3,))):
            s = np.zeros(maxlag)
            c = np.zeros(maxlag)
            for a in range(3):
                for q in qs:
                    for sin in (0, 1):
                        g, n = acf(o[:, col(a, m, q, sin)])
                        s += g
                        c += n
            box[m, key] = (s, c)
    boxes.append(box)
R = len(boxes)

fig, axs = plt.subplots(NM, 3, figsize=(9, 9), sharex=True)
t = np.arange(maxlag) * dt
for m in range(1, NM + 1):
    for j, (key, label) in enumerate((("trans", "transverse velocity"), ("S", "entropy"),
                                      ("rho", "density"))):
        s = sum(b[m, key][0] for b in boxes)
        c = sum(b[m, key][1] for b in boxes)
        full = s / c
        loo = np.array([(s - b[m, key][0]) / (c - b[m, key][1]) for b in boxes])
        loo = loo / loo[:, :1]
        val = full / full[0]
        err = np.sqrt((R - 1) / R * ((loo - loo.mean(0))**2).sum(0))
        p = pred[m][j]
        z = (val[1:] - p[1:]) / err[1:]
        ax = axs[m - 1, j]
        ax.plot(t, p, color=gray, lw=1.5, label="theory")
        ax.errorbar(t[::4], val[::4], 1.96 * err[::4], fmt="o", ms=2.5, color=blue,
                    capsize=0, lw=0.8, label="simulation")
        ax.axhline(0, color="#d8d7d2", lw=1, zorder=0)
        if m == 1:
            ax.set_title(label, fontsize=10, color=gray)
        if j == 0:
            ax.set_ylabel("m = %d" % m)
        print("m %d %-20s max |z| %.1f, rms z %.2f" % (m, label, np.abs(z).max(), np.sqrt((z**2).mean())))
for ax in axs[-1]:
    ax.set_xlabel("$t$")
axs[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("spectra.svg")
