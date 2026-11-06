import glob
import sys
import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np
import stats

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
d, kB, c = sys.argv[1], 0.01, 1.5
al = c / kB

groups = {}
for f in glob.glob(d + "/n.*"):
    if f.endswith(".k"):
        continue
    with open(f) as fh:
        N = int(fh.readline().split()[6])
    o = np.loadtxt(f)
    groups.setdefault(N, []).append(o[len(o) // 5:])

rows = []
for N in sorted(groups):
    R = groups[N]
    r = stats.uwerr([o[:, [4, 5]] for o in R], lambda m: m[0] / m[1])
    k = stats.uwerr([np.column_stack([o[:, 4], o[:, 4]**2]) for o in R],
                    lambda m: m[1] / m[0]**2 - 1)
    t = stats.uwerr([o[:, 6] for o in R])
    rows.append((N, 1 - r[0], r[1], k[0], k[1], t[0], t[1]))
    print("N %5d replicas %d: 1-ratio %.3e +- %.1e | var K %.3e +- %.1e | spread %.4e +- %.1e" %
          (N, len(R), 1 - r[0], r[1], k[0], k[1], t[0], t[1]))
rows = np.array(rows)

n = np.logspace(np.log10(rows[0, 0] / 1.5), np.log10(rows[-1, 0] * 1.5), 200)
a = 3 * (n - 1) / 2
b = n * al
exact = (1 / n, b / (a * (a + b + 1)), (n - 1) / (n * al + 1))
fig, axs = plt.subplots(3, 1, figsize=(6.4, 6.4), sharex=True)
titles = (r"$1 - \langle m v_\alpha^2\rangle/\langle k_B T\rangle$",
          r"$\mathrm{var}(K)/\langle K\rangle^2$", r"$\langle \delta T^2\rangle/\langle T\rangle^2$")
for q, ax in enumerate(axs):
    ax.plot(n, exact[q], color=gray, lw=1.5, label="exact")
    ax.errorbar(rows[:, 0], rows[:, 1 + 2 * q], 1.96 * rows[:, 2 + 2 * q], fmt="D",
                color=blue, ms=5, capsize=3, label="simulation")
    ax.set_xscale("log")
    if q < 2:
        ax.set_yscale("log")
    ax.set_ylabel(titles[q])
axs[2].set_xlabel("$N$")
axs[0].legend(frameon=False)
fig.tight_layout()
fig.savefig("equipartition.svg")
