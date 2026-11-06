import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np

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
blue, orange, gray = "#2a78d6", "#eb6834", "#52514e"
for dim in 2, 3:
    th = np.loadtxt(f"theory{dim}")
    fig, axs = plt.subplots(2, 1, figsize=(5.6, 5.6), sharex=True)
    for ax, col, name, label in ((axs[0], 1, "nu", r"$\nu/\nu_0$"),
                                 (axs[1], 2, "c", r"$c/c_0$")):
        ax.plot(th[:, 0], th[:, col], color=gray, lw=1.5, label="theory")
        for h, color, marker in ((3, blue, "D"), (4, orange, "x")):
            s = np.loadtxt(f"{name}{dim}.{h}")
            ax.plot(s[:, 0], s[:, 1], marker, color=color, ms=5,
                    label=f"$h/l_0 = {h}$")
        ax.axhline(1, color="#d8d7d2", lw=1, zorder=0)
        ax.set_xlim(0, 16.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(label)
    axs[1].set_xlabel(r"$\lambda/h$")
    axs[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"sdpd{dim}.svg")
