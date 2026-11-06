import glob
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
blue, orange, green, gray = "#2a78d6", "#eb6834", "#1f9e6e", "#52514e"
eta, kappa, cp = 0.5, 2, 2.5


def read(path):
    with open(path) as f:
        head = f.readline().split()
    return float(head[2]), float(head[4]), np.loadtxt(path)


def heat(path):
    l, k, o = read(path)
    t, a = o[:, 0], np.abs(o[:, 1])
    g0 = k * k / cp
    m = (t > 0.3 / g0) & (t < 2.5 / g0)
    return l, -np.polyfit(t[m], np.log(a[m]), 1)[0] / g0


def sound(path):
    l, k, o = read(path)
    t, u = o[:, 0], o[:, 1]
    e = np.where((np.abs(u[1:-1]) > np.abs(u[:-2])) & (np.abs(u[1:-1]) >= np.abs(u[2:])))[0] + 1
    rate = -np.polyfit(t[e], np.log(np.abs(u[e])), 1)[0]
    z = np.where(np.sign(u[1:]) != np.sign(u[:-1]))[0]
    tz = t[z] - u[z] * (t[z + 1] - t[z]) / (u[z + 1] - u[z])
    w = np.pi / np.mean(np.diff(tz))
    return l, k, rate, w / (k * np.sqrt(5 / 3))


th = np.loadtxt("modes3")
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True)
ax = axs[0]
ax.plot(th[:, 0], th[:, 2], color=gray, lw=1.5, label="theory")
s0 = sorted(sound(f) for f in glob.glob("sound.0.*"))
ax.plot([s[0] for s in s0], [s[2] / (2 / 3 * eta * s[1]**2) for s in s0], "D",
        color=blue, ms=5, label="sound, $\\kappa=0$")
ax.set_title(r"$\nu_l(k)/\nu_l$", fontsize=11, color=gray)
ax = axs[1]
ax.plot(th[:, 0], th[:, 3], color=gray, lw=1.5, label="theory")
hh = sorted(heat(f) for f in glob.glob("heat.*"))
ax.plot([h[0] for h in hh], [h[1] for h in hh], "D", color=blue, ms=5,
        label="heat mode")
s2 = {s[0]: s for s in (sound(f) for f in glob.glob("sound.2.*"))}
x, y = [], []
for s in s0:
    if s[0] in s2:
        x.append(s[0])
        y.append((s2[s[0]][2] - s[2]) / (2 / 15 * kappa * s[1]**2))
ax.plot(x, y, "x", color=orange, ms=6, label="sound, $\\kappa=2$ minus 0")
ax.set_title(r"$D_T(k)/D_T$", fontsize=11, color=gray)
ax = axs[2]
ax.plot(th[:, 0], th[:, 4], color=gray, lw=1.5, label="theory")
ax.plot([s[0] for s in s0], [s[3] for s in s0], "D", color=blue, ms=5,
        label="$\\kappa=0$")
ax.plot([s[0] for s in s2.values()], [s[3] for s in s2.values()], "x",
        color=orange, ms=6, label="$\\kappa=2$")
ax.set_title(r"$c(k)/c_0$", fontsize=11, color=gray)
for ax in axs:
    ax.axhline(1, color="#d8d7d2", lw=1, zorder=0)
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(r"$\lambda/h$")
    ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("modes.svg")

for s in s0:
    print("sound kappa=0 l/h %5.2f nu_l %.4f c %.4f" % (s[0], s[2] / (2 / 3 * eta * s[1]**2), s[3]))
for h in hh:
    print("heat l/h %5.2f D_T %.4f" % h)
for x0, y0 in zip(x, y):
    print("sound difference l/h %5.2f D_T %.4f" % (x0, y0))
for f in sorted(glob.glob("rest.*")):
    with open(f) as fh:
        n = int(fh.readline().split()[6])
    o = np.loadtxt(f)
    o = o[len(o) // 5:]
    kB = float(f[5:])
    a, b, al = 3 * (n - 1) / 2, 1.5 * n / kB, 1.5 / kB
    print("%s n %d <dT^2>/<T>^2 %.6f theory %.6f  <v_a^2>/<kB T> %.5f theory %.5f" %
          (f, n, o[:, 6].mean(), (a + b - al) / (al * (a + b + 1)),
           o[:, 4].mean() / o[:, 5].mean(), 1 - 1 / n))
