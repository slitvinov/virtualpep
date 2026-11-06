import numpy as np


def equilibration(a):
    """Start index that maximizes the number of uncorrelated samples
    (Chodera, J. Chem. Theory Comput. 12, 1799, 2016)."""
    n = len(a)
    best, t0 = -1, 0
    for t in range(0, n // 2, max(1, n // 100)):
        tau = tauint(a[t:])[0]
        neff = (n - t) / (2 * tau)
        if neff > best:
            best, t0 = neff, t
    return t0


def gamma(a):
    a = a - a.mean()
    n = len(a)
    f = np.fft.rfft(a, 2 * n)
    return np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)


def tauint(a, S=1.5):
    """Integrated autocorrelation time with Wolff's automatic window."""
    g = gamma(a)
    return window(g, len(a), S)


def window(g, n, S=1.5):
    if g[0] == 0:
        return 0.5, 0.0, 1
    rho = g / g[0]
    tint = 0.5
    for W in range(1, len(rho)):
        tint += rho[W]
        tau = S / np.log((2 * tint + 1) / (2 * tint - 1)) if tint > 0.5 else 1e-300
        if np.exp(-W / tau) - tau / np.sqrt(W * n) < 0:
            break
    dt = tint * np.sqrt(4 * (W + 0.5 - tint) / n)
    return tint, dt, W


def uwerr(reps, f=None, S=1.5):
    """Wolff's Gamma method. reps: list of arrays (time, observable) from
    independent replicas; f: function of the observable means. Returns
    value, error, error of the error, tau_int and its error."""
    reps = [np.atleast_2d(np.asarray(r, float).T).T for r in reps]
    nobs = reps[0].shape[1]
    N = sum(len(r) for r in reps)
    mean = sum(r.sum(0) for r in reps) / N
    if f is None:
        f = lambda m: m[0]
    h = 1e-7 * (np.abs(mean) + 1e-12)
    grad = np.array([(f(mean + h * e) - f(mean - h * e)) / (2 * h[k])
                     for k, e in enumerate(np.eye(nobs))])
    L = min(len(r) for r in reps)
    G = np.zeros(L)
    for r in reps:
        x = (r - mean) @ grad
        n = len(x)
        fx = np.fft.rfft(x, 2 * n)
        G[:n if n < L else L] += np.fft.irfft(fx * np.conj(fx))[:min(n, L)]
    G /= N - len(reps) * np.arange(L)
    tint, dtint, W = window(G, N, S)
    C = G[0] + 2 * G[1:W + 1].sum()
    C *= 1 + (2 * W + 1) / N
    tint = C / (2 * G[0])
    err = np.sqrt(C / N)
    return f(mean), err, err * np.sqrt((W + 0.5) / N), tint, dtint


def blocking(a):
    """Flyvbjerg-Petersen: errors of the mean for block lengths 2^k."""
    out = []
    a = np.asarray(a, float)
    while len(a) >= 32:
        out.append((len(a), a.std(ddof=1) / np.sqrt(len(a))))
        a = 0.5 * (a[:len(a) // 2 * 2:2] + a[1:len(a) // 2 * 2:2])
    return out


def jackknife(values):
    """Leave-one-replica-out jackknife. values: function value with each
    replica left out, and the full value first."""
    full, loo = values[0], np.asarray(values[1:])
    R = len(loo)
    bias = (R - 1) * (loo.mean() - full)
    err = np.sqrt((R - 1) / R * ((loo - loo.mean())**2).sum())
    return full - bias, err


def replicas(reps, f=None):
    """Independent replicas: leave-one-replica-out jackknife of f of the
    pooled means, Student-t 95% interval, and a chi^2 test of the replica
    means against their own Gamma-method errors."""
    from scipy import stats as st
    reps = [np.atleast_2d(np.asarray(r, float).T).T for r in reps]
    if f is None:
        f = lambda m: m[0]
    R = len(reps)
    sums = np.array([r.sum(0) for r in reps])
    counts = np.array([len(r) for r in reps])
    full = f(sums.sum(0) / counts.sum())
    loo = [f((sums.sum(0) - sums[k]) / (counts.sum() - counts[k])) for k in range(R)]
    value, err = jackknife([full] + loo)
    ci = st.t.ppf(0.975, R - 1) * err
    vals = np.array([uwerr([r], f)[:2] for r in reps])
    chi2 = (((vals[:, 0] - value) / vals[:, 1])**2).sum()
    q = st.chi2.sf(chi2, R - 1)
    return value, err, ci, chi2 / (R - 1), q


def acf(reps, maxlag, zero_mean=False):
    """Autocorrelation C(t) for t < maxlag averaged over independent
    replicas and time origins, with leave-one-replica-out jackknife
    errors per lag. Returns C, error of C and the jackknife samples."""
    reps = [np.asarray(r, float) for r in reps]
    m = 0 if zero_mean else np.mean(np.concatenate(reps))
    S, Nt = [], []
    for r in reps:
        x = r - m
        n = len(x)
        f = np.fft.rfft(x, 2 * n)
        S.append(np.fft.irfft(f * np.conj(f))[:maxlag])
        Nt.append(n - np.arange(maxlag))
    S, Nt = np.array(S), np.array(Nt)
    C = S.sum(0) / Nt.sum(0)
    loo = np.array([(S.sum(0) - S[k]) / (Nt.sum(0) - Nt[k]) for k in range(len(reps))])
    R = len(reps)
    err = np.sqrt((R - 1) / R * ((loo - loo.mean(0))**2).sum(0))
    return C, err, loo


def equilibration_replicas(reps):
    """One equilibration time for all replicas: maximize the summed
    effective sample size, Eq. 14 of Chodera 2016 averaged over replicas."""
    n = min(len(r) for r in reps)
    best, t0 = -1, 0
    for t in range(0, n // 2, max(1, n // 100)):
        neff = sum((len(r) - t) / (2 * tauint(np.asarray(r[t:], float))[0]) for r in reps)
        if neff > best:
            best, t0 = neff, t
    return t0
