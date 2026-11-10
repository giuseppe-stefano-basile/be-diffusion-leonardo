import numpy as np
from scipy.stats import norm, rankdata

def rank_diagnostics(chains):
    """Rank-normalized/folded split R-hat and Geyer positive-sequence ESS."""
    chains = np.asarray(chains)
    m, n, d = chains.shape
    half = n // 2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    rhat = np.full(d, np.inf); ess = np.zeros(d)
    if half < 4:
        return rhat, ess
    def normalized(x):
        ranks = rankdata(x.ravel()).reshape(x.shape)
        return norm.ppf((ranks - .375) / (x.size + .25))
    def rhat_one(x):
        within = np.mean(np.var(x, axis=1, ddof=1))
        between = x.shape[1] * np.var(np.mean(x, axis=1), ddof=1)
        if within <= 0:
            return np.inf
        varplus = (x.shape[1] - 1) / x.shape[1] * within + between / x.shape[1]
        return np.sqrt(varplus / within)
    for j in range(d):
        x = normalized(split[:, :, j])
        folded = normalized(np.abs(split[:, :, j] - np.median(split[:, :, j])))
        rhat[j] = max(rhat_one(x), rhat_one(folded))
        centered = x - x.mean(axis=1, keepdims=True)
        fft = np.fft.rfft(centered, n=2 * half, axis=1)
        acov = np.fft.irfft(fft * fft.conj(), n=2 * half, axis=1)[:, :half] / half
        within = np.mean(np.var(x, axis=1, ddof=1))
        varplus = (half - 1) / half * within + np.var(x.mean(axis=1), ddof=1)
        if varplus <= 0:
            continue
        rho = 1 - (within - acov.mean(axis=0)) / varplus
        rho[0] = 1.
        pairs = rho[::2][:half // 2] + rho[1::2][:half // 2]
        total = 0.; previous = np.inf
        for pair in pairs:
            if pair < 0:
                break
            previous = min(previous, pair)
            total += previous
        autocorrelation_time = max(1., -1 + 2 * total)
        ess[j] = min(m * n, len(x) * half / autocorrelation_time)
    return rhat, ess
