"""
Algothon 2026 submission strategy.

Big picture: we trade 50 of the 51 instruments (instrument 0, "ALGO", is just the
equal-weight index of the other 50, so we leave it flat). The core bet is mean
reversion - things that have drifted apart tend to come back together - run in two
independent flavours that are then blended:

  1. Pairs book (70%)    - Find instruments that move together ("tethered"), and when
                           one drifts too far from its partner, bet the gap closes.
  2. Residual book (30%) - Strip out the few big forces that push all names at once,
                           then fade each name's remaining private drift.

The two signals are only ~25% correlated, so blending them is steadier than either
alone. Everything is kept market-neutral (equal long and short) and sized to a fixed
gross exposure, capped at $10k per name. NumPy only - no extra packages.
"""

import numpy as np

nInst = 51  # total instruments (index 0 = ALGO, indices 1..50 = the tradeable names)

# ---------------------------------------------------------------------------
# Tuning knobs. These were chosen by out-of-sample validation. Everything the
# strategy does can be re-tuned here without touching the logic below.
# ---------------------------------------------------------------------------
ALPHA = 0.70        # blend weight: 70% pairs book, 30% residual book
GROSS = 700_000     # target gross dollar exposure (how much capital we deploy)

# Pairs-book settings:
ZW    = 30          # window (days) used to measure how "stretched" a spread is now
MAXP  = 25          # keep at most this many pairs (the best-reverting ones)
FW    = 250         # look-back window (days) used to find pairs and their hedge ratios
HLMAX = 15.0        # only trade pairs whose spread reverts within this many days
REBAL = 20          # re-pick the pair list every this many days (not every day)
ZCLIP = 2.5         # cap each spread's z-score so no single pair dominates

# Residual-book settings:
KF = 5              # number of common factors to strip out before looking at residuals
M  = 5              # window (days) over which we measure each name's recent drift
W  = 250            # look-back window (days) used to estimate the factors

# Cache so we only re-scan for pairs every REBAL days instead of every single day.
_cache = {"day": -10**9, "pairs": None}


def _select_pairs_if_needed(lp, nt):
    """Every REBAL days, scan all pairs and keep the best-reverting ones."""
    # Only re-select when the cache is empty, enough days have passed, or the day
    # counter went backwards (a safety check in case of a fresh run).
    if _cache["pairs"] is not None and nt - _cache["day"] < REBAL and nt >= _cache["day"]:
        return

    L = lp[:, -min(FW, nt - 1):]          # recent log-prices of the 50 names
    Lc = L - L.mean(1, keepdims=True)     # de-meaned, so we can compute covariances
    var = (Lc * Lc).sum(1)               # variance of each name over the window
    cand = []
    for i in range(50):                   # loop over every unordered pair (i, j)
        for j in range(i + 1, 50):
            if var[j] <= 0:
                continue
            b = (Lc[i] * Lc[j]).sum() / var[j]   # hedge ratio: how much of j hedges i
            if b <= 0:                            # ignore pairs that move oppositely
                continue
            s = L[i] - b * L[j]                   # the spread (gap) between the pair
            s = s - s.mean()                      # centre it on zero
            d0 = (s[:-1] ** 2).sum()
            if d0 <= 0:
                continue
            phi = (s[:-1] * s[1:]).sum() / d0     # how persistent the spread is (AR1)
            if 0 < phi < 1:                       # 0<phi<1 means it mean-reverts
                hl = -np.log(2) / np.log(phi)     # half-life: days to close half the gap
                if 1 < hl <= HLMAX:               # keep only reasonably fast reverters
                    cand.append((i, j, b, hl))

    cand.sort(key=lambda t: t[3])         # sort by half-life, fastest reverters first
    _cache["pairs"] = cand[:MAXP]         # keep the best MAXP pairs
    _cache["day"] = nt


def _pairs(lp, nt):
    """Turn each selected pair's current spread into per-name position weights."""
    _select_pairs_if_needed(lp, nt)
    zw = min(ZW, nt - 1)
    u = np.zeros(50)                       # weight to accumulate for each name
    for i, j, b, hl in _cache["pairs"]:
        s = lp[i, -zw:] - b * lp[j, -zw:]         # the spread over the recent window
        sd = s.std()
        if sd > 0:
            # z = how many std-devs the spread is stretched right now
            z = np.clip((s[-1] - s.mean()) / sd, -ZCLIP, ZCLIP)
            u[i] -= z                     # spread high -> name i is rich -> short it
            u[j] += b * z                 # ...and go long its partner j to hedge
    return u


def _residual(prc, nt):
    """Strip out common factors, then fade each name's recent private drift."""
    R = np.diff(np.log(prc[:, -(min(W, nt - 1) + 1):]), axis=1)   # daily returns
    X = R[1:] - R[1:].mean(1, keepdims=True)     # the 50 names, de-meaned
    # SVD finds the main shared movement patterns; the top KF rows of V are the
    # biggest common factors (already perpendicular to each other).
    V = np.linalg.svd(X, full_matrices=False)[2][:KF]
    resid = X - (X @ V.T) @ V             # subtract the common factors -> private wiggle
    rv = np.maximum(resid.std(1), 1e-9)   # each name's own noise level (avoid /0)
    # recent drift, scaled by noise; minus sign = fade it (mean-revert)
    return -(resid[:, -M:].sum(1) / (rv * np.sqrt(M)))


def _neutral(w):
    """Make a set of weights market-neutral and scale to one unit of exposure."""
    w = w - w.mean()                      # equal dollars long and short
    t = np.abs(w).sum()
    return w / t if t > 0 else w          # rescale so |weights| sum to 1


def getMyPosition(prcSoFar):
    """Called once per day with all prices so far; returns target share positions."""
    prc = np.asarray(prcSoFar, float)
    nt = prc.shape[1]                     # number of days of history available
    if nt < ZW + 3:                       # not enough history yet -> stay flat
        return np.zeros(prc.shape[0], dtype=int)

    lp = np.log(prc)[1:]                  # log-prices of the 50 names (drop ALGO)

    # Run both books, neutralise each, and blend them 70/30. This is the strategy.
    w = ALPHA * _neutral(_pairs(lp, nt)) + (1 - ALPHA) * _neutral(_residual(prc, nt))

    t = np.abs(w).sum()
    pos = np.zeros(prc.shape[0])
    if t > 0 and np.isfinite(t):          # safety: if weights are bad, stay flat
        # scale to GROSS dollars, then divide by price to convert dollars -> shares
        pos[1:] = GROSS * (w / t) / prc[1:, -1]   # index 0 (ALGO) stays 0 = flat

    # replace any bad values with 0 and round to whole shares
    return np.nan_to_num(pos).astype(int)
