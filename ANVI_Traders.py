
import numpy as np

ENGINE_W = 0.85      # position-level blend weight on the engine book
REFILL   = 1.5       # cap-refill (holdout-confirmed +14%): rescale blended names toward caps

# ===================== ENGINE BOOK =====================
MIN_HISTORY = 60          # days of returns needed before trading
BETA_HL     = 120.0       # EWMA half-life (days) for the ALGO market-strip beta
FLOOR_SE    = 1.0         # noise floor on matrix entries, in standard errors
HYSTERESIS  = 0.15        # weak-signal band that keeps the prior direction
ASSET_CAP   = 10_000.0    # per-name dollar limit (competition rule)
ALGO_CAP    = 100_000.0   # ALGO dollar limit (competition rule)

_state = {"prev_dir": None, "last_nt": None}


def _reset_hysteresis():
    _state["prev_dir"] = None


def _engine_pos(prcSoFar):
    prc = np.asarray(prcSoFar, float)
    n, nt = prc.shape
    pos = np.zeros(n)

    # fresh run detection: if the day counter went backwards, reset hysteresis
    if _state["last_nt"] is not None and nt < _state["last_nt"]:
        _reset_hysteresis()
    _state["last_nt"] = nt

    R = np.diff(np.log(prc), axis=1)              # daily log returns (51, nt-1)
    T = R.shape[1]
    if T < MIN_HISTORY or n != 51:
        return pos.astype(int)

    # 1. market-strip with an EWMA (recency-weighted) beta per name
    ra = R[0]                                     # ALGO = the market
    X = R[1:]
    lam = 0.5 ** (1.0 / BETA_HL)
    w = lam ** np.arange(T - 1, -1, -1); w = w / w.sum()   # weights, recent-heavy
    rac = ra - (ra * w).sum()                     # weighted-demeaned ALGO
    var_a = (w * rac * rac).sum() + 1e-18         # weighted var(ALGO)
    Xc = X - (X * w).sum(1, keepdims=True)         # weighted-demeaned names
    beta = ((Xc * w) @ rac) / var_a               # weighted cov(name, ALGO) / var
    res = X - beta[:, None] * ra[None, :]         # market-neutral residuals

    # 2. standardize each residual series over its history
    m = res.mean(1, keepdims=True)
    sd = np.maximum(res.std(1, keepdims=True), 1e-12)
    Z = (res - m) / sd

    # 3. spillover matrix: does i today predict j tomorrow?
    P, Q = Z[:, :-1], Z[:, 1:]
    n_obs = P.shape[1]
    C = P @ Q.T / n_obs
    np.fill_diagonal(C, 0.0)                      # own-lag excluded
    C = np.where(np.abs(C) >= FLOOR_SE / np.sqrt(n_obs), C, 0.0)

    # 4. aggregate prediction, demeaned (market-neutral)
    sig = C.T @ Z[:, -1]
    sig = sig - sig.mean()

    # If the floor zeroed the whole matrix (or the prediction is otherwise
    # degenerate), there is no signal - stay flat rather than force spurious
    # positions off np.sign(0).
    if not np.any(np.abs(sig) > 1e-12):
        _reset_hysteresis()
        return pos.astype(int)

    # 5. sign sizing at full caps, with a hysteresis band that keeps weak-signal
        #   names at their prior direction to control commission churn.
    tgt = np.sign(sig)
    if _state["prev_dir"] is not None:
        weak = np.abs(sig) < HYSTERESIS * (np.mean(np.abs(sig)) or 1.0)
        tgt = np.where(weak, _state["prev_dir"], tgt)
    _state["prev_dir"] = np.sign(tgt).copy()

    px = prc[:, -1]
    caps = (ASSET_CAP / px[1:]).astype(int)       # whole-share cap per name
    pos[1:] = np.clip(np.rint(tgt * caps), -caps, caps)

    # 6a. backstop: sign sizing can leave a net dollar imbalance. Trim the
    #     weakest-signal names on the heavy side so the net never exceeds what
    #     ALGO can hedge (and the book never carries a naked directional bet).
    dollars = pos[1:] * px[1:]
    net = dollars.sum()
    order = np.argsort(np.abs(sig))               # weakest signal first
    k = 0
    while abs(net) > 0.999 * ALGO_CAP and k < 50:
        idx = order[k]
        if pos[1 + idx] != 0 and np.sign(dollars[idx]) == np.sign(net):
            net -= dollars[idx]
            pos[1 + idx] = 0.0
            dollars[idx] = 0.0
        k += 1

    # 6b. ALGO hedges the remaining net dollar imbalance (0.2bp commission).
    hedge = np.clip(-net, -0.999 * ALGO_CAP, 0.999 * ALGO_CAP)
    algo_cap_sh = int(ALGO_CAP / px[0])
    pos[0] = np.clip(np.trunc(hedge / px[0]), -algo_cap_sh, algo_cap_sh)

    return np.nan_to_num(pos).astype(int)

# ===================== PAIRS / RESIDUAL BOOK =====================

# ---------------------------------------------------------------------------
# Tuning knobs. These were chosen by out-of-sample validation. Everything the
# strategy does can be re-tuned here without touching the logic below.
# ---------------------------------------------------------------------------
ALPHA = 0.70        # blend weight: 70% pairs book, 30% residual book
GROSS = 700_000     # target gross dollar exposure (how much capital we deploy)

# Pairs-book settings:
ZW    = 30          # window (days) used to measure how "stretched" a spread is now
MAXP  = 20          # keep at most this many pairs (the best-reverting ones)
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
    """#Every REBAL days, scan all pairs and keep the best-reverting ones."""
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


def _pairs_pos(prcSoFar):
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

# ===================== COMBO =====================
def getMyPosition(prcSoFar):
    prc = np.asarray(prcSoFar, float)
    n, nt = prc.shape
    px = prc[:, -1]
    pe = _engine_pos(prc).astype(float)
    pp = _pairs_pos(prc).astype(float)
    if n != 51:
        return np.zeros(n, dtype=int)

    # blend the NAMES books at position level (share blend == dollar blend, same px)
    names = ENGINE_W * pe[1:] + (1.0 - ENGINE_W) * pp[1:]
    caps = (ASSET_CAP / px[1:]).astype(int)
    pos = np.zeros(n)
    # cap refill (holdout-confirmed): rescale the blended book up and let the
    # caps clip it - reclaims the deployment the blend dilutes.
    pos[1:] = np.clip(np.rint(REFILL * names), -caps, caps)

    # net-trim (added 16/07): refill scales imbalances too, and the blended net
    # occasionally exceeded ALGO's $100k hedge cap. Zero the weakest blended
    # positions on the heavy side until the net is fully hedgeable - the book
    # never carries naked directional exposure.
    dollars = pos[1:] * px[1:]
    net = dollars.sum()
    order = np.argsort(np.abs(names))             # weakest blended conviction first
    k = 0
    while abs(net) > 0.999 * ALGO_CAP and k < 50:
        idx = order[k]
        if pos[1 + idx] != 0 and np.sign(dollars[idx]) == np.sign(net):
            net -= dollars[idx]
            pos[1 + idx] = 0.0
            dollars[idx] = 0.0
        k += 1

    # ALGO hedges the blended book's residual net dollar imbalance
    net = (pos[1:] * px[1:]).sum()
    hedge = np.clip(-net, -0.999 * ALGO_CAP, 0.999 * ALGO_CAP)
    algo_cap_sh = int(ALGO_CAP / px[0])
    pos[0] = np.clip(np.trunc(hedge / px[0]), -algo_cap_sh, algo_cap_sh)
    return np.nan_to_num(pos).astype(int)