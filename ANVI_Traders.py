"""
ANVI Traders - cross-asset spillover engine (Algothon 2026).

The signal: a dense web of weak "spillover" effects - asset i's move today
carries faint information about asset j's move tomorrow. No single link is
statistically significant on its own (which is why top-pairs tests miss it),
but 800+ of them aggregated into one prediction are decisively real. The effect
was validated with a strict train/test diagnostic: a spillover matrix estimated
on the FIRST half of the data and frozen predicts the SECOND half with IC ~ 0.045
(about 5 sigma from zero), confirming it is real and out-of-sample. That frozen
test is a diagnostic, not the deployed estimator. In live trading below we do the
causal, data-efficient thing instead - re-estimate the matrix each day on ALL
history available so far (an expanding walk-forward). This uses strictly past
prices (no look-ahead), and because the noise floor is measured in standard
errors it auto-tightens as the sample grows: early, small-sample matrices are
filtered hard - often to nothing, in which case we stay flat - and only sharpen
over time. Ordering-based tuning was done on three disjoint local windows,
selected on worst-window score (levels don't transfer to unseen data; orderings
do).

Pipeline each day:
  1. Residualize each of the 50 names against ALGO using an ESTIMATED regression
     beta, EWMA-weighted (half-life BETA_HL days) toward recent data. ALGO is the
     equal-weight index of the names (verified: corr 0.993), so this strips the
     common market move properly. Betas drift ~15% between the two halves of the
     data, so a recency-weighted beta residualizes the recent regime more cleanly
     than an equal-weight full-history beta - it lifts the score in every tested
     window, robustly across half-lives from 90 to 150 days. We deliberately do
     NOT remove PCA factors - testing showed the spillover web lives in and around
     the factor/cluster structure, and factor-stripping destroys it (worst-window
     score 17 vs 246).
  2. Standardize each residual series over the expanding history.
  3. Estimate the 50x50 lagged cross-correlation matrix (i today -> j tomorrow)
     and zero every entry below a 1-standard-error noise floor. Harder floors
     (2SE) or softer ones (soft-thresholding, none) all validated worse. If the
     floor zeros everything, the signal is empty and we stay flat this day.
  4. Predicted next-day residual per name = matrix^T . today's residuals, demeaned
     (market-neutral).
  5. Size by SIGN at the full $10k per-name cap - deploying the most capital,
     which the score rewards (it pays deployed mean at high Sharpe). A 30%
     hysteresis band keeps weak signal flips at their prior direction to control
     commission churn.
  6. ALGO hedges the net dollar imbalance that integer rounding leaves (cheapest
     hedge available: 0.2bp commission, $100k cap). As a backstop, if the names
     book's net would exceed ALGO's hedge capacity we first trim the weakest names
     on the heavy side until it fits, so the book can never carry a naked bet.
     NB: an ALGO own-autocorrelation (AR(1)) timing sleeve was tested here and
     REJECTED - its apparent edge was market-drift capture, not alpha (null on a
     walk-forward: hit-rate 51%, IC +0.019 +/- 0.051). See the team-brief graveyard.

NumPy only - no other packages required.
"""

import numpy as np

MIN_HISTORY = 60          # days of returns needed before trading
BETA_HL     = 120.0       # EWMA half-life (days) for the ALGO market-strip beta
FLOOR_SE    = 1.0         # noise floor on matrix entries, in standard errors
HYSTERESIS  = 0.30        # weak-signal band that keeps the prior direction
ASSET_CAP   = 10_000.0    # per-name dollar limit (competition rule)
ALGO_CAP    = 100_000.0   # ALGO dollar limit (competition rule)

_state = {"prev_dir": None, "last_nt": None}


def _reset_hysteresis():
    _state["prev_dir"] = None


def getMyPosition(prcSoFar):
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
    #    names at their prior direction to control commission churn.
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
