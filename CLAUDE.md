# ANVI Traders — Algothon 2026 team brief

This file is read automatically by Claude Code at session start. It contains everything
this team has measured and learned. **Read the graveyard before proposing any signal,
and follow the methodology rules for anything new.** Numbers below were all reproduced
on `prices.txt` with scripts that existed at time of writing (`eval.py` for the leaderboard
window, `validate.py` for the three-window worst-case check).

## The competition in 10 lines

- Implement `getMyPosition(prcSoFar)` in a file named after the registered team, zipped flat.
- Input: prices, shape (51, nt). Instrument 0 = "ALGO" = equal-weight index of the other 50
  (verified: corr 0.993, beta 1.006, OOS R² 0.988). Output: 51 integer share positions.
- Score = mu · SR²/(SR²+1) with SR = √250·mu/σ of daily PnL. Negative mean is returned RAW
  (losses get no shrinkage). Scale-covariant: 2× positions = 2× score → deploy capital.
  SR 1 keeps 50% of mean, SR 2 keeps 80%, SR 3 keeps 90%.
- Limits: $10k/name, $100k ALGO (max possible gross ≈ $600k). Commission 1bp names, 0.2bp ALGO.
- $25k minimum total traded volume or auto-zero (NOT enforced in eval.py — check locally).
- Grading sandbox: numpy/pandas/scipy/sklearn/statsmodels/matplotlib ONLY, Python 3.12,
  no network. Pure-numpy submissions need no requirements.txt. Never redeclare the six.
- Timeline: Testing Round scored on hidden days 501–750 (leaderboard = same fixed window,
  1 submission/day). General Round Jul 16–30 on FRESH data (this is the round that counts).
  Finalists Aug 3. Finals Aug 13: **50% of finals score is a live methodology presentation to SIG.**

## eval.py mechanics (verified to the cent — see hand-prediction test)

- Full price history from day 0 is passed on every call (`startDay = nt − 250`,
  slices start at column 0). There is NO cold start.
- 250 days scored; entry day unscored; final day is a mark with no trade.
- Commission is charged with a ONE-DAY LAG (yesterday's commission hits today's cash).
  Any local harness must replicate this or Sharpe won't match. `validate.py` does.
- `prices.txt` has a header row; `header=0` in loadPrices is correct for it.
- The `ret` printout is cosmetic. Local harness reproduces eval.py exactly
  (`validate.py` evalW score matches `eval.py` to the cent).

## Current state (as of Jul 15)

Hidden-window submission log (fixed window, days 501–750):

| date  | strategy                          | score | mean PL | std  |
|-------|-----------------------------------|-------|---------|------|
| 09/07 | pairs+residual blend, MAXP=25     | 46.6  | +94.4   | 1509 |
| 12/07 | same, MAXP=20 (teammate tweak)    | 56.1  | +100.5  | 1413 |
| 13/07 | simple 20d cross-sectional book   | −8.9  | −8.9    | 1251 |
| 14/07 | spillover engine (pure)           | _record result_ | | |
| 15/07 | engine + 15% pairs combo (planned dress rehearsal) | | | |

Leaders run mean PL ~800 at std ~1600–1900 on the hidden window (Sharpe ~5–8).

**July 16 candidate: spillover engine (now with EWMA-beta market-strip) + 15% pairs-blend
combo** (see below), re-validate on the fresh data before submission. Implemented in
`ANVI_Traders.py`; local numbers reproduce via `python validate.py`.

**Clean-A/B ordering (rule 8).** Two isolated changes are staged as separate commits so the
leaderboard can attribute each: (a) engine + EWMA-beta only — one controlled change vs the
14/07 pure engine; (b) the +15% pairs combo on top. Submit in that order, one per day.

## Validated results (what works, with the evidence)

**1. Spillover engine — the primary edge.** Dense aggregated lead-lag on market-stripped
residuals. No single cross-asset link is significant; ~800 of 2,450 above a 1-SE floor
aggregate into a real prediction. Proof: matrix estimated on H1 and FROZEN predicts H2
with IC +0.045 (≈5σ on 12,450 obs); element-wise matrix stability H1↔H2 = +0.11.
Local (pure, expanding beta): evalW 420 / A 251 / B 406 (worst 251). Design choices (each
swept one-at-a-time on 3 windows, selected on WORST window):
- Residuals = returns minus estimated-beta × ALGO. Do NOT strip PCA factors — the signal
  lives in the factor/cluster structure; PCA-5 residuals collapse it (worst 17 vs 246).
- **EWMA-weighted beta beats equal-weight (NEW, Jul 15).** Per-name betas drift ~15%
  between halves (mean |Δ|=0.145, cross-sectional corr 0.78), so a recency-weighted
  market-strip residualizes the current regime more cleanly. EWMA half-life 120d lifts
  every window: evalW 420→447, A 251→258, B 406→422 (worst 251→258). Robust across
  half-lives 90–150; hard rolling windows and slower/faster EWMAs are mixed or worse.
  Drift-checked (rule 4): it also helps the UP-market window A, so it is signal quality,
  not drift. (reproduce: `python validate.py`)
- Hard 1-SE floor. Soft-threshold (93), no floor (209), 2-SE (29) all worse.
- Expanding estimation window for the spillover matrix (rolling-250 loses, 318 vs 389).
- Sign sizing at full $10k caps + 30% hysteresis. tanh (204) and rank (178) lose deployment.
  A median-threshold 25/25 split + MAX_HOLD hysteresis cap tested neutral (worst +1) and was
  dropped to keep the engine a single controlled change vs the last submission.
- Lag-2 term hurts (152). Own-lag (diagonal) excluded.

**2. Pairs+residual blend (MAXP=20)** — the only strategy with PROVEN positive hidden-window
transfer (+100.5 mean). 70% cointegrated pairs (top 20 by half-life, 1–15d, reselected
every 20d) + 30% PCA-5 residual fade, 700k gross. Basis: 67/1225 pairs with 3–7d train
half-lives that persist OOS.

**3. The combo (July 16 shape): engine 85% + pairs blend 15% at position level.**
Correlation between the two books: −0.07. Combo beats engine alone on ALL windows.
- Original (expanding-beta engine): 450 / 288 / 437 vs 420 / 251 / 406.
- Current (EWMA-beta engine, `ENGINE_W=0.85`, MAXP=20): evalW 458 / A 283 / B 437
  (worst 283) vs engine-alone 447 / 258 / 422 (worst 258). Sharpe 5.92; turnover falls
  (95M vs 112M) as the books offset each other's churn. (reproduce: `python validate.py`)
- Blend-weight sweep is consistent (all of 80/85/90 beat engine on every window); local
  worst-window marginally prefers 80/20 (293) over 85/15 (283), but that is within noise —
  do NOT parameter-climb the fixed window (rule 8); confirm the weight on the oracle.

**4. MAXP 20 > 25** — validated on all 3 local windows AND confirmed on the hidden window
(56.1 vs 46.6). First demonstration that local ORDERING transfers even though levels don't.

## Graveyard — dead hypotheses. Do NOT re-propose without new evidence.

- **20-day cross-sectional reversal**: strong on days 1–500 (local 57.5), scored **−8.9 on
  hidden days 501–750**. Structure rotated across the day-500 boundary. Anything built on
  this signal (incl. the ALGO tilt's laggard allocation) is suspect on unseen data.
- **ALGO own-autocorrelation / AR(1) timing**: null THREE ways — lag-1 = +0.007; multi-lag
  unstable between halves; the trailing 60/120 AR(1) ensemble walk-forward gives hit-rate
  51.1%, IC +0.019 (SE 0.051). A trailing autocorr estimate on 60 days has SE ±0.13:
  readings of ±0.09 that "flip between regimes" are the ruler shaking, not regimes.
  The AR(1) sleeve's backtest profits were market-drift capture (ALGO fell ~20% on evalW
  and B, rose on A where the sleeve made +4/day). Drift ≠ alpha.
  **Re-confirmed Jul 15**: an AR(1) sleeve deployed on ALGO's $100k scored +26% on evalW but
  its ALGO-only PnL was +90/day (evalW) and +129/day (B, both down markets) vs +6/day on A
  (up market) — textbook drift. By worst-window selection it HURT (259→237). Reverted; ALGO
  stays a hedge. If you feel tempted again, run the A-window drift check first.
- **ALGO-vs-basket spread**: not stationary (ADF p≈0.9, half-life 119d). No index arb.
- **Top-pairs lead-lag**: individual entries are noise and collapse OOS. (The AGGREGATE is
  real — see engine. Test dense structures with aggregate statistics, never top-N.)
- **Lead-lag at lags 2–10**: noise (max |corr| ≈ expected max under null; top-5 fail OOS).
- **Tight/mirror pairs**: none exist. Max |return corr| 0.47; most negative −0.04; zero
  spreads with half-life < 1 day.
- **Calendar / day-of-week**: per-instrument phase patterns ANTI-correlate between halves.
- **Price-level structure**: no persistent trends (slope corr −0.22 between halves), no
  cycles (the "500-day period" is a detrended-random-walk artifact), 0/51 stationary levels.
- **ML (GBM/RF, 15 features)**: OOS IC ≈ shuffled-target control. No nonlinear edge found.
- **Regime-switching lag-1 beta**: beta corr between halves +0.08 ≈ noise; 22/50 sign flips.
  (NB: this is the LAGGED predictive beta — distinct from the contemporaneous market-strip
  beta in result #1, whose cross-sectional ordering is stable at +0.78.)
- **Conditional reversion after large moves**: sign flips between halves.
- **OU s-score (Avellaneda-Lee)**: no improvement over crude residual drift (est. noise).
- **Inverse-vol risk weighting**: hurts at scale (concentrates into caps).
- **EWMA-vol standardization of residuals**: only 2/51 names show volatility clustering
  (Ljung-Box on squared returns), so recency-weighting the vol adds noise not signal — helps
  the recent window, hurts the earlier one. Regime-dependent, not robust. (Distinct from the
  EWMA-*beta* in result #1, which is robust.)
- **Quantile books (top/bottom-N at cap)**: N is window-luck (N=10 spiked, collapsed on
  disjoint window).
- **ALGO tilt (laggard-longs vs ALGO short)**: real, drift-proof alpha on days 1–500
  (+50–66/day per 100k, worked in up AND down windows) — retired because its allocation
  signal is the dead 20-day reversal. Revisit only with a live allocation signal.

## Methodology rules (non-negotiable)

1. **No capital without measurement.** Every idea — from a teammate, from Claude, from
   another team's file — gets a walk-forward test on `prices.txt` before deployment.
2. **Three disjoint windows**: evalW (days 250–500), A (100–300), B (300–500). Reset all
   strategy state between windows. **Select on the WORST window**, average second.
   `validate.py` runs all three and prints the worst-window score.
3. **Levels don't transfer; orderings do.** Observed compression ≈ 5× (271 local → 56
   hidden). Use local backtests to RANK closely-related variants, never to predict scores.
4. **Decompose drift.** Any component with directional exposure must profit in up AND down
   market windows (ALGO fell ~20% on evalW/B, rose +7.6% on A — use A as the up-market test).
5. **Aggregate vs max.** Dense-weak structures are invisible to top-N tests. Check both.
6. **Respect sampling error.** Autocorr on 60d: SE ±0.13. IC on N obs: SE 1/√N. Below
   ~2–3 SE it isn't real. In-sample IC shrinks ~8× OOS here (0.37 → 0.045).
7. **No number enters a docstring without a reproducing script that still exists.**
   AI-generated numbers (any Claude session included) are hypotheses until reproduced.
   The engine/combo numbers above reproduce via `python validate.py`.
8. **The leaderboard is an oracle, one query per day.** One controlled change per
   submission (the MAXP 25→20 A/B was perfect). Never parameter-climb the fixed window.
9. **Never submit another team's code or a close derivative.** Ideas may be studied;
   implementations must be independent, improved, and defensible live at finals.
10. **Log everything.** Update the submission table and this file when results land —
    this document is the skeleton of the finals presentation.

## Data facts (days 1–500)

- Names: annualised vol 18–65% (median 33.5%); ALGO 15.7% (below every name — it's the index).
- Factor structure: top PCA factor ≈ 20% of variance, next five ≈ 19%.
- ALGO drift by window: evalW −19.7%, A +7.6%, B −20.0% (for drift-decomposition tests).
- 67/1225 pairs cointegrate with 3–7d half-lives that persist out of sample.
- Per-name betas to ALGO: mean 0.98, range [0.50, 1.68], avg regression R² ≈ 0.20; betas
  drift ~15% between halves but keep cross-sectional order (corr 0.78) — motivates EWMA beta.
