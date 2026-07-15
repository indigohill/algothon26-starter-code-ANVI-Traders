import numpy as np
import pandas as pd
 
nInst = 51
currentPos = np.zeros(nInst, dtype=int)
 
# --- signal parameters (swap this block out for your real signal later) ---
FAST, SLOW, VOL_LB = 3, 20, 20
THRESH = 0.3
DECAY = 0.8  # shrink toward flat when signal weakens
 
# --- position limits ---
DOLLAR_TARGET = np.full(nInst, 10000.0)
DOLLAR_TARGET[0] = 100000.0  # ALGO's cap is 10x every other instrument
 
# --- beta-hedge parameters ---
BETA_LOOKBACK = 60   # trailing window (days) used to estimate beta to ALGO
BETA_SHRINK = 0.5    # 0 = trust raw estimate fully, 1 = ignore data, assume beta=1.0
BETA_PRIOR = 1.0     # cross-sectional average beta observed in EDA (~0.93)
 
 
def estimate_betas(logret, lookback=BETA_LOOKBACK):
    """
    Rolling OLS beta of each instrument's returns against ALGO's returns
    (column 0), shrunk toward BETA_PRIOR for stability on short windows.
    """
    window = logret[-lookback:] if logret.shape[0] >= lookback else logret
    algo_r = window[:, 0]
    var_algo = np.var(algo_r) + 1e-12
 
    betas = np.zeros(logret.shape[1])
    for i in range(1, logret.shape[1]):
        cov = np.cov(window[:, i], algo_r)[0, 1]
        raw_beta = cov / var_algo
        betas[i] = BETA_SHRINK * BETA_PRIOR + (1 - BETA_SHRINK) * raw_beta
    return betas
 
 
def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape
    if nt < SLOW + 2:
        return np.zeros(nins, dtype=int)
 
    logp = np.log(prcSoFar.T)
    df = pd.DataFrame(logp)
 
    fastE = df.ewm(span=FAST, adjust=False).mean().iloc[-1].values
    slowE = df.ewm(span=SLOW, adjust=False).mean().iloc[-1].values
    logret = df.diff().dropna().values
    vol = np.nanstd(logret[-VOL_LB:], axis=0) + 1e-8
 
    # --- existing mean-reversion signal (replace with your real one) ---
    signal = (fastE - slowE) / vol
    z = np.clip(-signal / 3.0, -1, 1)
    active = np.abs(signal) > THRESH
 
    curPrices = prcSoFar[:, -1]
 
    # Step 1: raw dollar targets from the signal, for instruments 1..50
    rawDollar = np.zeros(nins)
    rawDollar[active] = z[active] * DOLLAR_TARGET[active]
 
    # Step 2: rolling betas to ALGO (need at least ~30 days of returns)
    betas = estimate_betas(logret) if nt >= 30 else np.ones(nins)
 
    # Step 3: net beta-dollar exposure across the whole book (excl. ALGO itself)
    netBetaDollars = np.sum(betas[1:] * rawDollar[1:])
 
    # Step 4: hedge the net exposure with a single ALGO trade, capped
    algoHedgeDollars = np.clip(-netBetaDollars, -DOLLAR_TARGET[0], DOLLAR_TARGET[0])
 
    # Step 5: convert dollar targets to share counts
    target = currentPos.copy().astype(float)
    target[active] = rawDollar[active] / curPrices[active]
    target[~active] *= DECAY  # no active signal: unwind toward flat
    target[0] = algoHedgeDollars / curPrices[0]  # ALGO position = pure hedge
 
    currentPos = target.astype(int)
    return currentPos