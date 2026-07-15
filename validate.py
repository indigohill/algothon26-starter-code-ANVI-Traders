#!/usr/bin/env python
"""
validate.py - local walk-forward scorer for ANVI Traders strategies.

Replicates eval.py's accounting EXACTLY - one-day commission lag, integer-share
position clipping, population-std Sharpe - and scores a strategy on the team's
three disjoint windows with worst-window selection (methodology rule 2):

    evalW = days 250-500   (down market: ALGO -19.7%)
    A     = days 100-300   (up market:   ALGO  +7.6%)   <- drift-decomposition test
    B     = days 300-500   (down market: ALGO -20.0%)

Strategy module-level state (hysteresis, pair cache) is reset between windows by
loading the strategy file fresh for each window.

Usage:
    python validate.py                 # scores ANVI_Traders.py
    python validate.py path/to/alt.py  # scores another strategy file
"""
import sys
import itertools
import importlib.util

import numpy as np
import pandas as pd

PRICES_FILE = "prices.txt"
DEFAULT_STRATEGY = "ANVI_Traders.py"

# Competition constants (mirror eval.py exactly).
DEFAULT_COMM = 0.0001      # 1bp on the names
INST0_COMM = 0.00002       # 0.2bp on ALGO
DEFAULT_LIMIT = 10_000     # $ per-name position limit
INST0_LIMIT = 100_000      # $ ALGO position limit
SCORE_PARAM = 1.0

# (label, price-history cutoff in days, number of scored test days)
WINDOWS = [
    ("evalW (250-500, down)", 500, 250),
    ("A     (100-300, up)",   300, 200),
    ("B     (300-500, down)", 500, 200),
]

_ctr = itertools.count()


def load_prices(fn=PRICES_FILE):
    return pd.read_csv(fn, sep=r"\s+", header=0, index_col=None).values.T


def load_strategy(path):
    """Load a fresh module instance so module-level state resets between runs."""
    spec = importlib.util.spec_from_file_location(f"strat{next(_ctr)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.getMyPosition


def score(mu, sigma, param=SCORE_PARAM):
    """Competition score: mu * SR^2 / (SR^2 + param^2); negative mean returned raw."""
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr ** 2 / (sr ** 2 + param ** 2)
    return mu * frac


def walk_forward(get_position, prc, num_test_days):
    """Exact replica of eval.py's calcPL. Returns (mean, std, sharpe, score, dvol)."""
    n_inst, nt = prc.shape
    comm = np.full(n_inst, DEFAULT_COMM); comm[0] = INST0_COMM
    limit = np.full(n_inst, DEFAULT_LIMIT); limit[0] = INST0_LIMIT

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    value = 0.0
    day_comm = 0.0                       # charged with a one-day lag, like eval.py
    tot_dvol = 0.0
    pnl = []
    start_day = nt - num_test_days

    for t in range(start_day, nt + 1):
        hist = prc[:, :t]
        px = hist[:, -1]
        if t < nt:
            raw = get_position(hist)
            lim = (limit / px).astype(int)
            new_pos = np.clip(raw, -lim, lim).astype(int)
        else:
            new_pos = np.array(cur_pos)      # final day is a mark, no trade

        d_pos = new_pos - cur_pos
        cash -= px.dot(d_pos) + day_comm     # yesterday's commission hits today
        dvol = px * np.abs(d_pos)
        tot_dvol += dvol.sum()
        day_comm = (dvol * comm).sum()
        cur_pos = np.array(new_pos)
        pos_value = cur_pos.dot(px)
        today_pl = cash + pos_value - value
        value = cash + pos_value
        if t > start_day:
            pnl.append(today_pl)

    pnl = np.array(pnl)
    mu, sd = pnl.mean(), pnl.std()
    sr = np.sqrt(250) * mu / sd if sd > 0 else 0.0
    return mu, sd, sr, score(mu, sd), tot_dvol


def main(path=DEFAULT_STRATEGY):
    prc_full = load_prices()
    n_inst, n_days = prc_full.shape
    print(f"strategy: {path}   data: {n_inst} x {n_days}\n")
    print(f"{'window':24s} {'mean':>8s} {'std':>8s} {'Sharpe':>7s} {'score':>8s} {'turnover':>10s}")
    scores = []
    for label, cut, ntd in WINDOWS:
        get_position = load_strategy(path)       # fresh state per window
        prc = prc_full[:, :cut].copy()
        mu, sd, sr, sc, dv = walk_forward(get_position, prc, ntd)
        scores.append(sc)
        print(f"{label:24s} {mu:8.1f} {sd:8.0f} {sr:7.2f} {sc:8.1f} {dv / 1e6:9.1f}M")
    print(f"\nWORST-WINDOW score (rule 2 selection): {min(scores):.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STRATEGY)
