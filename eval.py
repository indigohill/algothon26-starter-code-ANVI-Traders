#!/usr/bin/env python
"""Algothon 2026 evaluation script.

Participants: write getMyPosition(prcSoFar) in teamName.py and update the imports below
"""

import numpy as np
import pandas as pd
from ANVI_Traders import getMyPosition as getPosition

nInst = 0
nt = 0

pricesFile = "./prices.txt"
numTestDays = 250

# parameter for scoring function
scoreDefaultParam = 1.0

# commission rates (0.0001 = 1bp)
# SPECIAL rate for instrument 0
defaultCommRate = 0.0001
inst0CommRate = 0.00002

# position limits (dollars)
# SPECIAL position limit for instrument 0
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

def loadPrices(fn):
    """
    Load prices from csv file (one instrument per column) and transpose into one instrument per row
    """
    global nt, nInst
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    nt, nInst = df.shape
    return (df.values).T


def chargeFees(dvolumes, commRate):
    """
    Total commission for one day's trades.
    """
    return np.sum(dvolumes * commRate)


def score(mu, sigma, param=scoreDefaultParam):
    """
    Final score from the daily-PnL mean & std, plus a scoring parameter.
    """
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac

prcAll = loadPrices(pricesFile)
print(f"Loaded {nInst} instruments for {nt} days")

# initialise the per-instrument commissions and position limits
commRate = np.full(nInst, defaultCommRate)
commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, defaultDlrPosLimit)
dlrPosLimit[0] = inst0DlrPosLimit

def calcPL(prcHist, numTestDays):
    """
    Function to loop over days and calculate/store PnLs
    """
    
    # initial values
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    comm = 0
    
    todayPLL = []
    _, nt = prcHist.shape
    # start day is the first day to run getPosition() on
    # e.g. startDay=500 if last 250 of 750 days used as test days
    startDay = nt - numTestDays
    
    for t in range(startDay, nt + 1):
        # price history up to and including t, e.g. if t=500, gets first 500 days
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]

        # trading loop, do not do it on the very last day of the test
        if t < nt:
            # get new positions
            newPosOrig = getPosition(prcHistSoFar)

            # clip to position limits, and enforce integer shares
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            # the final day is only used as 'mark' of final PnL
            newPos = np.array(curPos)

        # change in positions
        deltaPos = newPos - curPos
        
        cash -= curPrices.dot(deltaPos) + comm

        # calculate commissions
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = chargeFees(dvolumes, commRate)
            
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        # PnL is the daily change in portfolio value (cash plus positions)
        todayPL = cash + posValue - value
        
        value = cash + posValue

        # calculate return (portfolio value over total dollar volume)
        ret = 0.0
        if totDVolume > 0:
            ret = value / totDVolume

        # only score for test days
        if t > startDay:
            print(
                f"Day {t} value: {value:.2f} todayPL: ${todayPL:.2f} $-traded: {totDVolume:.0f} return: {ret:.5f}"
            )
            todayPLL.append(todayPL)
            
    pll = np.array(todayPLL)
    plmu, plstd = (np.mean(pll), np.std(pll))

    # calculate annualised Sharpe
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd
        
    return (plmu, ret, plstd, annSharpe, totDVolume)

meanpl, ret, plstd, sharpe, dvol = calcPL(prcAll, numTestDays)
scoreVal = score(meanpl, plstd, scoreDefaultParam)
print("=====")
print(f"mean(PL): {meanpl:.1f}")
print(f"return: {ret:.5f}")
print(f"StdDev(PL): {plstd:.2f}")
print(f"annSharpe(PL): {sharpe:.2f}")
print(f"totDvolume: {dvol:.0f}")
print(f"Score: {scoreVal:.2f}")
