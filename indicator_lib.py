# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
from sklearn.linear_model import LinearRegression


# =============================================================================
# Heiken Ashi Candles
# =============================================================================
def heikenashi(o, h, l, c):
    HAc = (o + h + l + c) / 4
    HAo, HAh, HAl = HAc.copy(), HAc.copy(), HAc.copy()

    for i in range(1, o.shape[0]):
        HAo[i] = (HAo[i - 1] + HAc[i - 1]) / 2
        HAh[i] = np.array((h[i], HAo[i], HAc[i])).max()
        HAl[i] = np.array((l[i], HAo[i], HAc[i])).min()

    return HAo, HAh, HAl, HAc


# =============================================================================
# compute period for adaptive methods
# =============================================================================
def hilbert(series):
    """
    Hilbert transformation
    :param series: (np.array) price
    :return: (np.array) InPhase and Quadrature term
    """
    Q = 0.0962 * series + 0.5796 * np.roll(series, 2) \
        - 0.5796 * np.roll(series, 4) - 0.0962 * np.roll(series, 6)
    Q[0:6] = 0
    I = np.roll(series, 3)
    I[0:3] = 0
    return Q, I


def compute_period(series, cutoff):
    smooth = sma4(series)
    cycle = highpass2pole(smooth, cutoff)
    for i in range(2, 7):
        cycle[i] = (series[i] - 2 * series[i - 1] + series[i - 2]) / 4
    delta_phase = np.zeros_like(series)
    inst_period = np.zeros_like(series)
    period = np.zeros_like(series)
    Q, I = hilbert(cycle)
    for i in range(6, series.shape[0]):
        Q[i] *= 0.5 + 0.08 * inst_period[i - 1]
        if Q[i] != 0 and Q[i - 1] != 0:
            delta_phase[i] = (I[i] / Q[i] - I[i - 1] / Q[i - 1]) / (1 + I[i] * I[i - 1] / (Q[i] * Q[i - 1]))
            delta_phase[i] = max(0.1, min(1.1, delta_phase[i]))
        median_delta = np.median(delta_phase[i - 4:i + 1])
        if median_delta == 0:
            DC = 15
        else:
            DC = 6.29318 / median_delta + 0.5
        inst_period[i] = 0.33 * DC + 0.67 * inst_period[i - 1]
        period[i] = 0.15 * inst_period[i] + 0.85 * period[i - 1]
    return period, cycle


# =============================================================================
# filters
# =============================================================================
def sma4(series):
    """
    4 term simple moving average
    :param series: (np.array) price
    :return: (np.array) smoothed price
    """
    newseries = (series + 2 * np.roll(series, 1) + 2 * np.roll(series, 2)
                 + np.roll(series, 3)) / 6
    newseries[:3] = series[:3]
    return newseries


def ema(series, cutoff):
    """
    exponential moving average
    alpha = 1/(lag+1)
    :param series: (np.array) price
    :param cutoff: (float) cutoff period of the filter
    :return: (np.array) filtered price
    """
    K = 1
    alpha = 1 + (np.sin(2 * np.pi * K / cutoff) - 1) / np.cos(2 * np.pi * K / cutoff)
    for i in range(1, series.shape[0]):
        series[i] = alpha * series[i] \
                        + (1 - alpha) * series[i - 1]
    return series


def regularized_ema(series, cutoff):
    """
    add an additional penalty term to enhance filter effect while introducing no more lag
    :param series: (np.array) price
    :param cutoff: (float) cutoff period of the filter
    :return: (np.array) filtered price
    """
    K = 1
    alpha = 1 + (np.sin(2 * np.pi * K / cutoff) - 1) / np.cos(2 * np.pi * K / cutoff)
    l = np.exp(0.16 / alpha)
    newseries = np.copy(series)
    for i in range(2, series.shape[0]):
        newseries[i] = alpha / (1 + l) * series[i] \
                       + (1 - alpha - 2 * l) / (1 + l) * newseries[i - 1] - l / (l + 1) * newseries[i - 2]
    return newseries


def lowpass2pole(series, cutoff):
    """
    2 pole low-pass filter
    :param series: (np.array) price
    :param cutoff: (float) cutoff period of the filter
    :return: (np.array) filtered price
    """
    K = 1.414
    alpha = 1 + (np.sin(2 * np.pi * K / cutoff) - 1) / np.cos(2 * np.pi * K / cutoff)
    for i in range(2, series.shape[0]):
        series[i] = alpha ** 2 * series[i] \
                        + 2 * (1 - alpha) * series[i - 1] + (1 - alpha) ** 2 * series[i - 2]
    return series


def decycler(series, cutoff):
    """
    subtract high frequency component from the original series to decycle
    :param series: (np.array) price
    :param cutoff: (float) cutoff period of the filter
    :return: (np.array) decycled series
    """
    K = 1
    alpha = 1 + (np.sin(2 * np.pi * K / cutoff) - 1) / np.cos(2 * np.pi * K / cutoff)
    newseries = np.copy(series)
    for i in range(1, series.shape[0]):
        newseries[i] = alpha / 2 * (series[i] + series[i - 1]) \
                       + (1 - alpha) * newseries[i - 1]
    return newseries


def highpass(series, cutoff):
    """
    (1 pole) high-pass filter
    :param series: (np.array) price
    :param cutoff: (float) cutoff period of the filter
    :return: (np.array) filtered price
    """
    K = 1
    alpha = 1 + (np.sin(2 * np.pi * K / cutoff) - 1) / np.cos(2 * np.pi * K / cutoff)
    newseries = np.copy(series)
    for i in range(1, series.shape[0]):
        newseries[i] = (1 - alpha / 2) * series[i] - (1 - alpha / 2) * series[i - 1] \
                       + (1 - alpha) * newseries[i - 1]
    return newseries


def highpass2pole(series, cutoff):
    """
    2 pole high-pass filter
    :param series: (np.array) price
    :param cutoff: (float) cutoff period of the filter
    :return: (np.array) filtered price
    """
    K = 0.707
    alpha = 1 + (np.sin(2 * np.pi * K / cutoff) - 1) / np.cos(2 * np.pi * K / cutoff)
    newseries = np.copy(series)
    for i in range(2, series.shape[0]):
        newseries[i] = (1 - alpha / 2) ** 2 * series[i] \
                       - 2 * (1 - alpha / 2) ** 2 * series[i - 1] \
                       + (1 - alpha / 2) ** 2 * series[i - 2] \
                       + 2 * (1 - alpha) * newseries[i - 1] - (1 - alpha) ** 2 * newseries[i - 2]
    return newseries


def ad_highpass2pole(series, lag):
    """
    2 pole adaptive high-pass filter (variable cutoff period)
    :param series: (np.array) price
    :param lag: (np.array) lag of the filter
    :return: (np.array) filtered price
    """
    alpha = 1 / (1 + lag)
    newseries = np.copy(series)
    for i in range(2, series.shape[0]):
        newseries[i] = (1 - alpha[i] / 2) ** 2 * series[i] \
                       - 2 * (1 - alpha[i] / 2) ** 2 * series[i - 1] \
                       + (1 - alpha[i] / 2) ** 2 * series[i - 2] \
                       + 2 * (1 - alpha[i]) * newseries[i - 1] - (1 - alpha[i]) ** 2 * newseries[i - 2]
    return newseries


def supersmoother2pole(series, cutoff):
    """
    simplified 2 pole butterworth smoother
    :param series: (np.array) price
    :param cutoff: (float) cutoff period of the filter
    :return: (np.array) smoothed price
    """
    a = np.exp(-1.414 * np.pi / cutoff)
    b = 2 * a * np.cos(1.414 * np.pi / cutoff)
    newseries = np.copy(series)
    for i in range(2, series.shape[0]):
        newseries[i] = (1 + a ** 2 - b) / 2 * (series[i] + series[i - 1]) \
                       + b * newseries[i - 1] - a ** 2 * newseries[i - 2]
    return newseries


def supersmoother3pole(series, cutoff):
    """
    simplified 3 pole butterworth smoother
    :param series: (np.array) price
    :param cutoff: (float) cutoff period of the filter
    :return: (np.array) smoothed price
    """
    a = np.exp(-np.pi / cutoff)
    b = 2 * a * np.cos(1.738 * np.pi / cutoff)
    c = a ** 2
    newseries = np.copy(series)
    for i in range(3, series.shape[0]):
        newseries[i] = (1 - c ** 2 - b + b * c) * series[i] \
                       + (b + c) * newseries[i - 1] + (-c - b * c) * newseries[i - 2] + (c ** 2) * newseries[i - 3]
    return newseries


def allpass(series, alpha):
    """
    allpass filter (used in laguerre filter)
    :param series: (np.array) price
    :param alpha: (float) damping factor
    :return: (np.array) filtered price
    """
    newseries = np.copy(series)
    for i in range(1, series.shape[0]):
        newseries[i] = -alpha * series[i] + series[i - 1] \
                       + alpha * newseries[i - 1]
    return newseries


def laguerre(series, gamma):
    """
    Laguerre filter
    :param series: (np.array) price
    :param gamma: (float) damping factor
    :return: (np.array) filtered price
    """
    l0 = ema(series, gamma)
    l1 = allpass(l0, gamma)
    l2 = allpass(l1, gamma)
    l3 = allpass(l2, gamma)
    return l0, l1, l2, l3


def roofing(series, cutoff_hp, cutoff_lp):
    hp = highpass2pole(series, cutoff_hp)
    newseries = supersmoother3pole(hp, cutoff_lp)
    return newseries


# =============================================================================
# indicators transformer
# =============================================================================
def stoch(series, period):
    """
    stochaticize the indicator
    :param series: (np.array) indicator or price
    :param period: (int) window length
    :return: (np.array) series of value within [0,1]
    """
    df = pd.Series(series)
    df_max = df.rolling(period, min_periods=1).max()
    df_min = df.rolling(period, min_periods=1).min()
    series = (df - df_min) / (df_max - df_min)
    return series.to_numpy()


def fisher(series, period, stoch_time):
    """
    fisher transformer
    :param series: (np.array) indicator or price
    :param period: (int) window length
    :param stoch_time: (int) number of time applying stochastic transformation
    :return: (np.array) normalized series satisfying statistic inference of normally distributed data
    """
    # stochaticize
    for i in range(stoch_time):
        series = stoch(series, period)
    # transform data to [-0.9999,0.9999]
    series = 2 * (series - 0.5)
    for i in range(series.shape[0]):
        series[i] = max(-0.9999, min(0.9999, series[i]))
    # apply fisher transformation
    series = np.log((1 + series) / (1 - series)) / 2
    return series


def inverse_fisher(series, amplifying_factor):
    """
    inverse fisher transformer: serve as a soft limiter
    :param series: (np.array) indicator or price
    :param amplifying_factor: (double>1) if the indicator is already between 1 and -1
        we can amplifying it before inverse transformation to get the best of the soft limiter
    """
    return (np.exp(amplifying_factor*series)-1)/(np.exp(amplifying_factor*series)+1)


def cube(series):
    """
    cube transformation: for compressing the squiggles near 0
    :param series: (np.array) indicator or price
    """
    return series**3


# =============================================================================
# signal generator
# =============================================================================
def lag_signal(indicator, lag):
    # generate lag
    lag1 = np.roll(indicator, lag)
    lag1[0] = indicator[0]
    # generate signal
    signal = (indicator > lag1) * 1 - (indicator < lag1) * 1
    return signal


def fix_channel_break(indicator, up=2, mid=0, dn=-2):
    """
    fix channel break signal generator
    :param indicator: (np.array) indicator series
    :param up: (float) buy when cross over
    :param mid: (float) close out current position
    :param dn: (float) sell short when cross under
    :return: signal
    """
    signal = np.zeros_like(indicator)
    for i in range(indicator.shape[0]):
        if indicator[i] > up or (signal[i - 1] > 0 and indicator[i] > mid):
            signal[i] = 1
        if indicator[i] < dn or (signal[i - 1] < 0 and indicator[i] < mid):
            signal[i] = -1
    return signal


def lead(series, alpha1, alpha2):
    """
    lead indicator
    :param series: (np.array) price
    :param alpha1: (float) alpha to generate lead
    :param alpha2: (float) alpha to smooth while offsetting some lead
    :return: (np.array) netlead
    """
    assert alpha1 < alpha2
    lead = np.zeros_like(series)
    netlead = np.zeros_like(series)
    for i in range(1, series.shape[0]):
        lead[i] = 2 * series[i] + (alpha1 - 2) * series[i - 1] \
                  + (1 - alpha1) * lead[i - 1]
        netlead[i] = alpha2 * lead[i] + (1 - alpha2) * netlead[i - 1]
    return netlead


# =============================================================================
# combined signal
# =============================================================================
def itrend_bandpass(series, cutoff, cycle, bandwidth):
    i = i_trend(series, cutoff)[0]
    b = bandpass(series, cycle, bandwidth)[0]
    buy = i & b
    sell = i | b
    return (buy+sell)/2,

def adm_bandpass(series, cutoff_period, cutoff_signal, cycle, bandwidth):
    return ad_momentum(series, cutoff_period, cutoff_signal)[0] & bandpass(series, cycle, bandwidth)[0],

def roof_rsi(series, cutoff_hp, cutoff_lp, length):
    roof = roofing(series, cutoff_hp, cutoff_lp)
    return rsi(roof, length)[0],
# =============================================================================
# indicators - trend
# =============================================================================
def i_trend(series, cutoff):
    """
    instantaneous trendline
    :param series: (np.array) price
    :param cutoff: (float) cutoff period of the hp
    :return: (np.array) signal, trend and its trigger
    """
    # compute inst trend
    K = 0.707
    alpha = 1 + (np.sin(2 * np.pi * K / cutoff) - 1) / np.cos(2 * np.pi * K / cutoff)
    it = np.copy(series)
    for i in range(2, 7):
        it[i] = (series[i] + 2 * series[i - 1] + series[i - 2]) / 4
    for i in range(7, series.shape[0]):
        it[i] = (alpha - alpha ** 2 / 4) * series[i] \
                + alpha ** 2 / 2 * series[i - 1] \
                - (alpha - alpha ** 2 * 3 / 4) * series[i - 2] \
                + 2 * (1 - alpha) * it[i - 1] - (1 - alpha) ** 2 * it[i - 2]

    # compute lead 2 trigger & signal
    lag2 = np.roll(it, 20)
    lag2[:20] = it[:20]
    trigger = 2 * it - lag2
    signal = (trigger > it) * 1 - (trigger < it) * 1
    return signal, it, trigger


def ad_momentum(series, cutoff_period, cutoff_signal):
    """
    smoothed adaptive momentum indicator
    compare the price in the current cycle with that in the previous cycle (same phase)
    to indicate an uptrend or downtrend
    :param series: (np.array) price
    :param cutoff_period: (float) the cutoff period used to compute the period using Hilbert Transformation
    :param cutoff_signal: (float) the cutoff period used to smooth the signal
    :return: (np.array) signal & momentum
    """
    period, _ = compute_period(series, cutoff_period)
    period = period.astype(np.int)
    momen = np.zeros_like(series)
    for i in range(series.shape[0]):
        if (i - period[i]) >= 0:
            momen[i] = series[i] - series[i - period[i]]
    momen = supersmoother3pole(momen, cutoff_signal)
    signal = (momen > 0) * 1 - (momen < 0) * 1
    return signal, momen


# =============================================================================
# indicators - oscillator
# =============================================================================
def decycler_oscillator(series, cutoff1, times):
    """
    decycler oscillator
    take the difference of 2 decyclers with different cutoff
    :param series: (np.array) price
    :param cutoff1: (float) the smaller cutoff period
    :param times: (float, >1) larger cutoff / smaller cutoff
    :return: (np.array) signal & indicator
    """
    cutoff2 = cutoff1 * times
    hp1 = highpass(series, cutoff1)
    hp2 = highpass(series, cutoff2)
    delta_hp = hp2 - hp1
    # >0: uptrend, <0: downtrend
    signal = (delta_hp > 0) * 1 - (delta_hp < 0) * 1
    return signal, delta_hp


def bandpass(series, cycle, bandwidth):
    """
    bandpass filter
    :param series: (np.array) price
    :param cycle: (float) cycle period
    :param bandwidth: (float, >0 <2) length between left and right cutoffs / cycle period
    :return: (np.array) signal & indicator
    """
    # pass a HP to avoid spectral dilation of BP
    hp = highpass(series, 4 * cycle / bandwidth)
    # bandpass filter
    lmd = np.cos(2 * np.pi / cycle)
    gamma = np.cos(2 * np.pi * bandwidth / cycle)
    sigma = 1 / gamma - np.sqrt(1 / gamma ** 2 - 1)
    bp = np.copy(hp)
    for i in range(2, series.shape[0]):
        bp[i] = (1 - sigma) / 2 * hp[i] - (1 - sigma) / 2 * hp[i - 2] \
                + lmd * (1 + sigma) * bp[i - 1] - sigma * bp[i - 2]
    # fast attack-slow decay AGC
    K = 0.991
    peak = np.copy(bp)
    for i in range(series.shape[0]):
        if i > 0:
            peak[i] = peak[i - 1] * K
        if abs(bp[i]) > peak[i]:
            peak[i] = abs(bp[i])
    bp_normalized = bp / peak
    # trigger(lead) & signal
    trigger = highpass(bp_normalized, cycle / bandwidth / 1.5)
    signal = (bp_normalized < trigger) * 1 - (trigger < bp_normalized) * 1
    return signal, bp, bp_normalized, trigger


def cci(series, cutoff1, cutoff2, fperiod=None, stoch_time=None):
    """
    CCI - cyber cycle index
    delay is less than half a cycle: buy when signal cross under lag1, sell when signal cross over lag1
    need a 'stop loss' strategy, close out when profit<0 and bars since entry > 8(period)
    :param series: (np.array) price
    :param cutoff1: (float) cutoff period for hp
    :param cutoff2: (float) cutoff period for ema
    :param period, stoch_time: (tuple: ((int)period, (int)stoch_time)) fisher transformation parameters
                        if = empty tuple, no fisher transformation
    :return: (np.array) trading signal & cycle
    """
    # compute the cycle
    smooth = sma4(series)
    cycle = highpass2pole(smooth, cutoff1)
    for t in range(2, 7):
        cycle[t] = (series[t] - 2 * series[t - 1] + series[t - 2]) / 4
    signal = ema(cycle, cutoff2)
    # apply fisher transformation
    if fperiod != None:
        signal = fisher(signal, fperiod, stoch_time)
    return lag_signal(-signal, 1), signal


def ad_cci(series, cutoff_period, cutoff_signal, fperiod=None, stoch_time=None):
    """
    adaptive cyber cycle
    :param series: (np.array) price
    :param cutoff_period: (float) the cutoff period used to compute the period using Hilbert Transformation
    :param cutoff_signal: (float) the cutoff period used to smooth the signal
    :param period: (int) fisher transformation parameter
    :param stoch_time: (int) fisher transformation parameter
    :return: (np.array) trading signal & cycle
    """
    # compute period
    period, _ = compute_period(series, cutoff_period)
    # compute the cycle
    smooth = sma4(series)
    cycle = ad_highpass2pole(smooth, period)
    for t in range(2, 7):
        cycle[t] = (series[t] - 2 * series[t - 1] + series[t - 2]) / 4
    signal = ema(cycle, cutoff_signal)
    # apply fisher transformation
    if fperiod:
        signal = fisher(signal, fperiod, stoch_time)
    return lag_signal(-signal, 1), signal


def cg(series, length, fperiod=None, stoch_time=None):
    """
    CG - center of gravity
    view the price as weight to compute the center of gravity of the filter
    :param series: (np.array) price
    :param length: (int) length of the filter
    :param period: (int) fisher transformation parameter
    :param stoch_time: (int) fisher transformation parameter
    :return: (np.array) trading signal & cg
    """
    # compute cg
    num = np.zeros_like(series)
    denom = np.ones_like(series)
    for i in range(length - 1, series.shape[0]):
        num[i] = np.sum((np.array(range(length)) + 1) * series[i - length + 1:i + 1][::-1])
        denom[i] = np.sum(series[i - length + 1:i + 1])
    cg = -num / denom + (1 + length) / 2
    # apply fisher transformation
    if fperiod:
        cg = fisher(cg, fperiod, stoch_time)
    return lag_signal(cg, 1), cg


def ad_cg(series, cutoff_period, fperiod=None, stoch_time=None):
    """

    :param series: (np.array) price
    :param cutoff_period: (float) the cutoff period used to compute the period using Hilbert Transformation
    :param period: (int) fisher transformation parameter
    :param stoch_time: (int) fisher transformation parameter
    :return: (np.array) trading signal & cg
    """
    # compute period
    period, _ = compute_period(series, cutoff_period)
    length = (period / 2).astype(np.int)
    # compute cg
    num = np.zeros_like(series)
    denom = np.ones_like(series)
    for i in range(length[1] - 1, series.shape[0]):
        num[i] = np.sum((np.array(range(length[i])) + 1) * series[i - length[i] + 1:i + 1][::-1])
        denom[i] = np.sum(series[i - length[i] + 1:i + 1])
    cg = -num / denom + (1 + length) / 2
    # apply fisher transformation
    if fperiod:
        cg = fisher(cg, fperiod, stoch_time)
    return lag_signal(cg, 1), cg


def rvi(o, h, l, c, length, fperiod=None, stoch_time=None):
    """
    RVI - relative vigor index
    :param o: (np.array) open
    :param h: (np.array) high
    :param l: (np.array) low
    :param c: (np.array) close
    :param length: length to sum the num & denom
    :param period: (int) fisher transformation parameter
    :param stoch_time: (int) fisher transformation parameter
    :return: (np.array) signal & rvi
    """
    co = c - o
    hl = h - l
    num = sma4(co)
    denom = sma4(hl)
    rvi = np.zeros_like(o)
    for i in range(2 + length, o.shape[0]):
        rvi[i] = np.sum(num[i - length + 1:i + 1]) / np.sum(denom[i - length + 1:i + 1])
    # apply fisher transformation
    if fperiod:
        rvi = fisher(rvi, fperiod, stoch_time)
    return lag_signal(rvi, 1), rvi


def rsi(series, length, fperiod=None, stoch_time=None):
    """
    Relative Strength Index
    :param series: (np.array) price
    :param length: length to sum the num & denom
    :param period: (int) fisher transformation parameter
    :param stoch_time: (int) fisher transformation parameter
    :return: (np.array) signal & rsi
    """
    rsi = ta.RSI(series, length)
    # apply fisher transformation
    if fperiod:
        rsi = fisher(rsi, fperiod, stoch_time)
    return lag_signal(rsi, 1), rsi


def laguerre_rsi(series, gamma, up=2, mid=0, dn=-2, fperiod=None, stoch_time=None):
    """
    Laguerre RSI
    :param series: (np.array) price
    :param gamma: (float) damping factor
    :param period: (int) fisher transformation parameter
    :param stoch_time: (int) fisher transformation parameter
    :param up: (float) fixed channel parameter
    :param mid: (float) fixed channel parameter
    :param dn: (float) fixed channel parameter
    :return: (np.array) signal & rsi
    """
    l0, l1, l2, l3 = laguerre(series, gamma)
    rsi = np.zeros_like(series)
    for i in range(series.shape[0]):
        cu = 0
        cd = 0
        if l1[i] > l0[i]:
            cu += l1[i] - l0[i]
        else:
            cd -= l1[i] - l0[i]
        if l2[i] > l1[i]:
            cu += l2[i] - l1[i]
        else:
            cd -= l2[i] - l1[i]
        if l3[i] > l2[i]:
            cu += l3[i] - l2[i]
        else:
            cd -= l3[i] - l2[i]
        rsi[i] = cu / (cu + cd)
    # apply fisher transformation
    if fperiod:
        rsi = fisher(rsi, fperiod, stoch_time)
    signal = fix_channel_break(rsi, up, mid, dn)
    return signal, rsi


def sinewave(series, cutoff_period, lead=0.25 * np.pi):
    """
    sinewave indicator
    :param series: (np.array) price
    :param cutoff_period: (float) the cutoff period used to compute the period using Hilbert Transformation
    :param lead: (float) lead angle, in radians
    :return: (np.array) signal, sinewave and leadsine
    """
    # compute period
    period, cycle = compute_period(series, cutoff_period)
    dcperiod = period.astype(np.int)
    # compute dominant cycle phase
    real = np.zeros_like(series)
    imag = np.zeros_like(series)
    dcphase = np.zeros_like(series)
    for i in range(series.shape[0]):
        for j in range(dcperiod[i]):
            real[i] += np.sin(2 * np.pi * j / dcperiod[i]) * cycle[i]
            imag[i] += np.cos(2 * np.pi * j / dcperiod[i]) * cycle[i]
        if abs(imag[i] > 0.001):
            dcphase[i] = np.arctan(real[i] / imag[i])
        else:
            dcphase[i] = 0.5 * np.pi * np.sign(real[i])
        dcphase[i] += 0.5 * np.pi
        if imag[i] < 0:
            dcphase[i] += np.pi
        if dcphase[i] > 1.75 * np.pi:
            dcphase[i] -= 2 * np.pi
    # compute sinewave
    sinewave = np.sin(dcphase)
    leadsine = np.sin(dcphase + lead)
    signal = (leadsine > sinewave) * 1 - (leadsine < sinewave) * 1
    return signal, sinewave, leadsine


def better_sinewave(series, hp_period, lp_period, upper_bound, lower_bound):
    """
    the even better sinewave indicator: profits better when the market is in a trend mode
    :param hp_period: highpass period to remove the trend from the original price
    :param lp_period: lowpass period to smooth the original price
    :param upper_bound: long when the wave is above the upper_bound
    :param lower_bound: short when the wave is below the lower_bound
    """
    HP = highpass(series, hp_period)
    filt = supersmoother2pole(HP, lp_period)
    wave = (filt + np.roll(filt,1) + np.roll(filt,2))/3
    wave[0] = filt[0]
    wave[1] = (filt[0]+filt[1])/2
    pwr = (filt**2 + np.roll(filt,1)**2 + np.roll(filt,2)**2)/3
    pwr[0] = filt[0]**2
    pwr[1] = (filt[0]**2+filt[1]**2)/2
    wave = wave/np.sqrt(pwr)
    wave[np.isnan(wave)] = 0
    signal = (wave>upper_bound)*1 - (wave<lower_bound)*1
    return signal, wave


def corr_periodogram(series, hp_period, lp_period, average_len, max_lag, min_lag):
    """
    auto-correlation periodogram indicator: a preferred method to compute the dominant cycle
    :param hp_period: highpass period to remove the trend from the original price
    :param lp_period: lowpass period to smooth the original price
    :param average_len: # period to compute auto-correlation
    :param max_lag: max lag of period to compute auto_correlation
    :param min_lag: min lag of period to compute auto_correlation
    """
    num = np.zeros_like(series)
    denom = np.zeros_like(series)
    for lag in range(min_lag,max_lag):
        
        rho = corr(series, hp_period, lp_period, average_len, lag)
        cos_part = np.zeros_like(series)
        sin_part = np.zeros_like(series)
        for i in range(max_lag):
            cos_part += np.roll(rho,i)*np.cos(2*np.pi*i/max_lag)
            sin_part += np.roll(rho,i)*np.sin(2*np.pi*i/max_lag)
        sqsum = cos_part**2+sin_part**2
        
        for i in range(1, sqsum.shape[0]):
            sqsum[i] = 0.2 * sqsum[i] + 0.8 * sqsum[i - 1]
            
        K = 0.991
        peak = np.copy(sqsum)
        for i in range(sqsum.shape[0]):
            if i > 0:
                peak[i] = peak[i - 1] * K
            if abs(sqsum[i]) > peak[i]:
                peak[i] = abs(sqsum[i])
        sqsum = sqsum / peak
        sqsum[np.isnan(sqsum)] = 0
        
        num += (sqsum>0.5)*sqsum*lag
        denom += (sqsum>0.5)*sqsum
        
    dc = num/denom
    dc[np.isnan(dc)] = 0
    return lag_signal(dc, 1), dc
        

def dft(series, hp_period, lp_period, dft_period):
    """
    discrete Fourier Transformation 
    :param hp_period: highpass period to remove the trend from the original price
    :param lp_period: lowpass period to smooth the original price
    :param dft_period: period to compute Fourier transformation
    """
    HP = highpass2pole(series, hp_period)
    filt = supersmoother2pole(HP, lp_period)
    
    cos_part = np.zeros_like(series)
    sin_part = np.zeros_like(series)
    for i in range(dft_period):
        cos_part += np.roll(filt,i)*np.cos(2*np.pi*i/dft_period)/dft_period
        sin_part += np.roll(filt,i)*np.sin(2*np.pi*i/dft_period)/dft_period
    pwr = cos_part**2+sin_part**2
    
    K = 0.991
    peak = np.copy(pwr)
    for i in range(series.shape[0]):
        if i > 0:
            peak[i] = peak[i - 1] * K
        if abs(pwr[i]) > peak[i]:
            peak[i] = abs(pwr[i])
    pwr = pwr / peak
    pwr[np.isnan(pwr)] = 0
    return pwr,


def dft_cg(series, hp_period, lp_period, max_period, min_period):
    """
    center of gravity indicator based on discrete Fourier tansformation: indicates dominant cycle
    :param hp_period: highpass period to remove the trend from the original price
    :param lp_period: lowpass period to smooth the original price
    :param max_period: max period to compute Fourier transformation
    :param min_period: min period to compute Fourier transformation
    """
    num = np.zeros_like(series)
    denom = np.zeros_like(series)
    for period in range(min_period, max_period):
        pwr = dft(series, hp_period, lp_period, period)[0]
        num += (pwr>0.5)*pwr*period
        denom += (pwr>0.5)*pwr
    dc = num/denom
    dc[np.isnan(dc)] = 0
    return lag_signal(dc, 1), dc


def comb(series, hp_period, lp_period, max_period, min_period, bandwidth):
    """
    comb filter spectral estimate: compute the dominant cycle
    :param hp_period: highpass period to remove the trend from the original price
    :param lp_period: lowpass period to smooth the original price
    :param max_period: max period to compute bandpass
    :param min_period: min period to compute bandpass
    :param bandwidth: bandwidth for bandpass filter
    """
    num = np.zeros_like(series)
    denom = np.zeros_like(series)
    HP = highpass2pole(series, hp_period)
    filt = supersmoother2pole(HP, lp_period)
    for period in range(min_period, max_period):
        
        _, bp, _, _ =  bandpass(filt, period, bandwidth)
        pwr = np.zeros_like(bp)
        for i in range(period):
            pwr+=np.roll(bp,i)**2/period**2
            
        K = 0.991
        peak = np.copy(pwr)
        for i in range(pwr.shape[0]):
            if i > 0:
                peak[i] = peak[i - 1] * K
            if abs(pwr[i]) > peak[i]:
                peak[i] = abs(pwr[i])
        pwr = pwr / peak
        pwr[np.isnan(pwr)] = 0
        
        num += (pwr>0.5)*pwr*period
        denom += (pwr>0.5)*pwr
        
    dc = num/denom
    dc[np.isnan(dc)] = 0
    return lag_signal(dc, 1), dc


def hilbert_indicator(series, hp_period, lp_period, smooth_period):
    """
    hilbert transformation indicator: the real line moves as the original price, while the imaginary line as predictor
    :param series: (np.array) price
    :param hp_period: highpass period to remove the trend from the original price
    :param lp_period: lowpass period to smooth the original price
    :param smooth_period: lowpass period to smooth the imaginary line
    """
    HP = highpass2pole(series, hp_period)
    filt = supersmoother2pole(HP, lp_period)
    
    K = 0.991
    peak = np.copy(filt)
    for i in range(series.shape[0]):
        if i > 0:
            peak[i] = peak[i - 1] * K
        if abs(filt[i]) > peak[i]:
            peak[i] = abs(filt[i])
    real = filt / peak
    real[np.isnan(real)] = 0
    
    quad = real - np.roll(real,1)
    quad[0] = 0
    K = 0.991
    peak = np.copy(quad)
    for i in range(series.shape[0]):
        if i > 0:
            peak[i] = peak[i - 1] * K
        if abs(quad[i]) > peak[i]:
            peak[i] = abs(quad[i])
    quad = quad / peak
    quad[np.isnan(quad)] = 0

    imag = supersmoother2pole(quad, smooth_period)
    
    signal = (imag>real)*1 - (imag<real)*1
    return signal, real, imag
    

# =============================================================================
# turning point indicator
# =============================================================================
def corr(series, hp_period, lp_period, average_len, lag):
    """
    auto-correlation indicator: indicates reversal when it's near -1
    :param hp_period: highpass period to remove the trend from the original price
    :param lp_period: lowpass period to smooth the original price
    :param average_len: # period to compute auto-correlation
    :param lag: lag of period to compute auto_correlation
    """
    HP = highpass2pole(series, hp_period)
    filt = supersmoother2pole(HP, lp_period)
    corr = np.zeros_like(series)
    for i in range(average_len+lag, series.shape[0]+1):
        s1 = filt[i-average_len:i]
        s2 = filt[i-average_len-lag:i-lag]
        corr[i-1] = np.corrcoef(s1, s2)[0,1]
    return corr    


def corr_reversal(series, hp_period, lp_period, average_len, lag, thresh):
    """
    auto-correlation reversal indicatorL indicates the reversals of the price
    :param hp_period: highpass period to remove the trend from the original price
    :param lp_period: lowpass period to smooth the original price
    :param average_len: # period to compute auto-correlation
    :param lag: lag of period to compute auto_correlation
    :param: thresh: if the num of reversal-indicator delta happens more than thresh times in the lag period, 
                    it indicates a overall reversal
    """
    HP = highpass2pole(series, hp_period)
    filt = supersmoother2pole(HP, lp_period)
    corr = np.zeros_like(series)
    for i in range(average_len+lag, series.shape[0]+1):
        s1 = filt[i-average_len:i]
        s2 = filt[i-average_len-lag:i-lag]
        corr[i-1] = np.corrcoef(s1, s2)[0,1]
    corr_1 = corr>0
    corr_2 = np.roll(corr_1,1)
    delta = np.abs(corr_1-corr_2)/2
    delta[0] = 0
    sumdelta = np.zeros_like(delta)
    for i in range(lag):
        sumdelta += np.roll(delta, i)
    reversal = sumdelta>=thresh
    return reversal, sumdelta


def convolution(series, hp_period, lp_period, lookback_period):
    """
    convolution indicator: use convolution within a lookback period to determine whether a turning point has occurred
    :param series: (np.array) price
    :param hp_period: highpass period to remove the trend from the original price
    :param lp_period: lowpass period to smooth the original price
    :param lookback_period: lookback period to compute convolution
    """
    HP = highpass2pole(series, hp_period)
    filt = supersmoother2pole(HP, lp_period)
    
    corr = np.zeros_like(series)
    for i in range(lookback_period, series.shape[0]+1):
        lookback_ = filt[i-lookback_period:i]
        corr[i-1] = np.corrcoef(lookback_, np.flip(lookback_))[0,1]
    conv = (1+(np.exp(3*corr)-1)/(np.exp(3*corr)+1))/2
    
    return conv, corr
        
        

def get_weights(diff_amt, min_weight, max_window):
    """
    compute the weights for fractional differentiation
    :param diff_amt: (float) order of fractional differentiation
    :param min_weight: (float) lower bound for weights
    :param max_window: (int) upper bound for window length
    :return: (np.array) weights
    """
    weights = [1.]
    k, ctr = 1, 1
    while True:
        weights_ = -weights[-1] * (diff_amt - k + 1) / k
        if abs(weights_) < min_weight:
            break
        else:
            weights.append(weights_)
        k += 1
        ctr += 1
        if ctr == max_window:
            break
    return np.array(weights)


def ffd(series, diff_amt, min_weight=1e-5):
    """
    fractional differentiation (fixed window)
    :param series: (np.array) price
    :param diff_amt: (float) order of fractional differentiation
    :param min_weight: (float) lower bound for weights
    :return: (np.array) differentiated series
    """
    weights = get_weights(diff_amt, min_weight, series.shape[0])
    window = len(weights)
    frac_diff = np.full(series.shape[0], np.nan)
    for i in range(window-1, series.shape[0]):
        frac_diff[i] = np.sum(weights * series[i-window+1 : i+1])
    return frac_diff


def ffd_ma(series, diff_amt, ma_period, min_weight=1e-5):
    """
    moving average of fractional differentiation
    :param series: (np.array) price
    :param diff_amt: (float) order of fractional differentiation
    :param ma_period: (int) ma window length
    :param min_weight: (float) lower bound for weights
    :return: (np.array) ma of differentiated series
    """
    frac_diff = ffd(series, diff_amt, min_weight)
    feac_diff_ma = ta.SMA(frac_diff, ma_period)
    
    return feac_diff_ma


def HA_candle(o, h, l, c):
    """
    Heiken Ashi Candles
    :param o: (np.array) open
    :param h: (np.array) high
    :param l: (np.array) low
    :param c: (np.array) close
    :return: (np.array) HA ohlc
    """
    HAc = (o+h+l+c) / 4
    HAo, HAh, HAl = HAc.copy(), HAc.copy(), HAc.copy()

    for i in range(1, o.shape[0]):
        HAo[i] = (HAo[i - 1] + HAc[i - 1]) / 2
        HAh[i] = np.array((h[i], HAo[i], HAc[i])).max()
        HAl[i] = np.array((l[i], HAo[i], HAc[i])).min()

    return HAo, HAh, HAl, HAc


def diff(series):
    """
    differentiation
    :param series: (np.array) price
    :return: (np.array) differentiated series
    """
    series = series - np.roll(series, 1)
    series[0] = 0
    return series

