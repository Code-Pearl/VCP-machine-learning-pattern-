"""
Turns a window of daily OHLCV data into:
  1. a normalized feature VECTOR for the SOM (shape description), and
  2. an interpretable rule-based VCP score built from named sub-scores.

Both are derived from the same underlying swing/leg decomposition, so the
SOM's unsupervised clustering and the rule-based score are looking at the
same structure, just consuming it differently (numeric shape vs. named
criteria).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── swing / leg detection ────────────────────────────────────────────────

def zigzag_pivots(close: np.ndarray, pct_threshold: float = 0.035) -> list[tuple[int, float, str]]:
    """
    Percentage-threshold zigzag: the standard way to find swing highs/lows
    in price data without drowning in noise from tiny local peaks (which a
    naive argrelextrema-style detector would produce in a flat, choppy
    region). Returns [(index, price, 'H'|'L'), ...] in chronological order.

    A new pivot is only registered once price has moved pct_threshold away
    from the running extreme in the opposite direction.
    """
    n = len(close)
    if n < 3:
        return []

    pivots: list[tuple[int, float, str]] = []
    trend = None  # None -> 'up' -> 'down' -> 'up' ...
    extreme_idx, extreme_price = 0, close[0]

    for i in range(1, n):
        price = close[i]
        if trend is None:
            if price >= extreme_price * (1 + pct_threshold):
                trend = "up"
                extreme_idx, extreme_price = i, price
            elif price <= extreme_price * (1 - pct_threshold):
                trend = "down"
                extreme_idx, extreme_price = i, price
            else:
                if price > extreme_price:
                    extreme_idx, extreme_price = i, price
        elif trend == "up":
            if price > extreme_price:
                extreme_idx, extreme_price = i, price
            elif price <= extreme_price * (1 - pct_threshold):
                pivots.append((extreme_idx, float(extreme_price), "H"))
                trend = "down"
                extreme_idx, extreme_price = i, price
        else:  # trend == 'down'
            if price < extreme_price:
                extreme_idx, extreme_price = i, price
            elif price >= extreme_price * (1 + pct_threshold):
                pivots.append((extreme_idx, float(extreme_price), "L"))
                trend = "up"
                extreme_idx, extreme_price = i, price

    # trailing pending extreme, useful for the "tightening near the pivot" read
    pivots.append((extreme_idx, float(extreme_price), "H" if trend in ("up", None) else "L"))
    return pivots


@dataclass
class LegInfo:
    depth: float          # peak-to-trough drawdown, as a positive fraction
    peak_idx: int
    trough_idx: int
    avg_volume: float


def extract_pullback_legs(close: np.ndarray, volume: np.ndarray,
                           pivots: list[tuple[int, float, str]]) -> list[LegInfo]:
    """From a zigzag pivot sequence, pull out H->L (pullback) legs in
    chronological order, each with its drawdown depth and average volume
    over the leg."""
    legs: list[LegInfo] = []
    for k in range(len(pivots) - 1):
        idx1, price1, typ1 = pivots[k]
        idx2, price2, typ2 = pivots[k + 1]
        if typ1 == "H" and typ2 == "L" and price1 > 0:
            depth = (price1 - price2) / price1
            avg_vol = float(np.mean(volume[idx1:idx2 + 1])) if idx2 > idx1 else float(volume[idx1])
            legs.append(LegInfo(depth=depth, peak_idx=idx1, trough_idx=idx2, avg_volume=avg_vol))
    return legs


# ── feature vector + rule score ──────────────────────────────────────────

FEATURE_NAMES = [
    "n_legs",               # count of pullback legs found in the window (capped)
    "depth_ratio_mean",     # mean of consecutive leg-depth ratios (want < 1)
    "contraction_monotonic",# fraction of consecutive leg pairs that contracted
    "last_leg_depth",       # depth of the most recent pullback (want small)
    "volume_trend_slope",   # normalized slope of leg-avg-volume across legs (want negative)
    "volume_dryup_ratio",   # recent 10d avg vol / 50d avg vol (want < 1)
    "atr_contraction_ratio",# recent ATR% / prior ATR% (want < 1)
    "prior_advance_pct",    # % gain into the base (want sizably positive)
    "proximity_to_pivot",   # close / base-high (want close to 1, from below)
    "above_ma_score",       # trend-template-ish: close vs rising 50/150 SMA
]


@dataclass
class WindowFeatures:
    vector: np.ndarray                 # raw (unnormalized) feature vector, len(FEATURE_NAMES)
    rule_score: float                  # 0-1 interpretable composite
    sub_scores: dict = field(default_factory=dict)
    pivot_price: float = float("nan")
    n_legs_found: int = 0


def _atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, start: int, end: int) -> float:
    end = max(end, start + 1)
    rng = (high[start:end] - low[start:end]) / np.maximum(close[start:end], 1e-9)
    return float(np.mean(rng)) if end > start else float("nan")


def extract_window_features(df: pd.DataFrame, end_idx: int, lookback: int = 160,
                             pct_threshold: float = 0.035) -> WindowFeatures | None:
    """
    Extract VCP shape features for the window df.iloc[start:end_idx+1],
    where end_idx is the "as of today" bar (the candidate pivot date).
    Returns None if there isn't enough history in the window.
    """
    start = max(0, end_idx - lookback + 1)
    if end_idx - start < 60:
        return None

    close = df["Close"].values[start:end_idx + 1]
    high = df["High"].values[start:end_idx + 1]
    low = df["Low"].values[start:end_idx + 1]
    volume = df["Volume"].values[start:end_idx + 1]
    n = len(close)

    pivots = zigzag_pivots(close, pct_threshold=pct_threshold)
    legs = extract_pullback_legs(close, volume, pivots)
    # keep the most recent up-to-4 legs - VCP theory typically expects 2-4
    recent_legs = legs[-4:] if len(legs) > 4 else legs
    n_legs_found = len(recent_legs)

    depths = np.array([leg.depth for leg in recent_legs]) if recent_legs else np.array([])
    if len(depths) >= 2:
        ratios = depths[1:] / np.maximum(depths[:-1], 1e-9)
        depth_ratio_mean = float(np.mean(ratios))
        contraction_monotonic = float(np.mean(ratios < 1.0))
    else:
        depth_ratio_mean = 1.0
        contraction_monotonic = 0.0
    last_leg_depth = float(depths[-1]) if len(depths) else 0.30  # penalize "no legs found"

    leg_vols = np.array([leg.avg_volume for leg in recent_legs]) if recent_legs else np.array([])
    if len(leg_vols) >= 2:
        x = np.arange(len(leg_vols))
        slope = np.polyfit(x, leg_vols / max(np.mean(leg_vols), 1e-9), 1)[0]
        volume_trend_slope = float(slope)
    else:
        volume_trend_slope = 0.0

    recent10_vol = float(np.mean(volume[-10:])) if n >= 10 else float(np.mean(volume))
    recent50_vol = float(np.mean(volume[-50:])) if n >= 50 else float(np.mean(volume))
    volume_dryup_ratio = recent10_vol / max(recent50_vol, 1e-9)

    recent_atr = _atr_pct(high, low, close, n - 8, n)
    prior_atr = _atr_pct(high, low, close, max(0, n - 48), max(1, n - 8))
    atr_contraction_ratio = recent_atr / max(prior_atr, 1e-9) if np.isfinite(prior_atr) and prior_atr > 0 else 1.0

    base_start = recent_legs[0].peak_idx if recent_legs else max(0, n - 60)
    pre_base_price = close[max(0, base_start - 1)] if base_start > 0 else close[0]
    base_high = float(np.max(close[base_start:])) if base_start < n else float(close[-1])
    prior_advance_pct = (base_high - pre_base_price) / max(pre_base_price, 1e-9)

    proximity_to_pivot = float(close[-1] / max(base_high, 1e-9))

    if n >= 150:
        sma50 = float(np.mean(close[-50:]))
        sma150 = float(np.mean(close[-150:]))
        sma150_prev = float(np.mean(close[-160:-10])) if n >= 160 else sma150
        above_ma_score = float(
            (close[-1] > sma50) + (sma50 > sma150) + (sma150 > sma150_prev)
        ) / 3.0
    else:
        above_ma_score = 0.5  # not enough history to judge - neutral

    vector = np.array([
        min(n_legs_found, 4),
        depth_ratio_mean,
        contraction_monotonic,
        last_leg_depth,
        volume_trend_slope,
        volume_dryup_ratio,
        atr_contraction_ratio,
        prior_advance_pct,
        proximity_to_pivot,
        above_ma_score,
    ], dtype=float)

    # --- interpretable rule sub-scores (each in [0, 1]) ---
    def clip01(v):
        return float(np.clip(v, 0.0, 1.0))

    sub_scores = {
        "legs_present": clip01(n_legs_found / 3.0),
        "contraction": clip01(contraction_monotonic * 0.6 + (1.0 - clip01(depth_ratio_mean)) * 0.4),
        "tight_pullback": clip01(1.0 - last_leg_depth / 0.25),
        "volume_dryup": clip01(1.0 - volume_dryup_ratio) * 0.5 + clip01(-volume_trend_slope) * 0.5,
        "range_contraction": clip01(1.0 - atr_contraction_ratio),
        "prior_uptrend": clip01(prior_advance_pct / 0.30),
        "near_pivot": clip01(1.0 - abs(1.0 - proximity_to_pivot) / 0.15),
        "trend_template": above_ma_score,
    }
    weights = {
        "legs_present": 0.12, "contraction": 0.22, "tight_pullback": 0.13,
        "volume_dryup": 0.18, "range_contraction": 0.10, "prior_uptrend": 0.15,
        "near_pivot": 0.05, "trend_template": 0.05,
    }
    rule_score = float(sum(sub_scores[k] * weights[k] for k in weights))

    return WindowFeatures(
        vector=vector, rule_score=rule_score, sub_scores=sub_scores,
        pivot_price=base_high, n_legs_found=n_legs_found,
    )
