"""
Synthetic daily OHLCV generator for VCP (Volatility Contraction Pattern) research.

Generates price/volume series with KNOWN ground truth so the detector can be
evaluated honestly instead of just eyeballed. Two families:

  - 'vcp'      : prior uptrend -> N contracting pullback/recovery legs
                 (each pullback shallower than the last, volume drying up) ->
                 tightening near the pivot -> optional breakout.
  - negatives  : random_walk, strong_uptrend_no_pullback, choppy_equal_swings,
                 downtrend, expanding_volatility_top - chosen specifically to
                 stress-test the parts of a VCP detector that are easy to get
                 wrong (e.g. "any uptrend" or "any pullback" triggering a
                 false positive).

This is synthetic data. It is useful for validating that the pattern-matching
logic behaves the way VCP theory says it should - it is NOT a substitute for
testing against real historical data before trusting this for anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class SeriesMeta:
    symbol: str
    pattern: str                    # 'vcp' or the negative archetype name
    is_vcp: bool
    pivot_idx: int | None = None    # index of the resistance/pivot high (None if not applicable)
    base_start_idx: int | None = None
    base_end_idx: int | None = None
    breakout_idx: int | None = None
    n_legs: int | None = None


def _phase_prices(start_price: float, n_days: int, daily_drift: float, daily_vol: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Geometric random walk with drift for one phase. Returns close prices."""
    rets = rng.normal(daily_drift, daily_vol, n_days)
    return start_price * np.cumprod(1 + rets)


def _make_ohlcv(closes: np.ndarray, base_volume: float, vol_noise: float,
                 vol_multiplier: np.ndarray, rng: np.random.Generator,
                 intraday_range_pct: float = 0.012) -> pd.DataFrame:
    """Build a plausible OHLCV frame from a close-price path."""
    n = len(closes)
    opens = np.empty(n)
    opens[0] = closes[0] * (1 + rng.normal(0, intraday_range_pct * 0.3))
    opens[1:] = closes[:-1] * (1 + rng.normal(0, intraday_range_pct * 0.3, n - 1))
    ranges = np.abs(rng.normal(intraday_range_pct, intraday_range_pct * 0.4, n))
    ranges = np.clip(ranges, 0.002, None)
    highs = np.maximum(opens, closes) * (1 + ranges * rng.uniform(0.3, 1.0, n))
    lows = np.minimum(opens, closes) * (1 - ranges * rng.uniform(0.3, 1.0, n))
    volume = base_volume * vol_multiplier * np.exp(rng.normal(0, vol_noise, n))
    volume = np.clip(volume, base_volume * 0.05, None)
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume
    })


def generate_vcp_series(symbol: str, rng: np.random.Generator, total_days: int = 500,
                         breakout: bool | None = None) -> tuple[pd.DataFrame, SeriesMeta]:
    base_price = rng.uniform(20, 150)
    base_volume = rng.uniform(5e5, 3e6)

    # Phase 1: prior flat accumulation
    p1_days = rng.integers(30, 70)
    p1 = _phase_prices(base_price, p1_days, 0.0002, 0.014, rng)

    # Phase 2: Stage-2 advance (the move a VCP forms after)
    p2_days = rng.integers(70, 160)
    advance_drift = rng.uniform(0.0025, 0.006)
    p2 = _phase_prices(p1[-1], p2_days, advance_drift, 0.017, rng)

    # Phase 3: N contracting legs
    n_legs = int(rng.integers(2, 5))
    base_depth = rng.uniform(0.22, 0.35)
    decay = rng.uniform(0.55, 0.72)
    leg_prices = []
    leg_boundaries = []  # (peak_idx_in_leg_prices_concat, trough_idx)
    cursor_price = p2[-1]
    for i in range(n_legs):
        depth = base_depth * (decay ** i) * rng.uniform(0.85, 1.15)
        depth = float(np.clip(depth, 0.02, 0.5))
        pullback_days = int(rng.integers(6, 22))
        recovery_days = int(rng.integers(6, 20))
        peak_price = cursor_price
        trough_target = peak_price * (1 - depth)
        pullback_drift = (np.log(trough_target / peak_price)) / pullback_days
        pullback = _phase_prices(peak_price, pullback_days, pullback_drift, 0.012, rng)
        # recovery back toward (not necessarily past) the peak; each leg's
        # recovery high creeps slightly lower than the prior one, tightening.
        recovery_target = peak_price * rng.uniform(0.90, 1.0) * (0.97 + 0.03 * (i / max(n_legs - 1, 1)))
        recovery_drift = (np.log(recovery_target / pullback[-1])) / recovery_days
        recovery = _phase_prices(pullback[-1], recovery_days, recovery_drift, 0.011, rng)
        leg = np.concatenate([pullback, recovery])
        leg_boundaries.append((len(pullback) - 1, depth))  # local trough index, depth
        leg_prices.append(leg)
        cursor_price = recovery[-1]

    p3 = np.concatenate(leg_prices) if leg_prices else np.array([p2[-1]])

    # Phase 4: tightening near the pivot (narrow range, low vol drift)
    p4_days = int(rng.integers(5, 16))
    p4 = _phase_prices(p3[-1], p4_days, 0.0002, 0.005, rng)

    pivot_price = float(max(p3.max(), p4.max()))

    # Phase 5: optional breakout
    do_breakout = rng.random() < 0.5 if breakout is None else breakout
    if do_breakout:
        p5_days = int(rng.integers(3, 8))
        p5 = _phase_prices(p4[-1], p5_days, rng.uniform(0.012, 0.03), 0.012, rng)
        breakout_vol_mult = rng.uniform(1.8, 3.2)
    else:
        p5_days = 0
        p5 = np.array([])
        breakout_vol_mult = 1.0

    # Phase 6: trailing continuation so windows can slide past the pattern too
    remaining = max(total_days - (p1_days + p2_days + len(p3) + p4_days + p5_days), 20)
    p6 = _phase_prices((p5[-1] if len(p5) else p4[-1]), remaining, rng.normal(0.0005, 0.0008), 0.016, rng)

    closes = np.concatenate([p1, p2, p3, p4, p5, p6])

    # Volume multiplier per phase: elevated on the advance, DRYING UP across
    # the contraction legs (this is the "quiet base" signature), very low at
    # the tightening pivot, then a sharp expansion on breakout.
    vol_mult = np.ones(len(closes))
    idx = 0
    vol_mult[idx:idx + p1_days] = rng.uniform(0.8, 1.1); idx += p1_days
    vol_mult[idx:idx + p2_days] = rng.uniform(1.1, 1.6); idx += p2_days
    leg_start = idx
    for k, leg in enumerate(leg_prices):
        leg_len = len(leg)
        dryup = max(0.35, 1.0 - 0.18 * (k + 1))  # each leg quieter than the last
        vol_mult[idx:idx + leg_len] = dryup * rng.uniform(0.85, 1.15, leg_len)
        idx += leg_len
    base_end_idx = idx - 1
    vol_mult[idx:idx + p4_days] = rng.uniform(0.25, 0.45)  # the dry-up right before the pivot
    pivot_idx = idx + p4_days - 1
    idx += p4_days
    if p5_days:
        vol_mult[idx:idx + p5_days] = breakout_vol_mult
        breakout_idx = idx
        idx += p5_days
    else:
        breakout_idx = None
    vol_mult[idx:] = rng.uniform(0.7, 1.3, len(closes) - idx)

    df = _make_ohlcv(closes, base_volume, vol_noise=0.35, vol_multiplier=vol_mult, rng=rng)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(df))
    df.index = dates

    meta = SeriesMeta(
        symbol=symbol, pattern="vcp", is_vcp=True,
        pivot_idx=pivot_idx, base_start_idx=leg_start, base_end_idx=base_end_idx,
        breakout_idx=breakout_idx, n_legs=n_legs,
    )
    return df, meta


def generate_negative_series(symbol: str, archetype: str, rng: np.random.Generator,
                              total_days: int = 500) -> tuple[pd.DataFrame, SeriesMeta]:
    base_price = rng.uniform(20, 150)
    base_volume = rng.uniform(5e5, 3e6)

    if archetype == "random_walk":
        closes = _phase_prices(base_price, total_days, 0.0002, 0.018, rng)
        vol_mult = np.exp(rng.normal(0, 0.3, total_days))

    elif archetype == "strong_uptrend_no_pullback":
        # A genuine "runaway" - persistent drift, no meaningful legs. This is
        # a deliberately hard negative: it LOOKS bullish but has no
        # contraction structure, so a naive "stock is up + near highs"
        # detector would misfire on it.
        closes = _phase_prices(base_price, total_days, rng.uniform(0.003, 0.0055), 0.011, rng)
        vol_mult = np.linspace(1.0, 1.6, total_days) * np.exp(rng.normal(0, 0.2, total_days))

    elif archetype == "choppy_equal_swings":
        # Legs exist, but depths are roughly CONSTANT (not contracting) -
        # the direct negative control for the contraction feature.
        n_legs = int(rng.integers(3, 6))
        depth = rng.uniform(0.15, 0.28)
        cursor = base_price
        legs = []
        for _ in range(n_legs):
            pullback_days = int(rng.integers(8, 20))
            recovery_days = int(rng.integers(8, 20))
            trough = cursor * (1 - depth * rng.uniform(0.85, 1.15))
            pullback = _phase_prices(cursor, pullback_days, np.log(trough / cursor) / pullback_days, 0.014, rng)
            recover_target = cursor * rng.uniform(0.97, 1.03)
            recovery = _phase_prices(pullback[-1], recovery_days,
                                      np.log(recover_target / pullback[-1]) / recovery_days, 0.013, rng)
            legs.append(np.concatenate([pullback, recovery]))
            cursor = recovery[-1]
        closes = np.concatenate(legs)
        if len(closes) < total_days:
            closes = np.concatenate([closes, _phase_prices(closes[-1], total_days - len(closes), 0.0002, 0.016, rng)])
        else:
            closes = closes[:total_days]
        vol_mult = np.exp(rng.normal(0, 0.3, len(closes)))

    elif archetype == "downtrend":
        closes = _phase_prices(base_price, total_days, rng.uniform(-0.004, -0.0015), 0.017, rng)
        vol_mult = np.exp(rng.normal(0, 0.3, total_days))

    elif archetype == "expanding_volatility_top":
        # The opposite of VCP: successive legs get BIGGER, not smaller
        # (broadening top). Direct negative control on contraction direction.
        n_legs = int(rng.integers(2, 4))
        depth = rng.uniform(0.08, 0.14)
        growth = rng.uniform(1.35, 1.7)
        cursor = base_price * rng.uniform(1.2, 1.6)
        legs = []
        for k in range(n_legs):
            d = float(np.clip(depth * (growth ** k), 0.03, 0.6))
            pullback_days = int(rng.integers(8, 20))
            recovery_days = int(rng.integers(8, 20))
            trough = cursor * (1 - d)
            pullback = _phase_prices(cursor, pullback_days, np.log(trough / cursor) / pullback_days, 0.014, rng)
            recovery = _phase_prices(pullback[-1], recovery_days,
                                      np.log((cursor * rng.uniform(0.95, 1.05)) / pullback[-1]) / recovery_days,
                                      0.013, rng)
            legs.append(np.concatenate([pullback, recovery]))
            cursor = recovery[-1]
        closes = np.concatenate(legs)
        if len(closes) < total_days:
            closes = np.concatenate([closes, _phase_prices(closes[-1], total_days - len(closes), 0.0001, 0.02, rng)])
        else:
            closes = closes[:total_days]
        vol_mult = np.exp(rng.normal(0, 0.35, len(closes)))

    else:
        raise ValueError(f"unknown archetype: {archetype}")

    closes = closes[:total_days] if len(closes) >= total_days else np.concatenate(
        [closes, _phase_prices(closes[-1], total_days - len(closes), 0.0002, 0.016, rng)])
    vol_mult = vol_mult[:total_days] if len(vol_mult) >= total_days else np.concatenate(
        [vol_mult, np.exp(rng.normal(0, 0.3, total_days - len(vol_mult)))])

    df = _make_ohlcv(closes, base_volume, vol_noise=0.3, vol_multiplier=vol_mult, rng=rng)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(df))
    df.index = dates

    meta = SeriesMeta(symbol=symbol, pattern=archetype, is_vcp=False)
    return df, meta


NEGATIVE_ARCHETYPES = [
    "random_walk", "strong_uptrend_no_pullback", "choppy_equal_swings",
    "downtrend", "expanding_volatility_top",
]


def generate_universe(n_vcp: int = 25, n_negative_each: int = 10, seed: int = 42,
                       total_days: int = 500) -> tuple[dict[str, pd.DataFrame], dict[str, SeriesMeta]]:
    rng = np.random.default_rng(seed)
    data: dict[str, pd.DataFrame] = {}
    meta: dict[str, SeriesMeta] = {}

    for i in range(n_vcp):
        sym = f"VCP{i:03d}"
        df, m = generate_vcp_series(sym, rng, total_days=total_days)
        data[sym], meta[sym] = df, m

    for archetype in NEGATIVE_ARCHETYPES:
        for i in range(n_negative_each):
            sym = f"{archetype[:4].upper()}{i:03d}"
            df, m = generate_negative_series(sym, archetype, rng, total_days=total_days)
            data[sym], meta[sym] = df, m

    return data, meta
