
"""
V13 AI Trading Platform — V12.1 Protected Baseline + Advanced Additions
Single-file Streamlit trading research / paper-trading terminal.

Pipeline:
Market Data -> Technical/Fundamental/Positioning Intelligence -> MTF/Regime
-> Confluence -> Signal Validation -> Risk Veto -> Forex/Binary Entry
-> Paper Execution -> Trade Monitor -> Journal -> Backtest/Walk-forward/Monte Carlo
-> Optimizer -> Dashboard.

This file preserves every engine/tool/feature in the supplied V12.1 source and
adds strict dashboard-pair/data synchronization so scanners and analysis cannot silently
use candles belonging to a different currency pair. Live scanner pairs are fetched from
the configured live source; synthetic data is never substituted for a missing live pair.

Research/paper-trading only. Live execution requires an official broker/data API.
Binary options availability/legality varies by jurisdiction and broker.
"""

from __future__ import annotations

import math
import os
import socket
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone, time as dt_time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional MetaTrader 5 market-data adapter. READ-ONLY: no order functions are used.
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    initial_balance: float = 10000.0
    risk_per_trade: float = 0.005
    max_daily_loss: float = 0.03
    max_weekly_loss: float = 0.07
    max_drawdown: float = 0.15
    max_open_positions: int = 3
    max_symbol_exposure: float = 0.02
    min_score: float = 72.0
    min_binary_confidence: float = 72.0
    binary_payout: float = 0.80
    max_spread_pips: float = 2.0
    slippage_pips: float = 0.3
    commission_per_lot: float = 0.0
    binary_expiries: Tuple[int, ...] = (5, 10, 15, 30)
    news_blackout_before: int = 30
    news_blackout_after: int = 15
    demo_rows: int = 1200
    data_source: str = "FXCM"
    twelve_data_api_key: str = ""
    twelve_data_outputsize: int = 500
    # MT5 is the authoritative live source. Twelve Data remains available only when explicitly selected.
    allow_live_source_failover: bool = False
    live_refresh_seconds: int = 15
    mt5_outputsize: int = 500
    mt5_bridge_url: str = ""
    mt5_bridge_token: str = ""
    mt5_bridge_timeout_seconds: int = 10
    fxcm_outputsize: int = 500
    fxcm_timeout_seconds: int = 20
    fxcm_app_name: str = "Forex AI Pro V13"
    data_max_age_seconds: int = 420
    ml_min_probability: float = 0.60
    signal_max_age_seconds: int = 60
    no_trade_conflict_threshold: float = 18.0
    min_signal_confidence: float = 72.0
    ai_soft_floor: float = 25.0
    min_engine_agreement: float = 60.0


# ============================================================
# MARKET DATA ENGINE
# ============================================================

class MarketDataEngine:
    TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x.columns = [str(c).strip().lower().replace(" ", "_") for c in x.columns]
        aliases = {
            "datetime": "time",
            "date": "time",
            "timestamp": "time",
            "vol": "volume",
            "tick_volume": "volume",
        }
        x = x.rename(columns={c: aliases.get(c, c) for c in x.columns})

        for c in ["open", "high", "low", "close"]:
            if c not in x.columns:
                raise ValueError(f"Missing required OHLC column: {c}")
            x[c] = pd.to_numeric(x[c], errors="coerce")

        if "volume" not in x.columns:
            x["volume"] = 0.0
        else:
            x["volume"] = pd.to_numeric(x["volume"], errors="coerce").fillna(0)

        if "spread_pips" in x.columns:
            x["spread_pips"] = pd.to_numeric(x["spread_pips"], errors="coerce")
        else:
            x["spread_pips"] = np.nan

        if "time" in x.columns:
            x["time"] = pd.to_datetime(x["time"], errors="coerce", utc=True)
        else:
            x["time"] = pd.date_range(
                end=pd.Timestamp.now(tz="UTC"), periods=len(x), freq="5min"
            )

        before = len(x)
        x = (
            x.dropna(subset=["open", "high", "low", "close"])
            .drop_duplicates(subset=["time"])
            .sort_values("time")
            .reset_index(drop=True)
        )
        x.attrs["duplicates_removed"] = before - len(x)
        return x

    @staticmethod
    def synthetic(symbol="EURUSD", rows=1200, seed=7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        t = pd.date_range(
            end=pd.Timestamp.now(tz="UTC"), periods=rows, freq="5min"
        )
        regimes = np.where(
            (np.arange(rows) // 180) % 3 == 0,
            0.00008,
            np.where((np.arange(rows) // 180) % 3 == 1, -0.00003, 0.0),
        )
        noise = rng.normal(0, 0.00045, rows)
        close = 1.08 + np.cumsum(regimes + noise)
        close = np.maximum(close, 0.5)
        op = np.r_[close[0], close[:-1]]
        spread = np.abs(rng.normal(0.00008, 0.000025, rows))
        high = np.maximum(op, close) + spread
        low = np.minimum(op, close) - spread
        vol = rng.integers(80, 1200, rows)
        return pd.DataFrame(
            {
                "time": t,
                "open": op,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        )

    @staticmethod
    def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        rule = {
            "M1": "1min",
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H4": "4h",
            "D1": "1D",
        }[timeframe]
        x = df.copy().set_index("time")
        out = (
            x.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        return out.reset_index()

    @staticmethod
    def validate(df: pd.DataFrame) -> Dict[str, Any]:
        x = MarketDataEngine.normalize(df)
        gaps = x["time"].diff().dt.total_seconds().div(60).dropna()
        return {
            "rows": len(x),
            "duplicates_removed": int(df.shape[0] - x.shape[0]),
            "missing_ohlc": int(
                x[["open", "high", "low", "close"]].isna().sum().sum()
            ),
            "large_gaps": int((gaps > 60).sum()) if len(gaps) else 0,
            "timezone": "UTC",
            "data_ok": len(x) >= 100,
        }


# ============================================================
# INDICATOR UTILITIES
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(close):
    line = ema(close, 12) - ema(close, 26)
    sig = ema(line, 9)
    return line, sig, line - sig


def stochastic(df, n=14):
    lo, hi = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    d = k.rolling(3).mean()
    return k.fillna(50), d.fillna(50)


def roc(close, n=12):
    return close.pct_change(n) * 100


def cci(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mean = tp.rolling(n).mean()
    md = (tp - mean).abs().rolling(n).mean()
    return ((tp - mean) / (0.015 * md.replace(0, np.nan))).fillna(0)


def williams_r(df, n=14):
    hi = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    return (-100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)).fillna(-50)


def mfi(df, n=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"].replace(0, 1)
    pos = mf.where(tp.diff() > 0, 0).rolling(n).sum()
    neg = mf.where(tp.diff() < 0, 0).rolling(n).sum().abs()
    ratio = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50)


def adx(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    tr = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / tr.replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / tr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return (
        dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0),
        pdi.fillna(0),
        mdi.fillna(0),
    )


def supertrend_bias(df, n=10, mult=3.0):
    a = atr(df, n)
    mid = (df["high"] + df["low"]) / 2
    upper = mid + mult * a
    lower = mid - mult * a
    return np.where(
        df["close"] > upper.shift(1),
        "BULLISH",
        np.where(df["close"] < lower.shift(1), "BEARISH", "NEUTRAL"),
    )


def ichimoku_bias(df):
    ten = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    kij = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    return np.where(ten > kij, "BULLISH", np.where(ten < kij, "BEARISH", "NEUTRAL"))


# ============================================================
# ANALYSIS ENGINES
# ============================================================

class TrendEngine:
    @staticmethod
    def analyze(df):
        x = df.copy()
        for n in [9, 20, 50, 100, 200]:
            x[f"ema{n}"] = ema(x.close, n)
        x["sma50"] = sma(x.close, 50)
        x["macd"], x["macd_signal"], x["macd_hist"] = macd(x.close)
        x["adx"], x["pdi"], x["mdi"] = adx(x)
        x["ichimoku"] = ichimoku_bias(x)
        x["supertrend"] = supertrend_bias(x)
        last = x.iloc[-1]

        votes = 0
        for n in [9, 20, 50, 100, 200]:
            votes += 1 if last.close > last[f"ema{n}"] else -1
        votes += 1 if last.macd_hist > 0 else -1
        votes += 1 if last.pdi > last.mdi else -1
        votes += 1 if last.ichimoku == "BULLISH" else (-1 if last.ichimoku == "BEARISH" else 0)
        votes += 1 if last.supertrend == "BULLISH" else (-1 if last.supertrend == "BEARISH" else 0)

        direction = (
            "BULLISH" if votes >= 3 else "BEARISH" if votes <= -3 else "SIDEWAYS"
        )
        strength = min(100, 50 + abs(votes) * 6 + max(0, float(last.adx) - 20))
        label = (
            "STRONG BULLISH"
            if direction == "BULLISH" and strength >= 75
            else "WEAK BULLISH"
            if direction == "BULLISH"
            else "STRONG BEARISH"
            if direction == "BEARISH" and strength >= 75
            else "WEAK BEARISH"
            if direction == "BEARISH"
            else "SIDEWAYS"
        )
        return {
            "direction": direction,
            "strength": float(strength),
            "label": label,
            "adx": float(last.adx),
            "ema9": float(last.ema9),
            "ema20": float(last.ema20),
            "ema50": float(last.ema50),
            "ema100": float(last.ema100),
            "ema200": float(last.ema200),
        }


class MomentumEngine:
    @staticmethod
    def analyze(df):
        rr = rsi(df.close)
        k, d = stochastic(df)
        ml, ms, mh = macd(df.close)
        ro = roc(df.close)
        cc = cci(df)
        wr = williams_r(df)
        mf = mfi(df)
        a, _, _ = adx(df)

        score = (
            50
            + np.clip((rr.iloc[-1] - 50) * 0.45, -22, 22)
            + np.clip(
                mh.iloc[-1] / max(abs(mh.iloc[-50:]).mean(), 1e-9) * 8,
                -10,
                10,
            )
        )
        direction = "BULLISH" if score >= 58 else "BEARISH" if score <= 42 else "NEUTRAL"
        accel = float(mh.iloc[-1] - mh.iloc[-2]) if len(mh) > 1 else 0

        div = "NONE"
        if len(df) >= 30:
            price_change = df.close.iloc[-1] - df.close.iloc[-15]
            r_change = rr.iloc[-1] - rr.iloc[-15]
            if price_change > 0 and r_change < 0:
                div = "BEARISH"
            elif price_change < 0 and r_change > 0:
                div = "BULLISH"

        state = (
            "OVERBOUGHT"
            if rr.iloc[-1] >= 70
            else "OVERSOLD"
            if rr.iloc[-1] <= 30
            else "INCREASING"
            if accel > 0
            else "DECREASING"
            if accel < 0
            else "NORMAL"
        )
        return {
            "direction": direction,
            "score": float(np.clip(score, 0, 100)),
            "rsi": float(rr.iloc[-1]),
            "stochastic": float(k.iloc[-1]),
            "macd_hist": float(mh.iloc[-1]),
            "roc": float(ro.iloc[-1]),
            "cci": float(cc.iloc[-1]),
            "williams_r": float(wr.iloc[-1]),
            "mfi": float(mf.iloc[-1]),
            "adx": float(a.iloc[-1]),
            "divergence": div,
            "state": state,
        }


class VolatilityEngine:
    @staticmethod
    def analyze(df):
        a = atr(df)
        mid = df.close.rolling(20).mean()
        sd = df.close.rolling(20).std()
        bw = (4 * sd / mid.replace(0, np.nan)) * 100
        hv = df.close.pct_change().rolling(50).std() * math.sqrt(252 * 24 * 12) * 100
        ar = (df.high - df.low) / df.close * 100
        last = len(df) - 1

        atr_pct = float(a.iloc[last] / df.close.iloc[last] * 100)
        bwv = float(bw.iloc[last])
        hvv = float(hv.iloc[last]) if np.isfinite(hv.iloc[last]) else 0
        roll_mean = ar.rolling(50).mean().iloc[last]
        roll_std = ar.rolling(50).std().iloc[last]
        z = float((ar.iloc[last] - roll_mean) / (roll_std if roll_std not in (0, np.nan) else 1))
        composite = atr_pct * 0.5 + bwv * 0.3 + hvv * 0.2

        regime = (
            "EXTREME" if z > 3 else
            "HIGH" if z > 1.5 else
            "VERY LOW" if z < -1.5 else
            "LOW" if z < -0.5 else "NORMAL"
        )
        return {
            "atr": float(a.iloc[last]),
            "atr_pct": atr_pct,
            "bb_width": bwv,
            "historical_vol": hvv,
            "range_z": z,
            "composite": float(composite),
            "regime": regime,
            "expansion": bool(z > 0.5),
            "compression": bool(z < -0.5),
        }



# -------------------- FAIR VALUE GAP ENGINE --------------------
@dataclass
class FVGZone:
    kind: str
    lower: float
    upper: float
    size: float
    midpoint: float
    index: int
    timestamp: str
    age_bars: int
    fresh: bool
    mitigated: bool
    quality: float

class FairValueGapEngine:
    """Standalone three-candle Fair Value Gap / imbalance engine."""

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 2:
            return float("nan")
        h = pd.to_numeric(df["high"], errors="coerce")
        l = pd.to_numeric(df["low"], errors="coerce")
        c = pd.to_numeric(df["close"], errors="coerce")
        prev = c.shift(1)
        tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def detect(self, df: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
        base = {
            "detected": False, "direction": "NONE", "lower": np.nan,
            "upper": np.nan, "midpoint": np.nan, "size": 0.0,
            "size_atr": np.nan, "fresh": False, "mitigated": False,
            "age_bars": 0, "quality": 0.0, "count": 0,
            "bullish_count": 0, "bearish_count": 0, "signal": "NONE"
        }
        if df is None or len(df) < 5:
            return base

        d = df.copy().reset_index(drop=True)
        for col in ("high", "low", "close"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
        if len(d) < 5:
            return base

        start = max(2, len(d) - lookback)
        zones = []
        for i in range(start, len(d)):
            # Three-candle bullish imbalance.
            if d.at[i, "low"] > d.at[i-2, "high"]:
                zones.append(("BULLISH", float(d.at[i-2, "high"]), float(d.at[i, "low"]), i))
            # Three-candle bearish imbalance.
            if d.at[i, "high"] < d.at[i-2, "low"]:
                zones.append(("BEARISH", float(d.at[i, "high"]), float(d.at[i-2, "low"]), i))

        if not zones:
            return base

        active = []
        for kind, lower, upper, idx in zones:
            future = d.iloc[idx+1:]
            mitigated = False
            invalidated = False
            if not future.empty:
                if kind == "BULLISH":
                    mitigated = bool((future["low"] <= upper).any())
                    invalidated = bool((future["close"] < lower).any())
                else:
                    mitigated = bool((future["high"] >= lower).any())
                    invalidated = bool((future["close"] > upper).any())
            if not invalidated:
                active.append((kind, lower, upper, idx, mitigated))

        counts = {
            "count": len(zones),
            "bullish_count": sum(z[0] == "BULLISH" for z in zones),
            "bearish_count": sum(z[0] == "BEARISH" for z in zones),
        }
        if not active:
            return {**base, **counts}

        kind, lower, upper, idx, mitigated = active[-1]
        size = upper - lower
        age = len(d) - 1 - idx
        atr = self._atr(d)
        size_atr = size / atr if np.isfinite(atr) and atr > 0 else np.nan

        quality = 45.0
        if np.isfinite(size_atr):
            quality += min(30.0, max(0.0, size_atr * 15.0))
        quality += 20.0 if not mitigated else 5.0
        quality -= min(20.0, age * 1.5)
        quality = float(np.clip(quality, 0.0, 100.0))

        return {
            **counts,
            "detected": True,
            "direction": kind,
            "lower": lower,
            "upper": upper,
            "midpoint": (lower + upper) / 2.0,
            "size": size,
            "size_atr": size_atr,
            "fresh": not mitigated,
            "mitigated": mitigated,
            "age_bars": age,
            "quality": quality,
            "signal": kind if (not mitigated and quality >= 55.0) else "NONE",
        }

    def score(self, fvg: Dict[str, Any], direction: str) -> float:
        if not fvg.get("detected") or fvg.get("signal") != direction:
            return 0.0
        return float(np.clip(fvg.get("quality", 0.0), 0.0, 100.0))


class StructureEngine:
    @staticmethod
    def analyze(df, lookback=5):
        x = df.tail(max(100, lookback * 10)).copy()
        highs = x.high[
            (x.high.shift(lookback) < x.high) & (x.high.shift(-lookback) < x.high)
        ]
        lows = x.low[
            (x.low.shift(lookback) > x.low) & (x.low.shift(-lookback) > x.low)
        ]

        shs = highs.dropna()
        sls = lows.dropna()
        sh = float(shs.iloc[-1]) if len(shs) else float(x.high.iloc[-1])
        sl = float(sls.iloc[-1]) if len(sls) else float(x.low.iloc[-1])
        prev_h = float(shs.iloc[-2]) if len(shs) > 1 else sh
        prev_l = float(sls.iloc[-2]) if len(sls) > 1 else sl

        hh, hl = sh > prev_h, sl > prev_l
        lh, ll = sh < prev_h, sl < prev_l
        direction = "BULLISH" if hh and hl else "BEARISH" if lh and ll else "TRANSITION"

        last = float(x.close.iloc[-1])
        bos = "BULLISH_BOS" if last > sh else "BEARISH_BOS" if last < sl else "NONE"
        choch = (
            "BULLISH_CHOCH"
            if direction == "BEARISH" and last > sh
            else "BEARISH_CHOCH"
            if direction == "BULLISH" and last < sl
            else "NONE"
        )
        return {
            "direction": direction,
            "swing_high": sh,
            "swing_low": sl,
            "HH": hh,
            "HL": hl,
            "LH": lh,
            "LL": ll,
            "BOS": bos,
            "CHOCH": choch,
            "breakout": bos != "NONE",
            "failure": False,
        }


class PriceActionEngine:
    @staticmethod
    def analyze(df):
        x = df.iloc[-3:].copy()
        o, h, l, c = x.open.iloc[-1], x.high.iloc[-1], x.low.iloc[-1], x.close.iloc[-1]
        body = abs(c - o)
        rng = max(h - l, 1e-12)
        upper = h - max(o, c)
        lower = min(o, c) - l

        bull_eng = (
            c > o and x.close.iloc[-2] < x.open.iloc[-2]
            and c >= x.open.iloc[-2] and o <= x.close.iloc[-2]
        )
        bear_eng = (
            c < o and x.close.iloc[-2] > x.open.iloc[-2]
            and c <= x.open.iloc[-2] and o >= x.close.iloc[-2]
        )
        pin_bull = lower > body * 2 and upper < body
        pin_bear = upper > body * 2 and lower < body
        doji = body / rng < 0.12
        inside = h < x.high.iloc[-2] and l > x.low.iloc[-2]
        outside = h > x.high.iloc[-2] and l < x.low.iloc[-2]
        strong = body / rng > 0.7

        patterns = []
        if bull_eng: patterns.append("BULLISH_ENGULFING")
        if bear_eng: patterns.append("BEARISH_ENGULFING")
        if pin_bull: patterns.append("BULLISH_PIN")
        if pin_bear: patterns.append("BEARISH_PIN")
        if doji: patterns.append("DOJI")
        if inside: patterns.append("INSIDE_BAR")
        if outside: patterns.append("OUTSIDE_BAR")
        if strong: patterns.append("STRONG_MOMENTUM_CANDLE")

        direction = (
            "BULLISH" if any("BULL" in p for p in patterns)
            else "BEARISH" if any("BEAR" in p for p in patterns)
            else "NEUTRAL"
        )
        return {
            "patterns": patterns,
            "direction": direction,
            "body_ratio": body / rng,
            "upper_wick": upper / rng,
            "lower_wick": lower / rng,
        }


class SupportResistanceEngine:
    @staticmethod
    def analyze(df):
        x = df.copy()
        last = float(x.close.iloc[-1])
        levels = []

        for n, label in [(20, "SWING"), (50, "MEDIUM"), (100, "LONG")]:
            if len(x) >= n:
                levels += [
                    (float(x.high.tail(n).max()), "RESISTANCE_" + label),
                    (float(x.low.tail(n).min()), "SUPPORT_" + label),
                ]

        if not levels:
            return {
                "support": last,
                "resistance": last,
                "distance_support": 0,
                "distance_resistance": 0,
                "strength": 0,
                "touches": 0,
                "break_probability": 50,
                "retest_probability": 50,
            }

        supports = [v for v, _ in levels if v <= last]
        resistances = [v for v, _ in levels if v >= last]
        support = max(supports) if supports else min(v for v, _ in levels)
        resistance = min(resistances) if resistances else max(v for v, _ in levels)
        touches = sum(1 for v, _ in levels if abs(v - last) / max(last, 1e-9) < 0.002)

        return {
            "support": float(support),
            "resistance": float(resistance),
            "distance_support": float((last - support) / last * 100),
            "distance_resistance": float((resistance - last) / last * 100),
            "strength": float(min(100, 30 + touches * 12)),
            "touches": touches,
            "break_probability": float(
                np.clip(50 + (last - support) / (resistance - support + 1e-12) * 20, 0, 100)
            ),
            "retest_probability": float(np.clip(60 - touches * 4, 10, 90)),
        }


class BreakoutEngine:
    @staticmethod
    def analyze(df, n=20):
        hi = df.high.shift(1).rolling(n).max()
        lo = df.low.shift(1).rolling(n).min()
        last = df.iloc[-1]
        up = bool(last.close > hi.iloc[-1])
        dn = bool(last.close < lo.iloc[-1])

        mean_v = df.volume.rolling(50).mean().iloc[-1]
        std_v = df.volume.rolling(50).std().iloc[-1]
        volz = (last.volume - mean_v) / (std_v if std_v not in (0, np.nan) else 1)

        breakout = "BULLISH_BREAKOUT" if up else "BEARISH_BREAKOUT" if dn else "NONE"
        confirmed = (up or dn) and volz > 0.5
        fakeout = (up or dn) and volz < -0.5

        return {
            "state": "FAKEOUT" if fakeout else "BREAKOUT" if breakout != "NONE" else "RANGE",
            "direction": "BULLISH" if up else "BEARISH" if dn else "NONE",
            "volume_z": float(volz),
            "confirmed": bool(confirmed),
            "retest": bool(up or dn),
        }


class LiquidityEngine:
    @staticmethod
    def analyze(df):
        x = df.tail(80)
        eqh = float(x.high.quantile(0.95))
        eql = float(x.low.quantile(0.05))
        last = float(x.close.iloc[-1])
        rng = float((x.high - x.low).tail(20).mean())
        sweep_up = bool(last < eqh and x.high.iloc[-1] >= eqh)
        sweep_dn = bool(last > eql and x.low.iloc[-1] <= eql)

        fvg = False
        if len(x) >= 3:
            fvg = bool(
                x.low.iloc[-1] > x.high.iloc[-3]
                or x.high.iloc[-1] < x.low.iloc[-3]
            )

        return {
            "liquidity_high": eqh,
            "liquidity_low": eql,
            "equal_highs": eqh,
            "equal_lows": eql,
            "sweep_up": sweep_up,
            "sweep_down": sweep_dn,
            "FVG": fvg,
            "liquidity_void": bool(rng > 2 * x.close.iloc[-1] * 0.001),
        }


class RegimeEngine:
    @staticmethod
    def classify(trend, vol, structure, breakout):
        if vol["regime"] == "EXTREME":
            return "ABNORMAL"
        if breakout["state"] == "BREAKOUT":
            return "BREAKOUT"
        if vol["regime"] in ("VERY LOW", "LOW") and trend["direction"] == "SIDEWAYS":
            return "LOW VOLATILITY"
        if trend["direction"] == "SIDEWAYS":
            return "RANGING"
        if trend["strength"] >= 72 and structure["direction"] in ("BULLISH", "BEARISH"):
            return "TRENDING"
        if structure["direction"] == "TRANSITION":
            return "TRANSITION"
        return "EXTENDED" if trend["strength"] >= 88 else "NO-TRADE"


class SessionEngine:
    """Authoritative Forex market-calendar and session engine.

    Important safety rule: a session label is NEVER allowed to imply that the
    Forex market is open. The weekly market calendar is checked first. Live
    dashboard analysis uses the current UTC clock; historical backtests can
    pass a candle timestamp explicitly. New York time is used for the weekly
    Sunday-open/Friday-close boundary and automatically observes DST.
    """

    @staticmethod
    def _to_utc(ts=None):
        if ts is None:
            return pd.Timestamp.now(tz="UTC")
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            return t.tz_localize("UTC")
        return t.tz_convert("UTC")

    @staticmethod
    def _market_open(ny):
        # Standard retail FX weekly schedule: opens Sunday 17:00 New York
        # and closes Friday 17:00 New York. This deliberately does not
        # attempt to model broker-specific holiday closures.
        wd = int(ny.weekday())  # Monday=0 ... Sunday=6
        tm = ny.time()
        if wd == 5:  # Saturday
            return False
        if wd == 6:  # Sunday
            return tm >= dt_time(17, 0)
        if wd == 4:  # Friday
            return tm < dt_time(17, 0)
        return True

    @staticmethod
    def _session_label(ny):
        # Session windows expressed in New York local time so DST is handled
        # consistently with the weekly market boundary. Overlap labels have
        # priority over the individual sessions.
        m = ny.hour * 60 + ny.minute
        if 8 * 60 <= m < 12 * 60:
            return "LONDON/NEW YORK OVERLAP"
        if 3 * 60 <= m < 8 * 60:
            return "LONDON"
        if 8 * 60 <= m < 17 * 60:
            return "NEW YORK"
        if 19 * 60 <= m or m < 4 * 60:
            return "TOKYO"
        return "SYDNEY"

    @classmethod
    def analyze(cls, ts=None):
        t = cls._to_utc(ts)
        try:
            ny = t.tz_convert("America/New_York")
        except Exception:
            # zoneinfo/tzdata is part of normal Python installations; retain
            # a safe UTC fallback rather than inventing a session.
            ny = t

        market_open = cls._market_open(ny)
        if not market_open:
            session = "WEEKEND / MARKET CLOSED"
            tradeable = False
            status = "MARKET CLOSED"
        else:
            session = cls._session_label(ny)
            tradeable = True
            status = "MARKET OPEN"

        return {
            "session": session,
            "current_session": session,
            "hour": int(t.hour),
            "weekday": t.day_name(),
            "utc_timestamp": t.isoformat(),
            "new_york_timestamp": ny.isoformat(),
            "market_open": bool(market_open),
            "market_status": status,
            "session_tradeable": bool(tradeable),
            "is_weekend": not bool(market_open) and int(ny.weekday()) in (5, 6),
            "timezone_basis": "America/New_York for weekly boundary; UTC stored",
        }


class CurrencyStrengthEngine:
    CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

    @staticmethod
    def analyze(df, symbol="EURUSD"):
        ret = float(df.close.pct_change(20).iloc[-1] * 100) if len(df) > 20 else 0
        base, quote = (symbol[:3], symbol[3:]) if len(symbol) >= 6 else ("EUR", "USD")
        out = {c: 0.0 for c in CurrencyStrengthEngine.CURRENCIES}
        if base in out:
            out[base] = float(np.clip(ret * 10, -5, 5))
        if quote in out:
            out[quote] = float(np.clip(-ret * 10, -5, 5))
        return {
            "matrix": out,
            "base": base,
            "quote": quote,
            "spread": out.get(base, 0) - out.get(quote, 0),
            "strongest": max(out, key=out.get),
            "weakest": min(out, key=out.get),
        }


class CorrelationEngine:
    @staticmethod
    def analyze(history: Dict[str, pd.DataFrame], symbol):
        if symbol not in history or len(history) < 2:
            return {"average_abs_corr": 0.0, "risk": "LOW", "pairs": []}

        s = history[symbol].close.pct_change().tail(200)
        rows = []
        for k, v in history.items():
            if k == symbol:
                continue
            a = v.close.pct_change().tail(200)
            n = min(len(s), len(a))
            if n > 30:
                corr = float(s.tail(n).corr(a.tail(n)))
                if np.isfinite(corr):
                    rows.append((k, corr))

        avg = float(np.mean([abs(c) for _, c in rows])) if rows else 0
        return {
            "average_abs_corr": avg,
            "risk": "HIGH" if avg > 0.75 else "MEDIUM" if avg > 0.5 else "LOW",
            "pairs": sorted(rows, key=lambda z: -abs(z[1])),
        }


class MarketTrackerEngine:
    @staticmethod
    def rank(symbols, analyses):
        rows = []
        for sym in symbols:
            a = analyses[sym]
            c = a["confluence"]
            rows.append(
                {
                    "SYMBOL": sym,
                    "DIRECTION": c["direction"],
                    "SCORE": round(c["score"], 1),
                    "REGIME": a["regime"],
                    "VOLATILITY": a["volatility"]["regime"],
                    "SESSION": a["session"]["session"],
                    "TREND": a["trend"]["label"],
                    "MOMENTUM": a["momentum"]["direction"],
                    "STRUCTURE": a["structure"]["direction"],
                    "ENTRY QUALITY": round(c["quality"], 1),
                    "RISK": a["risk"]["status"],
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("SCORE", ascending=False)
            .reset_index(drop=True)
        )


class MultiTimeframeEngine:
    """Independent multi-timeframe analysis for the exact selected pair.

    LIVE mode fetches every timeframe independently from the configured live
    source. DEMO/backtest mode may resample the supplied dataframe.
    """
    TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1"]
    MIN_HISTORY = 220

    @staticmethod
    def _independent_live_data(symbol, timeframe, cfg):
        return get_live_pair_data(canonical_symbol(symbol), timeframe, cfg, force=False)

    @staticmethod
    def analyze(df, symbol=None, cfg=None):
        states, rows, sources, errors = {}, {}, {}, {}
        live_mode = bool(
            cfg is not None
            and str(getattr(cfg, "data_source", "DEMO")).upper() in {"TWELVE DATA", "MT5", "MT5 REMOTE"}
            and symbol
        )

        for tf in MultiTimeframeEngine.TIMEFRAMES:
            x = None
            source = "RESAMPLED"
            try:
                if live_mode:
                    x, meta = MultiTimeframeEngine._independent_live_data(symbol, tf, cfg)
                    source = str((meta or {}).get("source", getattr(cfg, "data_source", "LIVE"))).upper()
                else:
                    x = MarketDataEngine.resample(df, tf)
                x = MarketDataEngine.normalize(x)
                rows[tf] = int(len(x))
                sources[tf] = source
                if len(x) < MultiTimeframeEngine.MIN_HISTORY:
                    states[tf] = "INSUFFICIENT"
                else:
                    states[tf] = TrendEngine.analyze(x)["direction"]
            except Exception as ex:
                states[tf] = "NO DATA" if live_mode else "INSUFFICIENT"
                rows[tf] = int(len(x)) if isinstance(x, pd.DataFrame) else 0
                sources[tf] = source
                errors[tf] = str(ex)

        valid = [v for v in states.values() if v in ("BULLISH", "BEARISH")]
        bull = sum(v == "BULLISH" for v in valid)
        bear = sum(v == "BEARISH" for v in valid)
        direction = (
            "BULLISH" if bull > bear and bull >= len(valid) * 0.5
            else "BEARISH" if bear > bull and bear >= len(valid) * 0.5
            else "MIXED"
        )
        align = 100 * max(bull, bear) / max(len(valid), 1)
        return {
            "states": states,
            "direction": direction,
            "alignment": float(align),
            "strength": (
                "VERY STRONG" if align >= 85 else "STRONG" if align >= 70
                else "MODERATE" if align >= 55 else "WEAK"
            ),
            "rows": rows,
            "sources": sources,
            "errors": errors,
            "live_independent": live_mode,
            "symbol": canonical_symbol(symbol) if symbol else None,
            "required_history": MultiTimeframeEngine.MIN_HISTORY,
        }


class EconomicEngine:
    @staticmethod
    def analyze(events: pd.DataFrame, symbol: str, now=None, before=30, after=15):
        now = now or pd.Timestamp.now(tz="UTC")
        if events is None or events.empty:
            return {
                "bias": "NEUTRAL",
                "score": 50.0,
                "risk": "UNKNOWN",
                "blocked": False,
                "next_event": "No event data",
                "events": [],
            }

        e = events.copy()
        e.columns = [str(c).lower().strip().replace(" ", "_") for c in e.columns]
        if "time" not in e:
            return {
                "bias": "NEUTRAL",
                "score": 50.0,
                "risk": "UNKNOWN",
                "blocked": False,
                "next_event": "Invalid event data",
                "events": [],
            }

        e["time"] = pd.to_datetime(e["time"], errors="coerce", utc=True)
        e = e.dropna(subset=["time"]).sort_values("time")
        currencies = [symbol[:3], symbol[3:]] if len(symbol) >= 6 else []

        if "currency" not in e:
            e["currency"] = ""
        if "importance" not in e:
            e["importance"] = "LOW"

        e["currency"] = e["currency"].astype(str).str.upper()
        e["importance"] = e["importance"].astype(str).str.upper()
        rel = e[e["currency"].isin(currencies)].copy()

        bias = 0
        blocked = False
        nearest = None

        for _, r in rel.iterrows():
            mins = (r["time"] - now).total_seconds() / 60
            imp = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(
                r["importance"], 1
            )
            if -after <= mins <= before:
                blocked = True
            if mins >= 0 and nearest is None:
                nearest = r
            d = str(r.get("bias", "NEUTRAL")).upper()
            bias += (1 if d == "BULLISH" else -1 if d == "BEARISH" else 0) * imp

        score = float(np.clip(50 + bias * 8, 0, 100))
        direction = "BULLISH" if score >= 58 else "BEARISH" if score <= 42 else "NEUTRAL"
        risk = (
            "CRITICAL" if blocked
            else "HIGH" if any(rel.importance.isin(["HIGH", "CRITICAL"]))
            else "NORMAL"
        )

        return {
            "bias": direction,
            "score": score,
            "risk": risk,
            "blocked": blocked,
            "next_event": nearest.get("event", "Event") if nearest is not None else "No upcoming relevant event",
            "events": rel.head(10).to_dict("records"),
        }


class COTEngine:
    @staticmethod
    def analyze(cot: pd.DataFrame, symbol="EURUSD"):
        if cot is None or cot.empty:
            return {
                "bias": "NEUTRAL",
                "score": 50.0,
                "status": "NO DATA",
                "net": 0,
                "weekly_change": 0,
                "percentile": 50,
            }

        x = cot.copy()
        x.columns = [str(c).lower().strip().replace(" ", "_") for c in x.columns]
        base = symbol[:3] if len(symbol) >= 6 else "EUR"

        if "currency" in x:
            x = x[x.currency.astype(str).str.upper() == base]

        if x.empty:
            return {
                "bias": "NEUTRAL",
                "score": 50.0,
                "status": "NO DATA",
                "net": 0,
                "weekly_change": 0,
                "percentile": 50,
            }

        for c in [
            "commercial_long",
            "commercial_short",
            "noncommercial_long",
            "noncommercial_short",
        ]:
            if c not in x:
                x[c] = 0.0
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)

        x["net_spec"] = x.noncommercial_long - x.noncommercial_short
        net = float(x.net_spec.iloc[-1])
        change = float(x.net_spec.diff().iloc[-1]) if len(x) > 1 else 0
        pct = float(x.net_spec.rank(pct=True).iloc[-1] * 100) if len(x) > 1 else 50

        mean_abs = abs(x.net_spec).mean() + 1e-9
        std = x.net_spec.std() + 1e-9
        score = float(
            np.clip(
                50
                + np.sign(net) * min(35, abs(net) / mean_abs * 20)
                + np.sign(change) * min(15, abs(change) / std * 5),
                0,
                100,
            )
        )
        bias = "BULLISH" if score >= 58 else "BEARISH" if score <= 42 else "NEUTRAL"

        return {
            "bias": bias,
            "score": score,
            "status": "OK",
            "net": net,
            "weekly_change": change,
            "percentile": pct,
            "commercial_long": float(x.commercial_long.iloc[-1]),
            "commercial_short": float(x.commercial_short.iloc[-1]),
            "spec_long": float(x.noncommercial_long.iloc[-1]),
            "spec_short": float(x.noncommercial_short.iloc[-1]),
        }


class ConfluenceEngine:
    @staticmethod
    def score(
        trend,
        momentum,
        volatility,
        structure,
        price_action,
        sr,
        breakout,
        liquidity,
        regime,
        session,
        mtf,
        economic,
        cot,
    ):
        direction_votes = []
        for d in [
            trend["direction"],
            momentum["direction"],
            structure["direction"],
            price_action["direction"],
            mtf["direction"],
            economic["bias"],
            cot["bias"],
        ]:
            if d in ("BULLISH", "BEARISH"):
                direction_votes.append(d)

        bull = direction_votes.count("BULLISH")
        bear = direction_votes.count("BEARISH")
        direction = "BULLISH" if bull > bear else "BEARISH" if bear > bull else "WAIT"

        parts = {
            "Trend": np.clip(trend["strength"] * 0.20, 0, 20),
            "Momentum": np.clip(momentum["score"] * 0.15, 0, 15),
            "Structure": 20 if structure["direction"] == direction else 10 if structure["direction"] == "TRANSITION" else 4,
            "Price Action": 12 if price_action["direction"] == direction else 6 if price_action["direction"] == "NEUTRAL" else 3,
            "S/R": np.clip(sr["strength"] * 0.10, 0, 10),
            "Breakout": 10 if breakout["state"] == "BREAKOUT" and breakout["direction"] == direction else 7 if breakout["state"] == "RANGE" else 2,
            "Volatility": 5 if volatility["regime"] in ("NORMAL", "LOW") else 2 if volatility["regime"] == "HIGH" else 0,
            "Regime": 5 if regime in ("TRENDING", "BREAKOUT", "RANGING") else 1,
            "Session": 3 if session["session_tradeable"] else 0,
            "Liquidity": 2 if (liquidity["sweep_up"] or liquidity["sweep_down"] or liquidity["FVG"]) else 1,
            "MTF": 10 * (mtf["alignment"] / 100),
            "Economic": 5 * (economic["score"] / 100),
            "COT": 3 * (cot["score"] / 100),
        }

        total = float(np.clip(sum(parts.values()), 0, 100))
        if (not session.get("market_open", session.get("session_tradeable", False))) or economic["blocked"] or volatility["regime"] == "EXTREME":
            total = min(total, 55)
        if not session.get("market_open", session.get("session_tradeable", False)):
            total = 0.0

        quality = float(
            np.clip(
                total - (20 if direction == "WAIT" else 0)
                + (5 if mtf["alignment"] >= 80 else 0),
                0,
                100,
            )
        )
        grade = (
            "HIGH QUALITY" if total >= 85
            else "GOOD" if total >= 72
            else "WATCH" if total >= 60
            else "NO-TRADE"
        )

        return {
            "score": total,
            "direction": direction,
            "quality": quality,
            "entry_quality": quality,  # compatibility alias
            "components": parts,
            "grade": grade,
            "market_open": bool(session.get("market_open", session.get("session_tradeable", False))),
        }


class RiskEngine:
    @staticmethod
    def evaluate(account, cfg: Config, confluence, volatility, economic, correlation, spread=0.8):
        reasons = []
        if account["daily_loss_pct"] >= cfg.max_daily_loss * 100:
            reasons.append("DAILY LOSS LIMIT")
        if account["drawdown_pct"] >= cfg.max_drawdown * 100:
            reasons.append("MAX DRAWDOWN")
        if account["open_positions"] >= cfg.max_open_positions:
            reasons.append("MAX OPEN POSITIONS")
        if spread > cfg.max_spread_pips:
            reasons.append("SPREAD TOO WIDE")
        if volatility["regime"] == "EXTREME":
            reasons.append("ABNORMAL VOLATILITY")
        if economic["blocked"]:
            reasons.append("ECONOMIC EVENT BLACKOUT")
        if correlation["risk"] == "HIGH":
            reasons.append("CORRELATED EXPOSURE")
        if not confluence.get("market_open", True):
            reasons.append("FOREX MARKET CLOSED")
        if confluence["score"] < cfg.min_score:
            reasons.append("SCORE BELOW THRESHOLD")

        ok = not reasons
        return {
            "approved": ok,
            "status": "APPROVED" if ok else "VETO",
            "reasons": reasons,
            "risk_pct": cfg.risk_per_trade * 100,
        }


# ============================================================
# ENTRY / EXECUTION / MONITORING
# ============================================================

class ForexEntryEngine:
    @staticmethod
    def calculate(df, direction, confluence, cfg, symbol="EURUSD"):
        last = float(df.close.iloc[-1])
        a = float(atr(df).iloc[-1])
        s = SupportResistanceEngine.analyze(df)

        if direction == "BULLISH":
            entry = last
            sl = min(s["support"], last - a * 1.5)
            tp = max(s["resistance"], last + a * 3)
            side = "BUY"
        elif direction == "BEARISH":
            entry = last
            sl = max(s["resistance"], last + a * 1.5)
            tp = min(s["support"], last - a * 3)
            side = "SELL"
        else:
            return {
                "approved": False,
                "direction": "WAIT",
                "entry": last,
                "sl": None,
                "tp": None,
                "rr": 0,
                "lot": 0,
                "zone_low": last - a * 0.25,
                "zone_high": last + a * 0.25,
            }

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / max(risk, 1e-9)
        risk_money = cfg.initial_balance * cfg.risk_per_trade
        pip = 0.01 if symbol.endswith("JPY") else 0.0001
        pip_value_per_lot = 9.0 if symbol.endswith("JPY") else 10.0
        lot = risk_money / max((risk / pip) * pip_value_per_lot, 1e-9)

        return {
            "approved": confluence["score"] >= cfg.min_score,
            "direction": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "lot": float(np.clip(lot, 0.01, 100)),
            "zone_low": min(entry, entry - a * 0.25),
            "zone_high": max(entry, entry + a * 0.25),
            "quality": confluence["quality"],
            "risk_money": risk_money,
        }


class BinaryEntryEngine:
    @staticmethod
    def calculate(df, direction, confluence, cfg):
        last = float(df.close.iloc[-1])
        a = float(atr(df).iloc[-1])
        r = float(rsi(df["close"]).iloc[-1])
        side = "CALL" if direction == "BULLISH" else "PUT" if direction == "BEARISH" else "WAIT"

        if side == "WAIT":
            return {
                "approved": False,
                "direction": "WAIT",
                "entry": last,
                "zone_low": last - a * 0.15,
                "zone_high": last + a * 0.15,
                "confidence": 0,
                "expiry_table": [],
            }

        base = float(
            np.clip(
                confluence["score"]
                + (5 if (side == "CALL" and r < 70) or (side == "PUT" and r > 30) else -5),
                0,
                100,
            )
        )

        vol = VolatilityEngine.analyze(df)["regime"]
        rows = []
        for mins in cfg.binary_expiries:
            penalty = abs(mins - 15) * 0.35
            if vol == "EXTREME":
                penalty += 15
            elif vol == "VERY LOW":
                penalty += 7
            conf = float(np.clip(base - penalty, 0, 100))
            rows.append(
                {
                    "expiry_min": mins,
                    "confidence": conf,
                    "payout": cfg.binary_payout,
                    "expected_value": conf / 100 * (1 + cfg.binary_payout) - 1,
                }
            )

        best = max(rows, key=lambda z: z["confidence"])
        return {
            "approved": best["confidence"] >= cfg.min_binary_confidence,
            "direction": side,
            "entry": last,
            "zone_low": last - a * 0.15,
            "zone_high": last + a * 0.15,
            "confidence": best["confidence"],
            "expiry": best["expiry_min"],
            "payout": cfg.binary_payout,
            "expected_value": best["expected_value"],
            "expiry_table": rows,
        }


class ExecutionEngine:
    @staticmethod
    def paper_order(market, symbol, direction, entry, sl=None, tp=None, expiry=None, lot=0):
        return {
            "id": uuid.uuid4().hex[:10],
            "time": datetime.now(timezone.utc).isoformat(),
            "market": market,
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "expiry": expiry,
            "lot": lot,
            "status": "OPEN",
            "paper": True,
        }


class TradeMonitorEngine:
    """Paper-trade monitor: marks open trades SL/TP/expiry status from latest candle."""

    @staticmethod
    def monitor_trade(trade: Dict[str, Any], latest_price: float) -> Dict[str, Any]:
        if trade.get("status") != "OPEN":
            return trade

        direction = trade.get("direction", "")
        sl, tp = trade.get("sl"), trade.get("tp")

        if trade.get("market") == "FOREX":
            if direction == "BUY":
                if sl is not None and latest_price <= sl:
                    trade["status"] = "STOPPED"
                elif tp is not None and latest_price >= tp:
                    trade["status"] = "TARGET"
            elif direction == "SELL":
                if sl is not None and latest_price >= sl:
                    trade["status"] = "STOPPED"
                elif tp is not None and latest_price <= tp:
                    trade["status"] = "TARGET"

        trade["last_price"] = float(latest_price)
        return trade

    @staticmethod
    def monitor_all(journal, latest_price):
        return [TradeMonitorEngine.monitor_trade(t, latest_price) for t in journal]


class TradeJournal:
    @staticmethod
    def append(trade):
        st.session_state.journal.append(trade)


# ============================================================
# BACKTEST / STATISTICS
# ============================================================

class BacktestEngine:
    @staticmethod
    def run(df, cfg: Config, threshold=None, binary=False):
        x = MarketDataEngine.normalize(df)
        threshold = threshold or (cfg.min_binary_confidence if binary else cfg.min_score)

        if len(x) < 250:
            return pd.DataFrame(), {
                "error": "At least 250 candles recommended for backtesting."
            }

        equity = cfg.initial_balance
        peak = equity
        rows = []
        wins = 0
        losses = 0

        for i in range(220, len(x) - 1):
            window = x.iloc[: i + 1]
            t = TrendEngine.analyze(window)
            m = MomentumEngine.analyze(window)
            v = VolatilityEngine.analyze(window)
            s = StructureEngine.analyze(window)
            pa = PriceActionEngine.analyze(window)
            sr = SupportResistanceEngine.analyze(window)
            bo = BreakoutEngine.analyze(window)
            li = LiquidityEngine.analyze(window)
            re = RegimeEngine.classify(t, v, s, bo)
            se = SessionEngine.analyze(window.time.iloc[-1])
            mtf = MultiTimeframeEngine.analyze(window)
            eco = {"bias": "NEUTRAL", "score": 50, "blocked": False}
            cot = {"bias": "NEUTRAL", "score": 50}

            c = ConfluenceEngine.score(
                t, m, v, s, pa, sr, bo, li, re, se, mtf, eco, cot
            )

            if c["score"] < threshold or c["direction"] == "WAIT" or re == "ABNORMAL":
                continue

            entry = float(x.close.iloc[i])
            horizon = 1 if binary else min(24, len(x) - i - 1)
            future = x.iloc[i + 1 : i + 1 + horizon]
            if future.empty:
                continue

            if binary:
                win = (
                    future.close.iloc[-1] > entry
                    if c["direction"] == "BULLISH"
                    else future.close.iloc[-1] < entry
                )
                ret = cfg.binary_payout if win else -1
            else:
                a = float(atr(window).iloc[-1])
                sl = entry - a * 1.5
                tp = entry + a * 3
                if c["direction"] == "BEARISH":
                    sl, tp = entry + a * 1.5, entry - a * 3

                hit = 0
                for _, r in future.iterrows():
                    if c["direction"] == "BULLISH":
                        if r.low <= sl:
                            hit = -1
                            break
                        if r.high >= tp:
                            hit = 3
                            break
                    else:
                        if r.high >= sl:
                            hit = -1
                            break
                        if r.low <= tp:
                            hit = 3
                            break

                ret = (
                    hit
                    if hit
                    else (
                        (future.close.iloc[-1] - entry) / a
                        if c["direction"] == "BULLISH"
                        else (entry - future.close.iloc[-1]) / a
                    )
                )

            pnl = equity * cfg.risk_per_trade * ret
            equity += pnl
            peak = max(peak, equity)
            wins += ret > 0
            losses += ret <= 0

            rows.append(
                {
                    "time": x.time.iloc[i],
                    "symbol": "TEST",
                    "direction": c["direction"],
                    "score": c["score"],
                    "return_R": ret,
                    "pnl": pnl,
                    "equity": equity,
                    "drawdown_pct": (peak - equity) / peak * 100,
                }
            )

        trades = pd.DataFrame(rows)
        if trades.empty:
            return trades, {"trades": 0}

        gp = trades.loc[trades.pnl > 0, "pnl"].sum()
        gl = abs(trades.loc[trades.pnl < 0, "pnl"].sum())
        metrics = {
            "trades": len(trades),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": wins / len(trades) * 100,
            "net_profit": trades.pnl.sum(),
            "gross_profit": gp,
            "gross_loss": gl,
            "profit_factor": gp / gl if gl else np.inf,
            "avg_win": trades.loc[trades.pnl > 0, "pnl"].mean() if wins else 0,
            "avg_loss": trades.loc[trades.pnl < 0, "pnl"].mean() if losses else 0,
            "expectancy": trades.pnl.mean(),
            "max_drawdown": trades.drawdown_pct.max(),
            "recovery_factor": trades.pnl.sum() / max(trades.drawdown_pct.max(), 1e-9),
            "final_equity": equity,
        }
        return trades, metrics


class WalkForwardEngine:
    @staticmethod
    def run(df, cfg):
        n = len(df)
        cut = int(n * 0.65)
        train = df.iloc[:cut]
        test = df.iloc[cut:]
        tr, tm = BacktestEngine.run(train, cfg)
        te, em = BacktestEngine.run(test, cfg)
        return {
            "train": tm,
            "out_of_sample": em,
            "train_trades": tr,
            "test_trades": te,
        }


class MonteCarloEngine:
    @staticmethod
    def run(trades: pd.DataFrame, simulations=500, seed=11):
        if trades.empty:
            return {"error": "Run a backtest first."}

        rng = np.random.default_rng(seed)
        vals = trades.pnl.to_numpy()
        finals = []
        maxdds = []

        for _ in range(simulations):
            p = rng.permutation(vals)
            eq = 10000 + np.cumsum(p)
            peak = np.maximum.accumulate(eq)
            dd = np.max((peak - eq) / peak * 100)
            finals.append(eq[-1])
            maxdds.append(dd)

        return {
            "simulations": simulations,
            "median_final": float(np.median(finals)),
            "p05_final": float(np.percentile(finals, 5)),
            "p95_final": float(np.percentile(finals, 95)),
            "median_max_dd": float(np.median(maxdds)),
            "p95_max_dd": float(np.percentile(maxdds, 95)),
        }


class OptimizerEngine:
    @staticmethod
    def run(df, cfg, thresholds=(65, 70, 75, 80, 85)):
        rows = []
        for th in thresholds:
            _, m = BacktestEngine.run(df, cfg, threshold=th)
            if m.get("trades", 0):
                rows.append(
                    {
                        "threshold": th,
                        "net_profit": m["net_profit"],
                        "win_rate": m["win_rate"],
                        "profit_factor": m["profit_factor"],
                        "max_drawdown": m["max_drawdown"],
                        "trades": m["trades"],
                    }
                )
        return (
            pd.DataFrame(rows).sort_values(
                ["profit_factor", "net_profit"], ascending=False
            )
            if rows
            else pd.DataFrame()
        )


# ============================================================
# APP STATE / HELPERS
# ============================================================


# ============================================================
# V13 ADDITIONS — ADVANCED LIVE DATA / AI / QUALITY / TIMING
# These additions are intentionally layered on top of V12.1.
# Existing V12.1 engines are not replaced.
# ============================================================

class DataIntegrityEngine:
    """Live-feed quality, freshness and OHLC integrity monitor."""

    @staticmethod
    def assess(df: pd.DataFrame, timeframe="M5", expected_age_seconds=420) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"score": 0.0, "status": "BAD", "reasons": ["NO DATA"], "stale": True}
        x = MarketDataEngine.normalize(df)
        reasons=[]
        score=100.0
        if len(x) < 100:
            score -= 30; reasons.append("INSUFFICIENT HISTORY")
        bad_ohlc = ((x.high < x.low) | (x.high < x.open) | (x.high < x.close) |
                    (x.low > x.open) | (x.low > x.close)).sum()
        if bad_ohlc:
            score -= min(30, bad_ohlc*5); reasons.append("INVALID OHLC")
        dup = int(x.time.duplicated().sum())
        if dup:
            score -= 10; reasons.append("DUPLICATE TIMESTAMPS")
        diffs = x.time.diff().dt.total_seconds().dropna()
        if len(diffs):
            expected = {"M1":60,"M5":300,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}.get(timeframe,300)
            missing = int((diffs > expected*1.8).sum())
            if missing:
                score -= min(20, missing*2); reasons.append(f"MISSING/GAPPED CANDLES: {missing}")
        last = x.time.iloc[-1]
        age = max(0.0, (pd.Timestamp.now(tz="UTC") - last).total_seconds())
        stale = age > expected_age_seconds
        if stale:
            score -= 30; reasons.append(f"STALE DATA ({age:.0f}s)")
        # frozen price: last 8 closes identical
        if len(x) >= 8 and x.close.tail(8).nunique() == 1:
            score -= 20; reasons.append("FROZEN PRICE")
        score=max(0.0,min(100.0,score))
        status="EXCELLENT" if score>=90 else "GOOD" if score>=75 else "DEGRADED" if score>=55 else "BAD"
        return {"score":score,"status":status,"reasons":reasons,"stale":stale,"age_seconds":age,
                "last_timestamp":str(last),"rows":len(x),"signal_allowed":score>=55 and not stale}


class TwelveDataLiveEngine:
    """Twelve Data REST adapter. It only reads market data; it has no order API."""
    BASE_URL = "https://api.twelvedata.com/time_series"
    TF_MAP = {"M1":"1min","M5":"5min","M15":"15min","M30":"30min","H1":"1h","H4":"4h","D1":"1day"}

    @staticmethod
    def fetch(api_key, symbol, timeframe="M5", outputsize=500, timeout=15):
        if not api_key:
            raise ValueError("Twelve Data API key is not configured.")
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("The requests package is required for Twelve Data REST mode.") from e
        params={"symbol":symbol,"interval":TwelveDataLiveEngine.TF_MAP.get(timeframe,timeframe),
                "outputsize":int(outputsize),"apikey":api_key,"format":"JSON"}
        r=requests.get(TwelveDataLiveEngine.BASE_URL,params=params,timeout=timeout)
        r.raise_for_status()
        payload=r.json()
        if payload.get("status") == "error":
            raise RuntimeError(payload.get("message", "Twelve Data returned an error."))
        values=payload.get("values",[])
        if not values:
            raise RuntimeError("Twelve Data returned no candles.")
        rows=[]
        for z in values:
            rows.append({"time":z.get("datetime"),"open":z.get("open"),"high":z.get("high"),
                         "low":z.get("low"),"close":z.get("close"),"volume":z.get("volume",0)})
        df=MarketDataEngine.normalize(pd.DataFrame(rows)).sort_values("time").reset_index(drop=True)
        return df, {"source":"TWELVE DATA","rows":len(df),"fetched_at":datetime.now(timezone.utc).isoformat(),
                    "credits_used":r.headers.get("api-credits-used"),"credits_left":r.headers.get("api-credits-left")}


class FXCMLiveDataEngine:
    """FXCM FCLite read-only adapter using demo/real account credentials."""
    AUTH_DEMO = "https://endpoints-demo.fxcm.com"
    AUTH_REAL = "https://endpoints.fxcm.com"
    # FXCM's current FCLite documentation uses endpoints-demo.fxcm.com /
    # endpoints.fxcm.com for authentication. The legacy REST market-data
    # service still documents api-demo.fxcm.com / api.fxcm.com for market
    # requests. Streamlit Cloud may fail normal DNS resolution for the
    # legacy data hostname, so the connector below includes a DNS-over-HTTPS
    # fallback while preserving TLS SNI and certificate verification.
    DATA_DEMO = "https://api-demo.fxcm.com"
    DATA_REAL = "https://api.fxcm.com"
    DNS_DOHEndpoints = (
        "https://dns.google/resolve",
        "https://cloudflare-dns.com/dns-query",
    )
    TF_MAP = {"M1":"m1","M5":"m5","M15":"m15","M30":"m30","H1":"H1","H4":"H4","D1":"D1"}

    @staticmethod
    def _session():
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("The requests package is required for FXCM mode.") from e
        s = requests.Session()
        s.headers.update({"Accept":"application/json","User-Agent":"Forex-AI-Pro-V13/FXCM"})
        return s

    @staticmethod
    def _secret(name, default=""):
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
        try:
            value = str(st.secrets.get(name, "") or "").strip()
            if value:
                return value
        except Exception:
            pass
        return default

    @classmethod
    def _credentials(cls):
        login = cls._secret("FXCM_LOGIN_ID")
        password = cls._secret("FXCM_PASSWORD")
        environment = cls._secret("FXCM_ENVIRONMENT", "demo").lower()
        app_name = cls._secret("FXCM_APP_NAME", "Forex AI Pro V13")
        if environment not in {"demo","real"}:
            raise RuntimeError("FXCM_ENVIRONMENT must be 'demo' or 'real'.")
        if not login or not password:
            raise RuntimeError("FXCM_LOGIN_ID and FXCM_PASSWORD are required in Streamlit Secrets.")
        return login, password, environment, app_name

    @classmethod
    def _ensure_token(cls, force=False):
        cache = st.session_state.setdefault("fxcm_auth", {})
        now = time.time()
        if not force and cache.get("access_token") and now < float(cache.get("expires_at",0)) - 10:
            return cache

        login, password, environment, app_name = cls._credentials()
        auth_base = cls.AUTH_REAL if environment == "real" else cls.AUTH_DEMO
        data_base = cls.DATA_REAL if environment == "real" else cls.DATA_DEMO
        session = cls._session()

        # FXCM documents a one-minute access-token lifetime; refresh before expiry.
        refresh_token = cache.get("refresh_token")
        xsrf = cache.get("xsrf_token")
        if refresh_token and xsrf:
            try:
                rr = session.post(
                    f"{auth_base}/iam/refresh/",
                    headers={"X-XSRF-TOKEN":xsrf,"X-COOKIE-DOMAIN":"fxcm.com"},
                    timeout=20,
                )
                rr.raise_for_status()
                payload = rr.json()
                access = payload.get("accessToken")
                if access:
                    cache.update({"access_token":access,
                                  "refresh_token":payload.get("refreshToken",refresh_token),
                                  "expires_at":now+50,"environment":environment,
                                  "data_base":data_base})
                    return cache
            except Exception:
                pass

        tr = session.get(
            f"{auth_base}/iam/trading-systems/{login}",
            headers={"X-COOKIE-DOMAIN":"fxcm.com","Accept":"*/*"},
            timeout=20,
        )
        tr.raise_for_status()
        systems = tr.json()
        if not isinstance(systems,list) or not systems:
            raise RuntimeError("FXCM returned no trading-system information for this login.")
        system = systems[0]
        session_id = system.get("tradingSessionId")
        sub_id = system.get("tradingSessionSubId")
        if not session_id or not sub_id:
            raise RuntimeError("FXCM did not return tradingSessionId/tradingSessionSubId.")

        xsrf = session.cookies.get("XSRF-TOKEN")
        if not xsrf:
            for c in session.cookies:
                if str(c.name).upper() == "XSRF-TOKEN":
                    xsrf = c.value
                    break
        if not xsrf:
            raise RuntimeError("FXCM did not return the required XSRF-TOKEN cookie.")

        ar = session.post(
            f"{auth_base}/iam/authenticate/",
            json={"loginId":login,"password":password,
                  "tradingSessionId":session_id,"tradingSessionSubId":sub_id,
                  "appName":app_name},
            headers={"Content-Type":"application/json","X-COOKIE-DOMAIN":"fxcm.com",
                     "X-XSRF-TOKEN":xsrf},
            timeout=20,
        )
        ar.raise_for_status()
        auth = ar.json()
        access = auth.get("accessToken")
        if not access:
            raise RuntimeError("FXCM authentication returned no access token.")
        cache.update({"access_token":access,"refresh_token":auth.get("refreshToken"),
                      "xsrf_token":xsrf,"expires_at":now+50,
                      "environment":environment,"data_base":data_base,
                      "trading_session_id":session_id,
                      "trading_session_sub_id":sub_id,
                      "authenticated_at":datetime.now(timezone.utc).isoformat()})
        return cache

    @classmethod
    def _resolve_ipv4(cls, hostname, timeout=10):
        """Resolve a host normally, then via DNS-over-HTTPS if normal DNS fails.

        This is specifically to work around hosted environments where the
        legacy FXCM market-data hostname can fail local DNS resolution.
        """
        try:
            infos = socket.getaddrinfo(hostname, 443, socket.AF_INET, socket.SOCK_STREAM)
            ips = []
            for info in infos:
                ip = info[4][0]
                if ip not in ips:
                    ips.append(ip)
            if ips:
                return ips
        except Exception:
            pass

        try:
            import requests
        except ImportError as e:
            raise RuntimeError("The requests package is required for FXCM mode.") from e

        last_error = None
        for doh in cls.DNS_DOHEndpoints:
            try:
                headers = {"Accept": "application/dns-json"}
                if "cloudflare" in doh:
                    headers = {"Accept": "application/dns-message"}
                    # Cloudflare's JSON endpoint is more convenient here.
                    doh_url = "https://cloudflare-dns.com/dns-query"
                    rr = requests.get(
                        doh_url,
                        params={"name": hostname, "type": "A"},
                        headers={"Accept": "application/dns-json"},
                        timeout=timeout,
                    )
                else:
                    rr = requests.get(
                        doh,
                        params={"name": hostname, "type": "A"},
                        headers=headers,
                        timeout=timeout,
                    )
                rr.raise_for_status()
                payload = rr.json()
                answers = payload.get("Answer", []) if isinstance(payload, dict) else []
                ips = [a.get("data") for a in answers if a.get("type") == 1 and a.get("data")]
                if ips:
                    return ips
                last_error = RuntimeError(f"DNS-over-HTTPS returned no A record for {hostname}.")
            except Exception as ex:
                last_error = ex

        raise RuntimeError(
            f"Unable to resolve FXCM market-data host '{hostname}'. "
            f"Local DNS and DNS-over-HTTPS both failed. Last error: {last_error}"
        )

    @classmethod
    def _raw_https_request(cls, base_url, method, path, params=None, data=None, headers=None, timeout=20):
        """Small HTTPS client with explicit DNS fallback and TLS SNI.

        The TCP connection goes to the resolved IP, but TLS SNI and HTTP Host
        remain the original FXCM hostname, so certificate validation still
        applies to the intended server. No verify=False shortcut is used.
        """
        import http.client
        import json
        import socket
        import ssl
        from urllib.parse import urlencode, urlparse

        parsed = urlparse(base_url)
        hostname = parsed.hostname
        if not hostname:
            raise RuntimeError(f"Invalid FXCM base URL: {base_url}")
        port = parsed.port or 443
        ips = cls._resolve_ipv4(hostname, timeout=min(timeout, 10))

        query = ""
        if params:
            query = "?" + urlencode(params, doseq=True)
        target = path if path.startswith("/") else "/" + path
        target = target + query

        body = None
        req_headers = {
            "Accept": "application/json",
            "User-Agent": "Forex-AI-Pro-V13/FXCM",
            "Host": hostname,
            "Connection": "close",
        }
        if headers:
            req_headers.update(headers)
        if data is not None:
            if isinstance(data, (dict, list, tuple)):
                if isinstance(data, dict):
                    body = urlencode(data, doseq=True).encode("utf-8")
                else:
                    body = urlencode(data, doseq=True).encode("utf-8")
                req_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
            elif isinstance(data, bytes):
                body = data
            else:
                body = str(data).encode("utf-8")
            req_headers["Content-Length"] = str(len(body))

        context = ssl.create_default_context()
        last_error = None
        for ip in ips:
            sock = None
            try:
                sock = socket.create_connection((ip, port), timeout=timeout)
                sock.settimeout(timeout)
                tls_sock = context.wrap_socket(sock, server_hostname=hostname)
                sock = tls_sock
                request_line = f"{method.upper()} {target} HTTP/1.1\r\n"
                header_blob = "".join(f"{k}: {v}\r\n" for k, v in req_headers.items())
                tls_sock.sendall((request_line + header_blob + "\r\n").encode("utf-8") + (body or b""))

                response = http.client.HTTPResponse(tls_sock)
                response.begin()
                raw = response.read()
                status = response.status
                reason = response.reason
                response_headers = dict(response.getheaders())
                tls_sock.close()

                text = raw.decode("utf-8", errors="replace")
                try:
                    payload = json.loads(text) if text else {}
                except Exception:
                    payload = text
                return status, reason, response_headers, payload
            except Exception as ex:
                last_error = ex
                try:
                    if sock:
                        sock.close()
                except Exception:
                    pass

        raise RuntimeError(
            f"FXCM HTTPS connection failed for {hostname}: {last_error}"
        )

    @classmethod
    def _request(cls, method, path, params=None, data=None, retry=True, timeout=20):
        token = cls._ensure_token()
        base = token["data_base"]
        status, reason, response_headers, payload = cls._raw_https_request(
            base, method, path, params=params, data=data,
            headers={"Authorization": f"Bearer {token['access_token']}"},
            timeout=timeout,
        )
        if status == 401 and retry:
            cls._ensure_token(force=True)
            return cls._request(method, path, params, data, retry=False, timeout=timeout)
        if status < 200 or status >= 300:
            if isinstance(payload, dict):
                msg = payload.get("error") or payload.get("message") or payload.get("response")
            else:
                msg = str(payload)[:300]
            raise RuntimeError(f"FXCM HTTP {status} {reason}: {msg}")
        if isinstance(payload, dict):
            info = payload.get("response", {})
            if isinstance(info, dict) and info.get("executed") is False:
                raise RuntimeError(info.get("error") or "FXCM request rejected.")
        return payload

    @classmethod
    def _offer_map(cls):
        payload = cls._request("GET","/trading/get_model/",params=[("models","Offer")])
        offers = payload.get("offers",[]) if isinstance(payload,dict) else []
        if not offers and isinstance(payload,dict):
            data = payload.get("data",{})
            offers = data.get("offers",[]) if isinstance(data,dict) else []
        mapping={}
        for offer in offers or []:
            if isinstance(offer,dict) and offer.get("offerId") is not None:
                sym=canonical_symbol(offer.get("currency",""))
                mapping[sym]={"offer_id":int(offer["offerId"]),"symbol":sym,
                              "bid":offer.get("sell"),"ask":offer.get("buy"),
                              "spread":offer.get("spread"),"time":offer.get("time")}
        return mapping

    @classmethod
    def _offer(cls,symbol):
        wanted=canonical_symbol(symbol)
        offers=cls._offer_map()
        if wanted not in offers:
            cls._request("POST","/trading/update_subscriptions/",
                         data={"symbol":display_symbol(wanted),"visible":"true"})
            offers=cls._offer_map()
        if wanted not in offers:
            raise RuntimeError(f"FXCM does not expose {display_symbol(wanted)}.")
        return offers[wanted]

    @classmethod
    def candles(cls,symbol,timeframe="M5",outputsize=500,timeout=20):
        timeframe=str(timeframe).upper()
        if timeframe not in cls.TF_MAP:
            raise ValueError(f"Unsupported FXCM timeframe: {timeframe}")
        offer=cls._offer(symbol)
        payload=cls._request(
            "GET",f"/candles/{offer['offer_id']}/{cls.TF_MAP[timeframe]}/",
            params={"num":max(1,min(int(outputsize),10000))},timeout=timeout)
        raw=payload.get("candles",[]) if isinstance(payload,dict) else []
        rows=[]
        for c in raw:
            if not isinstance(c,(list,tuple)) or len(c)<10:
                continue
            rows.append({
                "time":pd.to_datetime(float(c[0]),unit="s",utc=True),
                "open":c[1],"close":c[2],"high":c[3],"low":c[4],"volume":c[9],
                "fxcm_ask_open":c[5],"fxcm_ask_close":c[6],
                "fxcm_ask_high":c[7],"fxcm_ask_low":c[8],
            })
        df=MarketDataEngine.normalize(pd.DataFrame(rows))
        if df.empty:
            raise RuntimeError(f"FXCM returned no usable candles for {display_symbol(symbol)}/{timeframe}.")
        try:
            pip=0.01 if "JPY" in canonical_symbol(symbol) else 0.0001
            df["spread_pips"]=(df["fxcm_ask_close"].astype(float)-df["close"].astype(float)).abs()/pip
        except Exception:
            pass
        meta={"source":"FXCM","symbol":canonical_symbol(symbol),
              "requested_symbol":canonical_symbol(symbol),"provider_symbol":display_symbol(symbol),
              "timeframe":timeframe,"rows":len(df),"offer_id":offer["offer_id"],
              "tick":{"bid":offer.get("bid"),"ask":offer.get("ask"),
                      "spread":offer.get("spread"),"time":offer.get("time")},
              "fetched_at":datetime.now(timezone.utc).isoformat(),
              "read_only":True,"execution_enabled":False,"synthetic":False,
              "fxcm_environment":st.session_state.get("fxcm_auth",{}).get("environment","demo")}
        return df,meta



class LiveConnectionManager:
    """Source selector and safe read-only connection state. No execution endpoints."""
    @staticmethod
    def status(source, df, validation, data_quality, metadata=None):
        metadata=metadata or {}
        return {"source":source,"connected":bool(df is not None and not df.empty),
                "quality":data_quality.get("score",0),"status":data_quality.get("status","BAD"),
                "last_fetch":metadata.get("fetched_at", data_quality.get("last_timestamp","-")),
                "credits_left":metadata.get("credits_left"),"read_only":True,
                "execution_enabled":False,"validation":validation}


class MomentumDirectionEngine:
    """Advanced momentum direction, acceleration and persistence layer."""
    @staticmethod
    def analyze(df):
        x=MarketDataEngine.normalize(df)
        close=x.close
        if len(x)<60:
            return {"direction":"NEUTRAL","score":50.0,"strength":"LOW","acceleration":0.0,"persistence":0.0,"divergence":"UNKNOWN"}
        r=rsi(close,14).iloc[-1]
        macd_line, macd_signal, macd_hist = macd(close)
        hist=float(macd_hist.iloc[-1])
        hist_prev=float(macd_hist.iloc[-2])
        adx_series, plus_series, minus_series = adx(x)
        adxv=float(adx_series.iloc[-1]); plus=float(plus_series.iloc[-1]); minus=float(minus_series.iloc[-1])
        e20=ema(close,20).iloc[-1]; e50=ema(close,50).iloc[-1]
        roc12=float(roc(close,12).iloc[-1])
        accel=float(hist-hist_prev)
        signs=np.sign(close.diff().tail(12).dropna())
        persistence=float((signs>0).mean()*100) if len(signs) else 50
        bull=0; bear=0
        bull += 1 if r>50 else 0; bear += 1 if r<50 else 0
        bull += 1 if hist>0 else 0; bear += 1 if hist<0 else 0
        bull += 1 if hist>=hist_prev else 0; bear += 1 if hist<hist_prev else 0
        bull += 1 if plus>minus else 0; bear += 1 if minus>plus else 0
        bull += 1 if close.iloc[-1]>e20>e50 else 0; bear += 1 if close.iloc[-1]<e20<e50 else 0
        bull += 1 if roc12>0 else 0; bear += 1 if roc12<0 else 0
        if bull-bear>=2: direction="BULLISH"
        elif bear-bull>=2: direction="BEARISH"
        else: direction="NEUTRAL"
        score=50 + (bull-bear)*8 + min(12,max(-12,adxv-20)*0.5) + (persistence-50)*0.12
        score=float(max(0,min(100,score)))
        strength="HIGH" if score>=78 or score<=22 else "MEDIUM" if score>=65 or score<=35 else "LOW"
        return {"direction":direction,"score":score,"strength":strength,"acceleration":accel,
                "persistence":persistence,"rsi":float(r),"adx":adxv,"plus_di":plus,"minus_di":minus,
                "roc":roc12,"macd_hist":float(hist),"divergence":"NOT CONFIRMED"}


class AIProbabilityEngine:
    """Optional ML layer. Uses sklearn when installed; otherwise a transparent statistical fallback."""
    @staticmethod
    def _features(df):
        x=MarketDataEngine.normalize(df).copy()
        x["ret1"]=x.close.pct_change()
        x["ret3"]=x.close.pct_change(3)
        x["ret10"]=x.close.pct_change(10)
        x["ema_gap"]=ema(x.close,20)/ema(x.close,50)-1
        x["rsi"]=rsi(x.close,14)/100
        x["atr_pct"]=atr(x,14)/x.close
        x["vol_z"]=(x.volume-x.volume.rolling(30).mean())/(x.volume.rolling(30).std()+1e-9)
        # The optional spread field is not available on every feed (e.g. many OHLC-only
        # Twelve Data responses). Keep only model features required by this engine so
        # an absent spread does not discard the entire training set.
        feature_cols=["ret1","ret3","ret10","ema_gap","rsi","atr_pct","vol_z"]
        return x.replace([np.inf,-np.inf],np.nan).dropna(subset=feature_cols)

    @staticmethod
    def predict(df, horizon=1):
        x=AIProbabilityEngine._features(df)
        if len(x)<160:
            return {"up_probability":50.0,"down_probability":50.0,"confidence":0.0,"model":"INSUFFICIENT DATA","trade_quality":"C"}
        cols=["ret1","ret3","ret10","ema_gap","rsi","atr_pct","vol_z"]
        y=(x.close.shift(-horizon)>x.close).astype(int)
        train=x.iloc[:-horizon].copy(); target=y.iloc[:-horizon]
        latest=x[cols].iloc[[-1]]
        try:
            from sklearn.ensemble import RandomForestClassifier
            model=RandomForestClassifier(n_estimators=120,max_depth=5,min_samples_leaf=6,random_state=42,class_weight="balanced_subsample")
            model.fit(train[cols],target)
            prob=float(model.predict_proba(latest)[0][1])
            model_name="RANDOM FOREST"
        except Exception:
            # deterministic fallback when sklearn is unavailable
            z=0.0
            z += float(train.ret1.tail(20).mean())/max(float(train.ret1.tail(60).std()),1e-8)*0.8
            z += float(train.ret3.tail(20).mean())/max(float(train.ret3.tail(60).std()),1e-8)*0.6
            z += float(latest.ema_gap.iloc[0])*100*0.8
            z += (float(latest.rsi.iloc[0])-0.5)*1.2
            prob=float(1/(1+math.exp(-max(-8,min(8,z)))))
            model_name="STATISTICAL FALLBACK"
        conf=abs(prob-0.5)*200
        quality="A" if conf>=65 else "B" if conf>=45 else "C"
        return {"up_probability":prob*100,"down_probability":(1-prob)*100,"confidence":conf,
                "model":model_name,"trade_quality":quality,"training_rows":len(train)}


class SignalConfidenceEngine:
    """Final signal-confidence layer built from independent engine agreement.

    AI probability is advisory rather than a single hard gate. A signal still needs
    strong multi-engine agreement, valid data, and risk/news/session safety.
    """
    @staticmethod
    def calculate(a, ai, data_quality, direction):
        if direction not in ("BULLISH", "BEARISH"):
            return {"confidence": 0.0, "agreement": 0.0, "ai_alignment": 50.0, "components": {}}

        def aligned(value, neutral=50.0):
            v = str(value or "").upper()
            return 100.0 if v == direction else 0.0 if v in ("BULLISH", "BEARISH") else neutral

        votes = [
            a.get("trend", {}).get("direction"),
            a.get("momentum", {}).get("direction"),
            a.get("structure", {}).get("direction"),
            a.get("price_action", {}).get("direction"),
            a.get("mtf", {}).get("direction"),
            a.get("advanced_momentum", {}).get("direction"),
        ]
        valid_votes = [v for v in votes if v in ("BULLISH", "BEARISH")]
        matching = sum(v == direction for v in valid_votes)
        agreement = 100.0 * matching / max(len(valid_votes), 1)

        trend_score = float(a.get("trend", {}).get("strength", 50.0))
        momentum_score = float(a.get("momentum", {}).get("score", 50.0))
        if direction == "BEARISH":
            momentum_score = 100.0 - momentum_score
        adv_score = float(a.get("advanced_momentum", {}).get("score", 50.0))
        if direction == "BEARISH":
            adv_score = 100.0 - adv_score

        mtf_dir = a.get("mtf", {}).get("direction", "MIXED")
        mtf_align = float(a.get("mtf", {}).get("alignment", 0.0))
        mtf_score = mtf_align if mtf_dir == direction else (100.0 - mtf_align if mtf_dir in ("BULLISH", "BEARISH") else 50.0)

        ai_dir = "BULLISH" if ai.get("up_probability", 50.0) > 55 else "BEARISH" if ai.get("down_probability", 50.0) > 55 else "NEUTRAL"
        ai_conf = float(ai.get("confidence", 0.0))
        if ai_dir == direction:
            ai_alignment = 50.0 + ai_conf * 0.50
        elif ai_dir in ("BULLISH", "BEARISH"):
            ai_alignment = 50.0 - ai_conf * 0.50
        else:
            ai_alignment = 50.0

        pa_score = aligned(a.get("price_action", {}).get("direction"))
        structure_score = aligned(a.get("structure", {}).get("direction"))
        data_score = float(data_quality.get("score", 0.0))

        confidence = (
            agreement * 0.25
            + trend_score * 0.15
            + momentum_score * 0.12
            + adv_score * 0.12
            + mtf_score * 0.12
            + ai_alignment * 0.10
            + data_score * 0.08
            + ((pa_score + structure_score) / 2.0) * 0.06
        )
        confidence = float(np.clip(confidence, 0.0, 100.0))
        return {
            "confidence": confidence,
            "agreement": float(agreement),
            "ai_alignment": float(ai_alignment),
            "components": {
                "engine_agreement": float(agreement),
                "trend": trend_score,
                "momentum": momentum_score,
                "advanced_momentum": adv_score,
                "mtf": mtf_score,
                "ai_alignment": ai_alignment,
                "data_quality": data_score,
                "price_action_structure": (pa_score + structure_score) / 2.0,
            },
        }


class EnsembleDecisionEngine:
    @staticmethod
    def decide(a, ai, data_quality, cfg):
        score = float(a["confluence"]["score"])
        direction = a["confluence"]["direction"]
        ai_dir = "BULLISH" if ai["up_probability"] > 55 else "BEARISH" if ai["down_probability"] > 55 else "NEUTRAL"
        momentum = a.get("advanced_momentum", {}).get("direction", "NEUTRAL")

        # Determine a directional consensus before confidence is calculated.
        vote_set = [
            a.get("trend", {}).get("direction"),
            a.get("momentum", {}).get("direction"),
            a.get("structure", {}).get("direction"),
            a.get("price_action", {}).get("direction"),
            a.get("mtf", {}).get("direction"),
            momentum,
            ai_dir,
        ]
        bull = sum(v == "BULLISH" for v in vote_set)
        bear = sum(v == "BEARISH" for v in vote_set)
        if direction not in ("BULLISH", "BEARISH"):
            direction = "BULLISH" if bull > bear else "BEARISH" if bear > bull else "WAIT"

        confidence = SignalConfidenceEngine.calculate(a, ai, data_quality, direction)
        final = confidence["confidence"]
        reasons = []

        if not data_quality.get("signal_allowed", False):
            reasons.append("DATA QUALITY / STALE FEED")
        if direction == "WAIT":
            reasons.append("NO CLEAR CONFLUENCE DIRECTION")
        if confidence["agreement"] < cfg.min_engine_agreement:
            reasons.append("ENGINE DIRECTION CONFLICT")
        if score < cfg.min_score:
            reasons.append("CONFLUENCE BELOW THRESHOLD")
        if not a.get("session", {}).get("market_open", a.get("session", {}).get("session_tradeable", False)):
            reasons.append("FOREX MARKET CLOSED")
            direction = "WAIT"
            final = 0.0

        # Directional AI conflict is a hard safety veto. A technical signal cannot
        # remain BUY/SELL while the probability engine strongly predicts the opposite.
        if direction == "BULLISH" and float(ai.get("down_probability", 50.0)) >= 65.0:
            reasons.append("AI DIRECTIONAL CONFLICT: DOWN >= 65%")
        elif direction == "BEARISH" and float(ai.get("up_probability", 50.0)) >= 65.0:
            reasons.append("AI DIRECTIONAL CONFLICT: UP >= 65%")
        elif ai["confidence"] < cfg.ai_soft_floor and confidence["agreement"] < 80:
            reasons.append("AI MODEL TOO UNCERTAIN")

        allowed = (
            direction in ("BULLISH", "BEARISH")
            and final >= cfg.min_signal_confidence
            and not reasons
        )
        return {
            "score": final,
            "confidence": final,
            "direction": direction,
            "ai_direction": ai_dir,
            "agreement": confidence["agreement"],
            "ai_alignment": confidence["ai_alignment"],
            "approved": allowed,
            "reasons": list(dict.fromkeys(reasons)),
            "components": confidence["components"],
            "confluence_score": score,
            "grade": "A" if final >= 85 else "B" if final >= 75 else "C" if final >= 65 else "D",
        }


class NoTradeEngine:
    @staticmethod
    def evaluate(a, ensemble, data_quality, cfg, signal_age=0):
        reasons = list(ensemble.get("reasons", []))
        if data_quality.get("score", 0) < 55:
            reasons.append("DATA QUALITY BELOW 55")
        if signal_age > cfg.signal_max_age_seconds:
            reasons.append("SIGNAL STALE")
        if a["volatility"].get("regime") == "EXTREME":
            reasons.append("EXTREME VOLATILITY")
        if a["economic"].get("blocked"):
            reasons.append("HIGH-IMPACT NEWS BLACKOUT")
        if not a["session"].get("market_open", a["session"].get("session_tradeable", False)):
            reasons.append("FOREX MARKET CLOSED")
            reasons.append("NO SESSION IS TRADEABLE WHILE MARKET IS CLOSED")
        if not a.get("risk", {}).get("approved", False):
            reasons.append("RISK VETO")
        return {
            "trade_allowed": len(reasons) == 0,
            "reasons": list(dict.fromkeys(reasons)),
            "status": "TRADE" if not reasons else "NO TRADE",
        }


class SignalTimingEngine:
    @staticmethod
    def assess(signal_time=None, max_age_seconds=60):
        if signal_time is None: return {"fresh":True,"age_seconds":0.0,"status":"FRESH"}
        t=pd.Timestamp(signal_time)
        if t.tzinfo is None: t=t.tz_localize("UTC")
        age=max(0,(pd.Timestamp.now(tz="UTC")-t).total_seconds())
        return {"fresh":age<=max_age_seconds,"age_seconds":age,"status":"FRESH" if age<=max_age_seconds else "EXPIRED"}


class TradeQualityEngine:
    @staticmethod
    def evaluate(a, ensemble, no_trade, data_quality):
        trend_dir = a["trend"].get("direction", "NEUTRAL")
        momentum_dir = a["momentum"].get("direction", "NEUTRAL")
        structure_dir = a["structure"].get("direction", "NEUTRAL")
        price_dir = a["price_action"].get("direction", "NEUTRAL")
        direction = ensemble.get("direction", "WAIT")
        checks = {
            "DATA QUALITY": data_quality.get("score", 0) >= 70,
            "TREND": trend_dir == direction,
            "MOMENTUM": momentum_dir == direction,
            "STRUCTURE": structure_dir == direction,
            "PRICE ACTION": price_dir == direction,
            "VOLATILITY": a["volatility"].get("regime") != "EXTREME",
            "REGIME": a.get("regime") not in (None, "NO-TRADE", "ABNORMAL"),
            "MTF": a["mtf"].get("alignment", 0) >= 50,
            "SESSION": a["session"].get("market_open", a["session"].get("session_tradeable", False)),
            "NEWS": not a["economic"].get("blocked", False),
            "AI": a["ai"].get("confidence", 0) >= 30 or ensemble.get("agreement", 0) >= 80,
            "RISK": a["risk"].get("approved", False),
            "SIGNAL CONFIDENCE": ensemble.get("confidence", 0) >= 72,
        }
        passed = sum(bool(v) for v in checks.values())
        pct = passed / len(checks) * 100
        grade = "A" if pct >= 90 else "B" if pct >= 75 else "C" if pct >= 60 else "D"
        decision = "TRADE" if grade in ("A", "B") and no_trade.get("trade_allowed") else "NO TRADE"
        return {"checks": checks, "score": pct, "grade": grade, "decision": decision}


class PerformanceIntelligenceEngine:
    @staticmethod
    def summarize(journal):
        if not journal: return {"signals":0,"wins":0,"losses":0,"win_rate":0.0,"by_key":pd.DataFrame()}
        j=pd.DataFrame(journal)
        result={"signals":len(j),"wins":int((j.get("result",pd.Series(dtype=str))=="WIN").sum()),"losses":int((j.get("result",pd.Series(dtype=str))=="LOSS").sum())}
        result["win_rate"]=result["wins"]/max(1,result["wins"]+result["losses"])*100
        keys=[c for c in ["symbol","market","session","regime","expiry"] if c in j.columns]
        result["by_key"]=j.groupby(keys).size().reset_index(name="trades") if keys else pd.DataFrame()
        return result


class VolumeFlowEngine:
    """Volume-flow confirmation layer using the feed's available volume/tick volume."""
    @staticmethod
    def analyze(df):
        x = MarketDataEngine.normalize(df)
        if len(x) < 30:
            return {"direction":"NEUTRAL","score":50.0,"volume_z":0.0,"flow_ratio":1.0,"state":"INSUFFICIENT"}
        v = pd.to_numeric(x["volume"], errors="coerce").fillna(0.0)
        ret = x["close"].diff()
        signed = v * np.sign(ret).fillna(0)
        flow_fast = float(signed.tail(10).sum())
        flow_slow = float(signed.tail(30).sum())
        mean = float(v.tail(50).mean())
        std = float(v.tail(50).std())
        vz = (float(v.iloc[-1]) - mean) / (std if np.isfinite(std) and std > 0 else 1.0)
        ratio = (flow_fast / max(abs(flow_slow), 1e-9)) if flow_slow else 0.0
        direction = "BULLISH" if flow_fast > 0 and flow_slow >= 0 else "BEARISH" if flow_fast < 0 and flow_slow <= 0 else "NEUTRAL"
        score = float(np.clip(50 + np.sign(flow_fast) * min(35, abs(ratio) * 25) + np.clip(vz, -3, 3) * 4, 0, 100))
        return {"direction":direction,"score":score,"volume_z":float(vz),"flow_ratio":float(ratio),
                "state":"EXPANDING" if vz > 0.75 else "CONTRACTING" if vz < -0.75 else "NORMAL"}


class DirectProbabilityEngine:
    """Transparent non-ML directional probability from independent engine votes."""
    @staticmethod
    def calculate(a):
        dirs = [a.get(k, {}).get("direction") for k in ("trend","momentum","structure","price_action","mtf","advanced_momentum","volume_flow")]
        valid = [d for d in dirs if d in ("BULLISH","BEARISH")]
        bull = valid.count("BULLISH"); bear = valid.count("BEARISH")
        if not valid or bull == bear:
            return {"up_probability":50.0,"down_probability":50.0,"direction":"WAIT","confidence":0.0,"votes":len(valid)}
        p = 50.0 + 50.0 * (bull - bear) / len(valid)
        direction = "BULLISH" if bull > bear else "BEARISH"
        return {"up_probability":float(p),"down_probability":float(100-p),"direction":direction,
                "confidence":float(abs(p-50)*2),"votes":len(valid),"bull_votes":bull,"bear_votes":bear}


class CandleTimingEngine:
    """Checks whether the current candle is sufficiently formed for a signal."""
    TF_SECONDS = {"M1":60,"M5":300,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
    @classmethod
    def assess(cls, df, timeframe, session):
        if df is None or df.empty or not session.get("market_open", False):
            return {"ready":False,"progress_pct":0.0,"remaining_seconds":0.0,"status":"MARKET CLOSED / NO CANDLE"}
        last = pd.Timestamp(df["time"].iloc[-1])
        if last.tzinfo is None: last = last.tz_localize("UTC")
        else: last = last.tz_convert("UTC")
        now = pd.Timestamp.now(tz="UTC")
        sec = cls.TF_SECONDS.get(str(timeframe).upper(), 300)
        elapsed = max(0.0, (now-last).total_seconds())
        progress = float(np.clip(elapsed/sec*100, 0, 100))
        remaining = max(0.0, sec-elapsed)
        # Require a materially formed candle and reject a candle that is too old.
        ready = 20.0 <= progress <= 100.0 and elapsed <= sec*1.5
        return {"ready":ready,"progress_pct":progress,"remaining_seconds":remaining,
                "status":"FORMING" if ready else "WAIT", "last_candle_utc":last.isoformat()}


class MarketRegionEngine:
    """Human-readable regional/session context built on the authoritative calendar."""
    @staticmethod
    def analyze(session):
        if not session.get("market_open", False):
            return {"region":"WEEKEND / CLOSED","active":False,"overlap":False}
        name = session.get("session", "")
        return {"region":name,"active":True,"overlap":"OVERLAP" in name}


class BreakEvenEngine:
    """Calculates a paper-trade break-even trigger; it never modifies broker orders."""
    @staticmethod
    def calculate(entry, sl, direction, trigger_rr=1.0, buffer=0.0):
        if entry is None or sl is None or direction not in ("BUY","SELL"):
            return {"enabled":False,"trigger":None,"status":"UNAVAILABLE"}
        risk = abs(float(entry)-float(sl))
        trigger = float(entry) + risk*trigger_rr + buffer if direction == "BUY" else float(entry) - risk*trigger_rr - buffer
        return {"enabled":True,"trigger":trigger,"status":"ARMED","trigger_rr":trigger_rr}


class SignalExplanationEngine:
    """Produces explicit, auditable reasons for a final signal or veto."""
    @staticmethod
    def explain(a):
        ens=a.get("ensemble",{}); reasons=[]
        if a.get("session",{}).get("market_open") is False: reasons.append("Forex market is closed")
        if a.get("data_quality",{}).get("signal_allowed") is False: reasons.append("Market data is not verified/fresh enough")
        if ens.get("direction") in ("BULLISH","BEARISH"): reasons.append(f"Final directional stack: {ens['direction']}")
        if a.get("direct_probability",{}).get("direction") not in (None,"WAIT"): reasons.append(f"Direct probability: {a['direct_probability']['direction']} ({a['direct_probability'].get('confidence',0):.1f}% confidence)")
        if a.get("volume_flow",{}).get("direction") in ("BULLISH","BEARISH"): reasons.append(f"Volume flow: {a['volume_flow']['direction']}")
        reasons.extend(ens.get("reasons",[]))
        return {"summary":"; ".join(dict.fromkeys(reasons)) if reasons else "No verified signal conditions.","reasons":list(dict.fromkeys(reasons))}


class RiskVetoEngine:
    """Final immutable safety gate. Any hard veto forces NO TRADE."""
    @staticmethod
    def evaluate(a):
        veto=[]
        if not a.get("session",{}).get("market_open",False): veto.append("FOREX MARKET CLOSED")
        if not a.get("data_quality",{}).get("signal_allowed",False): veto.append("DATA QUALITY / STALE FEED")
        if not a.get("risk",{}).get("approved",False): veto.append("RISK VETO")
        if not a.get("no_trade",{}).get("trade_allowed",False): veto.extend(a.get("no_trade",{}).get("reasons",[]))
        ai=a.get("ai",{}); d=a.get("ensemble",{}).get("direction")
        if d=="BULLISH" and ai.get("down_probability",0)>=65: veto.append("AI DOWN PROBABILITY CONFLICT")
        if d=="BEARISH" and ai.get("up_probability",0)>=65: veto.append("AI UP PROBABILITY CONFLICT")
        return {"veto":bool(veto),"approved":not veto,"reasons":list(dict.fromkeys(veto))}


class AnalysisEngine:
    """Orchestration marker: all analytical engines are consumed before decision."""
    REQUIRED = ("trend","momentum","volatility","structure","price_action","sr","breakout","liquidity",
                "regime","session","currency_strength","mtf","economic","cot","correlation",
                "volume_flow","direct_probability","ai","ensemble","risk","no_trade","trade_quality")
    @classmethod
    def audit(cls, analysis):
        missing=[k for k in cls.REQUIRED if k not in analysis]
        return {"complete":not missing,"missing":missing,"engine_count":len(cls.REQUIRED)}

def canonical_symbol(symbol: str) -> str:
    """Normalize a dashboard/feed symbol to one canonical key."""
    s = str(symbol or "").upper().strip().replace("/", "").replace("_", "").replace("-", "")
    return s


def display_symbol(symbol: str) -> str:
    """Human-readable dashboard symbol, e.g. EURUSD -> EUR/USD."""
    s = canonical_symbol(symbol)
    return f"{s[:3]}/{s[3:]}" if len(s) == 6 else s


def _empty_market_data() -> pd.DataFrame:
    """Return a typed empty OHLCV frame. Empty means NO DATA, never synthetic data."""
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


def _get_fxcm_setting(name: str, default: str = "") -> str:
    """Resolve FXCM settings without exposing credentials in source code."""
    value = str(os.getenv(name, "") or "").strip()
    if value:
        return value
    try:
        value = str(st.secrets.get(name, "") or "").strip()
        if value:
            return value
    except Exception:
        pass
    return default


def _get_twelve_data_key(cfg: Config) -> str:
    """Resolve the Twelve Data key from the UI, environment, or Streamlit secrets."""
    key = str(getattr(cfg, "twelve_data_api_key", "") or "").strip()
    if key:
        return key
    try:
        session_key = str(st.session_state.get("td_key", "") or "").strip()
        if session_key:
            return session_key
    except Exception:
        pass
    for env_name in ("TWELVE_DATA_API_KEY", "TWELVEDATA_API_KEY"):
        value = str(os.getenv(env_name, "") or "").strip()
        if value:
            return value
    try:
        value = str(st.secrets.get("TWELVE_DATA_API_KEY", "") or "").strip()
        if value:
            return value
    except Exception:
        pass
    return ""


def _get_mt5_bridge_setting(name: str, cfg_value: str = "") -> str:
    """Resolve remote MT5 bridge settings from session, environment, or Streamlit secrets."""
    value = str(cfg_value or "").strip()
    if value:
        return value
    try:
        session_key = "mt5_bridge_url" if name == "MT5_BRIDGE_URL" else "mt5_bridge_token"
        session_value = str(st.session_state.get(session_key, "") or "").strip()
        if session_value:
            return session_value
    except Exception:
        pass
    value = str(os.getenv(name, "") or "").strip()
    if value:
        return value
    try:
        value = str(st.secrets.get(name, "") or "").strip()
        if value:
            return value
    except Exception:
        pass
    return ""


def data_matches_selection(symbol: str, timeframe: str, df=None, source: str | None = None) -> bool:
    """Strict identity check: pair + timeframe + source must match the dashboard."""
    if df is None or df.empty:
        return False
    meta = st.session_state.get("data_meta", {}) or {}
    identity_ok = (
        canonical_symbol(meta.get("symbol", "")) == canonical_symbol(symbol)
        and str(meta.get("timeframe", "")).upper() == str(timeframe).upper()
    )
    if not identity_ok:
        return False
    if source is None:
        return True
    return str(meta.get("source", "")).upper() == str(source).upper()


def store_market_data(symbol: str, timeframe: str, df: pd.DataFrame, meta=None, source=None):
    """Atomically replace candles and their full identity. Never mixes pair/source data."""
    symbol = canonical_symbol(symbol)
    x = MarketDataEngine.normalize(df)
    m = dict(meta or {})
    actual_source = str(source or m.get("source", "UNKNOWN")).upper()
    m.update({
        "symbol": symbol,
        "timeframe": str(timeframe).upper(),
        "rows": len(x),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "source": actual_source,
    })
    # Validate before replacing the currently loaded dataset.
    if x.empty:
        raise RuntimeError(f"No market data returned for {symbol}/{timeframe} from {actual_source}.")
    st.session_state.data = x
    st.session_state.data_meta = m
    st.session_state.data_symbol = symbol
    st.session_state.data_timeframe = str(timeframe).upper()
    st.session_state.data_source_loaded = actual_source
    return x, m


def clear_market_data_identity(clear_candles: bool = True):
    """Hard-reset loaded-data identity so stale candles cannot survive a failed pair switch."""
    st.session_state.data_symbol = None
    st.session_state.data_timeframe = None
    st.session_state.data_source_loaded = None
    st.session_state.data_meta = {}
    if clear_candles:
        st.session_state.data = _empty_market_data()


def _fetch_twelve_data(symbol: str, timeframe: str, cfg: Config):
    key = _get_twelve_data_key(cfg)
    if not key:
        raise RuntimeError(
            "Twelve Data is selected but no API key is configured. "
            "Enter the key in the sidebar or set TWELVE_DATA_API_KEY in Streamlit secrets."
        )
    # Keep one canonical internal key, but use standard slash notation first at the provider.
    provider_symbols = [display_symbol(symbol)]
    if symbol not in provider_symbols:
        provider_symbols.append(symbol)
    errors = []
    for td_symbol in provider_symbols:
        try:
            df, meta = TwelveDataLiveEngine.fetch(key, td_symbol, timeframe, cfg.twelve_data_outputsize)
            meta = dict(meta or {})
            meta.update({"symbol": symbol, "requested_symbol": symbol, "provider_symbol": td_symbol,
                         "timeframe": timeframe, "source": "TWELVE DATA", "read_only": True,
                         "execution_enabled": False})
            return df, meta
        except Exception as ex:
            errors.append(f"{td_symbol}: {ex}")
    raise RuntimeError("Twelve Data fetch failed for the selected pair. " + " | ".join(errors))


def get_live_pair_data(symbol: str, timeframe: str, cfg: Config, force=False):
    """Fetch exactly the requested pair/timeframe. No synthetic substitution in live modes."""
    symbol = canonical_symbol(symbol)
    timeframe = str(timeframe).upper()
    requested_source = str(cfg.data_source).upper()
    cache = st.session_state.setdefault("live_pair_cache", {})
    outputsize = (int(getattr(cfg, "mt5_outputsize", 500)) if requested_source in {"MT5", "MT5 REMOTE"} else int(getattr(cfg, "fxcm_outputsize", 500)) if requested_source == "FXCM" else int(getattr(cfg, "twelve_data_outputsize", 500)))
    key = (requested_source, symbol, timeframe, outputsize)

    if not force and key in cache:
        cached = cache[key]
        cached_df = cached.get("df")
        cached_meta = dict(cached.get("meta", {}) or {})
        cached_at = float(cached.get("cached_at", 0.0) or 0.0)
        age = time.time() - cached_at if cached_at else float("inf")
        # MT5 is a live feed: refresh its bars after the configured TTL.
        # Twelve Data intentionally remains cached unless explicitly refreshed so
        # rate limits cannot be hit repeatedly by Streamlit reruns.
        ttl = max(1, int(getattr(cfg, "live_refresh_seconds", 15))) if requested_source in {"MT5", "MT5 REMOTE", "FXCM"} else float("inf")
        if cached_df is not None and not cached_df.empty and age < ttl:
            cached_meta["cache_age_seconds"] = max(0.0, age)
            return cached_df.copy(), cached_meta

    if requested_source == "FXCM":
        try:
            df, meta = FXCMLiveDataEngine.candles(
                symbol, timeframe, int(getattr(cfg, "fxcm_outputsize", 500)),
                int(getattr(cfg, "fxcm_timeout_seconds", 20))
            )
            actual_source = "FXCM"
        except Exception as fxcm_error:
            raise RuntimeError(
                f"FXCM live data unavailable for {display_symbol(symbol)}/{timeframe}: {fxcm_error}"
            )
    elif requested_source == "TWELVE DATA":
        df, meta = _fetch_twelve_data(symbol, timeframe, cfg)
        actual_source = "TWELVE DATA"
    elif requested_source == "MT5":
        try:
            MT5LiveDataEngine.connect(symbol, path=st.session_state.get("mt5_path") or None)
            df = MT5LiveDataEngine.bars(symbol, timeframe, int(getattr(cfg, "mt5_outputsize", 500)))
            tick = MT5LiveDataEngine.tick(symbol)
            meta = {"symbol": symbol, "timeframe": timeframe, "source": "MT5",
                    "read_only": True, "execution_enabled": False, "tick": tick}
            actual_source = "MT5"
        except Exception as mt5_error:
            raise RuntimeError(f"MT5 local data unavailable for {symbol}/{timeframe}: {mt5_error}")
    elif requested_source == "MT5 REMOTE":
        try:
            df, meta = RemoteMT5BridgeEngine.snapshot(
                symbol, timeframe,
                int(getattr(cfg, "mt5_outputsize", 500)),
                base_url=str(getattr(cfg, "mt5_bridge_url", "") or ""),
                token=str(getattr(cfg, "mt5_bridge_token", "") or ""),
                timeout=int(getattr(cfg, "mt5_bridge_timeout_seconds", 10)),
            )
            actual_source = "MT5 REMOTE"
        except Exception as bridge_error:
            raise RuntimeError(f"MT5 remote bridge unavailable for {symbol}/{timeframe}: {bridge_error}")
    elif requested_source == "DEMO":
        # DEMO remains an explicit research mode only. Build a base M5 series and
        # resample it so D1 really represents daily candles rather than mislabeled M5 data.
        base_rows = int(cfg.demo_rows)
        if timeframe == "D1":
            # 288 M5 candles ≈ one 24-hour day; keep enough history for the full
            # daily indicator stack (trend, momentum, structure, AI, etc.).
            base_rows = max(base_rows, 120 * 288)
        elif timeframe == "H4":
            base_rows = max(base_rows, 120 * 48)
        base = MarketDataEngine.synthetic(symbol, base_rows, seed=7 + sum(map(ord, symbol)))
        df = MarketDataEngine.resample(base, timeframe) if timeframe != "M5" else base
        meta = {"symbol": symbol, "timeframe": timeframe, "source": "DEMO", "synthetic": True,
                "read_only": True, "execution_enabled": False}
        actual_source = "DEMO"
    else:
        raise RuntimeError(f"Unsupported data source: {requested_source}")

    df = MarketDataEngine.normalize(df)
    if df.empty:
        raise RuntimeError(f"No market data returned for {symbol}/{timeframe} from {actual_source}.")
    meta = dict(meta or {})
    meta.update({"symbol": symbol, "timeframe": timeframe, "source": actual_source,
                 "requested_source": requested_source})
    resolved_size = (int(getattr(cfg, "mt5_outputsize", 500)) if actual_source in {"MT5", "MT5 REMOTE"} else int(getattr(cfg, "fxcm_outputsize", 500)) if actual_source == "FXCM" else int(getattr(cfg, "twelve_data_outputsize", 500)))
    cache_meta = dict(meta)
    cache_meta["cached_at_utc"] = datetime.now(timezone.utc).isoformat()
    cache_entry = {"df": df.copy(), "meta": cache_meta, "cached_at": time.time()}
    cache_key = (actual_source, symbol, timeframe, resolved_size)
    cache[cache_key] = cache_entry
    # Also bind the request key to the resolved feed so subsequent reruns use the
    # same exact pair/source rather than resurrecting a prior dataset.
    cache[key] = {"df": df.copy(), "meta": dict(cache_meta), "cached_at": time.time()}
    return df, meta


def get_selected_market_data(symbol: str, timeframe: str, cfg: Config):
    """Resolve the dashboard selection with strict pair/timeframe/source isolation."""
    symbol = canonical_symbol(symbol)
    timeframe = str(timeframe).upper()
    requested_source = str(cfg.data_source).upper()

    # IMPORTANT: source is part of identity. A source change must trigger a new fetch.
    # For live MT5/remote MT5, also honor the configured refresh TTL so Streamlit
    # reruns actually request a fresh snapshot instead of returning the previous
    # selected dataset forever.
    if data_matches_selection(symbol, timeframe, st.session_state.get("data"), requested_source):
        if requested_source in {"MT5", "MT5 REMOTE", "FXCM"}:
            meta = st.session_state.get("data_meta", {}) or {}
            loaded_at = meta.get("loaded_at")
            try:
                loaded_ts = pd.to_datetime(loaded_at, utc=True)
                age = (pd.Timestamp.now(tz="UTC") - loaded_ts).total_seconds()
            except Exception:
                age = float("inf")
            if age < max(1, int(getattr(cfg, "live_refresh_seconds", 15))):
                return st.session_state.data, meta
        else:
            return st.session_state.data, st.session_state.get("data_meta", {})

    # A pair or source change invalidates the old dataset BEFORE fetching.
    clear_market_data_identity(clear_candles=True)
    df, meta = get_live_pair_data(symbol, timeframe, cfg, force=True)
    return store_market_data(symbol, timeframe, df, meta, meta.get("source", requested_source))


def load_events():
    return st.session_state.get("events", pd.DataFrame())


def load_cot():
    return st.session_state.get("cot", pd.DataFrame())


def build_no_data_analysis(symbol, cfg, data_quality=None, reason="NO MARKET DATA"):
    """Return a complete, dashboard-safe analysis schema when candles are unavailable.

    Every key used by the dashboard is present so a failed feed cannot turn into a
    secondary KeyError such as a["cs"]["matrix"]. No synthetic candles are created here.
    """
    dq = data_quality or {
        "score": 0.0, "status": "BAD", "reasons": [reason], "stale": True,
        "age_seconds": 0.0, "last_timestamp": "-", "rows": 0,
        "signal_allowed": False,
    }
    c = {"score": 0.0, "direction": "WAIT", "components": {}, "quality": 0.0}
    trend = {"label": "NO DATA", "direction": "SIDEWAYS"}
    momentum = {"direction": "NO DATA", "score": 0.0, "strength": "LOW"}
    volatility = {"regime": "NO DATA"}
    structure = {"direction": "NO DATA", "bias": "NO DATA"}
    price_action = {"direction": "NO DATA", "bias": "NO DATA"}
    currency_strength = {
        "matrix": {}, "base": str(symbol)[:3], "quote": str(symbol)[3:6],
        "spread": 0.0, "strongest": "-", "weakest": "-",
    }
    mtf = {"alignment": 0.0, "strength": "WEAK", "states": {}}
    economic = {"bias": "NEUTRAL", "score": 0.0, "risk": "UNKNOWN", "blocked": False, "next_event": "-"}
    cot = {"bias": "NEUTRAL", "net": 0.0, "weekly_change": 0.0, "percentile": 50.0}
    correlation = {"average_abs_corr": 0.0, "risk": "LOW", "pairs": []}
    risk = {"approved": False, "status": "VETO", "reasons": [reason], "risk_pct": cfg.risk_per_trade * 100}
    advanced_momentum = {"direction": "NO DATA", "score": 0.0, "strength": "LOW"}
    ai = {"up_probability": 50.0, "down_probability": 50.0, "confidence": 0.0}
    ensemble = {"score": 0.0, "direction": "WAIT", "approved": False, "reasons": [reason]}
    no_trade = {"trade_allowed": False, "reasons": [reason], "status": "NO TRADE"}
    trade_quality = {
        "grade": "D", "score": 0.0, "decision": "NO TRADE",
        "checks": {"DATA QUALITY": False, "NO MARKET DATA": False},
    }
    session = {"session": "-", "session_tradeable": False}
    breakout = {"direction": "NO DATA"}
    liquidity = {"direction": "NO DATA"}
    sr = {}
    regime = "NO-TRADE"

    return {
        "trend": trend, "momentum": momentum, "volatility": volatility,
        "structure": structure, "price_action": price_action, "sr": sr,
        "breakout": breakout, "liquidity": liquidity, "regime": regime,
        "session": session, "currency_strength": currency_strength, "mtf": mtf,
        "economic": economic, "cot": cot, "correlation": correlation,
        "confluence": c, "risk": risk, "advanced_momentum": advanced_momentum,
        "ai": ai, "ensemble": ensemble, "no_trade": no_trade,
        "trade_quality": trade_quality, "data_quality": dq,
        # V12.1 compatibility aliases.
        "t": trend, "m": momentum, "v": volatility, "s": structure,
        "pa": price_action, "bo": breakout, "li": liquidity, "re": regime,
        "se": session, "cs": currency_strength, "eco": economic,
        "corr": correlation, "c": c,
        "_error": reason,
    }


def analyze_market(df, symbol, cfg, timeframe=None):
    t = TrendEngine.analyze(df)
    m = MomentumEngine.analyze(df)
    v = VolatilityEngine.analyze(df)
    s = StructureEngine.analyze(df)
    pa = PriceActionEngine.analyze(df)
    sr = SupportResistanceEngine.analyze(df)
    bo = BreakoutEngine.analyze(df)
    li = LiquidityEngine.analyze(df)
    re = RegimeEngine.classify(t, v, s, bo)
    # The dashboard session must reflect the current market clock, not the timestamp
    # of the last historical candle. This prevents Friday's last candle from making
    # Sunday morning appear to be an active London/Asian session.
    se = SessionEngine.analyze()
    cs = CurrencyStrengthEngine.analyze(df, symbol)
    # In live mode MTF fetches the exact selected pair independently for every timeframe.
    mtf = MultiTimeframeEngine.analyze(df, symbol=symbol, cfg=cfg)
    eco = EconomicEngine.analyze(load_events(), symbol)
    cot = COTEngine.analyze(load_cot(), symbol)

    # Single-symbol correlation is LOW by definition unless peer histories are supplied.
    history = {symbol: df}
    corr = CorrelationEngine.analyze(history, symbol)

    c = ConfluenceEngine.score(
        t, m, v, s, pa, sr, bo, li, re, se, mtf, eco, cot
    )
    dq = DataIntegrityEngine.assess(df, timeframe or st.session_state.get("data_timeframe", "M5") or "M5", cfg.data_max_age_seconds)
    adv_m = MomentumDirectionEngine.analyze(df)
    volume_flow = VolumeFlowEngine.analyze(df)
    # Add advanced engines to the advisory graph before probability/ensemble decisions.
    advisory = {"trend":t,"momentum":m,"volatility":v,"structure":s,"price_action":pa,"sr":sr,
                "breakout":bo,"liquidity":li,"regime":re,"session":se,"mtf":mtf,"economic":eco,
                "cot":cot,"correlation":corr,"advanced_momentum":adv_m,"volume_flow":volume_flow}
    direct_probability = DirectProbabilityEngine.calculate(advisory)
    ai = AIProbabilityEngine.predict(df)

    risk = RiskEngine.evaluate(
        {
            "daily_loss_pct": 0,
            "drawdown_pct": 0,
            "open_positions": len(
                [x for x in st.session_state.journal if x.get("status") == "OPEN"]
            ),
        },
        cfg,
        c,
        v,
        eco,
        corr,
    )
    # Advanced layers are advisory gates; they do not replace V12.1 engines.
    advisory.update({"confluence":c,"risk":risk,"ai":ai,"direct_probability":direct_probability})
    ensemble = EnsembleDecisionEngine.decide(advisory, ai, dq, cfg)
    no_trade = NoTradeEngine.evaluate(advisory, ensemble, dq, cfg)
    quality = TradeQualityEngine.evaluate(advisory, ensemble, no_trade, dq)
    advisory["ensemble"] = ensemble
    advisory["no_trade"] = no_trade
    advisory["trade_quality"] = quality
    advisory["data_quality"] = dq
    timing = CandleTimingEngine.assess(df, timeframe or "M5", se)
    region = MarketRegionEngine.analyze(se)
    explanation = SignalExplanationEngine.explain(advisory)
    analysis_audit = AnalysisEngine.audit(advisory)
    veto = RiskVetoEngine.evaluate(advisory)
    if veto["veto"]:
        ensemble["approved"] = False
        ensemble["direction"] = "WAIT" if not se.get("market_open",False) else ensemble.get("direction","WAIT")
        ensemble["reasons"] = list(dict.fromkeys(list(ensemble.get("reasons",[])) + veto["reasons"]))
        no_trade["trade_allowed"] = False
        no_trade["reasons"] = list(dict.fromkeys(list(no_trade.get("reasons",[])) + veto["reasons"]))
        no_trade["status"] = "NO TRADE"
        quality["decision"] = "NO TRADE"

    return {
        "trend": t,
        "momentum": m,
        "volatility": v,
        "structure": s,
        "price_action": pa,
        "sr": sr,
        "breakout": bo,
        "liquidity": li,
        "regime": re,
        "session": se,
        "currency_strength": cs,
        "mtf": mtf,
        "economic": eco,
        "cot": cot,
        "correlation": corr,
        "confluence": c,
        "risk": risk,
        "advanced_momentum": adv_m,
        "volume_flow": volume_flow,
        "direct_probability": direct_probability,
        "candle_timing": timing,
        "market_region": region,
        "signal_explanation": explanation,
        "analysis_audit": analysis_audit,
        "risk_veto": veto,
        "break_even": BreakEvenEngine.calculate(None, None, "WAIT"),
        "ai": ai,
        "ensemble": ensemble,
        "no_trade": no_trade,
        "trade_quality": quality,
        "data_quality": dq,
        # Compatibility aliases used by earlier V12.1 code.
        "t": t, "m": m, "v": v, "s": s, "pa": pa, "bo": bo, "li": li,
        "re": re, "se": se, "cs": cs, "eco": eco, "corr": corr, "c": c,
    }


def fmt(v, n=2):
    try:
        return f"{float(v):,.{n}f}"
    except Exception:
        return str(v)


# ============================================================
# DASHBOARD
# ============================================================



# ============================================================
# V13 ADDITION: REMOTE MT5 BRIDGE CLIENT (READ-ONLY)
# ============================================================
class RemoteMT5BridgeEngine:
    """Read-only HTTP client for a Windows-hosted MT5 bridge.

    The Streamlit app does not attempt to run MetaTrader 5 itself. It requests
    exact pair/timeframe snapshots from the remote bridge, validates the
    response identity, and converts the returned OHLCV records into the same
    MarketDataEngine schema used by every existing V13 engine.
    """

    @staticmethod
    def _headers(token):
        if not token:
            raise RuntimeError("MT5 remote bridge token is not configured.")
        return {"Authorization": f"Bearer {token}", "X-Bridge-Token": token, "Accept": "application/json"}

    @staticmethod
    def snapshot(symbol, timeframe, limit, base_url, token, timeout=10):
        try:
            import requests
        except ImportError as ex:
            raise RuntimeError(f"requests package is required for the remote MT5 bridge: {ex}")
        base_url = str(base_url or "").strip().rstrip("/")
        if not base_url:
            raise RuntimeError("MT5 remote bridge URL is not configured.")
        url = f"{base_url}/v1/snapshot"
        requested = canonical_symbol(symbol)
        tf = str(timeframe).upper()
        params = {"symbol": requested, "timeframe": tf, "limit": int(limit)}
        try:
            response = requests.get(url, params=params, headers=RemoteMT5BridgeEngine._headers(token), timeout=int(timeout))
        except Exception as ex:
            raise RuntimeError(f"Bridge request failed: {ex}")
        if response.status_code != 200:
            detail = response.text[:500]
            raise RuntimeError(f"Bridge HTTP {response.status_code}: {detail}")
        try:
            payload = response.json()
        except Exception as ex:
            raise RuntimeError(f"Bridge returned invalid JSON: {ex}")
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error") or "Bridge returned ok=false"))
        returned_pair = canonical_symbol(payload.get("requested_symbol") or payload.get("symbol") or "")
        if returned_pair != requested:
            raise RuntimeError(f"Bridge pair mismatch: requested {requested}, returned {returned_pair or 'NONE'}")
        returned_tf = str(payload.get("timeframe") or "").upper()
        if returned_tf != tf:
            raise RuntimeError(f"Bridge timeframe mismatch: requested {tf}, returned {returned_tf or 'NONE'}")
        rows = payload.get("bars") or []
        if not rows:
            raise RuntimeError(f"Bridge returned no candles for {display_symbol(symbol)}/{tf}")
        df = MarketDataEngine.normalize(pd.DataFrame(rows))
        if df.empty:
            raise RuntimeError(f"Bridge returned unusable candles for {display_symbol(symbol)}/{tf}")
        meta = {
            "source": "MT5 REMOTE",
            "requested_source": "MT5 REMOTE",
            "symbol": requested,
            "requested_symbol": requested,
            "mt5_symbol": payload.get("mt5_symbol", requested),
            "timeframe": tf,
            "read_only": True,
            "execution_enabled": False,
            "tick": payload.get("tick"),
            "bridge_url": base_url,
            "bridge_server_time": payload.get("server_time"),
        }
        return df, meta


# ============================================================
# V13 ADDITION: MT5 LIVE MARKET DATA (READ-ONLY)
# ============================================================
class MT5LiveDataEngine:
    """Read-only MT5 bridge for live ticks and OHLCV bars."""
    TF_MAP = {"M1": "TIMEFRAME_M1", "M2": "TIMEFRAME_M2", "M3": "TIMEFRAME_M3",
              "M5": "TIMEFRAME_M5", "M10": "TIMEFRAME_M10", "M15": "TIMEFRAME_M15",
              "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1"}

    @staticmethod
    def connect(symbol, path=None, login=None, password=None, server=None):
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not installed. Install it with: pip install MetaTrader5")
        kwargs = {}
        if login and password and server:
            kwargs = {"login": int(login), "password": str(password), "server": str(server)}
        ok = mt5.initialize(path=path, **kwargs) if path else mt5.initialize(**kwargs)
        if not ok:
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        if not mt5.symbol_select(symbol, True):
            err = mt5.last_error()
            mt5.shutdown()
            raise RuntimeError(f"MT5 symbol_select failed for {symbol}: {err}")
        return True

    @staticmethod
    def tick(symbol):
        if mt5 is None:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        bid = float(getattr(tick, "bid", 0.0))
        ask = float(getattr(tick, "ask", 0.0))
        last = float(getattr(tick, "last", 0.0))
        return {"symbol": symbol, "bid": bid, "ask": ask,
                "last": last if last > 0 else (bid + ask) / 2.0,
                "timestamp": pd.to_datetime(int(getattr(tick, "time", 0)), unit="s", utc=True),
                "spread": max(0.0, ask - bid), "source": "MT5"}

    @staticmethod
    def bars(symbol, timeframe="M5", outputsize=500):
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not installed.")
        tf_name = MT5LiveDataEngine.TF_MAP.get(timeframe, "TIMEFRAME_M5")
        tf = getattr(mt5, tf_name)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(outputsize))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 bars unavailable for {symbol}/{timeframe}: {mt5.last_error()}")
        df = pd.DataFrame(rates).rename(columns={"time": "timestamp", "tick_volume": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return MarketDataEngine.normalize(df[cols])

    @staticmethod
    def shutdown():
        if mt5 is not None:
            try:
                mt5.shutdown()
            except Exception:
                pass

# ============================================================
# END MT5 LIVE MARKET DATA
# ============================================================


def _daily_analysis_config(cfg: Config) -> Config:
    """Daily analysis uses a daily-appropriate freshness window without altering intraday settings."""
    return replace(cfg, data_max_age_seconds=max(int(cfg.data_max_age_seconds), 172800))


def get_daily_market_data(symbol: str, cfg: Config, force: bool = False):
    """Fetch the exact selected pair on D1 for the upfront daily outlook."""
    return get_live_pair_data(canonical_symbol(symbol), "D1", cfg, force=force)


def analyze_daily_market(df: pd.DataFrame, symbol: str, cfg: Config):
    """Run the existing full V13 analysis stack on the selected pair's D1 candles."""
    return analyze_market(df, canonical_symbol(symbol), _daily_analysis_config(cfg), "D1")



def init_state():
    """Initialize every Streamlit session-state value used by V13.

    This function is intentionally conservative: it only creates missing keys and
    never overwrites existing user/session data.  The previous V13 daily-outlook
    patch called init_state() but did not include its definition, which caused the
    Streamlit NameError shown on the dashboard before any market-data logic could run.
    """
    defaults = {
        "data": _empty_market_data(),
        "data_meta": {},
        "data_symbol": None,
        "data_timeframe": None,
        "data_source_loaded": None,
        "live_pair_cache": {},
        "live_status": {
            "source": "NONE",
            "connected": False,
            "read_only": True,
            "execution_enabled": False,
        },
        "journal": [],
        "paper_balance": 10000.0,
        "bot_enabled": False,
        "emergency": False,
        "events": pd.DataFrame(),
        "cot": pd.DataFrame(),
        "backtest": pd.DataFrame(),
        "bt_metrics": {},
        "wf": {},
        "mc": {},
        "optimizer": pd.DataFrame(),
        "last_signal_time": None,
        "td_key": "",
        "mt5_path": "",
        "mt5_connected": False,
        "mt5_remote_connected": False,
        "mt5_refresh_seconds": 15,
        "mt5_bridge_url": "",
        "mt5_bridge_token": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            # Mutable defaults must receive a fresh object per newly-created key.
            if isinstance(value, pd.DataFrame):
                st.session_state[key] = value.copy()
            elif isinstance(value, dict):
                st.session_state[key] = dict(value)
            elif isinstance(value, list):
                st.session_state[key] = list(value)
            else:
                st.session_state[key] = value


def dashboard():
    st.set_page_config(
        page_title="V13 AI Trading Platform",
        page_icon="📈",
        layout="wide",
    )
    init_state()

    cfg = Config()
    st.title("V13 AI Trading Platform · MT5 Remote Bridge")
    st.caption(
        "Forex + Binary research, paper trading, market intelligence, "
        "risk control and backtesting terminal"
    )

    with st.sidebar:
        st.header("⚙️ Control Center")
        pair_options = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD", "AUD/CAD",
            "EUR/GBP", "EUR/JPY", "GBP/JPY", "XAU/USD",
        ]
        selected_pair_label = st.selectbox("Primary currency pair", pair_options, index=0)
        symbol = canonical_symbol(selected_pair_label)
        st.caption(f"Internal feed key: {symbol} · Dashboard: {display_symbol(symbol)}")
        market = st.selectbox("Market", ["FOREX", "BINARY OPTIONS"])
        timeframe = st.selectbox(
            "Primary timeframe",
            ["M5", "M15", "M30", "H1", "H4", "D1"],
            index=1,
        )

        cfg.initial_balance = st.number_input(
            "Paper balance", 100.0, 10000000.0, 10000.0, 100.0
        )
        st.session_state.paper_balance = cfg.initial_balance
        cfg.risk_per_trade = st.slider(
            "Risk / trade %", 0.1, 2.0, 0.5, 0.1
        ) / 100
        cfg.min_score = st.slider("Minimum confluence score", 50, 95, 72)
        cfg.min_binary_confidence = st.slider(
            "Minimum binary confidence", 50, 95, 72
        )
        cfg.binary_payout = st.slider(
            "Binary payout %", 50, 95, 80
        ) / 100
        cfg.max_spread_pips = st.slider(
            "Max spread (pips)", 0.2, 10.0, 2.0, 0.1
        )

        st.divider()
        st.write("**Live Data (read-only)**")
        configured_td_key = _get_twelve_data_key(cfg)
        # Remote MT5 is the primary cloud-compatible live source. Local MT5 is
        # retained for users running Streamlit on the same Windows machine.
        source_options = ["FXCM", "MT5 REMOTE", "MT5", "TWELVE DATA", "DEMO"]
        data_source = st.selectbox("Data source", source_options, index=0)
        cfg.data_source = data_source
        cfg.allow_live_source_failover = False
        if data_source == "MT5 REMOTE":
            st.caption("MT5 REMOTE is the recommended Streamlit Cloud mode: the MT5 terminal stays on a Windows PC/VPS and V13 reads it through the secure bridge.")
            bridge_url_default = _get_mt5_bridge_setting("MT5_BRIDGE_URL", st.session_state.get("mt5_bridge_url", ""))
            bridge_token_default = _get_mt5_bridge_setting("MT5_BRIDGE_TOKEN", st.session_state.get("mt5_bridge_token", ""))
            cfg.mt5_bridge_url = st.text_input(
                "MT5 bridge URL",
                value=bridge_url_default,
                placeholder="https://your-secure-bridge.example.com",
                help="HTTPS URL of the Windows MT5 bridge. Do not paste a local 127.0.0.1 URL when V13 is on Streamlit Cloud.",
            ).strip().rstrip("/")
            st.session_state.mt5_bridge_url = cfg.mt5_bridge_url
            cfg.mt5_bridge_token = st.text_input(
                "MT5 bridge token", type="password",
                value=bridge_token_default,
                help="Use the same bearer token configured on the Windows MT5 bridge.",
            )
            st.session_state.mt5_bridge_token = cfg.mt5_bridge_token
            cfg.mt5_outputsize = st.number_input("MT5 historical candles", 200, 5000, 500, 100, key="remote_mt5_outputsize")
            cfg.live_refresh_seconds = st.number_input("Remote MT5 refresh interval (seconds)", 5, 300, 15, 5, key="remote_mt5_refresh_seconds")
            cfg.mt5_bridge_timeout_seconds = st.number_input("Bridge request timeout (seconds)", 3, 60, 10, 1, key="mt5_bridge_timeout")
            auto_refresh = st.checkbox("Auto-refresh remote MT5 data", value=True, key="remote_mt5_auto_refresh")
            if auto_refresh:
                try:
                    from streamlit_autorefresh import st_autorefresh
                    st_autorefresh(interval=int(cfg.live_refresh_seconds) * 1000, key="v13_remote_mt5_refresh")
                except Exception:
                    st.caption("Install streamlit-autorefresh to enable automatic remote refresh; the manual refresh button remains available.")
            if st.button("🟢 Test / Refresh Remote MT5", use_container_width=True):
                try:
                    live_df, meta = RemoteMT5BridgeEngine.snapshot(symbol, timeframe, cfg.mt5_outputsize, cfg.mt5_bridge_url, cfg.mt5_bridge_token, cfg.mt5_bridge_timeout_seconds)
                    live_df, meta = store_market_data(symbol, timeframe, live_df, meta, "MT5 REMOTE")
                    st.session_state.mt5_remote_connected = True
                    st.session_state.live_status = LiveConnectionManager.status("MT5 REMOTE", live_df, MarketDataEngine.validate(live_df), DataIntegrityEngine.assess(live_df, timeframe, cfg.data_max_age_seconds), meta)
                    tick = meta.get("tick") or {}
                    if tick:
                        st.success(f"Remote MT5 LIVE: {display_symbol(symbol)} · {timeframe} · Bid {tick.get('bid')} · Ask {tick.get('ask')} · {len(live_df):,} candles.")
                    else:
                        st.success(f"Remote MT5 connected: {len(live_df):,} candles loaded for {display_symbol(symbol)} · {timeframe}.")
                except Exception as ex:
                    st.session_state.mt5_remote_connected = False
                    st.session_state.live_status = {"source":"MT5 REMOTE","connected":False,"read_only":True,"execution_enabled":False,"error":str(ex)}
                    st.error(f"Remote MT5 bridge error: {ex}")
        elif data_source == "FXCM":
            st.caption("FXCM LIVE is read-only. No FXCM order endpoint is used.")
            fxcm_env = _get_fxcm_setting("FXCM_ENVIRONMENT", "demo").lower()
            st.info(
                f"FXCM environment: {fxcm_env.upper()} · "
                f"Login configured: {'YES' if _get_fxcm_setting('FXCM_LOGIN_ID') else 'NO'}"
            )
            cfg.fxcm_outputsize = st.number_input(
                "FXCM historical candles", 200, 5000, 500, 100, key="fxcm_outputsize"
            )
            cfg.fxcm_timeout_seconds = st.number_input(
                "FXCM request timeout (seconds)", 5, 60, 20, 5, key="fxcm_timeout_seconds"
            )
            auto_refresh = st.checkbox("Auto-refresh FXCM data", value=True, key="fxcm_auto_refresh")
            if auto_refresh:
                try:
                    from streamlit_autorefresh import st_autorefresh
                    st_autorefresh(
                        interval=int(cfg.live_refresh_seconds) * 1000,
                        key="v13_fxcm_refresh"
                    )
                except Exception:
                    st.caption(
                        "Install streamlit-autorefresh for automatic FXCM refresh; "
                        "the manual refresh button remains available."
                    )
            if st.button("🟢 Connect / Refresh FXCM Live Data", use_container_width=True):
                try:
                    live_df, meta = FXCMLiveDataEngine.candles(
                        symbol, timeframe, cfg.fxcm_outputsize, cfg.fxcm_timeout_seconds
                    )
                    live_df, meta = store_market_data(symbol, timeframe, live_df, meta, "FXCM")
                    st.session_state.live_status = LiveConnectionManager.status(
                        "FXCM", live_df, MarketDataEngine.validate(live_df),
                        DataIntegrityEngine.assess(
                            live_df, timeframe, cfg.data_max_age_seconds
                        ), meta
                    )
                    tick = meta.get("tick") or {}
                    st.success(
                        f"FXCM LIVE: {display_symbol(symbol)} · {timeframe} · "
                        f"Bid {tick.get('bid')} · Ask {tick.get('ask')} · "
                        f"{len(live_df):,} candles loaded."
                    )
                except Exception as ex:
                    st.session_state.live_status = {
                        "source":"FXCM","connected":False,"read_only":True,
                        "execution_enabled":False,"error":str(ex)
                    }
                    st.error(f"FXCM error: {ex}")
        elif data_source == "TWELVE DATA":

            cfg.twelve_data_api_key = st.text_input("Twelve Data API key", type="password", value=st.session_state.get("td_key", configured_td_key))
            st.session_state.td_key = cfg.twelve_data_api_key
            cfg.twelve_data_outputsize = st.number_input("Historical candles", 200, 5000, 500, 100)
            if st.button("🔄 Refresh Twelve Data", use_container_width=True):
                try:
                    live_df, meta = TwelveDataLiveEngine.fetch(cfg.twelve_data_api_key, symbol, timeframe, cfg.twelve_data_outputsize)
                    live_df, meta = store_market_data(symbol, timeframe, live_df, meta, "TWELVE DATA")
                    st.session_state.live_status = LiveConnectionManager.status("TWELVE DATA", live_df, MarketDataEngine.validate(live_df), DataIntegrityEngine.assess(live_df, timeframe, cfg.data_max_age_seconds), meta)
                    st.success(f"Twelve Data: {len(live_df):,} candles loaded for {display_symbol(symbol)} · {timeframe}.")
                except Exception as ex:
                    st.session_state.live_status = {"source":"TWELVE DATA","connected":False,"read_only":True,"execution_enabled":False,"error":str(ex)}
                    st.error(f"Twelve Data error: {ex}")
        elif data_source == "MT5":
            st.caption("MT5 provides live broker/demo prices to V13. This adapter is read-only; it never places orders.")
            mt5_terminal_path = st.text_input("MT5 terminal path (optional)", value=st.session_state.get("mt5_path", ""), help="Leave blank when the default MT5 terminal is installed.")
            st.session_state.mt5_path = mt5_terminal_path
            cfg.mt5_outputsize = st.number_input("MT5 historical candles", 200, 5000, 500, 100, key="mt5_outputsize")
            cfg.live_refresh_seconds = st.number_input(
                "MT5 bar refresh interval (seconds)", 5, 300, 15, 5, key="mt5_refresh_seconds"
            )
            st.caption("MT5-FIRST: no Twelve Data fallback is used if MT5 fails. The bot fails closed rather than changing data source.")
            if st.button("🟢 Connect / Refresh MT5 Live Data", use_container_width=True):
                try:
                    MT5LiveDataEngine.connect(symbol, path=mt5_terminal_path or None)
                    live_df = MT5LiveDataEngine.bars(symbol, timeframe, cfg.mt5_outputsize)
                    tick = MT5LiveDataEngine.tick(symbol)
                    meta = {"source":"MT5", "read_only":True, "execution_enabled":False, "tick":tick}
                    live_df, meta = store_market_data(symbol, timeframe, live_df, meta, "MT5")
                    st.session_state.mt5_connected = True
                    st.session_state.live_status = LiveConnectionManager.status("MT5", live_df, MarketDataEngine.validate(live_df), DataIntegrityEngine.assess(live_df, timeframe, cfg.data_max_age_seconds), meta)
                    if tick:
                        st.success(f"MT5 live data: {symbol} · {timeframe} · Bid {tick['bid']} · Ask {tick['ask']} · {len(live_df):,} candles loaded.")
                    else:
                        st.success(f"MT5 connected: {len(live_df):,} candles loaded for {symbol} · {timeframe}.")
                except Exception as ex:
                    st.session_state.mt5_connected = False
                    st.session_state.live_status = {"source":"MT5","connected":False,"read_only":True,"execution_enabled":False,"error":str(ex)}
                    st.error(f"MT5 error: {ex}")
        else:
            st.info("Demo mode. No external data request is made.")

        st.divider()
        st.write("**Bot safety**")
        st.session_state.bot_enabled = st.toggle(
            "Enable paper bot", st.session_state.bot_enabled
        )

        if st.button("⏸ Pause New Trades", use_container_width=True):
            st.session_state.bot_enabled = False

        if st.button("🚨 EMERGENCY STOP", use_container_width=True):
            st.session_state.emergency = True
            st.session_state.bot_enabled = False

        if st.button("Reset Emergency", use_container_width=True):
            st.session_state.emergency = False

        st.caption(
            "Live broker execution is intentionally disabled until an official adapter is connected."
        )

    st.subheader("1 · Market Data")
    up = st.file_uploader("Upload OHLCV CSV", type=["csv"])

    if up:
        try:
            csv_df = MarketDataEngine.normalize(pd.read_csv(up))
            store_market_data(symbol, timeframe, csv_df, {"source": "CSV", "symbol": symbol, "timeframe": timeframe}, "CSV")
            st.success(f"Loaded {len(st.session_state.data):,} candles for {display_symbol(symbol)} · {timeframe}.")
        except Exception as e:
            st.error(f"CSV error: {e}")
    else:
        st.info(
            f"{cfg.data_source} mode active. Primary pair/timeframe: {display_symbol(symbol)} · {timeframe}. "
            "In MT5-first mode, the selected pair/timeframe is fetched directly from MT5; no other pair or data source is substituted."
        )

    # Pair/data synchronization gate: the dataframe must belong to the exact selected pair/timeframe.
    try:
        df, resolved_meta = get_selected_market_data(symbol, timeframe, cfg)
    except Exception as ex:
        # Hard fail closed: never keep the previous pair's candles after a failed switch.
        clear_market_data_identity(clear_candles=True)
        st.session_state.live_status = {
            "source": cfg.data_source, "connected": False, "read_only": True,
            "execution_enabled": False, "error": str(ex)
        }
        st.error(f"Market data error for {display_symbol(symbol)}/{timeframe}: {ex}")
        if str(cfg.data_source).upper() == "MT5":
            st.warning(
                "MT5 is the authoritative live source in this V13 build. MT5 is currently unavailable, "
                "so live analysis is blocked. Twelve Data is NOT used as an automatic fallback and synthetic "
                "candles are never substituted for MT5 live data. Select Twelve Data manually only if you "
                "intentionally want to switch sources."
            )
        df = _empty_market_data()
        resolved_meta = {}

    if cfg.data_source == "MT5" and st.session_state.get("mt5_connected") and not df.empty:
        tick = MT5LiveDataEngine.tick(symbol)
        if tick:
            st.info(f"🟢 MT5 LOCAL LIVE · {display_symbol(symbol)} · Bid {tick['bid']} · Ask {tick['ask']} · Spread {tick['spread']}")
    elif cfg.data_source == "MT5 REMOTE" and not df.empty:
        tick = (st.session_state.get("data_meta", {}) or {}).get("tick") or {}
        if tick:
            st.info(f"🟢 MT5 REMOTE LIVE · {display_symbol(symbol)} · Bid {tick.get('bid')} · Ask {tick.get('ask')} · Spread {tick.get('spread')} · Bridge verified")
    validation = MarketDataEngine.validate(df) if not df.empty else {
        "rows": 0, "data_ok": False, "duplicates_removed": 0,
        "missing_ohlc": 0, "large_gaps": 0, "timezone": "UTC"
    }
    data_quality = DataIntegrityEngine.assess(df, timeframe, cfg.data_max_age_seconds)

    analysis_error = None
    if not df.empty:
        try:
            a = analyze_market(df, symbol, cfg, timeframe)
        except Exception as ex:
            analysis_error = str(ex)
            a = build_no_data_analysis(symbol, cfg, data_quality, f"ANALYSIS ERROR: {ex}")
    else:
        a = build_no_data_analysis(symbol, cfg, data_quality, "NO MARKET DATA")

    if analysis_error:
        st.error(f"Analysis engine error for {display_symbol(symbol)}/{timeframe}: {analysis_error}")
        st.warning("All dependent signal/trade panels have been safely disabled until valid data is available.")

    resolved_source = str(st.session_state.get("data_meta", {}).get("source", cfg.data_source)).upper()
    st.session_state.live_status = LiveConnectionManager.status(
        resolved_source, df, validation, data_quality, st.session_state.get("data_meta", {})
    )
    if st.session_state.get("data_meta", {}).get("fallback_reason") and str(cfg.data_source).upper() != "MT5":
        st.warning(
            f"LIVE source fallback active: requested {cfg.data_source}, loaded {resolved_source}. "
            f"Reason: {st.session_state.data_meta.get('fallback_reason')}"
        )

    # Keep paper monitoring synchronized with the current price.
    if st.session_state.journal and not df.empty:
        TradeMonitorEngine.monitor_all(
            st.session_state.journal, float(df.close.iloc[-1])
        )

    loaded_symbol = canonical_symbol(st.session_state.get("data_symbol", ""))
    selected_symbol = canonical_symbol(symbol)
    loaded_tf = str(st.session_state.get("data_timeframe", "") or "").upper()
    loaded_source = str(st.session_state.get("data_source_loaded", "") or st.session_state.get("data_meta", {}).get("source", "")).upper()
    sync_ok = bool(not df.empty and loaded_symbol == selected_symbol and loaded_tf == str(timeframe).upper() and loaded_source)
    if sync_ok:
        st.success(f"🔗 DATA SYNC LOCKED · Dashboard {display_symbol(selected_symbol)}/{timeframe} = Loaded {display_symbol(loaded_symbol)}/{loaded_tf} · Source {loaded_source}")
    else:
        st.error(f"⛔ DATA SYNC BLOCKED · Dashboard {display_symbol(selected_symbol)}/{timeframe} does not match loaded data {display_symbol(loaded_symbol or 'NONE')}/{loaded_tf or 'NONE'} · Source {loaded_source or 'NONE'}")

    # ========================================================
    # DAILY MARKET OUTLOOK — runs before the intraday dashboard
    # ========================================================
    st.subheader(f"🌅 Daily Market Outlook · {display_symbol(symbol)}")
    daily_df = _empty_market_data()
    daily_a = None
    daily_meta = {}
    try:
        daily_df, daily_meta = get_daily_market_data(symbol, cfg)
        if daily_df is None or daily_df.empty:
            raise RuntimeError("No D1 candles returned for the selected pair.")
        daily_a = analyze_daily_market(daily_df, symbol, cfg)
    except Exception as daily_ex:
        st.warning(
            f"Daily outlook unavailable for {display_symbol(symbol)}: {daily_ex}. "
            "No substitute pair is used."
        )

    if daily_df is not None and not daily_df.empty and daily_a:
        daily_c = daily_a.get("c", {}) or {}
        daily_t = daily_a.get("t", {}) or {}
        daily_m = daily_a.get("m", {}) or {}
        daily_ai = daily_a.get("ai", {}) or {}
        daily_direction = str(daily_c.get("direction", "WAIT")).upper()
        daily_score = float(daily_c.get("score", 0.0) or 0.0)
        daily_trend = str(daily_t.get("label", "-"))
        daily_momentum = str(daily_m.get("direction", "-"))
        daily_regime = str(daily_a.get("re", "-"))
        daily_up = float(daily_ai.get("up_probability", 0.0) or 0.0)
        daily_down = float(daily_ai.get("down_probability", 0.0) or 0.0)
        daily_quality = DataIntegrityEngine.assess(
            daily_df, "D1", max(int(cfg.data_max_age_seconds), 172800)
        )

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Daily Bias", daily_direction)
        d2.metric("Daily Score", f"{daily_score:.1f}/100")
        d3.metric("Trend", daily_trend)
        d4.metric("Momentum", daily_momentum)
        d5.metric("Regime", daily_regime)
        d6, d7, d8 = st.columns(3)
        d6.metric("AI Daily UP", f"{daily_up:.1f}%")
        d7.metric("AI Daily DOWN", f"{daily_down:.1f}%")
        d8.metric("D1 Candles", f"{len(daily_df):,}")

        daily_source = str(daily_meta.get("source", cfg.data_source)).upper()
        st.caption(
            f"Daily source: {daily_source} · {display_symbol(symbol)}/D1 · "
            f"Last candle: {daily_df['time'].iloc[-1]} UTC · "
            f"Age: {daily_quality.get('age_seconds', 0):.0f}s · Quality: {daily_quality.get('status', 'UNKNOWN')}"
        )
        close = float(daily_df["close"].iloc[-1])
        high20 = float(daily_df["high"].tail(20).max())
        low20 = float(daily_df["low"].tail(20).min())
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Daily Close", fmt(close, 5))
        dc2.metric("20-Day High", fmt(high20, 5))
        dc3.metric("20-Day Low", fmt(low20, 5))

        if daily_direction == "WAIT":
            st.info(
                "Daily planning bias: WAIT. The daily engines do not show enough agreement "
                "to force a directional view. Let the intraday timeframe confirm before considering a setup."
            )
        else:
            st.info(
                f"Daily planning bias: {daily_direction}. The D1 analysis is the higher-timeframe context; "
                "the selected intraday timeframe must still confirm before any setup is considered. "
                "This is research analysis, not a guaranteed forecast."
            )
        with st.expander("View full Daily Engine Analysis", expanded=False):
            st.json({
                "Trend": daily_a.get("trend"),
                "Momentum": daily_a.get("momentum"),
                "Volatility": daily_a.get("volatility"),
                "Structure": daily_a.get("structure"),
                "Price Action": daily_a.get("price_action"),
                "Support/Resistance": daily_a.get("sr"),
                "Breakout": daily_a.get("breakout"),
                "Regime": daily_a.get("regime"),
                "Confluence": daily_a.get("confluence"),
                "AI": daily_a.get("ai"),
                "No Trade": daily_a.get("no_trade"),
                "Trade Quality": daily_a.get("trade_quality"),
            })
    else:
        st.warning(
            f"NO VERIFIED DAILY DATA · {display_symbol(symbol)}/D1. "
            "The daily outlook will appear when the selected pair's D1 feed loads successfully."
        )

    st.subheader("2 · Command Center")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Balance", f"${st.session_state.paper_balance:,.2f}")
    c2.metric("Equity", f"${st.session_state.paper_balance:,.2f}")
    c3.metric("Daily P/L", "$0.00")
    c4.metric("Drawdown", "0.00%")
    c5.metric(
        "Bot",
        "🟢 PAPER ON" if st.session_state.bot_enabled else "⚪ PAUSED",
    )
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Data Source", st.session_state.live_status.get("source", cfg.data_source))
    q2.metric("Data Quality", f"{data_quality['score']:.0f}/100")
    q3.metric("Data Age", f"{data_quality.get('age_seconds',0):.0f}s")
    q4.metric("Read Only", "YES")
    q5.metric("Execution", "DISABLED")
    st.caption(
        f"Pair: {display_symbol(symbol)} · Loaded data pair: {display_symbol(st.session_state.get('data_symbol') or 'NONE')} · "
        f"Timeframe: {timeframe} · Loaded source: {st.session_state.get('data_meta', {}).get('source', 'NONE')} · "
        f"Data: {'OK' if validation['data_ok'] else 'INSUFFICIENT'} · UTC · "
        f"{validation['rows']:,} candles · Quality: {data_quality['status']} · "
        f"Emergency: {'STOPPED' if st.session_state.emergency else 'NORMAL'}"
    )
    if str(cfg.data_source).upper() in {"TWELVE DATA", "MT5 REMOTE", "MT5"} and not st.session_state.live_status.get("connected",False):
        st.warning(f"{cfg.data_source} is selected but no successful live fetch is currently loaded. Signal generation is disabled until the selected pair returns valid candles.")

    # Authoritative market-clock banner. This is independent of candle age.
    if a.get("session", {}).get("market_open"):
        st.success(
            f"🟢 FOREX MARKET OPEN · {a['session']['session']} · "
            f"UTC {a['session']['utc_timestamp']} · New York {a['session']['new_york_timestamp']}"
        )
    else:
        st.error(
            f"🔴 FOREX MARKET CLOSED · {a['session']['session']} · "
            f"UTC {a['session']['utc_timestamp']} · New York {a['session']['new_york_timestamp']} · "
            "ALL NEW SIGNALS AND PAPER ENTRIES ARE BLOCKED."
        )

    if not df.empty:
        st.info(
            f"PAIR DATA LOCK · {display_symbol(selected_symbol)}/{timeframe} · {resolved_source} · {len(df):,} candles. "
            "All analysis engines below receive this exact dataset; no other pair is substituted."
        )
    else:
        st.warning(
            f"NO VERIFIED MARKET DATA · {display_symbol(selected_symbol)}/{timeframe}. All signal/trade calculations are fail-closed; "
            "no synthetic candles are used in live modes."
        )

    tabs = st.tabs(
        [
            "📡 Market Scanner",
            "🧠 Intelligence",
            "🎯 Trade Desk",
            "🛡️ Risk",
            "🧪 Backtest Lab",
            "📓 Journal",
            "🚀 V13 AI/Data",
        ]
    )

    with tabs[0]:
        st.subheader("Market Scanner")
        watch = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
            "AUDUSD", "USDCAD", "NZDUSD", "AUDCAD",
        ]
        analyses = {}
        scanner_rows = []
        for sym in watch:
            try:
                if canonical_symbol(sym) == canonical_symbol(symbol):
                    x = df
                else:
                    x, _ = get_live_pair_data(sym, timeframe, cfg)
                if x is None or x.empty:
                    raise RuntimeError("empty feed")
                aa = analyze_market(x, sym, cfg, timeframe)
                analyses[sym] = aa
            except Exception as ex:
                # A scanner row with NO DATA is safer than silently analyzing synthetic data.
                analyses[sym] = {
                    "confluence": {"direction":"WAIT","score":0.0,"quality":0.0},
                    "regime":"NO DATA",
                    "volatility":{"regime":"NO DATA"},
                    "session":{"session":"-"},
                    "trend":{"label":"NO DATA"},
                    "momentum":{"direction":"NO DATA"},
                    "structure":{"direction":"NO DATA"},
                    "risk":{"status":"NO DATA"},
                    "_error": str(ex),
                }

        for sym in watch:
            aa = analyses[sym]
            if aa.get("_error"):
                scanner_rows.append({
                    "SYMBOL": sym, "DIRECTION":"NO DATA", "SCORE":0.0, "REGIME":"NO DATA",
                    "VOLATILITY":"NO DATA", "SESSION":"-", "TREND":"NO DATA", "MOMENTUM":"NO DATA",
                    "STRUCTURE":"NO DATA", "ENTRY QUALITY":0.0, "RISK":"NO DATA",
                })

        # Use the normal ranking engine only for successfully loaded pairs.
        valid_symbols = [sym for sym in watch if not analyses[sym].get("_error")]
        if valid_symbols:
            ranked = MarketTrackerEngine.rank(valid_symbols, analyses)
            scanner_rows.extend(ranked.to_dict("records"))
        scanner = pd.DataFrame(scanner_rows).sort_values("SCORE", ascending=False).reset_index(drop=True) if scanner_rows else pd.DataFrame()
        if not scanner.empty and "SYMBOL" in scanner.columns:
            scanner["SYMBOL"] = scanner["SYMBOL"].map(display_symbol)
        st.dataframe(scanner, use_container_width=True, hide_index=True)

        if not scanner.empty:
            top = scanner.iloc[0]
            if str(top.DIRECTION) == "NO DATA":
                st.warning(
                    f"TOP OPPORTUNITY: {top.SYMBOL} — NO VERIFIED DATA — 0.0/100. "
                    "No trade signal is generated until a valid feed is loaded."
                )
            else:
                st.success(
                    f"TOP OPPORTUNITY: {top.SYMBOL} — "
                    f"{top.DIRECTION} — {top.SCORE}/100"
                )

        st.subheader("Price Chart")
        if not df.empty:
            chart = df.set_index("time")[["close"]].tail(300)
            st.line_chart(chart)
        else:
            st.warning(f"No data available for {display_symbol(symbol)} · {timeframe}.")

    with tabs[1]:
        st.subheader("Market Intelligence Core")
        cols = st.columns(4)
        cols[0].metric("Confluence", f"{a['c']['score']:.1f}/100")
        cols[1].metric("Trend", a["t"]["label"])
        cols[2].metric("Momentum", a["m"]["direction"])
        cols[3].metric("Regime", a["re"])
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("AI UP", f"{a['ai']['up_probability']:.1f}%")
        x2.metric("AI DOWN", f"{a['ai']['down_probability']:.1f}%")
        x3.metric("AI Confidence", f"{a['ai']['confidence']:.1f}%")
        x4.metric("Signal Confidence", f"{a['ensemble'].get('confidence', a['ensemble'].get('score', 0)):.1f}%")
        st.markdown("### Advanced AI / Decision Layer")
        st.caption(
            f"Direction: {a['ensemble'].get('direction','WAIT')} · "
            f"Engine agreement: {a['ensemble'].get('agreement',0):.0f}% · "
            f"AI model confidence: {a['ai'].get('confidence',0):.1f}%"
        )
        st.json({"AI":a["ai"],"Direct Probability":a.get("direct_probability"),"Advanced Momentum":a["advanced_momentum"],"Volume Flow":a.get("volume_flow"),"Candle Timing":a.get("candle_timing"),"Market Region":a.get("market_region"),"Ensemble":a["ensemble"],"Signal Explanation":a.get("signal_explanation"),"Risk Veto":a.get("risk_veto"),"No Trade":a["no_trade"],"Trade Quality":a["trade_quality"],"Data Quality":a["data_quality"]})
        st.markdown("### Engine Scoreboard")
        comp = pd.DataFrame(
            {
                "Engine": list(a["c"]["components"].keys()),
                "Score": list(a["c"]["components"].values()),
            }
        )
        st.bar_chart(comp.set_index("Engine"))

        left, right = st.columns(2)
        with left:
            for title, key in [
                ("Trend", "t"),
                ("Momentum", "m"),
                ("Market Structure", "s"),
                ("Price Action", "pa"),
                ("Support / Resistance", "sr"),
            ]:
                st.markdown(f"**{title}**")
                st.json(a[key])

        with right:
            for title, key in [
                ("Volatility", "v"),
                ("Breakout", "bo"),
                ("Liquidity / FVG", "li"),
                ("Session", "se"),
            ]:
                st.markdown(f"**{title}**")
                st.json(a[key])

            st.markdown("**Multi-Timeframe Alignment**")
            st.dataframe(
                pd.DataFrame([a["mtf"]["states"]]),
                use_container_width=True,
            )
            st.metric(
                "MTF alignment",
                f"{a['mtf']['alignment']:.0f}% ({a['mtf']['strength']})",
            )

        st.markdown("### Currency Strength")
        cs = a.get("currency_strength") or a.get("cs") or {}
        cs_matrix = cs.get("matrix") if isinstance(cs, dict) else None
        if isinstance(cs_matrix, dict) and cs_matrix:
            st.bar_chart(pd.Series(cs_matrix, dtype=float))
        else:
            st.info(
                f"Currency strength unavailable for {display_symbol(symbol)} · {timeframe}. "
                "Waiting for valid market data; no fallback/synthetic values are displayed."
            )

        st.markdown("### Economic Engine")
        e1, e2, e3 = st.columns(3)
        e1.metric("Macro Bias", a["eco"]["bias"])
        e2.metric("Macro Score", f"{a['eco']['score']:.0f}")
        e3.metric("Event Risk", a["eco"]["risk"])
        if a["eco"]["blocked"]:
            st.error("ECONOMIC EVENT BLACKOUT — NO NEW TRADES")
        st.write("Next event:", a["eco"]["next_event"])

        st.markdown("### COT Engine")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("COT Bias", a["cot"]["bias"])
        q2.metric("Net Spec", fmt(a["cot"]["net"], 0))
        q3.metric("Weekly Change", fmt(a["cot"]["weekly_change"], 0))
        q4.metric("Percentile", f"{a['cot']['percentile']:.0f}%")
        st.info(
            "COT is a slower positioning/context signal; it is not treated "
            "as a precise short-term entry trigger."
        )

        st.markdown("### Economic / COT Data Upload")
        ec, cc = st.columns(2)
        with ec:
            ef = st.file_uploader(
                "Economic events CSV", type=["csv"], key="economic_csv"
            )
            if ef:
                try:
                    st.session_state.events = pd.read_csv(ef)
                    st.success("Economic event data loaded.")
                except Exception as ex:
                    st.error(str(ex))
            st.caption(
                "Expected columns: time, currency, importance, event, optional bias."
            )

        with cc:
            cf = st.file_uploader("COT CSV", type=["csv"], key="cot_csv")
            if cf:
                try:
                    st.session_state.cot = pd.read_csv(cf)
                    st.success("COT data loaded.")
                except Exception as ex:
                    st.error(str(ex))
            st.caption(
                "Expected: currency, commercial_long, commercial_short, "
                "noncommercial_long, noncommercial_short."
            )

    with tabs[2]:
        st.subheader("3 · Trade Desk")
        if df.empty:
            st.warning(
                f"NO TRADE DATA · {display_symbol(symbol)} · {timeframe}. Trade calculations are disabled until valid candles are loaded."
            )
            direction = "WAIT"
        else:
            direction = a["ensemble"].get("direction", a["c"]["direction"])

        if not df.empty and market == "FOREX":
            fx = ForexEntryEngine.calculate(df, direction, a["c"], cfg, symbol)
            b1, b2, b3, b4, b5, b6 = st.columns(6)
            b1.metric("Signal", fx["direction"])
            b2.metric("Entry", fmt(fx["entry"], 5))
            b3.metric("SL", fmt(fx["sl"], 5) if fx["sl"] else "-")
            b4.metric("TP", fmt(fx["tp"], 5) if fx["tp"] else "-")
            b5.metric("R:R", fmt(fx["rr"]))
            b6.metric("Lot", fmt(fx["lot"], 3))
            st.write(
                f"Entry zone: **{fmt(fx['zone_low'],5)} – {fmt(fx['zone_high'],5)}** · "
                f"Quality: **{fx.get('quality',0):.1f}/100**"
            )

            timing = SignalTimingEngine.assess(st.session_state.get("last_signal_time"), cfg.signal_max_age_seconds)
            approved = (
                a["risk"]["approved"] and fx["approved"] and a["ensemble"].get("approved", False)
                and a["no_trade"]["trade_allowed"] and a["trade_quality"]["decision"] == "TRADE" and timing["fresh"]
                and not st.session_state.emergency
                and a["session"].get("market_open", False)
            )

            if not a["risk"]["approved"]:
                st.error("RISK VETO: " + ", ".join(a["risk"]["reasons"]))
            if not a["no_trade"]["trade_allowed"]:
                st.error("NO TRADE: " + ", ".join(a["no_trade"]["reasons"]))
            elif not a["ensemble"].get("approved", False):
                st.warning("SIGNAL CONFIDENCE NOT READY: " + ", ".join(a["ensemble"].get("reasons", [])))

            if approved and st.button(
                "🟢 PAPER EXECUTE FOREX", use_container_width=True
            ):
                tr = ExecutionEngine.paper_order(
                    "FOREX",
                    symbol,
                    fx["direction"],
                    fx["entry"],
                    fx["sl"],
                    fx["tp"],
                    lot=fx["lot"],
                )
                TradeJournal.append(tr)
                st.success(f"Paper trade opened: {tr['id']}")
            elif not approved:
                st.warning("Forex trade is NOT approved.")

        elif not df.empty:
            bi = BinaryEntryEngine.calculate(df, direction, a["c"], cfg)
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Signal", bi["direction"])
            b2.metric("Entry/Strike", fmt(bi["entry"], 5))
            b3.metric("Confidence", f"{bi['confidence']:.1f}%")
            b4.metric("Expiry", f"{bi.get('expiry','-')} min")
            b5.metric("EV", f"{bi.get('expected_value',0)*100:.1f}%")

            st.write(
                f"Entry zone: **{fmt(bi['zone_low'],5)} – {fmt(bi['zone_high'],5)}** · "
                f"Payout: **{bi.get('payout',cfg.binary_payout)*100:.0f}%**"
            )

            if bi["expiry_table"]:
                st.dataframe(
                    pd.DataFrame(bi["expiry_table"]),
                    use_container_width=True,
                )

            timing = SignalTimingEngine.assess(st.session_state.get("last_signal_time"), cfg.signal_max_age_seconds)
            approved = (
                a["risk"]["approved"] and bi["approved"] and a["ensemble"].get("approved", False)
                and a["no_trade"]["trade_allowed"] and a["trade_quality"]["decision"] == "TRADE" and timing["fresh"]
                and not st.session_state.emergency
                and a["session"].get("market_open", False)
            )

            if not a["risk"]["approved"]:
                st.error("RISK VETO: " + ", ".join(a["risk"]["reasons"]))
            if not a["no_trade"]["trade_allowed"]:
                st.error("NO TRADE: " + ", ".join(a["no_trade"]["reasons"]))
            elif not a["ensemble"].get("approved", False):
                st.warning("SIGNAL CONFIDENCE NOT READY: " + ", ".join(a["ensemble"].get("reasons", [])))

            if approved and st.button(
                "🟢 PAPER EXECUTE BINARY", use_container_width=True
            ):
                tr = ExecutionEngine.paper_order(
                    "BINARY",
                    symbol,
                    bi["direction"],
                    bi["entry"],
                    expiry=bi["expiry"],
                )
                TradeJournal.append(tr)
                st.success(f"Paper binary trade opened: {tr['id']}")
            elif not approved:
                st.warning("Binary trade is NOT approved.")

        st.markdown("### Decision Hierarchy")
        st.code(
            "Live/History Data → Integrity → V12.1 Engines → MTF/Regime → "
            "AI + Advanced Momentum → Engine Consensus → Signal Confidence → NO-TRADE → Risk → Forex/Binary → PAPER ONLY → Monitoring → Journal"
        )

    with tabs[3]:
        st.subheader("4 · Risk Control Center")
        r = a["risk"]
        cols = st.columns(4)
        cols[0].metric("Risk Status", r["status"])
        cols[1].metric("Risk / Trade", f"{r['risk_pct']:.2f}%")
        cols[2].metric("Max Daily Loss", f"{cfg.max_daily_loss*100:.1f}%")
        cols[3].metric("Max DD", f"{cfg.max_drawdown*100:.1f}%")

        if r["reasons"]:
            st.error("\n".join("• " + x for x in r["reasons"]))
        else:
            st.success("All configured risk veto checks currently pass.")

        st.markdown("### Safety Gates")
        gates = [
            ("SIGNAL VALID", a["c"]["score"] >= cfg.min_score),
            ("RISK VALID", r["approved"]),
            ("VOLATILITY VALID", a["v"]["regime"] != "EXTREME"),
            ("ECONOMIC EVENT VALID", not a["eco"]["blocked"]),
            ("FOREX MARKET OPEN", a["se"].get("market_open", False)),
            ("SESSION VALID", a["se"].get("session_tradeable", False)),
            ("DATA VALID", validation["data_ok"]),
            ("CORRELATION VALID", a["corr"]["risk"] != "HIGH"),
            ("EMERGENCY STOP OFF", not st.session_state.emergency),
        ]
        st.dataframe(
            pd.DataFrame(gates, columns=["Gate", "PASS"]),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[4]:
        st.subheader("5 · Backtest / Walk-Forward / Monte Carlo / Optimizer")
        btcol1, btcol2 = st.columns(2)

        with btcol1:
            if st.button("▶ Run Backtest", use_container_width=True, disabled=df.empty):
                trades, metrics = BacktestEngine.run(
                    df, cfg, binary=(market == "BINARY OPTIONS")
                )
                st.session_state.backtest = trades
                st.session_state.bt_metrics = metrics

        with btcol2:
            if st.button("↔ Run Walk-Forward", use_container_width=True, disabled=df.empty):
                st.session_state.wf = WalkForwardEngine.run(df, cfg)

        if st.button("🎲 Run Monte Carlo", use_container_width=True, disabled=st.session_state.backtest.empty):
            st.session_state.mc = MonteCarloEngine.run(
                st.session_state.backtest
            )

        if st.button("⚙ Run Threshold Optimizer", use_container_width=True, disabled=df.empty):
            st.session_state.optimizer = OptimizerEngine.run(df, cfg)

        if st.session_state.bt_metrics:
            st.json(st.session_state.bt_metrics)

        if not st.session_state.backtest.empty:
            st.line_chart(
                st.session_state.backtest.set_index("time")[["equity"]]
            )
            st.dataframe(
                st.session_state.backtest.tail(100),
                use_container_width=True,
            )

        if st.session_state.wf:
            st.markdown("### Walk-Forward")
            st.json(
                {
                    "train": st.session_state.wf["train"],
                    "out_of_sample": st.session_state.wf["out_of_sample"],
                }
            )

        if st.session_state.mc:
            st.markdown("### Monte Carlo Stress")
            st.json(st.session_state.mc)

        if not st.session_state.optimizer.empty:
            st.markdown("### Optimization")
            st.dataframe(
                st.session_state.optimizer,
                use_container_width=True,
                hide_index=True,
            )

    with tabs[5]:
        st.subheader("6 · Trade Journal / Performance")

        if st.session_state.journal:
            j = pd.DataFrame(st.session_state.journal)
            st.dataframe(j, use_container_width=True, hide_index=True)
            st.download_button(
                "Export Journal CSV",
                j.to_csv(index=False),
                "v12_1_trade_journal.csv",
                "text/csv",
            )
        else:
            st.info("No paper trades yet.")

        st.markdown("### Open Trade Monitor")
        open_trades = [
            t for t in st.session_state.journal if t.get("status") == "OPEN"
        ]
        if open_trades:
            st.dataframe(
                pd.DataFrame(open_trades),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No open paper trades.")

        st.markdown("### Engine Status")
        status = pd.DataFrame(
            [
                ["Market Data", "READY"],
                ["Indicator Utilities", "READY"],
                ["Trend", "READY"],
                ["Momentum", "READY"],
                ["Volatility", "READY"],
                ["Structure", "READY"],
                ["Price Action", "READY"],
                ["Support/Resistance", "READY"],
                ["Breakout", "READY"],
                ["Liquidity/FVG", "READY"],
                ["Regime", "READY"],
                ["Session", "READY"],
                ["Currency Strength", "READY"],
                ["Correlation", "READY"],
                ["MTF", "READY"],
                ["Economic", "READY"],
                ["COT", "READY"],
                ["Confluence", "READY"],
                ["Risk", "READY"],
                ["Forex Entry", "READY"],
                ["Binary Entry", "READY"],
                ["Paper Execution", "READY"],
                ["Trade Monitor", "READY"],
                ["Backtest", "READY"],
                ["Walk-Forward", "READY"],
                ["Monte Carlo", "READY"],
                ["Optimizer", "READY"],
                ["Journal", "READY"],
                ["Advanced Live Data", "READY"],
                ["Twelve Data REST", "READY"],
                ["Data Integrity", "READY"],
                ["Live Connection Manager", "READY"],
                ["Advanced Momentum", "READY"],
                ["Volume Flow", "READY"],
                ["Direct Probability", "READY"],
                ["Candle Timing", "READY"],
                ["Market Region", "READY"],
                ["Break-Even", "READY (PAPER)"],
                ["Analysis Orchestrator", "READY"],
                ["Signal Explanation", "READY"],
                ["Risk Veto", "READY"],
                ["AI Probability", "READY (OPTIONAL ML)"],
                ["Ensemble Decision", "READY"],
                ["Confidence / Trade Quality", "READY"],
                ["Signal Confidence / Consensus", "READY"],
                ["NO TRADE", "READY"],
                ["Signal Timing", "READY"],
                ["Performance Intelligence", "READY"],
                ["Dashboard", "READY"],
            ],
            columns=["Engine / Tool", "Status"],
        )
        st.dataframe(status, use_container_width=True, hide_index=True)

    with tabs[6]:
        st.subheader("V13 Advanced AI / Live Data Control")
        st.info("V13 additions are layered on top of V12.1. Market-data connectivity is read-only; automatic broker order execution is disabled in this build.")
        l1,l2,l3,l4=st.columns(4)
        l1.metric("Source", st.session_state.live_status.get("source",cfg.data_source))
        l2.metric("Connection", "CONNECTED" if st.session_state.live_status.get("connected") else "NOT CONNECTED")
        l3.metric("Quality", f"{data_quality['score']:.0f}/100")
        l4.metric("Execution", "DISABLED")
        st.markdown("### AI Prediction")
        st.dataframe(pd.DataFrame([a["ai"]]), use_container_width=True, hide_index=True)
        st.markdown("### Momentum Direction")
        st.dataframe(pd.DataFrame([a["advanced_momentum"]]), use_container_width=True, hide_index=True)
        st.markdown("### Signal Confidence / Engine Consensus")
        st.dataframe(pd.DataFrame([a["ensemble"]]), use_container_width=True, hide_index=True)
        st.markdown("### Trade Quality Checklist")
        st.dataframe(pd.DataFrame(list(a["trade_quality"]["checks"].items()),columns=["Gate","PASS"]),use_container_width=True,hide_index=True)
        st.metric("Final Trade Quality", f"{a['trade_quality']['grade']} · {a['trade_quality']['score']:.0f}/100")
        st.markdown("### Data Integrity")
        st.json(a["data_quality"])
        st.markdown("### NO-TRADE Engine")
        if a["no_trade"]["trade_allowed"]:
            st.success("TRADE CONDITIONS PASS")
        else:
            st.error("NO TRADE — " + " | ".join(a["no_trade"]["reasons"]))
        st.markdown("### Performance Intelligence")
        perf=PerformanceIntelligenceEngine.summarize(st.session_state.journal)
        p1,p2,p3,p4=st.columns(4)
        p1.metric("Signals",perf["signals"]); p2.metric("Wins",perf["wins"]); p3.metric("Losses",perf["losses"]); p4.metric("Win Rate",f"{perf['win_rate']:.1f}%")
        if not perf["by_key"].empty: st.dataframe(perf["by_key"],use_container_width=True,hide_index=True)

    st.divider()
    st.caption(
        "V13 is a research/paper-trading system. V12.1 engines are preserved; advanced live-data/AI layers are advisory and read-only.  No strategy is guaranteed "
        "profitable. Connect only official, permitted broker/data APIs after "
        "independent testing and compliance review."
    )


if __name__ == "__main__":
    dashboard()



# -------------------- DATA QUALITY & ENGINE HEALTH VERIFICATION --------------------
@dataclass
class EngineHealth:
    name: str
    status: str
    runtime_ms: float
    output_valid: bool
    critical: bool
    error: str = ""

class DataQualityEngine:
    """Independent pre-trade verification of market data quality."""

    REQUIRED = ("time", "open", "high", "low", "close")

    def verify(self, df: pd.DataFrame, max_age_seconds: int = 420) -> Dict[str, Any]:
        checks = {}
        errors = []
        if df is None or df.empty:
            return {"status": "FAILED", "score": 0.0, "checks": {"non_empty": False},
                    "errors": ["No market data"]}

        checks["non_empty"] = True
        missing = [c for c in self.REQUIRED if c not in df.columns]
        checks["required_columns"] = not missing
        if missing:
            errors.append("Missing columns: " + ",".join(missing))
            return {"status": "FAILED", "score": 0.0, "checks": checks, "errors": errors}

        x = df.copy()
        checks["ohlc_numeric"] = True
        for c in ("open", "high", "low", "close"):
            x[c] = pd.to_numeric(x[c], errors="coerce")
        if x[list(("open","high","low","close"))].isna().any().any():
            checks["ohlc_numeric"] = False
            errors.append("Invalid OHLC values")

        checks["positive_prices"] = bool((x[["open","high","low","close"]] > 0).all().all())
        if not checks["positive_prices"]:
            errors.append("Non-positive price detected")

        checks["ohlc_structure"] = bool(
            (x["high"] >= x[["open","close","low"]].max(axis=1)).all()
            and (x["low"] <= x[["open","close","high"]].min(axis=1)).all()
        )
        if not checks["ohlc_structure"]:
            errors.append("Impossible OHLC structure")

        t = pd.to_datetime(x["time"], utc=True, errors="coerce")
        checks["timestamps_valid"] = bool(t.notna().all())
        if not checks["timestamps_valid"]:
            errors.append("Invalid timestamps")
        else:
            checks["chronological"] = bool(t.is_monotonic_increasing)
            checks["duplicates"] = int(t.duplicated().sum())
            if not checks["chronological"]:
                errors.append("Timestamps are not chronological")
            if checks["duplicates"]:
                errors.append("Duplicate timestamps detected")

            latest_age = (pd.Timestamp.now(tz="UTC") - t.iloc[-1]).total_seconds()
            checks["latest_age_seconds"] = float(max(0.0, latest_age))
            checks["fresh"] = latest_age <= max_age_seconds
            if not checks["fresh"]:
                errors.append(f"Stale market data ({latest_age:.0f}s old)")

        checks["finite_values"] = bool(
            np.isfinite(x[["open","high","low","close"]].to_numpy(dtype=float)).all()
        )
        if not checks["finite_values"]:
            errors.append("Non-finite OHLC values")

        score = 100.0
        score -= min(40.0, 15.0 * len(errors))
        score = float(np.clip(score, 0.0, 100.0))
        status = "HEALTHY" if not errors else ("DEGRADED" if score >= 60 else "FAILED")
        return {"status": status, "score": score, "checks": checks, "errors": errors}


class EngineHealthVerificationLayer:
    """Runs analysis engines under a common health contract.

    A failed critical engine or failed market-data verification must veto a
    trade. Advisory failures are exposed as DEGRADED rather than hidden.
    """

    def __init__(self):
        self.data_quality = DataQualityEngine()
        self.history: Dict[str, EngineHealth] = {}

    @staticmethod
    def _valid_output(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            return all(v is not None for v in value.values())
        if isinstance(value, pd.DataFrame):
            return not value.empty
        if isinstance(value, (float, int, np.floating, np.integer)):
            return bool(np.isfinite(value))
        return True

    def run_engine(self, name: str, fn, critical: bool = False, *args, **kwargs):
        import time
        start = time.perf_counter()
        try:
            output = fn(*args, **kwargs)
            valid = self._valid_output(output)
            status = "HEALTHY" if valid else "FAILED"
            err = "" if valid else "Invalid/empty engine output"
        except Exception as exc:
            output = None
            valid = False
            status = "FAILED"
            err = f"{type(exc).__name__}: {exc}"

        health = EngineHealth(
            name=name,
            status=status,
            runtime_ms=(time.perf_counter() - start) * 1000.0,
            output_valid=valid,
            critical=critical,
            error=err,
        )
        self.history[name] = health
        return output, health

    def summary(self) -> Dict[str, Any]:
        items = list(self.history.values())
        failed_critical = [x.name for x in items if x.critical and x.status == "FAILED"]
        failed = [x.name for x in items if x.status == "FAILED"]
        status = "FAILED" if failed_critical else ("DEGRADED" if failed else "HEALTHY")
        return {
            "status": status,
            "total_engines": len(items),
            "healthy_engines": sum(x.status == "HEALTHY" for x in items),
            "failed_engines": len(failed),
            "failed_critical": failed_critical,
            "failed": failed,
            "details": {
                x.name: {
                    "status": x.status,
                    "runtime_ms": round(x.runtime_ms, 2),
                    "output_valid": x.output_valid,
                    "critical": x.critical,
                    "error": x.error,
                } for x in items
            },
            "trade_veto": bool(failed_critical),
        }

    def verify_data(self, df: pd.DataFrame, max_age_seconds: int = 420):
        return self.data_quality.verify(df, max_age_seconds=max_age_seconds)


health_layer = EngineHealthVerificationLayer()

# -------------------- FVG ENGINE REGISTRY --------------------
# Exposed as a standalone engine so the orchestrator/dashboard can consume it
# without replacing or weakening any existing V13 engine.
fvg_engine = FairValueGapEngine()

def analyze_fair_value_gap(df: pd.DataFrame) -> Dict[str, Any]:
    """Return the latest active FVG state for the supplied OHLC data."""
    return fvg_engine.detect(df)



# -------------------- V13 ENGINE HEALTH REGISTRY --------------------
# Every analysis engine is registered here. The registry verifies execution
# and output without replacing the engine's own logic.
V13_ENGINE_REGISTRY = {
    "TrendEngine": ("TrendEngine", True),
    "MomentumEngine": ("MomentumEngine", True),
    "VolatilityEngine": ("VolatilityEngine", True),
    "TechnicalAnalysis": ("TechnicalAnalysis", True),
    "PriceAction": ("PriceAction", True),
    "MarketStructure": ("MarketStructure", True),
    "SupportResistance": ("SupportResistance", True),
    "Breakout": ("Breakout", False),
    "FairValueGap": ("FairValueGap", False),
    "VolumeFlow": ("VolumeFlow", False),
    "MarketRegion": ("MarketRegion", False),
    "CurrencyStrength": ("CurrencyStrength", False),
    "Correlation": ("Correlation", False),
    "MultiTimeframe": ("MultiTimeframe", True),
    "DirectProbability": ("DirectProbability", True),
    "AIMLPrediction": ("AIMLPrediction", True),
    "AIConfluence": ("AIConfluence", True),
    "EnsembleDecision": ("EnsembleDecision", True),
    "Confidence": ("Confidence", True),
    "TradeQuality": ("TradeQuality", True),
    "SignalTiming": ("SignalTiming", True),
    "RiskControl": ("RiskControl", True),
    "RiskVeto": ("RiskVeto", True),
    "NewsEconomicFilter": ("NewsEconomicFilter", True),
    "CandleTiming": ("CandleTiming", False),
    "MomentumDirection": ("MomentumDirection", False),
    "CurrentVolatility": ("CurrentVolatility", False),
    "MarketTracker": ("MarketTracker", False),
}

def _health_call(name: str, fn, df: pd.DataFrame, critical: bool):
    """Execute one engine through the health layer."""
    return health_layer.run_engine(name, fn, critical, df)

def run_all_v13_engine_health(df: pd.DataFrame,
                              engine_functions: Dict[str, Any],
                              max_age_seconds: int = 420) -> Dict[str, Any]:
    """
    Central health gate for V13.

    engine_functions contains the actual callable for each available engine.
    Missing callables are reported explicitly instead of being counted healthy.
    """
    data_quality = health_layer.verify_data(df, max_age_seconds=max_age_seconds)
    health_layer.history.clear()

    if data_quality["status"] == "FAILED":
        return {
            "status": "FAILED",
            "data_quality": data_quality,
            "engine_health": health_layer.summary(),
            "healthy_count": 0,
            "total_required": len(V13_ENGINE_REGISTRY),
            "trade_veto": True,
            "reason": "DATA QUALITY FAILURE",
        }

    missing = []
    for engine_name, (_, critical) in V13_ENGINE_REGISTRY.items():
        fn = engine_functions.get(engine_name)
        if not callable(fn):
            missing.append(engine_name)
            health_layer.history[engine_name] = EngineHealth(
                name=engine_name,
                status="FAILED",
                runtime_ms=0.0,
                output_valid=False,
                critical=critical,
                error="Engine callable not wired into health registry",
            )
            continue
        _health_call(engine_name, fn, df, critical)

    summary = health_layer.summary()
    healthy = summary["healthy_engines"]
    total = len(V13_ENGINE_REGISTRY)
    failed_critical = summary["failed_critical"]

    # A missing/failed critical engine always blocks a trade.
    trade_veto = bool(
        data_quality["status"] == "FAILED"
        or failed_critical
        or len(missing) > 0 and any(
            V13_ENGINE_REGISTRY[x][1] for x in missing
        )
    )

    if trade_veto:
        overall = "FAILED"
    elif summary["failed"]:
        overall = "DEGRADED"
    else:
        overall = "HEALTHY"

    return {
        "status": overall,
        "data_quality": data_quality,
        "engine_health": summary,
        "healthy_count": healthy,
        "total_required": total,
        "failed_count": total - healthy,
        "missing_engines": missing,
        "health_percent": round((healthy / total) * 100.0, 2) if total else 0.0,
        "trade_veto": trade_veto,
    }


def build_v13_engine_functions(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapter registry.

    The existing V13 engine implementations remain authoritative. The
    orchestrator supplies callables here; the health layer only verifies that
    they execute and return valid output.
    """
    return {
        # These adapters use existing engine classes/functions when present.
        # Unavailable optional engines remain explicitly FAILED rather than
        # being falsely reported as healthy.
        "TrendEngine": context.get("trend_fn"),
        "MomentumEngine": context.get("momentum_fn"),
        "VolatilityEngine": context.get("volatility_fn"),
        "TechnicalAnalysis": context.get("technical_fn"),
        "PriceAction": context.get("price_action_fn"),
        "MarketStructure": context.get("structure_fn"),
        "SupportResistance": context.get("support_resistance_fn"),
        "Breakout": context.get("breakout_fn"),
        "FairValueGap": context.get("fvg_fn"),
        "VolumeFlow": context.get("volume_flow_fn"),
        "MarketRegion": context.get("market_region_fn"),
        "CurrencyStrength": context.get("currency_strength_fn"),
        "Correlation": context.get("correlation_fn"),
        "MultiTimeframe": context.get("mtf_fn"),
        "DirectProbability": context.get("probability_fn"),
        "AIMLPrediction": context.get("ml_fn"),
        "AIConfluence": context.get("ai_confluence_fn"),
        "EnsembleDecision": context.get("ensemble_fn"),
        "Confidence": context.get("confidence_fn"),
        "TradeQuality": context.get("trade_quality_fn"),
        "SignalTiming": context.get("signal_timing_fn"),
        "RiskControl": context.get("risk_control_fn"),
        "RiskVeto": context.get("risk_veto_fn"),
        "NewsEconomicFilter": context.get("news_fn"),
        "CandleTiming": context.get("candle_timing_fn"),
        "MomentumDirection": context.get("momentum_direction_fn"),
        "CurrentVolatility": context.get("current_volatility_fn"),
        "MarketTracker": context.get("market_tracker_fn"),
    }


def verify_v13_health(df: pd.DataFrame, context: Optional[Dict[str, Any]] = None,
                      max_age_seconds: int = 420) -> Dict[str, Any]:
    """Single entry point for the dashboard/decision pipeline."""
    context = context or {}
    functions = build_v13_engine_functions(context)
    return run_all_v13_engine_health(
        df, functions, max_age_seconds=max_age_seconds
    )


# -------------------- HEALTH DISPLAY HELPERS --------------------
def v13_health_label(report: Dict[str, Any]) -> str:
    status = report.get("status", "FAILED")
    pct = report.get("health_percent", 0.0)
    return f"{status} — {pct:.1f}% engines healthy"


def v13_trade_gate(report: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (allowed, reason). This is fail-closed."""
    if report.get("trade_veto"):
        if report.get("data_quality", {}).get("status") == "FAILED":
            return False, "NO TRADE: DATA QUALITY FAILURE"
        failed = report.get("engine_health", {}).get("failed_critical", [])
        if failed:
            return False, "NO TRADE: CRITICAL ENGINE FAILURE — " + ", ".join(failed)
        return False, "NO TRADE: ENGINE HEALTH DEGRADED"
    return True, "ENGINE HEALTH PASS"
