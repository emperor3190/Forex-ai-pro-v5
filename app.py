
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
import uuid
import json
import threading
import time as _time
from dataclasses import dataclass, replace
from datetime import datetime, timezone, time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
import streamlit as st

# Optional WebSocket client for Deriv public market-data streaming. READ-ONLY.
try:
    import websocket
except ImportError:
    websocket = None

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
    data_source: str = "DEMO"
    twelve_data_api_key: str = ""
    twelve_data_outputsize: int = 500
    allow_live_source_failover: bool = True
    live_refresh_seconds: int = 300
    data_max_age_seconds: int = 420
    ml_min_probability: float = 0.60
    signal_max_age_seconds: int = 60
    no_trade_conflict_threshold: float = 18.0
    min_signal_confidence: float = 72.0
    ai_soft_floor: float = 25.0
    min_engine_agreement: float = 60.0
    # Twelve Data Basic currently provides 8 API credits/minute. V13 keeps
    # one safety margin and never performs duplicate symbol requests.
    twelve_data_request_budget_per_minute: int = 7
    twelve_data_daily_budget: int = 760
    twelve_data_rate_guard_seconds: int = 60
    twelve_data_retry_after_429: bool = False
    twelve_data_cache_stale_seconds: int = 420
    scanner_refresh_seconds: int = 300
    # Additive Deriv real-Forex market-data stream settings. No order/trading API is used.
    deriv_stream_endpoint: str = "wss://api.derivws.com/trading/v1/options/ws/public"
    deriv_stream_enabled: bool = False
    deriv_stream_outputsize: int = 500
    deriv_stream_reconnect_seconds: int = 5
    deriv_stream_stale_seconds: int = 15


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
    """Authoritative live Forex session/market-calendar engine.

    Rules:
      * Live mode ALWAYS uses the current UTC clock.
      * Historical/backtest mode may use an explicit candle timestamp.
      * Weekly FX open/close is evaluated in New York local time so DST is
        handled correctly (Sunday 17:00 NY -> Friday 17:00 NY).
      * Individual session windows are evaluated in their own local timezone.
      * A session label can never imply that the weekly FX market is open.
    """

    MARKET_ZONES = {
        "SYDNEY": "Australia/Sydney",
        "TOKYO": "Asia/Tokyo",
        "LONDON": "Europe/London",
        "NEW YORK": "America/New_York",
    }

    SESSION_HOURS = {
        "SYDNEY": (8, 17),
        "TOKYO": (9, 18),
        "LONDON": (8, 17),
        "NEW YORK": (8, 17),
    }

    @staticmethod
    def _utc(ts=None, live=False):
        if live or ts is None:
            return pd.Timestamp.now(tz="UTC")
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.tz_convert("UTC")

    @staticmethod
    def _weekly_market_open(utc_ts):
        ny = utc_ts.to_pydatetime().astimezone(ZoneInfo("America/New_York"))
        wd = ny.weekday()  # Mon=0 ... Sun=6
        minute = ny.hour * 60 + ny.minute
        # Standard retail FX weekly boundary in New York local time.
        if wd == 5:  # Saturday
            return False, "WEEKEND / MARKET CLOSED"
        if wd == 6 and minute < 17 * 60:  # Sunday before 17:00 NY
            return False, "WEEKEND / MARKET CLOSED"
        if wd == 4 and minute >= 17 * 60:  # Friday from 17:00 NY
            return False, "WEEKEND / MARKET CLOSED"
        return True, "MARKET OPEN"

    @staticmethod
    def _session_window(name, utc_ts):
        zone = ZoneInfo(SessionEngine.MARKET_ZONES[name])
        local = utc_ts.to_pydatetime().astimezone(zone)
        start_h, end_h = SessionEngine.SESSION_HOURS[name]
        mins = local.hour * 60 + local.minute
        return start_h * 60 <= mins < end_h * 60, local

    @classmethod
    def analyze(cls, ts=None, *, live=False):
        utc_ts = cls._utc(ts, live=live)
        market_open, market_status = cls._weekly_market_open(utc_ts)

        active = []
        local_times = {}
        if market_open:
            for name in cls.MARKET_ZONES:
                is_active, local = cls._session_window(name, utc_ts)
                local_times[name] = local.isoformat()
                if is_active:
                    active.append(name)
        else:
            for name, zone in cls.MARKET_ZONES.items():
                local_times[name] = utc_ts.to_pydatetime().astimezone(ZoneInfo(zone)).isoformat()

        if not market_open:
            session = "MARKET CLOSED"
        elif "LONDON" in active and "NEW YORK" in active:
            session = "LONDON/NEW YORK OVERLAP"
        elif "LONDON" in active:
            session = "LONDON"
        elif "NEW YORK" in active:
            session = "NEW YORK"
        elif "TOKYO" in active:
            session = "TOKYO"
        elif "SYDNEY" in active:
            session = "SYDNEY"
        else:
            session = "OFF-HOURS"

        # Determine the next major session opening for dashboard transparency.
        next_open = None
        if market_open:
            candidates = []
            for name, zone_name in cls.MARKET_ZONES.items():
                local = utc_ts.to_pydatetime().astimezone(ZoneInfo(zone_name))
                start_h, _ = cls.SESSION_HOURS[name]
                candidate = local.replace(hour=start_h, minute=0, second=0, microsecond=0)
                if candidate <= local:
                    candidate = candidate.replace(day=local.day) + pd.Timedelta(days=1).to_pytimedelta()
                candidates.append((candidate.astimezone(ZoneInfo("UTC")), name))
            if candidates:
                nxt, nm = min(candidates, key=lambda z: z[0])
                next_open = {"session": nm, "utc": nxt.isoformat()}
        else:
            # Weekly reopen is Sunday 17:00 New York from any closed period.
            ny = utc_ts.to_pydatetime().astimezone(ZoneInfo("America/New_York"))
            days_ahead = (6 - ny.weekday()) % 7
            if days_ahead == 0 and (ny.hour > 17 or (ny.hour == 17 and ny.minute >= 0)):
                days_ahead = 7
            sunday = (ny + pd.Timedelta(days=days_ahead).to_pytimedelta()).replace(
                hour=17, minute=0, second=0, microsecond=0
            )
            next_open = {"session": "WEEKLY FX OPEN", "utc": sunday.astimezone(ZoneInfo("UTC")).isoformat()}

        return {
            "session": session,
            "current_session": session,
            "active_sessions": active,
            "hour": int(utc_ts.hour),
            "minute": int(utc_ts.minute),
            "weekday": utc_ts.day_name(),
            "timestamp_utc": utc_ts.isoformat(),
            "session_tradeable": bool(market_open and active),
            "market_open": bool(market_open),
            "market_status": market_status,
            "clock_source": "CURRENT_UTC" if (live or ts is None) else "CANDLE_TIMESTAMP",
            "timezone_basis": "UTC clock + local session zones + America/New_York weekly boundary",
            "local_times": local_times,
            "next_open": next_open,
            "session_health": "HEALTHY",
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
    """
    Current-market MTF engine.

    Live/dashboard mode:
      * Fetches each required timeframe directly from the selected live source.
      * Does NOT try to manufacture H1/H4/D1 from a small M5 window.
      * D1 is read directly so the current daily market can participate.
      * Each timeframe is evaluated through the core technical engines.
      * Data quality is checked independently for every timeframe.
      * A weak/failed timeframe cannot be silently converted to a bullish/bearish vote.

    Backtest mode:
      * When no symbol/cfg is supplied, the historical dataframe is resampled,
        preserving the original backtest behavior.
    """

    TIMEFRAMES = ("M5", "M15", "M30", "H1", "H4", "D1")

    @staticmethod
    def _engine_direction(x):
        """Use several existing technical engines to determine one TF direction."""
        if x is None or x.empty:
            return "INSUFFICIENT", {"votes": {}, "agreement": 0.0}

        try:
            t = TrendEngine.analyze(x)
            m = MomentumEngine.analyze(x)
            s = StructureEngine.analyze(x)
            pa = PriceActionEngine.analyze(x)
            bo = BreakoutEngine.analyze(x)
            v = VolatilityEngine.analyze(x)
            re = RegimeEngine.classify(t, v, s, bo)
        except Exception as exc:
            return "ERROR", {"votes": {}, "agreement": 0.0, "error": str(exc)}

        outputs = {
            "Trend": str(t.get("direction", "")).upper(),
            "Momentum": str(m.get("direction", "")).upper(),
            "Structure": str(s.get("direction", "")).upper(),
            "Price Action": str(pa.get("direction", "")).upper(),
            "Breakout": str(bo.get("direction", "")).upper(),
        }

        valid = [v for v in outputs.values() if v in ("BULLISH", "BEARISH")]
        if not valid:
            return "NEUTRAL", {
                "votes": outputs,
                "agreement": 0.0,
                "regime": re,
                "volatility": v.get("regime", "UNKNOWN"),
            }

        bull = valid.count("BULLISH")
        bear = valid.count("BEARISH")
        total = len(valid)

        # Require a real majority. A 2/5 split must not be advertised as alignment.
        if bull > bear:
            direction = "BULLISH"
            agreement = 100.0 * bull / total
        elif bear > bull:
            direction = "BEARISH"
            agreement = 100.0 * bear / total
        else:
            direction = "MIXED"
            agreement = 50.0

        return direction, {
            "votes": outputs,
            "agreement": float(agreement),
            "regime": re,
            "volatility": v.get("regime", "UNKNOWN"),
        }

    @staticmethod
    def _direct_live_data(df, symbol, timeframe, cfg):
        """
        Get the exact timeframe needed by MTF.

        The currently loaded dataframe is reused only when its identity exactly
        matches the requested pair/timeframe/source. Otherwise the live loader
        fetches that timeframe directly.
        """
        requested_source = str(getattr(cfg, "data_source", "")).upper()
        if (
            df is not None
            and not df.empty
            and canonical_symbol(symbol) == canonical_symbol(
                st.session_state.get("data_symbol") or symbol
            )
            and str(timeframe).upper() == str(
                st.session_state.get("data_timeframe") or ""
            ).upper()
            and str(st.session_state.get("data_source_loaded") or "").upper()
            == requested_source
        ):
            return MarketDataEngine.normalize(df), {
                "source": requested_source,
                "direct": True,
                "reused_selected_dataset": True,
            }

        # get_live_pair_data is defined later in the module but is available when
        # this method is called by the running Streamlit application.
        x, meta = get_live_pair_data(
            canonical_symbol(symbol), timeframe, cfg, force=False
        )
        return MarketDataEngine.normalize(x), dict(meta or {})

    @staticmethod
    def analyze(df, symbol=None, cfg=None):
        # ---------------- Historical/backtest compatibility ----------------
        if symbol is None or cfg is None:
            states = {}
            details = {}
            for tf in MultiTimeframeEngine.TIMEFRAMES:
                x = MarketDataEngine.resample(df, tf)
                if len(x) < 100:
                    states[tf] = "INSUFFICIENT"
                    details[tf] = {
                        "status": "INSUFFICIENT",
                        "rows": int(len(x)),
                        "source": "HISTORICAL_RESAMPLE",
                        "current": False,
                    }
                    continue
                direction, info = MultiTimeframeEngine._engine_direction(x)
                states[tf] = direction
                details[tf] = {
                    **info,
                    "status": "OK",
                    "rows": int(len(x)),
                    "source": "HISTORICAL_RESAMPLE",
                    "current": False,
                }

            valid = [v for v in states.values() if v in ("BULLISH", "BEARISH")]
            bull = valid.count("BULLISH")
            bear = valid.count("BEARISH")
            direction = (
                "BULLISH" if bull > bear
                else "BEARISH" if bear > bull
                else "MIXED"
            )
            align = 100.0 * max(bull, bear) / max(len(valid), 1)
            return {
                "states": states,
                "direction": direction,
                "alignment": float(align),
                "strength": (
                    "VERY STRONG" if align >= 85
                    else "STRONG" if align >= 70
                    else "MODERATE" if align >= 55
                    else "WEAK"
                ),
                "details": details,
                "live_current_market": False,
            }

        # ---------------- Current/live market mode ----------------
        states = {}
        details = {}
        valid = []

        for tf in MultiTimeframeEngine.TIMEFRAMES:
            try:
                x, meta = MultiTimeframeEngine._direct_live_data(
                    df, symbol, tf, cfg
                )
                if x.empty or len(x) < 100:
                    states[tf] = "INSUFFICIENT"
                    details[tf] = {
                        "status": "INSUFFICIENT",
                        "rows": int(len(x)),
                        "source": str(meta.get("source", "UNKNOWN")),
                        "current": tf == "D1",
                    }
                    continue

                dq_age = {
                    "M5": getattr(cfg, "data_max_age_seconds", 420),
                    "M15": 1200,
                    "M30": 2400,
                    "H1": 4800,
                    "H4": 18000,
                    # D1 must be current enough to represent today's market.
                    # This is deliberately tighter than "one calendar day".
                    "D1": 7200,
                }.get(tf, getattr(cfg, "data_max_age_seconds", 420))

                dq = DataIntegrityEngine.assess(x, tf, dq_age)
                direction, info = MultiTimeframeEngine._engine_direction(x)

                # A failed/stale timeframe is never allowed to vote.
                usable = bool(dq.get("signal_allowed", False)) and direction in (
                    "BULLISH", "BEARISH"
                )
                if usable:
                    valid.append(direction)
                    states[tf] = direction
                    status = "OK"
                else:
                    states[tf] = "UNAVAILABLE"
                    status = "DATA QUALITY BLOCKED"

                last_time = (
                    str(x["time"].iloc[-1])
                    if "time" in x.columns and not x.empty
                    else "-"
                )

                details[tf] = {
                    **info,
                    "status": status,
                    "rows": int(len(x)),
                    "source": str(meta.get("source", "UNKNOWN")),
                    "last_timestamp": last_time,
                    "age_seconds": float(dq.get("age_seconds", 0.0)),
                    "data_quality": dq.get("status", "UNKNOWN"),
                    "data_quality_score": float(dq.get("score", 0.0)),
                    "current": True,
                    "usable_vote": usable,
                }
            except Exception as exc:
                states[tf] = "UNAVAILABLE"
                details[tf] = {
                    "status": "FETCH/ENGINE ERROR",
                    "rows": 0,
                    "source": str(getattr(cfg, "data_source", "UNKNOWN")),
                    "current": True,
                    "usable_vote": False,
                    "error": str(exc),
                }

        bull = valid.count("BULLISH")
        bear = valid.count("BEARISH")

        # MTF must have enough independent live timeframes to make a reliable
        # alignment claim. One valid timeframe must NEVER become "100% aligned".
        min_valid_timeframes = 3
        if len(valid) < min_valid_timeframes:
            direction = "INSUFFICIENT"
            align = 0.0
        else:
            direction = (
                "BULLISH" if bull > bear
                else "BEARISH" if bear > bull
                else "MIXED"
            )
            # Coverage is part of the alignment score: unavailable timeframes
            # count against alignment rather than disappearing from the denominator.
            align = 100.0 * max(bull, bear) / len(MultiTimeframeEngine.TIMEFRAMES)

        # Explicit D1 status: this tells the user whether today's current
        # daily market was actually read, rather than inferred from M5.
        d1 = details.get("D1", {})
        d1_current = bool(d1.get("usable_vote", False))
        d1_direction = states.get("D1", "UNAVAILABLE")

        # Current D1 is a mandatory anchor for live MTF. If today's D1 cannot
        # be read and quality-checked, the system must not pretend to have a
        # complete daily/intraday alignment.
        if not d1_current:
            direction = "INSUFFICIENT"
            align = 0.0

        return {
            "states": states,
            "direction": direction if valid else "MIXED",
            "alignment": float(align if valid else 0.0),
            "strength": (
                "VERY STRONG" if align >= 85
                else "STRONG" if align >= 70
                else "MODERATE" if align >= 55
                else "WEAK"
            ),
            "coverage_percent": 100.0 * len(valid) / len(MultiTimeframeEngine.TIMEFRAMES),
            "details": details,
            "live_current_market": True,
            "daily_current_available": d1_current,
            "daily_current_direction": d1_direction,
            "usable_timeframes": len(valid),
            "total_timeframes": len(MultiTimeframeEngine.TIMEFRAMES),
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
        if economic["blocked"] or volatility["regime"] == "EXTREME":
            total = min(total, 55)

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
            # Daily FX data normally has a Friday-to-Monday weekend gap. Do not
            # misclassify that normal market closure as missing candles.
            gap_multiplier = 4.5 if str(timeframe).upper() == "D1" else 1.8
            missing = int((diffs > expected*gap_multiplier).sum())
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




class DailySignalAssessmentEngine:
    """Creates a validated, once-per-UTC-day D1 signal from the full V13 stack.

    The signal is based only on completed D1 candles.  It is cached for the
    current UTC calendar day so intraday refreshes cannot rewrite the daily
    signal.  Data-quality failure or missing critical analysis outputs prevents
    the signal from being locked.
    """

    REQUIRED_DIRECTIONAL = (
        "trend", "momentum", "structure", "price_action",
        "sr", "breakout", "liquidity", "mtf", "advanced_momentum"
    )

    @staticmethod
    def completed_d1(df: pd.DataFrame) -> pd.DataFrame:
        x = MarketDataEngine.normalize(df).copy()
        if x.empty:
            return x
        now = pd.Timestamp.now(tz="UTC")
        # Only completed calendar days are eligible.  This also removes
        # Sunday/weekend artefacts and the currently forming daily candle.
        x = x[x["time"].dt.tz_convert("UTC").dt.date < now.date()].copy()
        x = x[x["time"].dt.dayofweek < 5].copy()
        return x.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    @staticmethod
    def _engine_health(a: Dict[str, Any], dq: Dict[str, Any]) -> Dict[str, Any]:
        checks = {}
        for name in DailySignalAssessmentEngine.REQUIRED_DIRECTIONAL:
            value = a.get(name)
            ok = isinstance(value, dict) and bool(value)
            if name == "sr":
                ok = isinstance(value, dict) and any(k in value for k in ("support", "resistance", "score", "direction"))
            checks[name] = {"status": "HEALTHY" if ok else "FAILED", "output_valid": ok}
        for name in ("ai", "confluence", "ensemble", "risk", "economic", "correlation", "currency_strength", "regime", "volatility"):
            value = a.get(name)
            ok = isinstance(value, dict) and bool(value) if name != "regime" else bool(value)
            checks[name] = {"status": "HEALTHY" if ok else "FAILED", "output_valid": ok}
        healthy = sum(v["status"] == "HEALTHY" for v in checks.values())
        total = len(checks)
        return {
            "status": "HEALTHY" if healthy == total else "DEGRADED" if healthy >= total * .8 else "FAILED",
            "healthy": healthy, "total": total,
            "percent": round(healthy / max(total, 1) * 100, 1),
            "checks": checks,
            "data_quality": dq,
        }

    @staticmethod
    def evaluate(df: pd.DataFrame, symbol: str, cfg: Config) -> Dict[str, Any]:
        completed = DailySignalAssessmentEngine.completed_d1(df)
        if completed.empty:
            return {"status": "UNAVAILABLE", "signal": "WAIT", "reason": "NO COMPLETED D1 CANDLE", "locked": False}

        dq = DataIntegrityEngine.assess(completed, "D1", max(int(cfg.data_max_age_seconds), 432000))
        # Freshness is measured against the completed candle, so do not reject a
        # weekend/overnight D1 signal merely because it is older than intraday data.
        dq = dict(dq)
        dq["signal_allowed"] = dq.get("score", 0) >= 70 and not any("INVALID" in r for r in dq.get("reasons", []))
        a = analyze_market(completed, canonical_symbol(symbol), cfg, "D1")
        health = DailySignalAssessmentEngine._engine_health(a, dq)

        directions = []
        for key in DailySignalAssessmentEngine.REQUIRED_DIRECTIONAL:
            value = a.get(key, {})
            d = str(value.get("direction", "NEUTRAL")).upper() if isinstance(value, dict) else "NEUTRAL"
            if d in ("BULLISH", "BEARISH"):
                directions.append(d)
        bull = directions.count("BULLISH"); bear = directions.count("BEARISH")
        engine_direction = "BULLISH" if bull > bear else "BEARISH" if bear > bull else "WAIT"

        ai = a.get("ai", {}) or {}
        up = float(ai.get("up_probability", 50.0) or 50.0)
        down = float(ai.get("down_probability", 50.0) or 50.0)
        ai_direction = "BULLISH" if up >= 55 else "BEARISH" if down >= 55 else "NEUTRAL"

        # A materially opposing AI forecast is a daily conflict, not something
        # the trend engine is allowed to silently override.
        conflict = (engine_direction in ("BULLISH", "BEARISH") and
                    ai_direction in ("BULLISH", "BEARISH") and
                    ai_direction != engine_direction and
                    max(up, down) >= 60.0)

        ensemble = a.get("ensemble", {}) or {}
        final_direction = engine_direction
        reasons = []
        if not dq["signal_allowed"]:
            final_direction = "WAIT"; reasons.append("DAILY DATA QUALITY FAILED")
        if health["status"] == "FAILED":
            final_direction = "WAIT"; reasons.append("CRITICAL DAILY ENGINE FAILURE")
        if conflict:
            final_direction = "WAIT"; reasons.append("AI / TECHNICAL DIRECTION CONFLICT")
        if final_direction in ("BULLISH", "BEARISH") and engine_direction != final_direction:
            final_direction = "WAIT"
        if final_direction == "WAIT" and not reasons:
            reasons.append("INSUFFICIENT DAILY ENGINE CONSENSUS")

        agreement = max(bull, bear) / max(len(directions), 1) * 100.0
        confidence = float(ensemble.get("confidence", 0.0) or 0.0)
        if final_direction in ("BULLISH", "BEARISH"):
            confidence = min(confidence, agreement, float(dq.get("score", 0.0)))

        latest = completed["time"].iloc[-1]
        cycle = datetime.now(timezone.utc).date().isoformat()
        return {
            "status": "LOCKED" if final_direction in ("BULLISH", "BEARISH") else "UNCONFIRMED",
            "signal": final_direction,
            "cycle_utc": cycle,
            "completed_candle": str(latest),
            "locked": final_direction in ("BULLISH", "BEARISH"),
            "engine_direction": engine_direction,
            "ai_direction": ai_direction,
            "ai_up": up, "ai_down": down,
            "engine_agreement": round(agreement, 1),
            "confidence": round(confidence, 1),
            "reasons": list(dict.fromkeys(reasons)),
            "data_quality": dq,
            "engine_health": health,
            "analysis": a,
        }


class TwelveDataRateLimitError(RuntimeError):
    """Raised when Twelve Data rejects a request because the API quota is exhausted."""

    def __init__(self, message, retry_after=60, credits_left=None):
        super().__init__(message)
        self.retry_after = int(max(1, retry_after or 60))
        self.credits_left = credits_left


class TwelveDataLiveEngine:
    """Twelve Data REST adapter. Read-only; no order endpoints are used.

    Important reliability rules:
      * One canonical provider symbol is used; V13 never retries the same request
        as both GBP/USD and GBPUSD.
      * The API key is sent in the Authorization header, not in the URL.
      * 429 responses are surfaced as a controlled data-quality/rate-limit state
        rather than immediately issuing more requests and making the quota worse.
    """
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

        # Twelve Data accepts slash notation for forex. Use exactly one form so
        # a failed request never burns a second credit on a symbol alias.
        provider_symbol = display_symbol(canonical_symbol(symbol))
        params = {
            "symbol": provider_symbol,
            "interval": TwelveDataLiveEngine.TF_MAP.get(str(timeframe).upper(), str(timeframe)),
            "outputsize": int(outputsize),
            "format": "JSON",
        }
        headers = {"Authorization": f"apikey {api_key}"}

        r = requests.get(
            TwelveDataLiveEngine.BASE_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After") or 60
            try:
                retry_after = int(float(retry_after))
            except Exception:
                retry_after = 60
            credits_left = r.headers.get("api-credits-left")
            raise TwelveDataRateLimitError(
                f"Twelve Data rate limit reached for {provider_symbol}/{str(timeframe).upper()}. "
                f"No duplicate retry was attempted. Wait about {retry_after}s before the next request.",
                retry_after=retry_after,
                credits_left=credits_left,
            )

        r.raise_for_status()
        payload = r.json()
        if str(payload.get("status", "")).lower() == "error":
            code = payload.get("code")
            message = payload.get("message", "Twelve Data returned an error.")
            if str(code) == "429":
                raise TwelveDataRateLimitError(str(message))
            raise RuntimeError(str(message))

        values = payload.get("values", [])
        if not values:
            raise RuntimeError(f"Twelve Data returned no candles for {provider_symbol}/{str(timeframe).upper()}.")

        rows = []
        for z in values:
            rows.append({
                "time": z.get("datetime"),
                "open": z.get("open"),
                "high": z.get("high"),
                "low": z.get("low"),
                "close": z.get("close"),
                "volume": z.get("volume", 0),
            })
        df = MarketDataEngine.normalize(pd.DataFrame(rows)).sort_values("time").reset_index(drop=True)
        return df, {
            "source": "TWELVE DATA",
            "rows": len(df),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "credits_used": r.headers.get("api-credits-used"),
            "credits_left": r.headers.get("api-credits-left"),
            "provider_symbol": provider_symbol,
        }


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


class VolumeFlowEngine:
    """Volume/flow direction using available OHLCV data; no synthetic volume is invented."""
    @staticmethod
    def analyze(df):
        x = MarketDataEngine.normalize(df)
        if x.empty or len(x) < 30:
            return {"direction": "NO DATA", "score": 0.0, "flow": 0.0, "status": "INSUFFICIENT"}
        vol = pd.to_numeric(x.get("volume", pd.Series(0.0, index=x.index)), errors="coerce").fillna(0.0)
        if float(vol.abs().sum()) <= 0:
            return {"direction": "NEUTRAL", "score": 50.0, "flow": 0.0, "status": "NO VOLUME"}
        signed = np.sign(x.close.diff().fillna(0.0)) * vol
        flow = float(signed.tail(20).sum() / max(vol.tail(20).sum(), 1e-9))
        score = float(np.clip(50.0 + flow * 50.0, 0, 100))
        direction = "BULLISH" if score >= 58 else "BEARISH" if score <= 42 else "NEUTRAL"
        return {"direction": direction, "score": score, "flow": flow, "status": "OK"}


class CandleTimingEngine:
    """Current-candle state: age/progress plus directional body confirmation."""
    @staticmethod
    def analyze(df, timeframe="M5"):
        x = MarketDataEngine.normalize(df)
        if x.empty:
            return {"direction": "NO DATA", "score": 0.0, "progress_pct": 0.0, "status": "NO DATA"}
        tf_seconds = {"M1":60,"M5":300,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}.get(str(timeframe).upper(),300)
        now = pd.Timestamp.now(tz="UTC")
        last_ts = x.time.iloc[-1]
        age = max(0.0, (now - last_ts).total_seconds())
        progress = float(np.clip((age / tf_seconds) * 100.0, 0, 100)) if age <= tf_seconds else 100.0
        row = x.iloc[-1]
        rng = max(float(row.high - row.low), 1e-12)
        body = float(row.close - row.open) / rng
        direction = "BULLISH" if body > 0.10 else "BEARISH" if body < -0.10 else "NEUTRAL"
        # A forming candle is informational; it never becomes a hard veto merely because it is unfinished.
        score = float(np.clip(50 + body * 50, 0, 100))
        return {"direction": direction, "score": score, "progress_pct": progress,
                "age_seconds": age, "timeframe_seconds": tf_seconds,
                "forming": age < tf_seconds, "status": "OK"}


class MarketRegionEngine:
    """Maps current price to recent range regions: lower/middle/upper."""
    @staticmethod
    def analyze(df, lookback=100):
        x = MarketDataEngine.normalize(df).tail(lookback)
        if len(x) < 20:
            return {"region": "UNKNOWN", "direction": "NO DATA", "score": 0.0, "status": "INSUFFICIENT"}
        lo = float(x.low.min()); hi = float(x.high.max()); last = float(x.close.iloc[-1])
        pos = (last - lo) / max(hi - lo, 1e-12)
        if pos <= 0.33:
            region, direction = "LOWER", "BULLISH"
        elif pos >= 0.67:
            region, direction = "UPPER", "BEARISH"
        else:
            region, direction = "MIDDLE", "NEUTRAL"
        # Region is contextual, not a standalone trigger.
        score = float(np.clip(50 + (0.5 - abs(pos - 0.5)) * 40, 0, 100))
        return {"region": region, "direction": direction, "score": score, "position_pct": pos * 100, "status": "OK"}


class DirectProbabilityEngine:
    """Transparent non-ML probability estimate from independent price features."""
    @staticmethod
    def predict(df):
        x = MarketDataEngine.normalize(df)
        if len(x) < 60:
            return {"up_probability":50.0,"down_probability":50.0,"confidence":0.0,"direction":"NEUTRAL","status":"INSUFFICIENT"}
        r = float(rsi(x.close,14).iloc[-1])
        e20 = float(ema(x.close,20).iloc[-1]); e50 = float(ema(x.close,50).iloc[-1])
        ret = float(x.close.pct_change(10).iloc[-1])
        z = ((r - 50) / 20) + ((e20 / max(e50,1e-12) - 1) * 120) + ret * 100
        prob = float(1 / (1 + math.exp(-np.clip(z, -8, 8))))
        conf = float(abs(prob - 0.5) * 200)
        direction = "BULLISH" if prob >= 0.55 else "BEARISH" if prob <= 0.45 else "NEUTRAL"
        return {"up_probability":prob*100,"down_probability":(1-prob)*100,"confidence":conf,"direction":direction,"status":"OK"}


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

        # AI is a soft advisory gate: it can reduce confidence, but does not veto a
        # signal when the independent technical stack strongly agrees.
        if ai["confidence"] < cfg.ai_soft_floor and confidence["agreement"] < 80:
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
        if a["session"].get("session_tradeable") is False:
            reasons.append("SESSION NOT TRADEABLE")
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


class EngineInformationBus:
    """Central normalized evidence layer used by Trade Quality and diagnostics.

    Every engine publishes the same contract: direction, strength/confidence,
    health, role and whether it may contribute a directional vote. Missing or
    invalid outputs are marked unavailable rather than silently treated as neutral.
    """
    SPECS = [
        ("Trend", "trend", "directional", 1.00),
        ("Momentum", "momentum", "directional", 1.00),
        ("Current Volatility", "volatility", "context", 0.75),
        ("Market Structure", "structure", "directional", 1.00),
        ("Price Action", "price_action", "directional", 1.00),
        ("Support/Resistance", "sr", "context", 0.80),
        ("Breakout", "breakout", "directional", 0.90),
        ("Fair Value Gap", "fvg", "context", 0.80),
        ("Liquidity", "liquidity", "context", 0.80),
        ("Regime", "regime", "context", 0.80),
        ("Session", "session", "timing", 0.70),
        ("Currency Strength", "currency_strength", "directional", 0.90),
        ("Correlation", "correlation", "risk", 0.60),
        ("Multi-Timeframe Alignment", "mtf", "directional", 1.20),
        ("News/Economic", "economic", "risk", 0.80),
        ("COT/Positioning", "cot", "context", 0.40),
        ("Advanced Momentum", "advanced_momentum", "directional", 1.00),
        ("Volume Flow", "volume_flow", "directional", 0.90),
        ("Candle Timing", "candle_timing", "timing", 0.60),
        ("Market Region", "market_region", "context", 0.60),
        ("Direct Probability", "direct_probability", "directional", 0.90),
        ("AI/ML Prediction", "ai", "directional", 1.00),
        ("Confluence", "confluence", "summary", 1.00),
        ("Ensemble Decision", "ensemble", "summary", 1.10),
        ("Risk Control", "risk", "risk", 1.00),
        ("Risk Veto", "risk_veto", "risk", 1.20),
        ("Signal Timing", "signal_timing", "timing", 0.70),
        ("Data Integrity", "data_quality", "data", 1.30),
        ("Market Tracker", "market_tracker", "context", 0.50),
        ("Daily Signal", "daily_signal", "daily_bias", 1.30),
    ]

    @staticmethod
    def _direction(obj):
        if not isinstance(obj, dict):
            return "NEUTRAL"
        d = str(obj.get("direction", obj.get("bias", "NEUTRAL"))).upper()
        return d if d in {"BULLISH","BEARISH"} else "NEUTRAL"

    @staticmethod
    def _strength(obj):
        if not isinstance(obj, dict):
            return 0.0
        for key in ("confidence","score","strength","alignment","quality"):
            value = obj.get(key)
            if isinstance(value, (int,float,np.integer,np.floating)):
                return float(np.clip(value,0,100))
        return 0.0

    @classmethod
    def build(cls, a, data_quality, daily_signal=None):
        out = {}
        for label, key, role, weight in cls.SPECS:
            obj = a.get(key)
            if key == "fvg":
                li = a.get("liquidity", {}) or {}
                obj = {"direction": a.get("liquidity",{}).get("direction","NEUTRAL"),
                       "score": 70.0 if li.get("FVG") else 30.0,
                       "present": bool(li.get("FVG")), "status":"OK"}
            if key == "risk_veto":
                rv = a.get("risk_veto") or a.get("risk", {}) or {}
                obj = {"direction":"NEUTRAL","score":100.0 if not rv.get("reasons") else 0.0,
                       "status":"PASS" if not rv.get("reasons") else "VETO"}
            if key == "signal_timing":
                obj = a.get("signal_timing") or {"direction":"NEUTRAL","score":100.0,"status":"FRESH"}
            if key == "daily_signal":
                obj = daily_signal or st.session_state.get("daily_signal_locked") or {}
            if key == "market_tracker":
                obj = a.get("market_tracker") or {"direction":a.get("confluence",{}).get("direction","NEUTRAL"),"score":a.get("confluence",{}).get("score",0),"status":"CURRENT PAIR"}

            if isinstance(obj, str):
                raw_text = obj.upper()
                obj = {"direction": raw_text if raw_text in {"BULLISH","BEARISH"} else "NEUTRAL", "status": raw_text}
            direction = cls._direction(obj)
            strength = cls._strength(obj)
            if key == "data_quality":
                healthy = bool(obj) and not bool(obj.get("stale",False)) and bool(obj.get("signal_allowed",False)) and float(obj.get("score",0)) >= 70
            elif key == "session":
                healthy = bool(obj) and bool(obj.get("session_tradeable",False))
            elif key == "risk":
                healthy = bool(obj) and bool(obj.get("approved",False))
            elif key == "risk_veto":
                healthy = bool(obj) and not bool(obj.get("reasons"))
            elif key == "economic":
                healthy = bool(obj) and not bool(obj.get("blocked",False))
            elif key == "mtf":
                healthy = bool(obj) and bool(obj.get("daily_current_available",True)) and float(obj.get("alignment",0)) >= 60
            else:
                healthy = bool(obj) and str(obj.get("status","OK")).upper() not in {"FAILED","ERROR","INSUFFICIENT","NO DATA","UNAVAILABLE"}
            out[label] = {
                "key": key, "role": role, "weight": weight,
                "direction": direction, "strength": round(strength,1),
                "healthy": bool(healthy), "available": bool(obj),
                "raw": obj,
            }
        return out


class TradeQualityEngine:
    """Central trade-quality evaluator fed by the complete Engine Information Bus.

    Directional evidence is aggregated only from healthy, available engines.
    Context/risk/timing engines can strengthen or veto a setup. Data quality,
    session, daily bias, MTF coverage and risk remain hard gates.
    """

    @staticmethod
    def evaluate(a, ensemble, no_trade, data_quality, daily_signal=None):
        direction = str(ensemble.get("direction", a.get("confluence",{}).get("direction","WAIT"))).upper()
        bus = EngineInformationBus.build(a, data_quality, daily_signal)

        # Directional consensus across every healthy directional engine.
        votes = []
        for label, item in bus.items():
            if item["role"] not in {"directional", "daily_bias"} or not item["healthy"]:
                continue
            if item["direction"] in {"BULLISH","BEARISH"}:
                votes.append((label, item["direction"], item["weight"]))
        total_weight = sum(w for _,_,w in votes)
        aligned_weight = sum(w for _,d,w in votes if d == direction) if direction in {"BULLISH","BEARISH"} else 0.0
        agreement = 100.0 * aligned_weight / max(total_weight,1e-9)

        dq = bus["Data Integrity"]
        session = bus["Session"]
        mtf = bus["Multi-Timeframe Alignment"]
        risk = bus["Risk Control"]
        risk_veto = bus["Risk Veto"]
        news = bus["News/Economic"]
        daily = bus["Daily Signal"]
        ai = bus["AI/ML Prediction"]
        confluence = bus["Confluence"]
        ensemble_item = bus["Ensemble Decision"]

        daily_raw = daily["raw"] or {}
        daily_signal_dir = str(daily_raw.get("signal", "WAIT")).upper()
        daily_locked = bool(daily_raw.get("locked", False)) and str(daily_raw.get("status","")).upper() == "LOCKED"
        daily_ok = daily_locked and daily_signal_dir in {"BULLISH","BEARISH"} and direction == daily_signal_dir

        mtf_raw = mtf["raw"] or {}
        mtf_alignment = float(mtf_raw.get("alignment",0) or 0)
        mtf_coverage = float(mtf_raw.get("coverage_percent",0) or 0)
        mtf_ok = bool(mtf_raw.get("daily_current_available",False)) and mtf_alignment >= 60 and mtf_coverage >= 50

        dq_ok = dq["healthy"]
        session_ok = session["healthy"]
        risk_ok = risk["healthy"] and risk_veto["healthy"]
        news_ok = news["healthy"]
        ensemble_ok = bool(ensemble.get("approved",False))
        no_trade_ok = bool(no_trade.get("trade_allowed",False))

        # Weighted central quality score. Data/MTF/directional agreement carry
        # more weight than any single indicator so no isolated engine dominates.
        engine_direction_score = agreement
        ai_score = ai["strength"]
        confluence_score = confluence["strength"]
        ensemble_score = ensemble_item["strength"]
        context_scores = [
            bus["Current Volatility"]["strength"], bus["Volume Flow"]["strength"],
            bus["Candle Timing"]["strength"], bus["Market Region"]["strength"],
            bus["Fair Value Gap"]["strength"], bus["Currency Strength"]["strength"],
            bus["Correlation"]["strength"], bus["Breakout"]["strength"],
            bus["Support/Resistance"]["strength"], bus["Regime"]["strength"],
        ]
        context_score = float(np.mean([x for x in context_scores if x > 0])) if any(x > 0 for x in context_scores) else 0.0

        score = (
            engine_direction_score * 0.25
            + confluence_score * 0.15
            + ai_score * 0.10
            + ensemble_score * 0.10
            + mtf_alignment * 0.15
            + context_score * 0.10
            + float(dq["raw"].get("score",0)) * 0.10
            + (100.0 if risk_ok and news_ok and session_ok else 0.0) * 0.05
        )
        score = float(np.clip(score,0,100))

        hard_vetoes=[]
        if not dq_ok: hard_vetoes.append("DATA QUALITY / STALE OR INVALID FEED")
        if not session_ok: hard_vetoes.append("SESSION NOT TRADEABLE")
        if not daily_ok: hard_vetoes.append(f"DAILY BIAS CONFLICT OR UNLOCKED ({daily_signal_dir})")
        if direction not in {"BULLISH","BEARISH"}: hard_vetoes.append("NO VALID DIRECTION")
        if not mtf_ok: hard_vetoes.append("CURRENT-MARKET MTF INSUFFICIENT")
        if agreement < 65: hard_vetoes.append("ENGINE INFORMATION AGREEMENT BELOW 65")
        if not ensemble_ok: hard_vetoes.append("ENSEMBLE NOT APPROVED")
        if not no_trade_ok: hard_vetoes.extend(no_trade.get("reasons",[]))
        if not risk_ok: hard_vetoes.append("RISK CONTROL / RISK VETO")
        if not news_ok: hard_vetoes.append("HIGH-IMPACT NEWS BLACKOUT")
        if float(a.get("volatility",{}).get("regime_score",50) or 50) < 0: hard_vetoes.append("INVALID VOLATILITY")

        hard_vetoes=list(dict.fromkeys(hard_vetoes))
        decision = "TRADE" if score >= 85 and not hard_vetoes else "NO TRADE"
        grade = "A" if decision == "TRADE" and score >= 90 else "B" if decision == "TRADE" else "C" if score >= 70 else "D"

        checks = {
            "DATA QUALITY": dq_ok,
            "DAILY SIGNAL ALIGNED": daily_ok,
            "CURRENT-MARKET MTF": mtf_ok,
            "ALL-ENGINE INFORMATION BUS": agreement >= 65,
            "ENSEMBLE": ensemble_ok,
            "RISK CONTROL": risk_ok,
            "NEWS FILTER": news_ok,
            "SESSION": session_ok,
            "NO-TRADE GATE": no_trade_ok,
        }
        return {
            "checks": checks,
            "components": {
                "Engine information agreement": round(engine_direction_score*0.25,2),
                "Confluence": round(confluence_score*0.15,2),
                "AI/ML": round(ai_score*0.10,2),
                "Ensemble": round(ensemble_score*0.10,2),
                "MTF": round(mtf_alignment*0.15,2),
                "Technical context": round(context_score*0.10,2),
                "Data integrity": round(float(dq["raw"].get("score",0))*0.10,2),
                "Risk/news/session": round((100.0 if risk_ok and news_ok and session_ok else 0.0)*0.05,2),
            },
            "score": round(score,1), "grade": grade, "decision": decision,
            "direction": direction, "directional_agreement": round(agreement,1),
            "directional_engines": [(x,d,round(w,2)) for x,d,w in votes],
            "engine_information_bus": bus,
            "mtf_alignment": round(mtf_alignment,1),
            "mtf_coverage": round(mtf_coverage,1),
            "daily_signal": daily_signal_dir,
            "hard_vetoes": hard_vetoes,
            "reasons": hard_vetoes,
        }


class LiveTradeEligibilityEngine:
    """Final fail-closed gate separating analysis quality from live-data eligibility.

    Important separation:
      * analysis_quality = how strongly the analysis engines agree
      * ai_confidence = model confidence
      * data_integrity_score = structural quality of the loaded dataset
      * data_freshness = whether the latest candle is recent enough
      * trade_eligible = final permission to consider a live/paper entry

    High confluence or AI confidence can never override a stale/invalid feed.
    """

    @staticmethod
    def evaluate(
        a,
        ensemble,
        no_trade,
        trade_quality,
        data_quality,
        timing,
        emergency=False,
    ):
        dq = data_quality or {}
        reasons = list(no_trade.get("reasons", []))

        if dq.get("stale", False) or not dq.get("signal_allowed", False):
            if "DATA QUALITY / STALE FEED" not in reasons:
                reasons.append("DATA QUALITY / STALE FEED")

        if timing and not timing.get("fresh", True):
            reasons.append("SIGNAL TIMING EXPIRED")

        if emergency:
            reasons.append("EMERGENCY STOP")

        if not ensemble.get("approved", False):
            reasons.extend(ensemble.get("reasons", []))

        if trade_quality.get("decision") != "TRADE":
            reasons.append("TRADE QUALITY GATE")

        reasons = list(dict.fromkeys(reasons))

        analysis_score = float(
            np.clip(
                ensemble.get("confluence_score", a.get("confluence", {}).get("score", 0.0)),
                0,
                100,
            )
        )
        ai_confidence = float(np.clip(a.get("ai", {}).get("confidence", 0.0), 0, 100))
        integrity_score = float(np.clip(dq.get("score", 0.0), 0, 100))
        freshness_ok = bool(not dq.get("stale", True))
        quality_score = float(np.clip(trade_quality.get("score", 0.0), 0, 100))

        eligible = (
            not reasons
            and freshness_ok
            and bool(dq.get("signal_allowed", False))
            and bool(ensemble.get("approved", False))
            and bool(no_trade.get("trade_allowed", False))
            and trade_quality.get("decision") == "TRADE"
            and not emergency
        )

        return {
            "eligible": bool(eligible),
            "status": "APPROVED" if eligible else "BLOCKED",
            "reasons": reasons,
            "analysis_score": analysis_score,
            "ai_confidence": ai_confidence,
            "data_integrity_score": integrity_score,
            "data_freshness_seconds": float(dq.get("age_seconds", 0.0)),
            "data_freshness_ok": freshness_ok,
            "trade_quality_score": quality_score,
            "trade_quality_grade": trade_quality.get("grade", "D"),
        }


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


class V13DataBudgetManager:
    """Central gate for every Twelve Data request made by V13.

    The manager is intentionally conservative. It prevents Streamlit reruns,
    scanner rotation, MTF/D1 analysis and manual refreshes from competing for
    the same API allowance. A 429 starts a cooldown instead of triggering a
    retry storm. Cached data may be served only when it remains inside the
    configured maximum-staleness window; otherwise V13 fails closed.
    """

    @staticmethod
    def state() -> dict:
        return st.session_state.setdefault("td_budget", {
            "request_log": [],
            "daily_log": [],
            "cooldown_until": 0.0,
            "key_cooldowns": {},
            "last_request": None,
            "last_error": None,
            "last_error_at": None,
        })

    @staticmethod
    def _prune(now: float, minute_window: int = 60):
        state = V13DataBudgetManager.state()
        state["request_log"] = [t for t in state.get("request_log", []) if now - float(t) < minute_window]
        today = datetime.now(timezone.utc).date().isoformat()
        state["daily_log"] = [
            item for item in state.get("daily_log", [])
            if str(item.get("day", "")) == today
        ]
        state["key_cooldowns"] = {
            k: float(v) for k, v in state.get("key_cooldowns", {}).items() if float(v) > now
        }

    @staticmethod
    def allow(symbol: str, timeframe: str, cfg: Config) -> tuple[bool, int, str]:
        now = datetime.now(timezone.utc).timestamp()
        V13DataBudgetManager._prune(now, int(getattr(cfg, "twelve_data_rate_guard_seconds", 60)))
        state = V13DataBudgetManager.state()
        key = f"{canonical_symbol(symbol)}|{str(timeframe).upper()}"

        global_until = float(state.get("cooldown_until", 0.0) or 0.0)
        key_until = float(state.get("key_cooldowns", {}).get(key, 0.0) or 0.0)
        until = max(global_until, key_until)
        if until > now:
            return False, max(1, int(math.ceil(until - now))), "API_COOLDOWN"

        minute_budget = max(1, int(getattr(cfg, "twelve_data_request_budget_per_minute", 7)))
        if len(state["request_log"]) >= minute_budget:
            oldest = min(state["request_log"]) if state["request_log"] else now
            wait = max(1, int(math.ceil(60 - (now - oldest))))
            return False, wait, "MINUTE_BUDGET"

        daily_budget = max(minute_budget, int(getattr(cfg, "twelve_data_daily_budget", 760)))
        if len(state["daily_log"]) >= daily_budget:
            return False, 86400, "DAILY_BUDGET"

        return True, 0, "OK"

    @staticmethod
    def reserve(symbol: str, timeframe: str, cfg: Config):
        allowed, wait, reason = V13DataBudgetManager.allow(symbol, timeframe, cfg)
        if not allowed:
            raise TwelveDataRateLimitError(
                f"V13 API budget protection blocked {display_symbol(symbol)}/{str(timeframe).upper()} "
                f"({reason}). Wait about {wait}s; no duplicate request was sent.",
                retry_after=wait,
            )
        now = datetime.now(timezone.utc).timestamp()
        state = V13DataBudgetManager.state()
        state["request_log"].append(now)
        state["daily_log"].append({
            "time": now,
            "day": datetime.now(timezone.utc).date().isoformat(),
            "symbol": canonical_symbol(symbol),
            "timeframe": str(timeframe).upper(),
        })
        state["last_request"] = now
        return True

    @staticmethod
    def record_429(symbol: str, timeframe: str, retry_after: int):
        now = datetime.now(timezone.utc).timestamp()
        until = now + max(1, int(retry_after or 60))
        state = V13DataBudgetManager.state()
        state["cooldown_until"] = max(float(state.get("cooldown_until", 0.0) or 0.0), until)
        state["key_cooldowns"][f"{canonical_symbol(symbol)}|{str(timeframe).upper()}"] = until
        state["last_error"] = "429 RATE LIMIT"
        state["last_error_at"] = now

    @staticmethod
    def snapshot(cfg: Config) -> dict:
        now = datetime.now(timezone.utc).timestamp()
        V13DataBudgetManager._prune(now, int(getattr(cfg, "twelve_data_rate_guard_seconds", 60)))
        state = V13DataBudgetManager.state()
        cooldown = max(0, int(math.ceil(float(state.get("cooldown_until", 0.0) or 0.0) - now)))
        return {
            "minute_used": len(state["request_log"]),
            "minute_budget": int(getattr(cfg, "twelve_data_request_budget_per_minute", 7)),
            "daily_used": len(state["daily_log"]),
            "daily_budget": int(getattr(cfg, "twelve_data_daily_budget", 760)),
            "cooldown_seconds": cooldown,
            "last_error": state.get("last_error"),
        }


def _td_cache_ttl(timeframe: str) -> int:
    """How long a successfully fetched Twelve Data dataset may be reused."""
    return {
        "M1": 30,
        "M5": 75,
        "M15": 150,
        "M30": 300,
        "H1": 600,
        "H4": 1800,
        "D1": 3600,
    }.get(str(timeframe).upper(), 300)


def _td_cache_age_seconds(meta: dict) -> float:
    fetched = meta.get("fetched_at") or meta.get("loaded_at")
    if not fetched:
        return float("inf")
    try:
        ts = pd.Timestamp(fetched)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return max(0.0, (pd.Timestamp.now(tz="UTC") - ts.tz_convert("UTC")).total_seconds())
    except Exception:
        return float("inf")


def _td_request_allowed(cfg: Config) -> tuple[bool, int]:
    """Backward-compatible view of the central V13 budget manager."""
    allowed, wait, _ = V13DataBudgetManager.allow("GLOBAL", "GLOBAL", cfg)
    return allowed, wait


def _td_record_request():
    # Kept only for compatibility with older V13 code paths. New provider calls
    # reserve their credit atomically through V13DataBudgetManager.reserve().
    return None


def _td_cached_pair(symbol: str, timeframe: str, cfg: Config, allow_stale_seconds: int | None = None):
    """Return cached data without spending an API credit.

    Fresh cache is preferred. When the provider is rate-limited, a bounded stale
    cache may be returned only if its fetch age remains inside the data-integrity
    maximum. This is explicitly marked so downstream quality/health layers know
    that a provider refresh was unavailable.
    """
    cache = st.session_state.setdefault("live_pair_cache", {})
    source = str(getattr(cfg, "data_source", "TWELVE DATA")).upper()
    key = (source, canonical_symbol(symbol), str(timeframe).upper(), int(getattr(cfg, "twelve_data_outputsize", 500)))
    cached = cache.get(key)
    if not cached:
        return None
    cdf = cached.get("df")
    meta = dict(cached.get("meta", {}) or {})
    if cdf is None or cdf.empty:
        return None
    ttl = _td_cache_ttl(timeframe)
    age = _td_cache_age_seconds(meta)
    stale_limit = int(allow_stale_seconds if allow_stale_seconds is not None else ttl)
    if age > stale_limit:
        return None
    meta["cache_age_seconds"] = round(age, 1)
    meta["cache_fresh"] = bool(age <= ttl)
    meta["cache_stale"] = bool(age > ttl)
    return cdf.copy(), meta



def _fetch_twelve_data(symbol: str, timeframe: str, cfg: Config):
    key = _get_twelve_data_key(cfg)
    if not key:
        raise RuntimeError(
            "Twelve Data is selected but no API key is configured. "
            "Enter the key in the sidebar or set TWELVE_DATA_API_KEY in Streamlit secrets."
        )

    # Reserve exactly one provider credit before the network call. This is the
    # single gate used by the selected pair, D1 daily signal and scanner.
    V13DataBudgetManager.reserve(symbol, timeframe, cfg)
    try:
        df, meta = TwelveDataLiveEngine.fetch(key, symbol, timeframe, cfg.twelve_data_outputsize)
    except TwelveDataRateLimitError as ex:
        # The provider has authoritative quota information. Start a cooldown
        # immediately and never retry inside this call.
        V13DataBudgetManager.record_429(symbol, timeframe, ex.retry_after)
        raise
    meta = dict(meta or {})
    meta.update({
        "symbol": canonical_symbol(symbol),
        "requested_symbol": canonical_symbol(symbol),
        "timeframe": str(timeframe).upper(),
        "source": "TWELVE DATA",
        "read_only": True,
        "execution_enabled": False,
    })
    return df, meta


def get_live_pair_data(symbol: str, timeframe: str, cfg: Config, force=False):
    """Resolve exactly one pair/timeframe while respecting the central data budget.

    Normal calls are cache-first. ``force=True`` is an explicit operator refresh;
    even then, if the provider refuses the request, V13 may use a bounded stale
    cache rather than inventing data. Live modes never fall back to synthetic data.
    """
    symbol = canonical_symbol(symbol)
    timeframe = str(timeframe).upper()
    requested_source = str(cfg.data_source).upper()
    cache = st.session_state.setdefault("live_pair_cache", {})
    key = (requested_source, symbol, timeframe, int(getattr(cfg, "twelve_data_outputsize", 500)))

    # Cache-first is the default and is the critical protection against Streamlit
    # reruns repeatedly spending provider credits.
    if requested_source == "TWELVE DATA":
        fresh = _td_cached_pair(symbol, timeframe, cfg)
        if fresh is not None and not force:
            return fresh
    elif not force and key in cache:
        cached = cache[key]
        cached_df = cached.get("df")
        if cached_df is not None and not cached_df.empty:
            return cached_df.copy(), dict(cached.get("meta", {}))

    try:
        if requested_source == "TWELVE DATA":
            df, meta = _fetch_twelve_data(symbol, timeframe, cfg)
            actual_source = "TWELVE DATA"
        elif requested_source == "MT5":
            try:
                MT5LiveDataEngine.connect(symbol, path=st.session_state.get("mt5_path") or None)
                df = MT5LiveDataEngine.bars(symbol, timeframe, int(getattr(cfg, "mt5_outputsize", 500)))
                tick = MT5LiveDataEngine.tick(symbol)
                meta = {"symbol": symbol, "timeframe": timeframe, "source": "MT5",
                        "read_only": True, "execution_enabled": False, "tick": tick,
                        "fetched_at": datetime.now(timezone.utc).isoformat()}
                actual_source = "MT5"
            except Exception as mt5_error:
                td_key = _get_twelve_data_key(cfg)
                if bool(getattr(cfg, "allow_live_source_failover", True)) and td_key:
                    df, meta = _fetch_twelve_data(symbol, timeframe, cfg)
                    meta["requested_source"] = "MT5"
                    meta["fallback_reason"] = str(mt5_error)
                    meta["source"] = "TWELVE DATA"
                    actual_source = "TWELVE DATA"
                else:
                    raise RuntimeError(str(mt5_error))
        elif requested_source == "DERIV REAL FOREX":
            df, meta = get_deriv_real_forex_data(symbol, timeframe, cfg, force=force)
            actual_source = "DERIV REAL FOREX"
        elif requested_source == "DEMO":
            base_rows = int(cfg.demo_rows)
            if timeframe == "D1":
                base_rows = max(base_rows, 120 * 288)
            elif timeframe == "H4":
                base_rows = max(base_rows, 120 * 48)
            base = MarketDataEngine.synthetic(symbol, base_rows, seed=7 + sum(map(ord, symbol)))
            df = MarketDataEngine.resample(base, timeframe) if timeframe != "M5" else base
            meta = {"symbol": symbol, "timeframe": timeframe, "source": "DEMO", "synthetic": True,
                    "read_only": True, "execution_enabled": False,
                    "fetched_at": datetime.now(timezone.utc).isoformat()}
            actual_source = "DEMO"
        else:
            raise RuntimeError(f"Unsupported data source: {requested_source}")
    except TwelveDataRateLimitError as ex:
        # Provider quota is authoritative. Use stale cache only inside the
        # configured integrity window; otherwise fail closed.
        if requested_source == "TWELVE DATA" or str(getattr(ex, "provider", "")).upper() == "TWELVE DATA":
            stale = _td_cached_pair(symbol, timeframe, cfg, allow_stale_seconds=int(getattr(cfg, "twelve_data_cache_stale_seconds", cfg.data_max_age_seconds)))
            if stale is not None:
                stale_df, stale_meta = stale
                stale_meta.update({
                    "source": "TWELVE DATA",
                    "rate_limited": True,
                    "rate_limit_fallback": True,
                    "rate_limit_retry_after": ex.retry_after,
                    "read_only": True,
                    "execution_enabled": False,
                })
                return stale_df, stale_meta
        raise

    df = MarketDataEngine.normalize(df)
    if df.empty:
        raise RuntimeError(f"No market data returned for {symbol}/{timeframe} from {actual_source}.")
    meta = dict(meta or {})
    meta.update({"symbol": symbol, "timeframe": timeframe, "source": actual_source,
                 "requested_source": requested_source})
    if not meta.get("fetched_at"):
        meta["fetched_at"] = datetime.now(timezone.utc).isoformat()
    cache_key = (actual_source, symbol, timeframe, int(getattr(cfg, "twelve_data_outputsize", 500)))
    cache[cache_key] = {"df": df.copy(), "meta": dict(meta)}
    cache[key] = {"df": df.copy(), "meta": dict(meta)}
    return df, meta


def get_selected_market_data(symbol: str, timeframe: str, cfg: Config):
    """Resolve the dashboard selection with strict pair/timeframe/source isolation."""
    symbol = canonical_symbol(symbol)
    timeframe = str(timeframe).upper()
    requested_source = str(cfg.data_source).upper()

    # IMPORTANT: source is part of identity. A source change must trigger a new fetch.
    if data_matches_selection(symbol, timeframe, st.session_state.get("data"), requested_source):
        return st.session_state.data, st.session_state.get("data_meta", {})

    # A pair or source change invalidates the old dataset BEFORE fetching.
    clear_market_data_identity(clear_candles=True)
    df, meta = get_live_pair_data(symbol, timeframe, cfg, force=False)
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
    session = SessionEngine.analyze(live=True)
    breakout = {"direction": "NO DATA", "state": "NO DATA"}
    liquidity = {"direction": "NO DATA", "FVG": False}
    sr = {}
    regime = "NO-TRADE"
    volume_flow = {"direction": "NO DATA", "score": 0.0, "status": "NO DATA"}
    candle_timing = {"direction": "NO DATA", "score": 0.0, "status": "NO DATA"}
    market_region = {"direction": "NO DATA", "score": 0.0, "status": "NO DATA"}
    direct_probability = {"up_probability": 50.0, "down_probability": 50.0, "confidence": 0.0, "direction": "NEUTRAL", "status": "NO DATA"}
    signal_timing = {"fresh": False, "age_seconds": float("inf"), "status": "NO DATA"}

    return {
        "trend": trend, "momentum": momentum, "volatility": volatility,
        "structure": structure, "price_action": price_action, "sr": sr,
        "breakout": breakout, "liquidity": liquidity, "regime": regime,
        "session": session, "currency_strength": currency_strength, "mtf": mtf,
        "economic": economic, "cot": cot, "correlation": correlation,
        "volume_flow": volume_flow, "candle_timing": candle_timing,
        "market_region": market_region, "direct_probability": direct_probability,
        "market_tracker": {"direction":"WAIT","score":0.0,"status":"NO DATA"},
        "signal_timing": signal_timing,
        "confluence": c, "risk": risk, "advanced_momentum": advanced_momentum,
        "ai": ai, "ensemble": ensemble, "no_trade": no_trade,
        "trade_quality": trade_quality, "data_quality": dq,
        # V12.1 compatibility aliases.
        "t": trend, "m": momentum, "v": volatility, "s": structure,
        "pa": price_action, "bo": breakout, "li": liquidity, "re": regime,
        "se": session, "cs": currency_strength, "eco": economic,
        "corr": correlation, "c": c, "vf": volume_flow, "ct": candle_timing,
        "mr": market_region, "dp": direct_probability,
        "_error": reason,
    }


def analyze_market(df, symbol, cfg, timeframe=None, *, deep_mtf=True):
    t = TrendEngine.analyze(df)
    m = MomentumEngine.analyze(df)
    v = VolatilityEngine.analyze(df)
    s = StructureEngine.analyze(df)
    pa = PriceActionEngine.analyze(df)
    sr = SupportResistanceEngine.analyze(df)
    bo = BreakoutEngine.analyze(df)
    li = LiquidityEngine.analyze(df)
    re = RegimeEngine.classify(t, v, s, bo)
    # LIVE session gate: use the current UTC clock, never the timestamp of
    # the last candle. This prevents stale/historical candles from showing
    # London during Tokyo/Sydney hours or during the weekend.
    se = SessionEngine.analyze(live=True)
    cs = CurrencyStrengthEngine.analyze(df, symbol)
    # The selected pair gets the full current-market MTF pass. Scanner and
    # daily-assessment calls can explicitly use the historical/resampled path so
    # they do not multiply external API requests for every pair.
    mtf = (
        MultiTimeframeEngine.analyze(df, symbol=symbol, cfg=cfg)
        if deep_mtf
        else MultiTimeframeEngine.analyze(df)
    )
    volume_flow = VolumeFlowEngine.analyze(df)
    candle_timing = CandleTimingEngine.analyze(df, timeframe or st.session_state.get("data_timeframe", "M5") or "M5")
    market_region = MarketRegionEngine.analyze(df)
    direct_probability = DirectProbabilityEngine.predict(df)
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
    advisory = {"trend":t,"momentum":m,"volatility":v,"structure":s,"price_action":pa,"sr":sr,
                "breakout":bo,"liquidity":li,"regime":re,"session":se,"mtf":mtf,"economic":eco,
                "cot":cot,"correlation":corr,"confluence":c,"risk":risk}
    advisory["advanced_momentum"] = adv_m
    advisory["ai"] = ai
    advisory["volume_flow"] = volume_flow
    advisory["candle_timing"] = candle_timing
    advisory["market_region"] = market_region
    advisory["direct_probability"] = direct_probability
    advisory["market_tracker"] = {"direction": c.get("direction","WAIT"), "score": c.get("score",0), "status":"CURRENT PAIR"}
    advisory["signal_timing"] = SignalTimingEngine.assess(pd.Timestamp.now(tz="UTC"), cfg.signal_max_age_seconds)
    ensemble = EnsembleDecisionEngine.decide(advisory, ai, dq, cfg)
    no_trade = NoTradeEngine.evaluate(advisory, ensemble, dq, cfg)
    quality = TradeQualityEngine.evaluate(advisory, ensemble, no_trade, dq)

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
        "volume_flow": volume_flow,
        "candle_timing": candle_timing,
        "market_region": market_region,
        "direct_probability": direct_probability,
        "market_tracker": advisory["market_tracker"],
        "signal_timing": advisory["signal_timing"],
        "mtf": mtf,
        "economic": eco,
        "cot": cot,
        "correlation": corr,
        "confluence": c,
        "risk": risk,
        "advanced_momentum": adv_m,
        "ai": ai,
        "ensemble": ensemble,
        "no_trade": no_trade,
        "trade_quality": quality,
        "data_quality": dq,
        # Compatibility aliases used by earlier V12.1 code.
        "t": t, "m": m, "v": v, "s": s, "pa": pa, "bo": bo, "li": li,
        "re": re, "se": se, "cs": cs, "eco": eco, "corr": corr, "c": c,
        "vf": volume_flow, "ct": candle_timing, "mr": market_region, "dp": direct_probability,
    }


def refresh_trade_quality_with_daily_signal(a, daily_signal):
    """Re-run the unified trade-quality gate after the locked Daily Signal is known."""
    if not isinstance(a, dict):
        return a
    a["trade_quality"] = TradeQualityEngine.evaluate(
        a,
        a.get("ensemble", {}),
        a.get("no_trade", {}),
        a.get("data_quality", {}),
        daily_signal=daily_signal,
    )
    return a


def fmt(v, n=2):
    try:
        return f"{float(v):,.{n}f}"
    except Exception:
        return str(v)


# ============================================================
# DASHBOARD
# ============================================================



# ============================================================
# V13 ADDITION: DERIV REAL-FOREX LIVE STREAM (READ-ONLY)
# ============================================================
class DerivRealForexStream:
    """Additive read-only Deriv real-Forex tick stream.

    This adapter accepts only instruments whose active_symbols metadata identifies
    both market=forex and underlying_symbol_type=forex. Synthetic/derived symbols
    are rejected before they can enter the V13 market-data layer.

    It bootstraps OHLC candles with ticks_history, then keeps the currently-forming
    candle updated from the live tick subscription. No account authentication and
    no order/trading endpoint is used.
    """
    TF_SECONDS = {"M1":60, "M5":300, "M15":900, "M30":1800, "H1":3600, "H4":14400, "D1":86400}
    _registry = {}
    _registry_lock = threading.RLock()

    @classmethod
    def symbol_map(cls):
        return {
            "EURUSD":"frxEURUSD", "GBPUSD":"frxGBPUSD", "USDJPY":"frxUSDJPY",
            "USDCHF":"frxUSDCHF", "AUDUSD":"frxAUDUSD", "USDCAD":"frxUSDCAD",
            "NZDUSD":"frxNZDUSD", "EURGBP":"frxEURGBP", "EURJPY":"frxEURJPY",
            "GBPJPY":"frxGBPJPY", "AUDCAD":"frxAUDCAD",
        }

    def __init__(self, symbol, endpoint, outputsize=500, reconnect_seconds=5):
        if websocket is None:
            raise RuntimeError("websocket-client is not installed. Add websocket-client to requirements.txt.")
        self.symbol = canonical_symbol(symbol)
        self.deriv_symbol = self.symbol_map().get(self.symbol)
        if not self.deriv_symbol:
            raise RuntimeError(f"Deriv real-Forex mapping is unavailable for {self.symbol}.")
        self.endpoint = endpoint
        self.outputsize = int(outputsize)
        self.reconnect_seconds = int(reconnect_seconds)
        self.ws = None
        self.thread = None
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.history_events = {}
        self.lock = threading.RLock()
        self.frames = {}
        self.status = {"connected":False,"validated":False,"source":"DERIV REAL FOREX","symbol":self.symbol,
                       "deriv_symbol":self.deriv_symbol,"last_tick":None,"last_quote":None,
                       "error":None,"streaming":False,"synthetic":False,"read_only":True,
                       "execution_enabled":False}

    @classmethod
    def get(cls, symbol, endpoint, outputsize=500, reconnect_seconds=5):
        key = canonical_symbol(symbol)
        with cls._registry_lock:
            obj = cls._registry.get(key)
            if obj is None:
                obj = cls(key, endpoint, outputsize, reconnect_seconds)
                cls._registry[key] = obj
            return obj

    def _send(self, payload):
        if self.ws is not None:
            self.ws.send(json.dumps(payload))

    def _validate_active_symbols(self, rows):
        for item in rows or []:
            sym = item.get("underlying_symbol", item.get("symbol"))
            market = str(item.get("market", "")).lower()
            typ = str(item.get("underlying_symbol_type", item.get("symbol_type", ""))).lower()
            if sym == self.deriv_symbol and market == "forex" and typ == "forex":
                with self.lock:
                    self.status["validated"] = True
                return True
        raise RuntimeError(f"{self.deriv_symbol} was not validated as a real Forex instrument by Deriv active_symbols.")

    def _on_open(self, ws):
        with self.lock:
            self.status.update({"connected":True,"streaming":False,"error":None,"endpoint":self.endpoint})
        self._send({"active_symbols":"brief","req_id":1001})

    def _on_message(self, ws, raw):
        try:
            data = json.loads(raw)
        except Exception:
            return
        if data.get("error"):
            err = data.get("error") or {}
            with self.lock:
                self.status["error"] = err.get("message") or err.get("code") or "Deriv WebSocket error"
            return
        if data.get("errors"):
            errors = data.get("errors") or []
            msg = "; ".join(str(e.get("message") or e.get("code") or e) for e in errors)
            with self.lock:
                self.status["error"] = msg or "Deriv WebSocket returned validation errors"
            return
        typ = data.get("msg_type")
        if typ == "active_symbols":
            try:
                self._validate_active_symbols(data.get("active_symbols", []))
                for tf in self.TF_SECONDS:
                    req_id = 2000 + list(self.TF_SECONDS).index(tf)
                    self._send({"ticks_history": self.deriv_symbol, "count": self.outputsize, "end": "latest",
                                "style": "candles", "granularity": self.TF_SECONDS[tf],
                                "subscribe": 0, "req_id": req_id,
                                "passthrough": {"v13_tf": tf, "v13_req_id": req_id}})
                self._send({"ticks": self.deriv_symbol, "subscribe": 1, "req_id": 3001,
                            "passthrough": {"v13_stream": "forex_ticks", "symbol": self.deriv_symbol}})
            except Exception as ex:
                with self.lock: self.status["error"] = str(ex)
            return
        if typ == "candles":
            echo_req = data.get("echo_req") or {}
            passthrough = data.get("passthrough") or echo_req.get("passthrough") or {}
            req_id = passthrough.get("v13_req_id", echo_req.get("req_id", 0))
            try:
                req_id = int(req_id or 0)
            except Exception:
                req_id = 0
            idx = req_id - 2000
            tfs = list(self.TF_SECONDS)
            tf = passthrough.get("v13_tf")
            if not tf and 0 <= idx < len(tfs):
                tf = tfs[idx]
            if tf in self.TF_SECONDS:
                rows = data.get("candles", [])
                frame = pd.DataFrame(rows)
                if not frame.empty:
                    frame = frame.rename(columns={"epoch":"time"})
                    frame["time"] = pd.to_datetime(pd.to_numeric(frame["time"], errors="coerce"), unit="s", utc=True)
                    for c in ["open","high","low","close"]:
                        frame[c] = pd.to_numeric(frame[c], errors="coerce")
                    frame["volume"] = 0.0
                    frame = frame[["time","open","high","low","close","volume"]].dropna()
                    with self.lock:
                        self.frames[tf] = MarketDataEngine.normalize(frame)
                        self.history_events[tf] = True
            return
        if typ == "tick" and data.get("tick"):
            tick = data["tick"]
            try:
                quote = float(tick["quote"]); epoch = int(tick["epoch"])
                if tick.get("symbol") != self.deriv_symbol or not math.isfinite(quote):
                    return
                ts = pd.Timestamp(epoch, unit="s", tz="UTC")
            except Exception:
                return
            with self.lock:
                sub = data.get("subscription") or {}
                self.status.update({"last_tick":ts.isoformat(),"last_quote":quote,"streaming":True,
                                    "subscription_id": sub.get("id", self.status.get("subscription_id"))})
                for tf, seconds in self.TF_SECONDS.items():
                    start = pd.Timestamp((epoch // seconds) * seconds, unit="s", tz="UTC")
                    frame = self.frames.get(tf)
                    if frame is None or frame.empty:
                        continue
                    if frame.iloc[-1]["time"] < start:
                        new = pd.DataFrame([{"time":start,"open":quote,"high":quote,"low":quote,"close":quote,"volume":0.0}])
                        frame = pd.concat([frame,new], ignore_index=True)
                    elif frame.iloc[-1]["time"] == start:
                        i = frame.index[-1]
                        frame.at[i,"high"] = max(float(frame.at[i,"high"]), quote)
                        frame.at[i,"low"] = min(float(frame.at[i,"low"]), quote)
                        frame.at[i,"close"] = quote
                    self.frames[tf] = frame.tail(self.outputsize).reset_index(drop=True)

    def _on_error(self, ws, error):
        with self.lock:
            self.status["error"] = str(error)
            self.status["connected"] = False

    def _on_close(self, ws, code, msg):
        with self.lock:
            self.status["connected"] = False
            self.status["streaming"] = False

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.ready_event.clear()
                self.ws = websocket.WebSocketApp(self.endpoint, on_open=self._on_open,
                    on_message=self._on_message, on_error=self._on_error, on_close=self._on_close)
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as ex:
                with self.lock: self.status["error"] = str(ex)
            if not self.stop_event.is_set():
                _time.sleep(self.reconnect_seconds)

    def start(self, wait_seconds=15):
        if self.thread and self.thread.is_alive():
            return self.snapshot()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name=f"deriv-forex-{self.symbol}", daemon=True)
        self.thread.start()
        deadline = _time.time() + wait_seconds
        while _time.time() < deadline:
            with self.lock:
                if self.status.get("validated") and self.frames:
                    return self.snapshot()
                err = self.status.get("error")
            if err:
                raise RuntimeError(err)
            _time.sleep(0.1)
        status = self.snapshot()
        detail = status.get("error") or "No validation/bootstrap response received before timeout."
        raise RuntimeError(f"Deriv real-Forex stream startup failed: {detail} | status={status}")

    def frame(self, timeframe):
        with self.lock:
            df = self.frames.get(str(timeframe).upper())
            return None if df is None else df.copy()

    def snapshot(self):
        with self.lock:
            s = dict(self.status)
        last = s.get("last_tick")
        if last:
            try:
                age = max(0.0, (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last)).total_seconds())
            except Exception:
                age = float("inf")
        else:
            age = float("inf")
        s["tick_age_seconds"] = round(age, 2) if math.isfinite(age) else None
        s["healthy"] = bool(s.get("connected") and s.get("validated") and s.get("streaming"))
        return s

    def stop(self):
        self.stop_event.set()
        try:
            if self.ws: self.ws.close()
        except Exception: pass
        with self.lock:
            self.status["connected"] = False
            self.status["streaming"] = False


def get_deriv_real_forex_data(symbol: str, timeframe: str, cfg: Config, force=False):
    """Return validated live Deriv real-Forex candles for the existing V13 data layer."""
    if websocket is None:
        raise RuntimeError("Deriv streaming requires websocket-client. Add websocket-client to requirements.txt.")
    symbol = canonical_symbol(symbol); timeframe = str(timeframe).upper()
    if symbol not in DerivRealForexStream.symbol_map():
        raise RuntimeError(f"No verified Deriv real-Forex mapping configured for {symbol}.")
    stream = DerivRealForexStream.get(symbol, cfg.deriv_stream_endpoint, cfg.deriv_stream_outputsize, cfg.deriv_stream_reconnect_seconds)
    stream.start()
    df = stream.frame(timeframe)
    if df is None or df.empty:
        raise RuntimeError(f"Deriv real-Forex stream has not produced {symbol}/{timeframe} candles yet.")
    status = stream.snapshot()
    if not status.get("healthy") or (status.get("tick_age_seconds") is not None and status["tick_age_seconds"] > cfg.deriv_stream_stale_seconds):
        raise RuntimeError(f"Deriv real-Forex stream is stale/unhealthy for {symbol}: {status}")
    meta = {"symbol":symbol,"timeframe":timeframe,"source":"DERIV REAL FOREX",
            "provider_symbol":stream.deriv_symbol,"synthetic":False,"market":"FOREX",
            "underlying_symbol_type":"forex","read_only":True,"execution_enabled":False,
            "streaming":True,"stream_health":status,"fetched_at":datetime.now(timezone.utc).isoformat()}
    return MarketDataEngine.normalize(df), meta

# ============================================================
# END DERIV REAL-FOREX LIVE STREAM
# ============================================================

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
    return replace(cfg, data_max_age_seconds=max(int(cfg.data_max_age_seconds), 432000))


def get_daily_market_data(symbol: str, cfg: Config, force: bool = False):
    """Fetch the exact selected pair on D1 for the upfront daily outlook."""
    return get_live_pair_data(canonical_symbol(symbol), "D1", cfg, force=force)


def analyze_daily_market(df: pd.DataFrame, symbol: str, cfg: Config):
    """Run the existing full V13 analysis stack on completed D1 candles only."""
    completed = DailySignalAssessmentEngine.completed_d1(df)
    if completed.empty:
        raise RuntimeError("No completed D1 candle is available for daily signal assessment.")
    return analyze_market(
        completed, canonical_symbol(symbol), _daily_analysis_config(cfg), "D1", deep_mtf=False
    )


def get_locked_daily_signal(symbol: str, cfg: Config, force: bool = False):
    """Return one validated Daily Signal for the current UTC day.

    Intraday reruns reuse the locked result. A new daily cycle is the only normal
    event that causes a fresh D1 assessment. Force is reserved for explicit
    operator revalidation after a data/source change.
    """
    cycle = datetime.now(timezone.utc).date().isoformat()
    key = f"{canonical_symbol(symbol)}|{cycle}|{str(cfg.data_source).upper()}"
    cache = st.session_state.setdefault("daily_signal_cache", {})
    if not force and key in cache:
        return cache[key]
    daily_df, daily_meta = get_daily_market_data(symbol, cfg, force=force)
    report = DailySignalAssessmentEngine.evaluate(daily_df, symbol, cfg)
    report["source"] = str(daily_meta.get("source", cfg.data_source)).upper()
    report["symbol"] = canonical_symbol(symbol)
    if report.get("locked"):
        cache[key] = report
        st.session_state["daily_signal_locked"] = report
    return report



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
        "daily_signal_cache": {},
        "daily_signal_locked": None,
        "td_request_log": [],
        "td_budget": {"request_log": [], "daily_log": [], "cooldown_until": 0.0, "key_cooldowns": {}, "last_request": None, "last_error": None, "last_error_at": None},
        "scanner_rotation_index": 0,
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
    st.title("V13 AI Trading Platform · V12.1 Protected Baseline")
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
        source_options = ["DEMO", "TWELVE DATA", "MT5", "DERIV REAL FOREX"]
        default_source_index = 1 if configured_td_key else 0
        data_source = st.selectbox("Data source", source_options, index=default_source_index)
        cfg.data_source = data_source
        cfg.allow_live_source_failover = st.checkbox(
            "Allow LIVE source failover (MT5 ↔ Twelve Data)",
            value=True,
            help="Only live currency-pair feeds may fail over. Synthetic DEMO data is never used as a live fallback."
        )
        if data_source == "TWELVE DATA":
            cfg.twelve_data_api_key = st.text_input("Twelve Data API key", type="password", value=st.session_state.get("td_key", configured_td_key))
            budget_view = V13DataBudgetManager.snapshot(cfg)
            st.caption(
                f"V13 central API budget: {budget_view['minute_used']}/{budget_view['minute_budget']} per minute · "
                f"{budget_view['daily_used']}/{budget_view['daily_budget']} per UTC day · "
                f"cooldown {budget_view['cooldown_seconds']}s. Cached candles are reused before new requests."
            )
            st.session_state.td_key = cfg.twelve_data_api_key
            cfg.twelve_data_outputsize = st.number_input("Historical candles", 200, 5000, 500, 100)
            if st.button("🔄 Refresh Twelve Data", use_container_width=True):
                try:
                    live_df, meta = _fetch_twelve_data(symbol, timeframe, cfg)
                    live_df, meta = store_market_data(symbol, timeframe, live_df, meta, "TWELVE DATA")
                    st.session_state.live_status = LiveConnectionManager.status("TWELVE DATA", live_df, MarketDataEngine.validate(live_df), DataIntegrityEngine.assess(live_df, timeframe, cfg.data_max_age_seconds), meta)
                    st.success(f"Twelve Data: {len(live_df):,} candles loaded for {display_symbol(symbol)} · {timeframe}.")
                except TwelveDataRateLimitError as ex:
                    st.session_state.live_status = {"source":"TWELVE DATA","connected":False,"read_only":True,"execution_enabled":False,"error":str(ex),"retry_after":ex.retry_after}
                    st.warning(f"Twelve Data quota protection is active. No duplicate request was sent. Wait about {ex.retry_after}s and V13 will reuse cached data where possible.")
                except Exception as ex:
                    st.session_state.live_status = {"source":"TWELVE DATA","connected":False,"read_only":True,"execution_enabled":False,"error":str(ex)}
                    st.error(f"Twelve Data error: {ex}")
        elif data_source == "MT5":
            st.caption("MT5 provides live broker/demo prices to V13. This adapter is read-only; it never places orders.")
            mt5_terminal_path = st.text_input("MT5 terminal path (optional)", value=st.session_state.get("mt5_path", ""), help="Leave blank when the default MT5 terminal is installed.")
            st.session_state.mt5_path = mt5_terminal_path
            cfg.mt5_outputsize = st.number_input("MT5 historical candles", 200, 5000, 500, 100, key="mt5_outputsize")
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
        elif data_source == "DERIV REAL FOREX":
            st.caption("Deriv public WebSocket · read-only real-Forex market data. Synthetic/derived symbols are rejected by the adapter.")
            cfg.deriv_stream_enabled = True
            cfg.deriv_stream_outputsize = st.number_input("Deriv historical candles", 200, 5000, 500, 100, key="deriv_outputsize")
            cfg.deriv_stream_stale_seconds = st.number_input("Live tick stale limit (seconds)", 5, 120, 15, 1, key="deriv_stale")
            if st.button("🟢 Connect Deriv Real-Forex Stream", use_container_width=True):
                try:
                    live_df, meta = get_deriv_real_forex_data(symbol, timeframe, cfg, force=True)
                    live_df, meta = store_market_data(symbol, timeframe, live_df, meta, "DERIV REAL FOREX")
                    dq = DataIntegrityEngine.assess(live_df, timeframe, cfg.data_max_age_seconds)
                    validation = MarketDataEngine.validate(live_df)
                    st.session_state.live_status = LiveConnectionManager.status("DERIV REAL FOREX", live_df, validation, dq, meta)
                    st.session_state.live_status.update(meta.get("stream_health", {}))
                    st.success(f"Deriv real Forex stream: {display_symbol(symbol)} · {timeframe} · {len(live_df):,} candles loaded.")
                except Exception as ex:
                    st.session_state.live_status = {"source":"DERIV REAL FOREX","connected":False,"read_only":True,"execution_enabled":False,"error":str(ex),"synthetic":False}
                    st.error(f"Deriv real-Forex stream error: {ex}")
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
            "In live modes, the selected pair is fetched directly; no other pair is substituted."
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
        if str(cfg.data_source).upper() == "MT5" and mt5 is None:
            if _get_twelve_data_key(cfg):
                st.warning("MT5 is unavailable in this environment. A Twelve Data key is configured, so enable LIVE source failover or select Twelve Data to read the selected currency pair.")
            else:
                st.warning("MT5 is unavailable in this environment. For Streamlit Cloud, select Twelve Data and provide your Twelve Data API key. No synthetic candles will be substituted for live forex data.")
        df = _empty_market_data()
        resolved_meta = {}

    if cfg.data_source == "MT5" and st.session_state.get("mt5_connected") and not df.empty:
        tick = MT5LiveDataEngine.tick(symbol)
        if tick:
            st.info(f"🟢 MT5 LIVE · {display_symbol(symbol)} · Bid {tick['bid']} · Ask {tick['ask']} · Spread {tick['spread']}")
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
    # Preserve the live-stream health details from the additive Deriv adapter.
    if resolved_source == "DERIV REAL FOREX":
        stream_health = st.session_state.get("data_meta", {}).get("stream_health", {})
        if stream_health:
            st.session_state.live_status.update(stream_health)
            if stream_health.get("tick_age_seconds") is not None:
                st.session_state.live_status["connected"] = bool(
                    stream_health.get("healthy") and
                    stream_health.get("tick_age_seconds", 999999) <= int(cfg.deriv_stream_stale_seconds)
                )
    if st.session_state.get("data_meta", {}).get("fallback_reason"):
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
    # DAILY MARKET SIGNAL — validated once per UTC day, then locked
    # ========================================================
    st.subheader(f"🌅 Daily Market Signal · {display_symbol(symbol)}")
    daily_report = None
    try:
        daily_report = get_locked_daily_signal(symbol, cfg)
    except Exception as daily_ex:
        st.warning(f"Daily signal unavailable for {display_symbol(symbol)}: {daily_ex}")

    if daily_report:
        # The Daily Signal is a locked directional filter for the UTC day.
        # Re-evaluate Trade Quality now that the daily assessment is available.
        a = refresh_trade_quality_with_daily_signal(a, daily_report)
        ds1, ds2, ds3, ds4, ds5 = st.columns(5)
        ds1.metric("DAILY SIGNAL", daily_report.get("signal", "WAIT"))
        ds2.metric("Status", daily_report.get("status", "UNCONFIRMED"))
        ds3.metric("Engine Agreement", f"{daily_report.get('engine_agreement', 0):.1f}%")
        ds4.metric("Confidence", f"{daily_report.get('confidence', 0):.1f}%")
        ds5.metric("Engine Health", f"{daily_report.get('engine_health', {}).get('percent', 0):.1f}%")
        ds6, ds7, ds8, ds9 = st.columns(4)
        ds6.metric("AI Daily UP", f"{daily_report.get('ai_up', 0):.1f}%")
        ds7.metric("AI Daily DOWN", f"{daily_report.get('ai_down', 0):.1f}%")
        ds8.metric("D1 Candle", str(daily_report.get('completed_candle', '-')).replace('+00:00',' UTC'))
        ds9.metric("Cycle", daily_report.get('cycle_utc', '-'))

        dq = daily_report.get('data_quality', {})
        st.caption(
            f"Daily source: {daily_report.get('source', cfg.data_source)} · "
            f"{display_symbol(symbol)}/D1 · COMPLETED CANDLE ONLY · "
            f"Data quality: {dq.get('status','UNKNOWN')} ({dq.get('score',0):.0f}/100) · "
            f"Signal is locked for the UTC day once validated."
        )
        if daily_report.get('status') == 'LOCKED':
            st.success(
                f"🔒 DAILY SIGNAL LOCKED: {daily_report.get('signal')} · "
                "Intraday refreshes cannot rewrite this Daily Signal. "
                "M5/M15/H1 engines may independently return BUY, SELL or WAIT."
            )
        else:
            st.warning(
                "⚠️ DAILY SIGNAL UNCONFIRMED — no directional daily signal is locked. "
                + ("; ".join(daily_report.get('reasons', [])) or "Insufficient validated consensus.")
            )

        with st.expander("View complete Daily Engine Assessment", expanded=False):
            st.json({
                "Daily Signal": daily_report.get('signal'),
                "Status": daily_report.get('status'),
                "Engine Agreement": daily_report.get('engine_agreement'),
                "AI": {"UP": daily_report.get('ai_up'), "DOWN": daily_report.get('ai_down'), "Direction": daily_report.get('ai_direction')},
                "Data Quality": daily_report.get('data_quality'),
                "Engine Health": daily_report.get('engine_health'),
                "Reasons": daily_report.get('reasons'),
                "Analysis": daily_report.get('analysis'),
            })
    else:
        st.warning(
            f"NO VERIFIED DAILY DATA · {display_symbol(symbol)}/D1. "
            "The Daily Signal remains unconfirmed until the completed D1 dataset passes quality and engine validation."
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
    if str(cfg.data_source).upper() == "TWELVE DATA":
        bv = V13DataBudgetManager.snapshot(cfg)
        st.caption(
            f"Central data budget · {bv['minute_used']}/{bv['minute_budget']} req/min · "
            f"{bv['daily_used']}/{bv['daily_budget']} req/day · cooldown {bv['cooldown_seconds']}s"
        )
    st.caption(
        f"Pair: {display_symbol(symbol)} · Loaded data pair: {display_symbol(st.session_state.get('data_symbol') or 'NONE')} · "
        f"Timeframe: {timeframe} · Loaded source: {st.session_state.get('data_meta', {}).get('source', 'NONE')} · "
        f"Data: {'OK' if validation['data_ok'] else 'INSUFFICIENT'} · UTC · "
        f"{validation['rows']:,} candles · Quality: {data_quality['status']} · "
        f"Emergency: {'STOPPED' if st.session_state.emergency else 'NORMAL'}"
    )
    if str(cfg.data_source).upper() == "TWELVE DATA" and not st.session_state.live_status.get("connected",False):
        st.warning("Twelve Data is selected but no successful live fetch is currently loaded. Signal generation is disabled until the selected pair returns valid candles.")

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
        # The free Twelve Data tier is deliberately protected: deep MTF for the
        # selected pair can consume several requests, so the scanner never fires
        # seven additional full-MTF requests on every Streamlit rerun. Other pairs
        # use fresh cache first, then at most ONE rotated live fetch per refresh.
        scanner_idx = int(st.session_state.get("scanner_rotation_index", 0)) % max(len(watch), 1)
        rotation_symbol = watch[scanner_idx] if watch else None
        st.session_state.scanner_rotation_index = (scanner_idx + 1) % max(len(watch), 1)
        for sym in watch:
            try:
                if canonical_symbol(sym) == canonical_symbol(symbol):
                    x = df
                else:
                    cached = _td_cached_pair(sym, timeframe, cfg) if str(cfg.data_source).upper() == "TWELVE DATA" else None
                    if cached is not None:
                        x, _ = cached
                    elif sym == rotation_symbol and str(cfg.data_source).upper() == "TWELVE DATA":
                        x, _ = get_live_pair_data(sym, timeframe, cfg, force=False)
                    else:
                        raise RuntimeError("NO FRESH SCANNER CACHE — ROTATION PENDING")
                if x is None or x.empty:
                    raise RuntimeError("empty feed")
                # Scanner ranking remains a real analysis pass, but it does not
                # recursively open six more live MTF requests for every symbol.
                aa = analyze_market(x, sym, cfg, timeframe, deep_mtf=False)
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
        st.markdown("### Authoritative Current Trading Session")
        se_live = a.get("session", {})
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Current Session", se_live.get("session", "UNKNOWN"))
        sc2.metric("Market", "OPEN" if se_live.get("market_open") else "CLOSED")
        sc3.metric("Tradeable", "YES" if se_live.get("session_tradeable") else "NO")
        sc4.metric("UTC", f"{se_live.get('hour',0):02d}:{se_live.get('minute',0):02d}")
        st.caption(f"Clock: {se_live.get('clock_source','UNKNOWN')} · Active: {', '.join(se_live.get('active_sessions',[])) or 'NONE'} · Next open: {(se_live.get('next_open') or {}).get('session','-')} {(se_live.get('next_open') or {}).get('utc','')}" )
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
        st.json({"AI":a["ai"],"Direct Probability":a.get("direct_probability"),"Volume Flow":a.get("volume_flow"),"Candle Timing":a.get("candle_timing"),"Market Region":a.get("market_region"),"Advanced Momentum":a["advanced_momentum"],"Ensemble":a["ensemble"],"No Trade":a["no_trade"],"Trade Quality":a["trade_quality"],"Data Quality":a["data_quality"]})

        st.markdown("### Central Engine Information Layer")
        bus = a.get("trade_quality", {}).get("engine_information_bus", {})
        if bus:
            bus_rows = []
            for name, item in bus.items():
                bus_rows.append({"Engine": name, "Role": item.get("role"), "Direction": item.get("direction"), "Strength": item.get("strength"), "Healthy": "YES" if item.get("healthy") else "NO", "Weight": item.get("weight")})
            st.dataframe(pd.DataFrame(bus_rows), use_container_width=True, hide_index=True)
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
                ("Volume Flow", "volume_flow"),
                ("Candle Timing", "candle_timing"),
                ("Market Region", "market_region"),
                ("Direct Probability", "direct_probability"),
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

            # Current-market MTF verification: D1 is fetched directly rather
            # than inferred from the selected intraday candle set.
            mtf_details = a["mtf"].get("details", {})
            d1_detail = mtf_details.get("D1", {})
            if a["mtf"].get("live_current_market"):
                d1_status = (
                    "AVAILABLE" if a["mtf"].get("daily_current_available")
                    else "UNAVAILABLE / BLOCKED"
                )
                st.info(
                    f"CURRENT D1 MARKET: {d1_status} · "
                    f"Direction: {a['mtf'].get('daily_current_direction','UNAVAILABLE')} · "
                    f"Quality: {d1_detail.get('data_quality','UNKNOWN')} · "
                    f"Age: {float(d1_detail.get('age_seconds',0.0)):.0f}s"
                )

                detail_rows = []
                for tf in MultiTimeframeEngine.TIMEFRAMES:
                    d = mtf_details.get(tf, {})
                    detail_rows.append({
                        "TF": tf,
                        "Direction": a["mtf"]["states"].get(tf, "UNAVAILABLE"),
                        "Data": d.get("data_quality", d.get("status", "UNKNOWN")),
                        "Age(s)": round(float(d.get("age_seconds", 0.0)), 0),
                        "Engine agreement": round(float(d.get("agreement", 0.0)), 0),
                        "Source": d.get("source", "UNKNOWN"),
                    })
                st.dataframe(
                    pd.DataFrame(detail_rows),
                    use_container_width=True,
                    hide_index=True,
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
            live_gate = LiveTradeEligibilityEngine.evaluate(
                a=a,
                ensemble=a["ensemble"],
                no_trade=a["no_trade"],
                trade_quality=a["trade_quality"],
                data_quality=a["data_quality"],
                timing=timing,
                emergency=st.session_state.emergency,
            )
            approved = bool(a["risk"]["approved"] and fx["approved"] and live_gate["eligible"])

            # Keep the three meanings separate on the Trade Desk:
            # analysis strength, AI confidence, and live-data eligibility.
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Analysis / Confluence", f"{live_gate['analysis_score']:.1f}%")
            g2.metric("AI Confidence", f"{live_gate['ai_confidence']:.1f}%")
            g3.metric("Data Integrity", f"{live_gate['data_integrity_score']:.1f}/100")
            g4.metric(
                "Live Eligibility",
                "APPROVED" if live_gate["eligible"] else "BLOCKED",
            )
            st.caption(
                f"Feed freshness: {live_gate['data_freshness_seconds']:.0f}s · "
                f"{'VALID' if live_gate['data_freshness_ok'] else 'STALE / INVALID'} · "
                f"Trade-quality gate: {live_gate['trade_quality_grade']} "
                f"({live_gate['trade_quality_score']:.1f}/100)"
            )

            if not live_gate["eligible"]:
                st.error(
                    "🔴 TRADE BLOCKED — " +
                    " | ".join(live_gate["reasons"] or ["LIVE TRADE ELIGIBILITY FAILED"])
                )
            else:
                st.success(
                    "🟢 LIVE TRADE ELIGIBILITY PASSED — "
                    "analysis, data integrity, freshness and risk gates are clear."
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
            live_gate = LiveTradeEligibilityEngine.evaluate(
                a=a,
                ensemble=a["ensemble"],
                no_trade=a["no_trade"],
                trade_quality=a["trade_quality"],
                data_quality=a["data_quality"],
                timing=timing,
                emergency=st.session_state.emergency,
            )
            approved = bool(a["risk"]["approved"] and bi["approved"] and live_gate["eligible"])

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Analysis / Confluence", f"{live_gate['analysis_score']:.1f}%")
            g2.metric("AI Confidence", f"{live_gate['ai_confidence']:.1f}%")
            g3.metric("Data Integrity", f"{live_gate['data_integrity_score']:.1f}/100")
            g4.metric(
                "Live Eligibility",
                "APPROVED" if live_gate["eligible"] else "BLOCKED",
            )
            st.caption(
                f"Feed freshness: {live_gate['data_freshness_seconds']:.0f}s · "
                f"{'VALID' if live_gate['data_freshness_ok'] else 'STALE / INVALID'} · "
                f"Trade-quality gate: {live_gate['trade_quality_grade']} "
                f"({live_gate['trade_quality_score']:.1f}/100)"
            )

            if not live_gate["eligible"]:
                st.error(
                    "🔴 TRADE BLOCKED — " +
                    " | ".join(live_gate["reasons"] or ["LIVE TRADE ELIGIBILITY FAILED"])
                )
            else:
                st.success(
                    "🟢 LIVE TRADE ELIGIBILITY PASSED — "
                    "analysis, data integrity, freshness and risk gates are clear."
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
            ("SESSION VALID", a["se"]["session_tradeable"]),
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
                ["Candle Timing", "READY"],
                ["Market Region", "READY"],
                ["Direct Probability", "READY"],
                ["Engine Information Bus", "READY"],
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


# -------------------- V13 TWO-LAYER DAILY SIGNAL MODEL --------------------
# Layer A: stable, validated daily thesis based on completed D1 candles.
# Layer B: live development of the currently-forming D1 candle.
# The live layer can confirm/contradict the thesis, but cannot rewrite it
# during the same UTC day.

DAILY_SIGNAL_STATES = {"BULLISH", "BEARISH", "UNCONFIRMED"}
LIVE_DAILY_STATES = {"BULLISH", "BEARISH", "NEUTRAL", "UNKNOWN"}

def _v13_utc_day_key(ts=None):
    t = pd.Timestamp(ts if ts is not None else datetime.now(timezone.utc))
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.date().isoformat()

def _v13_completed_daily_frame(df):
    """Return completed D1 candles only; never use the forming D1 candle."""
    x = MarketDataEngine.normalize(df)
    if x.empty:
        return x
    x = x.sort_index()
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    else:
        x.index = x.index.tz_convert("UTC")

    x = x[x.index.dayofweek < 5].copy()
    if x.empty:
        return x

    now = pd.Timestamp(datetime.now(timezone.utc))
    # Current UTC day is not yet a completed daily candle.
    if x.index[-1].date() >= now.date():
        x = x.iloc[:-1].copy()
    return x

def _v13_forming_daily_frame(df):
    """Return the live D1 dataset separately from the completed D1 dataset."""
    x = MarketDataEngine.normalize(df)
    if x.empty:
        return x
    x = x.sort_index()
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    else:
        x.index = x.index.tz_convert("UTC")
    return x[x.index.dayofweek < 5].copy()

def _v13_direction_from_score(score):
    if score > 0.10:
        return "BULLISH"
    if score < -0.10:
        return "BEARISH"
    return "UNCONFIRMED"

def build_v13_daily_signal_layers(df_d1, analysis_result=None):
    """
    Produce two independent D1 outputs:
      - validated Daily Signal from completed candles
      - live development from the forming candle

    The second layer is informational/confirmatory and never overwrites the
    first layer during the same UTC day.
    """
    completed = _v13_completed_daily_frame(df_d1)
    forming = _v13_forming_daily_frame(df_d1)
    ar = analysis_result or {}

    if completed.empty:
        return {
            "daily_signal": "UNCONFIRMED",
            "daily_signal_status": "DATA_INVALID",
            "daily_signal_locked": False,
            "daily_signal_day": _v13_utc_day_key(),
            "completed_d1_candle": None,
            "live_daily_state": "UNKNOWN",
            "live_daily_alignment": "UNKNOWN",
            "live_daily_warning": "Insufficient completed D1 data.",
        }

    # Normalize directional evidence from the existing V13 engines.
    directional = []
    keys = (
        "trend", "momentum", "structure", "price_action",
        "support_resistance", "breakout", "fvg", "volume_flow",
        "currency_strength", "correlation", "ai", "probability",
        "ml", "mtf", "confluence"
    )
    for key in keys:
        v = ar.get(key)
        blob = str(v).upper()
        if "BULL" in blob or "UP" in blob or "BUY" in blob:
            directional.append(1.0)
        elif "BEAR" in blob or "DOWN" in blob or "SELL" in blob:
            directional.append(-1.0)

    score = float(np.mean(directional)) if directional else 0.0
    daily_signal = _v13_direction_from_score(score)

    # AI is a validation layer: material opposition prevents a false daily
    # lock instead of allowing one engine to silently override all others.
    ai_up = ar.get("ai_up_probability", ar.get("up_probability"))
    ai_down = ar.get("ai_down_probability", ar.get("down_probability"))
    try:
        ai_up = float(ai_up) if ai_up is not None else None
        ai_down = float(ai_down) if ai_down is not None else None
    except (TypeError, ValueError):
        ai_up, ai_down = None, None

    if daily_signal == "BULLISH" and ai_down is not None and ai_down >= 0.55:
        daily_signal = "UNCONFIRMED"
    elif daily_signal == "BEARISH" and ai_up is not None and ai_up >= 0.55:
        daily_signal = "UNCONFIRMED"

    health = str(ar.get("health_status", "HEALTHY")).upper()
    data = str(ar.get("data_quality_status", "PASS")).upper()
    if health in {"FAILED", "CRITICAL"} or data in {"FAILED", "INVALID", "STALE"}:
        daily_signal = "UNCONFIRMED"

    live_state = "UNKNOWN"
    live_alignment = "UNKNOWN"
    warning = ""

    if not forming.empty:
        last = forming.iloc[-1]
        try:
            body = float(last["close"]) - float(last["open"])
            live_state = "BULLISH" if body > 0 else "BEARISH" if body < 0 else "NEUTRAL"
        except Exception:
            live_state = "UNKNOWN"

        if daily_signal in {"BULLISH", "BEARISH"} and live_state in {"BULLISH", "BEARISH"}:
            live_alignment = "CONFIRMING" if live_state == daily_signal else "CONTRADICTING"
            if live_alignment == "CONTRADICTING":
                warning = (
                    "The forming D1 candle contradicts the locked Daily Signal. "
                    "Do not rewrite the Daily Signal; require stronger intraday confirmation."
                )
        elif daily_signal == "UNCONFIRMED":
            live_alignment = "NO_CONFIRMED_THESIS"

    return {
        "daily_signal": daily_signal,
        "daily_signal_status": "VALIDATED" if daily_signal != "UNCONFIRMED" else "UNCONFIRMED",
        "daily_signal_locked": daily_signal != "UNCONFIRMED",
        "daily_signal_day": _v13_utc_day_key(),
        "completed_d1_candle": str(completed.index[-1]),
        "live_daily_state": live_state,
        "live_daily_alignment": live_alignment,
        "live_daily_warning": warning,
        "engine_directional_score": round(score, 4),
        "ai_up_probability": ai_up,
        "ai_down_probability": ai_down,
    }

def render_v13_daily_signal_layers(report):
    """Optional dashboard panel for the stable Daily Signal/live D1 split."""
    if not isinstance(report, dict):
        return
    st.subheader("Daily Signal")
    st.metric("Validated Daily Signal", report.get("daily_signal", "UNCONFIRMED"))
    st.caption(
        f"Status: {report.get('daily_signal_status', 'UNKNOWN')} · "
        f"UTC day: {report.get('daily_signal_day', '—')}"
    )
    st.metric("Developing D1 Candle", report.get("live_daily_state", "UNKNOWN"))
    st.caption(
        "Relationship to Daily Signal: "
        f"{report.get('live_daily_alignment', 'UNKNOWN')}"
    )
    if report.get("live_daily_warning"):
        st.warning(report["live_daily_warning"])


if __name__ == "__main__":
    dashboard()
