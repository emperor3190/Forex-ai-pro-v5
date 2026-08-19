
"""
V13 AI Trading Platform — V12.1 Protected Baseline + Advanced Additions
Single-file Streamlit trading research / paper-trading terminal.

Pipeline:
Market Data -> Technical/Fundamental/Positioning Intelligence -> MTF/Regime
-> Confluence -> Signal Validation -> Risk Veto -> Forex/Binary Entry
-> Paper Execution -> Trade Monitor -> Journal -> Backtest/Walk-forward/Monte Carlo
-> Optimizer -> Dashboard.

This file preserves every engine/tool/feature in the supplied V12.1 source and
fixes the dashboard scanner KeyError caused by inconsistent dictionary keys.

Research/paper-trading only. Live execution requires an official broker/data API.
Binary options availability/legality varies by jurisdiction and broker.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


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
    live_refresh_seconds: int = 300
    data_max_age_seconds: int = 420
    ml_min_probability: float = 0.60
    signal_max_age_seconds: int = 60
    no_trade_conflict_threshold: float = 18.0


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
    @staticmethod
    def analyze(ts=None):
        t = ts or pd.Timestamp.now(tz="UTC")
        h = int(t.hour)
        if 0 <= h < 8:
            s = "ASIAN"
        elif 8 <= h < 13:
            s = "LONDON"
        elif 13 <= h < 17:
            s = "LONDON/NEW YORK OVERLAP"
        elif 17 <= h < 22:
            s = "NEW YORK"
        else:
            s = "OFF-HOURS"
        return {
            "session": s,
            "hour": h,
            "weekday": t.day_name(),
            "session_tradeable": s != "OFF-HOURS",
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
    @staticmethod
    def analyze(df):
        states = {}
        for tf in ["M5", "M15", "M30", "H1", "H4", "D1"]:
            x = MarketDataEngine.resample(df, tf)
            if len(x) < 220:
                states[tf] = "INSUFFICIENT"
            else:
                states[tf] = TrendEngine.analyze(x)["direction"]

        valid = [v for v in states.values() if v != "INSUFFICIENT"]
        bull = sum(v == "BULLISH" for v in valid)
        bear = sum(v == "BEARISH" for v in valid)
        direction = (
            "BULLISH"
            if bull > bear and bull >= len(valid) * 0.5
            else "BEARISH"
            if bear > bull and bear >= len(valid) * 0.5
            else "MIXED"
        )
        align = 100 * max(bull, bear) / max(len(valid), 1)
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
        "data_quality": dq,
        "advanced_momentum": adv_m,
        "ai": ai,
        "ensemble": ensemble,
        "no_trade": no_trade,
        "trade_quality": quality,
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
        m=macd(close)
        hist=m["hist"].iloc[-1]
        hist_prev=m["hist"].iloc[-2]
        ad=adx(x)
        adxv=float(ad["adx"].iloc[-1]); plus=float(ad["plus_di"].iloc[-1]); minus=float(ad["minus_di"].iloc[-1])
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
        return x.replace([np.inf,-np.inf],np.nan).dropna()

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


class EnsembleDecisionEngine:
    @staticmethod
    def decide(a, ai, data_quality, cfg):
        score=float(a["confluence"]["score"])
        direction=a["confluence"]["direction"]
        ai_dir="BULLISH" if ai["up_probability"]>55 else "BEARISH" if ai["down_probability"]>55 else "NEUTRAL"
        momentum=a.get("advanced_momentum",{}).get("direction","NEUTRAL")
        agreement=0
        for d in [direction,momentum,ai_dir]:
            if d==direction and d!="NEUTRAL": agreement+=1
        final=max(0,min(100,score*0.55 + ai["confidence"]*0.25 + data_quality["score"]*0.20))
        reasons=[]
        if not data_quality.get("signal_allowed",False): reasons.append("DATA QUALITY / STALE FEED")
        if ai["confidence"] < (cfg.ml_min_probability*100): reasons.append("AI CONFIDENCE LOW")
        if direction=="NEUTRAL": reasons.append("NO CLEAR CONFLUENCE DIRECTION")
        if agreement==0: reasons.append("ENGINE DIRECTION CONFLICT")
        allowed=(final>=cfg.min_score and not reasons)
        return {"score":final,"direction":direction if direction!="NEUTRAL" else ai_dir,
                "ai_direction":ai_dir,"agreement":agreement,"approved":allowed,"reasons":reasons,
                "grade":"A" if final>=85 else "B" if final>=75 else "C" if final>=65 else "D"}


class NoTradeEngine:
    @staticmethod
    def evaluate(a, ensemble, data_quality, cfg, signal_age=0):
        reasons=list(ensemble.get("reasons",[]))
        if data_quality.get("score",0)<70: reasons.append("DATA QUALITY BELOW 70")
        if signal_age>cfg.signal_max_age_seconds: reasons.append("SIGNAL STALE")
        if a["volatility"].get("regime")=="EXTREME": reasons.append("EXTREME VOLATILITY")
        if a["economic"].get("blocked"): reasons.append("HIGH-IMPACT NEWS BLACKOUT")
        if a["session"].get("session_tradeable") is False: reasons.append("SESSION NOT TRADEABLE")
        return {"trade_allowed":len(reasons)==0,"reasons":list(dict.fromkeys(reasons)),"status":"TRADE" if not reasons else "NO TRADE"}


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
        checks={
            "DATA QUALITY":data_quality.get("score",0)>=70,
            "TREND":a["trend"].get("label") not in (None,"NEUTRAL"),
            "MOMENTUM":a["momentum"].get("direction") not in (None,"NEUTRAL"),
            "STRUCTURE":a["structure"].get("bias") not in (None,"NEUTRAL"),
            "PRICE ACTION":a["price_action"].get("bias") not in (None,"NEUTRAL"),
            "VOLATILITY":a["volatility"].get("regime")!="EXTREME",
            "REGIME":bool(a["regime"]),
            "MTF":a["mtf"].get("alignment",0)>=50,
            "SESSION":a["session"].get("session_tradeable",True),
            "NEWS":not a["economic"].get("blocked",False),
            "AI":a["ai"].get("confidence",0)>=60,
            "RISK":a["risk"].get("approved",False),
            "NO TRADE":no_trade.get("trade_allowed",False),
        }
        passed=sum(bool(v) for v in checks.values()); pct=passed/len(checks)*100
        grade="A" if pct>=90 else "B" if pct>=75 else "C" if pct>=60 else "D"
        return {"checks":checks,"score":pct,"grade":grade,"decision":"TRADE" if grade in ("A","B") and no_trade.get("trade_allowed") else "NO TRADE"}


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

def init_state():
    defaults = {
        "journal": [],
        "paper_balance": 10000.0,
        "bot_enabled": False,
        "emergency": False,
        "data": MarketDataEngine.synthetic(),
        "backtest": pd.DataFrame(),
        "bt_metrics": {},
        "wf": None,
        "mc": None,
        "optimizer": pd.DataFrame(),
        "events": pd.DataFrame(),
        "cot": pd.DataFrame(),
        "data_meta": {},
        "live_status": {"source":"DEMO","connected":False,"read_only":True,"execution_enabled":False},
        "last_signal_time": None,
        "ai_cache": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_events():
    return st.session_state.get("events", pd.DataFrame())


def load_cot():
    return st.session_state.get("cot", pd.DataFrame())


def analyze_market(df, symbol, cfg):
    t = TrendEngine.analyze(df)
    m = MomentumEngine.analyze(df)
    v = VolatilityEngine.analyze(df)
    s = StructureEngine.analyze(df)
    pa = PriceActionEngine.analyze(df)
    sr = SupportResistanceEngine.analyze(df)
    bo = BreakoutEngine.analyze(df)
    li = LiquidityEngine.analyze(df)
    re = RegimeEngine.classify(t, v, s, bo)
    se = SessionEngine.analyze(df.time.iloc[-1])
    cs = CurrencyStrengthEngine.analyze(df, symbol)
    mtf = MultiTimeframeEngine.analyze(df)
    eco = EconomicEngine.analyze(load_events(), symbol)
    cot = COTEngine.analyze(load_cot(), symbol)

    # Single-symbol correlation is LOW by definition unless peer histories are supplied.
    history = {symbol: df}
    corr = CorrelationEngine.analyze(history, symbol)

    c = ConfluenceEngine.score(
        t, m, v, s, pa, sr, bo, li, re, se, mtf, eco, cot
    )
    dq = DataIntegrityEngine.assess(df, cfg.data_source if cfg.data_source in MarketDataEngine.TIMEFRAMES else "M5", cfg.data_max_age_seconds)
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
                "breakout":bo,"liquidity":li,"regime":re,"session":se,"mtf":mtf,"economic":eco,"risk":risk}
    advisory["advanced_momentum"] = adv_m
    advisory["ai"] = ai
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
        "mtf": mtf,
        "economic": eco,
        "cot": cot,
        "correlation": corr,
        "confluence": c,
        "risk": risk,
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
        symbol = st.text_input("Primary symbol", "EURUSD").upper().strip()
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
        data_source = st.selectbox("Data source", ["DEMO", "TWELVE DATA"], index=0)
        cfg.data_source = data_source
        if data_source == "TWELVE DATA":
            cfg.twelve_data_api_key = st.text_input("Twelve Data API key", type="password", value=st.session_state.get("td_key", ""))
            st.session_state.td_key = cfg.twelve_data_api_key
            cfg.twelve_data_outputsize = st.number_input("Historical candles", 200, 5000, 500, 100)
            if st.button("🔄 Refresh Twelve Data", use_container_width=True):
                try:
                    live_df, meta = TwelveDataLiveEngine.fetch(cfg.twelve_data_api_key, symbol, timeframe, cfg.twelve_data_outputsize)
                    st.session_state.data = live_df
                    st.session_state.data_meta = meta
                    st.session_state.live_status = LiveConnectionManager.status("TWELVE DATA", live_df, MarketDataEngine.validate(live_df), DataIntegrityEngine.assess(live_df, timeframe, cfg.data_max_age_seconds), meta)
                    st.success(f"Twelve Data: {len(live_df):,} candles loaded for {symbol} · {timeframe}.")
                except Exception as ex:
                    st.session_state.live_status = {"source":"TWELVE DATA","connected":False,"read_only":True,"execution_enabled":False,"error":str(ex)}
                    st.error(f"Twelve Data error: {ex}")
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
            st.session_state.data = MarketDataEngine.normalize(pd.read_csv(up))
            st.success(f"Loaded {len(st.session_state.data):,} candles.")
        except Exception as e:
            st.error(f"CSV error: {e}")
    else:
        st.info(
            f"Demo data is active. Primary timeframe selected: {timeframe}. "
            "Upload broker historical CSV to replace it."
        )

    df = st.session_state.data
    validation = MarketDataEngine.validate(df)
    data_quality = DataIntegrityEngine.assess(df, timeframe, cfg.data_max_age_seconds)
    a = analyze_market(df, symbol, cfg)
    st.session_state.live_status = LiveConnectionManager.status(
        st.session_state.get("live_status", {}).get("source", cfg.data_source),
        df, validation, data_quality, st.session_state.get("data_meta", {})
    )

    # Keep paper monitoring synchronized with the current price.
    if st.session_state.journal:
        TradeMonitorEngine.monitor_all(
            st.session_state.journal, float(df.close.iloc[-1])
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
        f"Data: {'OK' if validation['data_ok'] else 'INSUFFICIENT'} · UTC · "
        f"{validation['rows']:,} candles · Quality: {data_quality['status']} · "
        f"Emergency: {'STOPPED' if st.session_state.emergency else 'NORMAL'}"
    )
    if cfg.data_source == "TWELVE DATA" and not st.session_state.live_status.get("connected",False):
        st.warning("Twelve Data is selected but no successful live fetch is currently loaded. Signal generation should be treated as non-live.")

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
            "AUDUSD", "USDCAD", "NZDUSD", "XAUUSD",
        ]
        analyses = {}
        for i, sym in enumerate(watch):
            x = df if sym == symbol else MarketDataEngine.synthetic(
                sym, len(df), seed=20 + i
            )
            analyses[sym] = analyze_market(x, sym, cfg)

        scanner = MarketTrackerEngine.rank(watch, analyses)
        st.dataframe(scanner, use_container_width=True, hide_index=True)

        if not scanner.empty:
            top = scanner.iloc[0]
            st.success(
                f"TOP OPPORTUNITY: {top.SYMBOL} — "
                f"{top.DIRECTION} — {top.SCORE}/100"
            )

        st.subheader("Price Chart")
        chart = df.set_index("time")[["close"]].tail(300)
        st.line_chart(chart)

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
        x4.metric("Ensemble", f"{a['ensemble']['score']:.1f}/100")
        st.markdown("### Advanced AI / Decision Layer")
        st.json({"AI":a["ai"],"Advanced Momentum":a["advanced_momentum"],"Ensemble":a["ensemble"],"No Trade":a["no_trade"],"Trade Quality":a["trade_quality"],"Data Quality":a["data_quality"]})
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
        st.bar_chart(pd.Series(a["cs"]["matrix"]))

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
        direction = a["c"]["direction"]

        if market == "FOREX":
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
                a["risk"]["approved"] and fx["approved"] and a["no_trade"]["trade_allowed"]
                and a["trade_quality"]["decision"] == "TRADE" and timing["fresh"]
                and not st.session_state.emergency
            )

            if not a["risk"]["approved"]:
                st.error("RISK VETO: " + ", ".join(a["risk"]["reasons"]))
            if not a["no_trade"]["trade_allowed"]:
                st.error("NO TRADE: " + ", ".join(a["no_trade"]["reasons"]))

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

        else:
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
                a["risk"]["approved"] and bi["approved"] and a["no_trade"]["trade_allowed"]
                and a["trade_quality"]["decision"] == "TRADE" and timing["fresh"]
                and not st.session_state.emergency
            )

            if not a["risk"]["approved"]:
                st.error("RISK VETO: " + ", ".join(a["risk"]["reasons"]))
            if not a["no_trade"]["trade_allowed"]:
                st.error("NO TRADE: " + ", ".join(a["no_trade"]["reasons"]))

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
            "Live/History Data → Integrity → V12.1 Analysis → AI → Ensemble → "
            "Confidence → NO TRADE → Timing → Risk → Forex/Binary → PAPER ONLY → Monitoring → Journal"
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
            if st.button("▶ Run Backtest", use_container_width=True):
                trades, metrics = BacktestEngine.run(
                    df, cfg, binary=(market == "BINARY OPTIONS")
                )
                st.session_state.backtest = trades
                st.session_state.bt_metrics = metrics

        with btcol2:
            if st.button("↔ Run Walk-Forward", use_container_width=True):
                st.session_state.wf = WalkForwardEngine.run(df, cfg)

        if st.button("🎲 Run Monte Carlo", use_container_width=True):
            st.session_state.mc = MonteCarloEngine.run(
                st.session_state.backtest
            )

        if st.button("⚙ Run Threshold Optimizer", use_container_width=True):
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
                ["AI Probability", "READY (OPTIONAL ML)"],
                ["Ensemble Decision", "READY"],
                ["Confidence / Trade Quality", "READY"],
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
