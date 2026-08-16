
import os
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# FOREX AI PRO V8
# Research-grade Streamlit application scaffold
# Data source: Twelve Data
#
# IMPORTANT:
# - This application is a research / paper-trading system.
# - It does not guarantee profitable predictions.
# - Real-money execution should be implemented separately
#   behind explicit broker controls and extensive validation.
# ============================================================

APP_VERSION = "V8.0"
TWELVE_DATA_BASE = "https://api.twelvedata.com"


# ----------------------------- Configuration -----------------------------

DEFAULT_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
]

TIMEFRAMES = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1day": "1day",
}

HIGHER_TFS = ["4h", "1h"]
ENTRY_TFS = ["15m", "5m", "3m", "1m"]


def get_api_key() -> str:
    """Read Twelve Data API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("TWELVE_DATA_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("TWELVE_DATA_API_KEY", "")


# ----------------------------- Twelve Data -------------------------------

@st.cache_data(ttl=20, show_spinner=False)
def twelve_time_series(
    symbol: str,
    interval: str,
    outputsize: int = 300,
    api_key: str = "",
) -> pd.DataFrame:
    """Fetch OHLCV candles from Twelve Data."""
    if not api_key:
        raise ValueError("Twelve Data API key is not configured.")

    url = f"{TWELVE_DATA_BASE}/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()

    if "status" in payload and payload.get("status") == "error":
        raise RuntimeError(payload.get("message", "Twelve Data returned an error."))

    values = payload.get("values")
    if not values:
        raise RuntimeError("No candle data returned by Twelve Data.")

    df = pd.DataFrame(values)

    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required candle fields: {missing}")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = np.nan

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").drop_duplicates("datetime")
    df = df.reset_index(drop=True)

    return df


@st.cache_data(ttl=20, show_spinner=False)
def twelve_quote(symbol: str, api_key: str = "") -> Dict:
    """Fetch current quote information when supported by the account."""
    if not api_key:
        raise ValueError("Twelve Data API key is not configured.")

    url = f"{TWELVE_DATA_BASE}/quote"
    params = {"symbol": symbol, "apikey": api_key}

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == "error":
        raise RuntimeError(payload.get("message", "Twelve Data quote error."))

    return payload


# ----------------------------- Data Validation ----------------------------

def validate_market_data(df: pd.DataFrame) -> Dict:
    """Detect stale, incomplete, duplicate and abnormal candle data."""
    result = {
        "valid": True,
        "issues": [],
        "rows": len(df),
        "last_timestamp": None,
    }

    if df.empty:
        result["valid"] = False
        result["issues"].append("No market data.")
        return result

    result["last_timestamp"] = df["datetime"].iloc[-1]

    if df[["open", "high", "low", "close"]].isna().any().any():
        result["valid"] = False
        result["issues"].append("Missing OHLC values.")

    if (df["high"] < df["low"]).any():
        result["valid"] = False
        result["issues"].append("Invalid high/low relationship.")

    if (df["high"] < df["open"]).any() or (df["high"] < df["close"]).any():
        result["valid"] = False
        result["issues"].append("Price above candle high detected.")

    if (df["low"] > df["open"]).any() or (df["low"] > df["close"]).any():
        result["valid"] = False
        result["issues"].append("Price below candle low detected.")

    if df["datetime"].duplicated().any():
        result["valid"] = False
        result["issues"].append("Duplicate timestamps detected.")

    if len(df) >= 20:
        returns = df["close"].pct_change().dropna()
        if len(returns) > 5:
            median_abs = returns.abs().median()
            extreme = returns.abs() > max(0.03, median_abs * 25)
            if extreme.any():
                result["issues"].append("Extreme price movement detected.")

    return result


# ----------------------------- Technical Engine ---------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=df.index,
    )

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_val = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_val

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for p in [9, 20, 50, 100, 200]:
        out[f"ema_{p}"] = ema(out["close"], p)

    out["sma_50"] = sma(out["close"], 50)
    out["sma_200"] = sma(out["close"], 200)

    out["rsi"] = rsi(out["close"])
    out["atr"] = atr(out)
    out["adx"] = adx(out)

    macd_line, macd_signal, macd_hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    mid = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["bb_mid"] = mid
    out["bb_upper"] = mid + 2 * std
    out["bb_lower"] = mid - 2 * std

    out["momentum"] = out["close"].pct_change(5)
    out["returns"] = out["close"].pct_change()

    if out["volume"].notna().any():
        out["relative_volume"] = (
            out["volume"] / out["volume"].rolling(20).mean()
        )
    else:
        out["relative_volume"] = np.nan

    return out


# ----------------------------- Price Action --------------------------------

def detect_structure(df: pd.DataFrame, lookback: int = 30) -> Dict:
    if len(df) < lookback:
        return {
            "trend": "UNKNOWN",
            "structure": "INSUFFICIENT_DATA",
            "bos": False,
            "choch": False,
        }

    recent = df.tail(lookback)
    highs = recent["high"].rolling(3, center=True).max()
    lows = recent["low"].rolling(3, center=True).min()

    swing_highs = recent[recent["high"] == highs]["high"].dropna()
    swing_lows = recent[recent["low"] == lows]["low"].dropna()

    close = recent["close"].iloc[-1]

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs.iloc[-1] > swing_highs.iloc[-2]
        hl = swing_lows.iloc[-1] > swing_lows.iloc[-2]
        lh = swing_highs.iloc[-1] < swing_highs.iloc[-2]
        ll = swing_lows.iloc[-1] < swing_lows.iloc[-2]

        if hh and hl:
            trend = "BULLISH"
        elif lh and ll:
            trend = "BEARISH"
        else:
            trend = "RANGE"

        recent_high = swing_highs.iloc[-2]
        recent_low = swing_lows.iloc[-2]

        bos_up = close > recent_high
        bos_down = close < recent_low

        return {
            "trend": trend,
            "structure": "BULLISH" if bos_up else "BEARISH" if bos_down else trend,
            "bos": bool(bos_up or bos_down),
            "choch": False,
            "swing_high": float(swing_highs.iloc[-1]),
            "swing_low": float(swing_lows.iloc[-1]),
        }

    return {
        "trend": "RANGE",
        "structure": "RANGE",
        "bos": False,
        "choch": False,
    }


def detect_liquidity(df: pd.DataFrame) -> Dict:
    if len(df) < 30:
        return {"sweep": "NONE", "equal_high": False, "equal_low": False}

    recent = df.tail(30)
    prev_high = recent["high"].iloc[:-1].max()
    prev_low = recent["low"].iloc[:-1].min()
    last = recent.iloc[-1]

    sweep = "NONE"

    if last["high"] > prev_high and last["close"] < prev_high:
        sweep = "BUY_SIDE_SWEEP"
    elif last["low"] < prev_low and last["close"] > prev_low:
        sweep = "SELL_SIDE_SWEEP"

    tolerance = recent["close"].iloc[-1] * 0.0002
    equal_high = abs(recent["high"].tail(5).max() - recent["high"].tail(5).min()) <= tolerance
    equal_low = abs(recent["low"].tail(5).max() - recent["low"].tail(5).min()) <= tolerance

    return {
        "sweep": sweep,
        "equal_high": bool(equal_high),
        "equal_low": bool(equal_low),
        "previous_high": float(prev_high),
        "previous_low": float(prev_low),
    }


def detect_fvg(df: pd.DataFrame) -> Dict:
    if len(df) < 5:
        return {"type": "NONE"}

    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]

    if a["high"] < c["low"]:
        return {
            "type": "BULLISH_FVG",
            "lower": float(a["high"]),
            "upper": float(c["low"]),
        }

    if a["low"] > c["high"]:
        return {
            "type": "BEARISH_FVG",
            "lower": float(c["high"]),
            "upper": float(a["low"]),
        }

    return {"type": "NONE"}


# ----------------------------- Regime Engine ------------------------------

def detect_regime(df: pd.DataFrame) -> str:
    if len(df) < 50:
        return "UNKNOWN"

    last = df.iloc[-1]
    atr_pct = (last["atr"] / last["close"]) if pd.notna(last["atr"]) else 0
    adx_value = last["adx"] if pd.notna(last["adx"]) else 0

    if adx_value >= 25:
        if last["ema_20"] > last["ema_50"]:
            return "TRENDING_BULLISH"
        if last["ema_20"] < last["ema_50"]:
            return "TRENDING_BEARISH"

    if atr_pct > 0.004:
        return "HIGH_VOLATILITY"

    if adx_value < 18:
        return "RANGING"

    return "TRANSITION"


# ----------------------------- Sessions -----------------------------------

def market_session(ts: Optional[pd.Timestamp] = None) -> str:
    ts = ts or pd.Timestamp.now(tz="UTC")
    hour = ts.hour

    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 12:
        return "LONDON"
    if 12 <= hour < 16:
        return "LONDON/NEW YORK OVERLAP"
    if 16 <= hour < 21:
        return "NEW YORK"
    return "LATE / LOW LIQUIDITY"


# ----------------------------- Currency Strength --------------------------

def currency_strength_snapshot(
    pair_frames: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """Simple relative-strength proxy from available pair returns."""
    currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]
    scores = {c: [] for c in currencies}

    for pair, df in pair_frames.items():
        if len(df) < 6:
            continue

        try:
            base, quote = pair.split("/")
            ret = float(df["close"].pct_change(5).iloc[-1])
            if not np.isfinite(ret):
                continue

            scores[base].append(ret)
            scores[quote].append(-ret)
        except Exception:
            continue

    return {
        c: float(np.mean(v)) if v else 0.0
        for c, v in scores.items()
    }


# ----------------------------- Scoring Engine -----------------------------

def score_setup(
    df: pd.DataFrame,
    higher_frames: List[pd.DataFrame],
    pair: str,
) -> Dict:
    last = df.iloc[-1]
    structure = detect_structure(df)
    liquidity = detect_liquidity(df)
    fvg = detect_fvg(df)
    regime = detect_regime(df)

    bullish = 0
    bearish = 0
    reasons = []

    # Trend
    if last["ema_20"] > last["ema_50"]:
        bullish += 12
        reasons.append("EMA20 > EMA50")
    elif last["ema_20"] < last["ema_50"]:
        bearish += 12
        reasons.append("EMA20 < EMA50")

    # Higher timeframe confirmation
    ht_bull = 0
    ht_bear = 0
    for htf in higher_frames:
        if len(htf) < 50:
            continue
        h = htf.iloc[-1]
        if h["ema_20"] > h["ema_50"]:
            ht_bull += 1
        elif h["ema_20"] < h["ema_50"]:
            ht_bear += 1

    if ht_bull > ht_bear:
        bullish += 18
        reasons.append("Higher-timeframe bullish alignment")
    elif ht_bear > ht_bull:
        bearish += 18
        reasons.append("Higher-timeframe bearish alignment")
    else:
        reasons.append("Higher-timeframe conflict")

    # RSI
    if 52 <= last["rsi"] <= 70:
        bullish += 8
    elif 30 <= last["rsi"] <= 48:
        bearish += 8

    # MACD
    if last["macd_hist"] > 0:
        bullish += 8
    elif last["macd_hist"] < 0:
        bearish += 8

    # ADX / trend strength
    if pd.notna(last["adx"]) and last["adx"] >= 25:
        if bullish > bearish:
            bullish += 8
        elif bearish > bullish:
            bearish += 8

    # Structure
    if structure["structure"] == "BULLISH":
        bullish += 12
        reasons.append("Bullish market structure")
    elif structure["structure"] == "BEARISH":
        bearish += 12
        reasons.append("Bearish market structure")

    # Liquidity sweep
    if liquidity["sweep"] == "SELL_SIDE_SWEEP":
        bullish += 10
        reasons.append("Sell-side liquidity sweep")
    elif liquidity["sweep"] == "BUY_SIDE_SWEEP":
        bearish += 10
        reasons.append("Buy-side liquidity sweep")

    # FVG
    if fvg["type"] == "BULLISH_FVG":
        bullish += 6
        reasons.append("Bullish FVG detected")
    elif fvg["type"] == "BEARISH_FVG":
        bearish += 6
        reasons.append("Bearish FVG detected")

    # Momentum
    momentum = last["momentum"]
    if pd.notna(momentum):
        if momentum > 0:
            bullish += 6
        elif momentum < 0:
            bearish += 6

    # Regime quality
    if regime in ("TRENDING_BULLISH", "TRENDING_BEARISH"):
        if regime.endswith("BULLISH"):
            bullish += 5
        else:
            bearish += 5
    elif regime == "RANGING":
        reasons.append("Ranging regime: reduced confidence")

    total = max(bullish + bearish, 1)
    raw_direction = "BUY" if bullish > bearish else "SELL" if bearish > bullish else "WAIT"

    directional_strength = abs(bullish - bearish) / total
    confidence = 50 + directional_strength * 45

    if raw_direction == "WAIT":
        confidence = min(confidence, 49)

    # Hard quality filters
    no_trade_reasons = []

    if len(df) < 200:
        no_trade_reasons.append("Insufficient history for long-term EMA context")

    if pd.isna(last["atr"]) or last["atr"] <= 0:
        no_trade_reasons.append("ATR unavailable")

    if regime == "UNKNOWN":
        no_trade_reasons.append("Unknown market regime")

    if ht_bull and ht_bear:
        no_trade_reasons.append("Higher-timeframe conflict")

    if raw_direction == "WAIT":
        no_trade_reasons.append("No directional edge")

    if confidence < 60:
        no_trade_reasons.append("Confidence below minimum threshold")

    signal = raw_direction if not no_trade_reasons else "NO TRADE"

    return {
        "pair": pair,
        "signal": signal,
        "direction": raw_direction,
        "confidence": round(float(confidence), 1),
        "bullish_score": bullish,
        "bearish_score": bearish,
        "trend": structure["trend"],
        "structure": structure["structure"],
        "regime": regime,
        "liquidity": liquidity["sweep"],
        "fvg": fvg["type"],
        "rsi": float(last["rsi"]) if pd.notna(last["rsi"]) else None,
        "adx": float(last["adx"]) if pd.notna(last["adx"]) else None,
        "atr": float(last["atr"]) if pd.notna(last["atr"]) else None,
        "session": market_session(df["datetime"].iloc[-1]),
        "reasons": reasons,
        "no_trade_reasons": no_trade_reasons,
    }


# ----------------------------- Risk Engine --------------------------------

def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("/JPY") else 0.0001


def calculate_forex_plan(
    pair: str,
    direction: str,
    entry: float,
    atr_value: float,
    account_balance: float,
    risk_pct: float,
    rr: float,
) -> Dict:
    p = pip_size(pair)

    stop_distance = max(atr_value * 1.5, p * 5)

    if direction == "BUY":
        sl = entry - stop_distance
        tp = entry + stop_distance * rr
    else:
        sl = entry + stop_distance
        tp = entry - stop_distance * rr

    risk_amount = account_balance * risk_pct / 100
    pip_distance = stop_distance / p

    # This is a simplified position-size estimate.
    # Broker-specific contract size/tick-value rules must be supplied
    # before live execution.
    position_units = risk_amount / max(stop_distance, 1e-12)

    return {
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "risk_amount": risk_amount,
        "pip_distance": pip_distance,
        "estimated_units": position_units,
        "rr": rr,
    }


def binary_plan(
    confidence: float,
    direction: str,
    regime: str,
    atr_value: float,
    price: float,
) -> Dict:
    """Heuristic binary-options research signal.

    This does NOT model a broker's actual payout/settlement mechanics.
    It should be replaced with a calibrated probability model.
    """
    if direction not in ("BUY", "SELL"):
        return {
            "signal": "NO TRADE",
            "expiry": "N/A",
            "probability": 50.0,
        }

    if confidence < 70:
        return {
            "signal": "NO TRADE",
            "expiry": "N/A",
            "probability": confidence,
        }

    if regime.startswith("TRENDING"):
        expiry = "5 min"
    elif regime == "RANGING":
        expiry = "3 min"
    elif regime == "HIGH_VOLATILITY":
        expiry = "1 min"
    else:
        expiry = "5 min"

    return {
        "signal": "CALL" if direction == "BUY" else "PUT",
        "expiry": expiry,
        "probability": round(confidence, 1),
        "reference_price": price,
        "atr": atr_value,
    }


# ----------------------------- Backtesting --------------------------------

def simple_backtest(df: pd.DataFrame, threshold: float = 65) -> Dict:
    """Simple research backtest using the same scoring philosophy.

    This is intentionally conservative and should not be treated as a
    complete execution simulator. It uses next-bar direction as a basic
    outcome proxy and avoids claiming broker-realistic fills.
    """
    if len(df) < 250:
        return {"error": "At least 250 candles are recommended."}

    work = add_indicators(df.copy())
    outcomes = []

    for i in range(210, len(work) - 1):
        window = work.iloc[: i + 1]
        result = score_setup(window, [window], "BACKTEST")

        if result["signal"] not in ("BUY", "SELL"):
            continue

        if result["confidence"] < threshold:
            continue

        next_close = work["close"].iloc[i + 1]
        current_close = work["close"].iloc[i]

        win = (
            next_close > current_close
            if result["signal"] == "BUY"
            else next_close < current_close
        )

        outcomes.append(
            {
                "timestamp": work["datetime"].iloc[i],
                "signal": result["signal"],
                "confidence": result["confidence"],
                "win": bool(win),
            }
        )

    if not outcomes:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
        }

    bt = pd.DataFrame(outcomes)
    wins = int(bt["win"].sum())
    trades = len(bt)
    losses = trades - wins

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / trades * 100, 2),
        "signals": bt,
    }


# ----------------------------- Charts -------------------------------------

def candlestick_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=pair,
        )
    )

    if "ema_20" in df:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["ema_20"],
                name="EMA 20",
                mode="lines",
            )
        )

    if "ema_50" in df:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["ema_50"],
                name="EMA 50",
                mode="lines",
            )
        )

    fig.update_layout(
        height=550,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=35, b=10),
        title=f"{pair} — V8 Market Structure",
    )

    return fig


# ----------------------------- Streamlit UI -------------------------------

st.set_page_config(
    page_title="Forex AI Pro V8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Forex AI Pro V8")
st.caption(
    "Multi-layer market analysis • Twelve Data • Forex + Binary research engines"
)

api_key = get_api_key()

if not api_key:
    st.warning(
        "Twelve Data API key not found. Add TWELVE_DATA_API_KEY to "
        "Streamlit Secrets or an environment variable."
    )

with st.sidebar:
    st.header("⚙️ V8 Controls")

    pair = st.selectbox("Forex pair", DEFAULT_PAIRS)
    entry_tf_label = st.selectbox(
        "Entry timeframe",
        ["1m", "3m", "5m", "15m", "30m"],
        index=2,
    )

    outputsize = st.slider(
        "Historical candles",
        min_value=250,
        max_value=5000,
        value=500,
        step=250,
    )

    mode = st.radio(
        "Trading engine",
        ["Forex", "Binary Options", "Research"],
    )

    account_balance = st.number_input(
        "Demo account balance",
        min_value=100.0,
        value=10000.0,
        step=100.0,
    )

    risk_pct = st.slider(
        "Risk per Forex trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

    rr = st.slider(
        "Forex target R:R",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.25,
    )

    confidence_threshold = st.slider(
        "Minimum signal confidence",
        min_value=50,
        max_value=90,
        value=70,
        step=1,
    )

    st.divider()
    st.caption(f"Application version: {APP_VERSION}")
    st.caption("Paper/research mode by default.")


if api_key:
    try:
        raw_entry = twelve_time_series(
            pair,
            TIMEFRAMES[entry_tf_label],
            outputsize,
            api_key,
        )

        entry_validation = validate_market_data(raw_entry)
        entry_df = add_indicators(raw_entry)

        higher_frames = []
        higher_status = {}

        for tf in HIGHER_TFS:
            try:
                htf_raw = twelve_time_series(
                    pair,
                    TIMEFRAMES[tf],
                    min(outputsize, 500),
                    api_key,
                )
                validation = validate_market_data(htf_raw)
                higher_status[tf] = validation
                higher_frames.append(add_indicators(htf_raw))
            except Exception as exc:
                higher_status[tf] = {
                    "valid": False,
                    "issues": [str(exc)],
                }

        if not entry_validation["valid"]:
            st.error("🚫 Market data failed validation. Signals are disabled.")
            st.json(entry_validation)
            st.stop()

        result = score_setup(
            entry_df,
            higher_frames,
            pair,
        )

        last_price = float(entry_df["close"].iloc[-1])

        # ---------------- Dashboard KPIs ----------------

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Signal", result["signal"])
        c2.metric("Confidence", f'{result["confidence"]:.1f}%')
        c3.metric("Regime", result["regime"])
        c4.metric("Trend", result["trend"])
        c5.metric("Session", result["session"])

        st.plotly_chart(
            candlestick_chart(entry_df.tail(250), pair),
            use_container_width=True,
        )

        tab_signal, tab_analysis, tab_risk, tab_binary, tab_backtest, tab_health = st.tabs(
            [
                "🎯 Signal",
                "🧠 Analysis",
                "🛡️ Risk",
                "⏱️ Binary",
                "🧪 Backtest",
                "🩺 System Health",
            ]
        )

        # ---------------- Signal ----------------

        with tab_signal:
            st.subheader("V8 Decision")

            if result["signal"] == "NO TRADE":
                st.error("NO TRADE — quality filters rejected this setup.")
                for reason in result["no_trade_reasons"]:
                    st.write(f"• {reason}")
            else:
                if result["signal"] == "BUY":
                    st.success("🟢 BUY")
                else:
                    st.error("🔴 SELL")

                st.write(
                    f"**Current price:** `{last_price:.6f}`  \n"
                    f"**Confidence:** `{result['confidence']:.1f}%`"
                )

                st.markdown("### Why?")
                for reason in result["reasons"]:
                    st.write(f"• {reason}")

                st.markdown("### Signal status")
                st.info(
                    "This is a research signal. It should not be interpreted "
                    "as a guaranteed profitable trade."
                )

        # ---------------- Analysis ----------------

        with tab_analysis:
            st.subheader("Market Intelligence")

            analysis_cols = st.columns(4)
            analysis_cols[0].metric("Structure", result["structure"])
            analysis_cols[1].metric("Liquidity", result["liquidity"])
            analysis_cols[2].metric("FVG", result["fvg"])
            analysis_cols[3].metric(
                "RSI",
                f"{result['rsi']:.1f}" if result["rsi"] is not None else "N/A",
            )

            st.markdown("#### Multi-timeframe context")

            htf_rows = []
            for tf, status in higher_status.items():
                if status.get("valid"):
                    htf_rows.append(
                        {
                            "Timeframe": tf,
                            "Status": "VALID",
                            "Last candle": status.get("last_timestamp"),
                        }
                    )
                else:
                    htf_rows.append(
                        {
                            "Timeframe": tf,
                            "Status": "INVALID",
                            "Last candle": status.get("last_timestamp"),
                        }
                    )

            st.dataframe(
                pd.DataFrame(htf_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Indicator snapshot")

            last = entry_df.iloc[-1]
            indicator_table = pd.DataFrame(
                {
                    "Indicator": [
                        "EMA 9", "EMA 20", "EMA 50", "EMA 100", "EMA 200",
                        "RSI", "MACD", "MACD histogram", "ADX", "ATR",
                    ],
                    "Value": [
                        last["ema_9"],
                        last["ema_20"],
                        last["ema_50"],
                        last["ema_100"],
                        last["ema_200"],
                        last["rsi"],
                        last["macd"],
                        last["macd_hist"],
                        last["adx"],
                        last["atr"],
                    ],
                }
            )
            st.dataframe(indicator_table, use_container_width=True, hide_index=True)

        # ---------------- Risk ----------------

        with tab_risk:
            st.subheader("Forex Risk Engine")

            if result["direction"] in ("BUY", "SELL") and result["atr"]:
                plan = calculate_forex_plan(
                    pair,
                    result["direction"],
                    last_price,
                    result["atr"],
                    account_balance,
                    risk_pct,
                    rr,
                )

                rc = st.columns(5)
                rc[0].metric("Entry", f'{plan["entry"]:.6f}')
                rc[1].metric("Stop Loss", f'{plan["stop_loss"]:.6f}')
                rc[2].metric("Take Profit", f'{plan["take_profit"]:.6f}')
                rc[3].metric("Risk", f'${plan["risk_amount"]:.2f}')
                rc[4].metric("R:R", f'{plan["rr"]:.2f}')

                st.warning(
                    "Position sizing shown here is a simplified research estimate. "
                    "Broker contract size, tick value, leverage, spread and "
                    "execution rules must be incorporated before live trading."
                )
            else:
                st.info("Risk plan unavailable because there is no valid directional setup.")

        # ---------------- Binary ----------------

        with tab_binary:
            st.subheader("Binary Options Engine")

            binary = binary_plan(
                result["confidence"],
                result["direction"],
                result["regime"],
                result["atr"] or 0,
                last_price,
            )

            bc = st.columns(4)
            bc[0].metric("Decision", binary["signal"])
            bc[1].metric("Estimated probability", f'{binary["probability"]:.1f}%')
            bc[2].metric("Suggested expiry", binary["expiry"])
            bc[3].metric("Reference price", f'{last_price:.6f}')

            st.warning(
                "Binary options require a calibrated expiry/probability model and "
                "broker-specific settlement assumptions. This V8 research engine "
                "does not claim that a confidence score equals a true win probability."
            )

        # ---------------- Backtest ----------------

        with tab_backtest:
            st.subheader("Historical Research Backtest")

            if st.button("▶ Run V8 Backtest", type="primary"):
                with st.spinner("Running research backtest..."):
                    bt = simple_backtest(
                        raw_entry,
                        threshold=confidence_threshold,
                    )

                if "error" in bt:
                    st.error(bt["error"])
                else:
                    bc = st.columns(4)
                    bc[0].metric("Trades", bt["trades"])
                    bc[1].metric("Wins", bt["wins"])
                    bc[2].metric("Losses", bt["losses"])
                    bc[3].metric("Win rate", f'{bt["win_rate"]:.2f}%')

                    if bt.get("trades", 0) > 0:
                        st.dataframe(
                            bt["signals"].tail(100),
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.info(
                        "This backtest is intentionally a basic research layer. "
                        "It does not simulate broker fills, spread, slippage, "
                        "latency, commissions or binary payout mechanics."
                    )

        # ---------------- Health ----------------

        with tab_health:
            st.subheader("Self-Diagnostic")

            health = {
                "Twelve Data API configured": bool(api_key),
                "Entry data valid": entry_validation["valid"],
                "Entry candle count": len(entry_df),
                "Latest candle": str(entry_df["datetime"].iloc[-1]),
                "Latest price": last_price,
                "Data issues": entry_validation["issues"],
            }

            st.json(health)

            if entry_validation["issues"]:
                st.warning("Data-quality warnings are present.")

            st.markdown("### Higher-timeframe health")
            for tf, status in higher_status.items():
                if status.get("valid"):
                    st.success(f"{tf}: healthy")
                else:
                    st.error(f"{tf}: unavailable / invalid")
                    for issue in status.get("issues", []):
                        st.write(f"• {issue}")

    except Exception as exc:
        st.error(f"V8 market-data/system error: {exc}")
        st.info(
            "Check your Twelve Data API key, symbol format, plan limits, "
            "network connection and requested timeframe."
        )

else:
    st.markdown(
        """
        ## 🚀 V8 initialization

        Add your Twelve Data API key to Streamlit Secrets:

        `TWELVE_DATA_API_KEY = "your_key_here"`

        Then reload the application.

        **Do not put your real API key directly into `app.py` or commit it to GitHub.**
        """
    )

st.divider()
st.caption(
    "Forex AI Pro V8 • Research-first architecture • "
    "No Martingale • No guaranteed-profit claims"
)
