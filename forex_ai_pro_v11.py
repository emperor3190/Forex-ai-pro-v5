
import os, json, sqlite3, math, warnings
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss

warnings.filterwarnings("ignore")

# ============================================================
# FOREX AI PRO V11.0
# Multi-engine market analysis + event-aligned walk-forward test
# Research / paper trading only. No guaranteed performance.
# ============================================================

APP_VERSION = "V11.0"
BASE = "https://api.twelvedata.com"
DATA_DIR = ".v11_data"
DB = os.path.join(DATA_DIR, "forex_ai_pro_v11.sqlite3")

PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","USD/CHF",
    "AUD/USD","USD/CAD","NZD/USD","EUR/GBP"
]

TF = {
    "1m":"1min","3m":"3min","5m":"5min","15m":"15min",
    "30m":"30min","1h":"1h","4h":"4h","1day":"1day"
}

# ----------------------------------------------------------------
# Utility
# ----------------------------------------------------------------

def now():
    return pd.Timestamp.now(tz="UTC")

def api_key():
    try:
        k = st.secrets.get("TWELVE_DATA_API_KEY", "")
    except Exception:
        k = ""
    return str(k or os.getenv("TWELVE_DATA_API_KEY", ""))

def pip_size(pair):
    return 0.01 if pair.endswith("/JPY") else 0.0001

def pip_value_distance(pair, price_distance):
    return price_distance / pip_size(pair)

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def db_init():
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(DB) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS signals(
            id INTEGER PRIMARY KEY,
            created_at TEXT, pair TEXT, engine TEXT, timeframe TEXT,
            direction TEXT, probability REAL, threshold REAL,
            entry REAL, strike REAL, expiry_minutes REAL,
            stop_loss REAL, take_profit REAL, regime TEXT,
            status TEXT, explanation TEXT, payload TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS backtests(
            id INTEGER PRIMARY KEY,
            created_at TEXT, pair TEXT, timeframe TEXT, engine TEXT,
            trades INTEGER, wins INTEGER, losses INTEGER, win_rate REAL,
            expectancy REAL, profit_factor REAL, max_drawdown REAL,
            brier REAL, payload TEXT
        )""")

def journal(payload):
    db_init()
    with sqlite3.connect(DB) as c:
        c.execute("""
        INSERT INTO signals(
            created_at,pair,engine,timeframe,direction,probability,threshold,
            entry,strike,expiry_minutes,stop_loss,take_profit,regime,
            status,explanation,payload
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            now().isoformat(),
            payload.get("pair",""),
            payload.get("engine","V11"),
            payload.get("timeframe",""),
            payload.get("direction","NO TRADE"),
            payload.get("probability"),
            payload.get("threshold"),
            payload.get("entry"),
            payload.get("strike"),
            payload.get("expiry_minutes"),
            payload.get("stop_loss"),
            payload.get("take_profit"),
            payload.get("regime"),
            payload.get("status"),
            payload.get("explanation",""),
            json.dumps(payload, default=str)
        ))

def load_journal(n=250):
    db_init()
    with sqlite3.connect(DB) as c:
        return pd.read_sql_query(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
            c, params=(n,)
        )

# ----------------------------------------------------------------
# Data
# ----------------------------------------------------------------

@st.cache_data(ttl=20, show_spinner=False)
def candles(symbol, interval, size, apikey):
    if not apikey:
        raise ValueError("TWELVE_DATA_API_KEY is not configured.")

    r = requests.get(
        f"{BASE}/time_series",
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": size,
            "apikey": apikey,
            "format": "JSON"
        },
        timeout=25
    )
    r.raise_for_status()
    p = r.json()

    if p.get("status") == "error":
        raise RuntimeError(p.get("message", "Twelve Data error."))

    d = pd.DataFrame(p.get("values", []))
    if d.empty:
        raise RuntimeError("Twelve Data returned no candles.")

    for c in ["open","high","low","close","volume"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "volume" not in d:
        d["volume"] = np.nan

    d["datetime"] = pd.to_datetime(
        d["datetime"], utc=True, errors="coerce"
    )

    d = (
        d.dropna(subset=["datetime","open","high","low","close"])
         .sort_values("datetime")
         .drop_duplicates("datetime")
         .reset_index(drop=True)
    )
    return d

def health(d, interval):
    out = {
        "valid": True, "fresh": True, "gaps_ok": True,
        "issues": []
    }
    if d.empty:
        return {
            "valid":False, "fresh":False, "gaps_ok":False,
            "issues":["No data."]
        }

    ohlc = ["open","high","low","close"]
    if d[ohlc].isna().any().any():
        out["valid"] = False
        out["issues"].append("Missing OHLC values.")

    if (d.high < d.low).any():
        out["valid"] = False
        out["issues"].append("High below low.")

    if (d.high < d[["open","close"]].max(axis=1)).any():
        out["valid"] = False
        out["issues"].append("OHLC inconsistency: high.")

    if (d.low > d[["open","close"]].min(axis=1)).any():
        out["valid"] = False
        out["issues"].append("OHLC inconsistency: low.")

    mins = {
        "1min":1,"3min":3,"5min":5,"15min":15,
        "30min":30,"1h":60,"4h":240,"1day":1440
    }.get(interval, 5)

    age = max(
        0,
        (now() - d.datetime.iloc[-1]).total_seconds()/60
    )
    out["age"] = age

    if age > mins*2 + 3:
        out["fresh"] = False
        out["issues"].append(f"Data stale ({age:.1f} min).")

    diff = d.datetime.diff().dt.total_seconds().div(60).dropna()
    if not diff.empty and (diff > mins*2.5).any():
        out["gaps_ok"] = False
        out["issues"].append("Large candle gap detected.")

    return out

# ----------------------------------------------------------------
# Indicators - causal only
# ----------------------------------------------------------------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100/(1+rs)

def true_range(d):
    pc = d.close.shift()
    return pd.concat([
        d.high-d.low,
        (d.high-pc).abs(),
        (d.low-pc).abs()
    ], axis=1).max(axis=1)

def atr(d, n=14):
    return true_range(d).ewm(
        alpha=1/n, min_periods=n, adjust=False
    ).mean()

def adx(d, n=14):
    up = d.high.diff()
    dn = -d.low.diff()

    plus = pd.Series(
        np.where((up > dn) & (up > 0), up, 0.0),
        index=d.index
    )
    minus = pd.Series(
        np.where((dn > up) & (dn > 0), dn, 0.0),
        index=d.index
    )

    tr = true_range(d)
    av = tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()

    pdi = 100 * plus.ewm(
        alpha=1/n, min_periods=n, adjust=False
    ).mean() / av.replace(0, np.nan)

    mdi = 100 * minus.ewm(
        alpha=1/n, min_periods=n, adjust=False
    ).mean() / av.replace(0, np.nan)

    dx = 100 * (pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)

    return dx.ewm(
        alpha=1/n, min_periods=n, adjust=False
    ).mean()

def macd_parts(close):
    m = ema(close,12)-ema(close,26)
    sig = ema(m,9)
    return m, sig, m-sig

def zscore(s, n=50):
    mu = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (s-mu)/sd.replace(0,np.nan)

# ----------------------------------------------------------------
# Feature factory
# Everything here is causal: no centered/future pivots.
# ----------------------------------------------------------------

FEATURES = [
    # returns / price action
    "ret1","ret3","ret5","ret10","ret20",
    "range_pct","body_pct","uwick_pct","lwick_pct",
    "cloc","body_direction","range_z","ret_z",

    # trend
    "e9_dist","e20_dist","e50_dist","e100_dist","e200_dist",
    "e20_50","e50_100","e100_200",
    "ema20_slope","ema50_slope","ema100_slope",
    "trend_strength","adx","adx_slope",

    # momentum
    "rsi","rsi_slope","macd_hist","macd_slope",
    "roc5","roc10","roc20","momentum_accel",
    "consistency","impulse_score",

    # volatility
    "atr_pct","atr_z","vol20","vol_z",
    "bbw","bbw_z","range_expansion",
    "volatility_ratio",

    # structure
    "structure_bias","bos_up","bos_down",
    "structure_distance_high","structure_distance_low",
    "swing_pressure",

    # breakout / mean reversion
    "breakout_up","breakout_down","bbpos",
    "mean_distance","mean_z","extension_score",

    # liquidity / S&R
    "dist_high20","dist_low20","dist_high50","dist_low50",
    "liquidity_sweep_up","liquidity_sweep_down",
    "support_pressure","resistance_pressure",

    # session / calendar
    "hsin","hcos","dsin","dcos",
    "asia","london","overlap","ny",

    # volume where provider supplies it
    "volume_z","volume_trend",

    # directional interaction
    "trend_momentum","trend_volatility",
    "structure_momentum","breakout_quality"
]

def build_features(d):
    x = d.copy()

    # Price / returns
    x["ret1"] = x.close.pct_change()
    x["ret3"] = x.close.pct_change(3)
    x["ret5"] = x.close.pct_change(5)
    x["ret10"] = x.close.pct_change(10)
    x["ret20"] = x.close.pct_change(20)

    x["range_pct"] = (x.high-x.low)/x.close
    x["body_pct"] = (x.close-x.open).abs()/x.close
    x["uwick_pct"] = (
        x.high-x[["open","close"]].max(axis=1)
    )/x.close
    x["lwick_pct"] = (
        x[["open","close"]].min(axis=1)-x.low
    )/x.close
    x["cloc"] = (
        (x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    )
    x["body_direction"] = np.sign(x.close-x.open)

    # EMAs
    for n in [9,20,50,100,200]:
        x[f"ema{n}"] = ema(x.close,n)

    x["e9_dist"] = (x.close-x.ema9)/x.close
    x["e20_dist"] = (x.close-x.ema20)/x.close
    x["e50_dist"] = (x.close-x.ema50)/x.close
    x["e100_dist"] = (x.close-x.ema100)/x.close
    x["e200_dist"] = (x.close-x.ema200)/x.close

    x["e20_50"] = (x.ema20-x.ema50)/x.close
    x["e50_100"] = (x.ema50-x.ema100)/x.close
    x["e100_200"] = (x.ema100-x.ema200)/x.close

    x["ema20_slope"] = x.ema20.pct_change(5)
    x["ema50_slope"] = x.ema50.pct_change(5)
    x["ema100_slope"] = x.ema100.pct_change(5)

    # Trend
    x["adx"] = adx(x)
    x["adx_slope"] = x.adx.diff(3)
    x["trend_strength"] = (
        x.e20_50.abs() * x.adx
    )

    # Momentum engine
    x["rsi"] = rsi(x.close)
    x["rsi_slope"] = x.rsi.diff(3)

    macd, macdsig, hist = macd_parts(x.close)
    x["macd"] = macd
    x["macd_signal"] = macdsig
    x["macd_hist"] = hist
    x["macd_slope"] = hist.diff(3)

    x["roc5"] = x.close.pct_change(5)
    x["roc10"] = x.close.pct_change(10)
    x["roc20"] = x.close.pct_change(20)
    x["momentum_accel"] = x.roc5-x.roc5.shift(3)

    x["consistency"] = x.ret1.rolling(10).apply(
        lambda z: abs(np.sign(z).sum())/len(z),
        raw=True
    )

    x["impulse_score"] = (
        x.body_pct /
        x.range_pct.replace(0,np.nan)
    ) * x.body_direction

    # Volatility engine
    x["atr"] = atr(x)
    x["atr_pct"] = x.atr/x.close
    x["atr_z"] = zscore(x.atr_pct,50)
    x["vol20"] = x.ret1.rolling(20).std()
    x["vol_z"] = zscore(x.vol20,50)

    mid = x.close.rolling(20).mean()
    sd = x.close.rolling(20).std()
    x["bbu"] = mid+2*sd
    x["bbl"] = mid-2*sd
    x["bbm"] = mid

    x["bbw"] = (x.bbu-x.bbl)/x.bbm.replace(0,np.nan)
    x["bbw_z"] = zscore(x.bbw,50)
    x["bbpos"] = (
        (x.close-x.bbl)/(x.bbu-x.bbl).replace(0,np.nan)
    )

    x["range_z"] = zscore(x.range_pct,50)
    x["ret_z"] = zscore(x.ret1,20)
    x["range_expansion"] = (
        x.range_pct /
        x.range_pct.rolling(20).mean().replace(0,np.nan)
    )
    x["volatility_ratio"] = (
        x.vol20 /
        x.vol20.rolling(50).mean().replace(0,np.nan)
    )

    # Causal market structure:
    # compare current close against PREVIOUS rolling highs/lows.
    prev_high20 = x.high.rolling(20).max().shift(1)
    prev_low20 = x.low.rolling(20).min().shift(1)
    prev_high50 = x.high.rolling(50).max().shift(1)
    prev_low50 = x.low.rolling(50).min().shift(1)

    x["bos_up"] = (x.close > prev_high20).astype(float)
    x["bos_down"] = (x.close < prev_low20).astype(float)

    x["structure_bias"] = (
        (x.close > x.close.rolling(10).mean()).astype(int)
        - (x.close < x.close.rolling(10).mean()).astype(int)
    )

    x["structure_distance_high"] = (
        prev_high20-x.close
    )/x.close

    x["structure_distance_low"] = (
        x.close-prev_low20
    )/x.close

    x["swing_pressure"] = (
        (x.close-prev_low20) /
        (prev_high20-prev_low20).replace(0,np.nan)
    )

    # Breakout engine
    x["breakout_up"] = (
        (x.close > prev_high20) &
        (x.range_expansion > 1.1)
    ).astype(float)

    x["breakout_down"] = (
        (x.close < prev_low20) &
        (x.range_expansion > 1.1)
    ).astype(float)

    # Mean reversion / extension
    x["mean_distance"] = (x.close-mid)/x.close
    x["mean_z"] = zscore(x.mean_distance,50)
    x["extension_score"] = (
        x.e20_dist.abs() /
        x.atr_pct.replace(0,np.nan)
    )

    # Liquidity / support-resistance
    x["dist_high20"] = (prev_high20-x.close)/x.close
    x["dist_low20"] = (x.close-prev_low20)/x.close
    x["dist_high50"] = (prev_high50-x.close)/x.close
    x["dist_low50"] = (x.close-prev_low50)/x.close

    # A causal sweep proxy: previous high/low breached intrabar,
    # but close returned back inside the prior range.
    x["liquidity_sweep_up"] = (
        (x.high > prev_high20) &
        (x.close < prev_high20)
    ).astype(float)

    x["liquidity_sweep_down"] = (
        (x.low < prev_low20) &
        (x.close > prev_low20)
    ).astype(float)

    x["support_pressure"] = (
        1/(1+x.dist_low20.abs()/x.atr_pct.replace(0,np.nan))
    )
    x["resistance_pressure"] = (
        1/(1+x.dist_high20.abs()/x.atr_pct.replace(0,np.nan))
    )

    # Session engine
    h = x.datetime.dt.hour
    w = x.datetime.dt.dayofweek

    x["hsin"] = np.sin(2*np.pi*h/24)
    x["hcos"] = np.cos(2*np.pi*h/24)
    x["dsin"] = np.sin(2*np.pi*w/7)
    x["dcos"] = np.cos(2*np.pi*w/7)

    x["asia"] = ((h>=0)&(h<7)).astype(int)
    x["london"] = ((h>=7)&(h<12)).astype(int)
    x["overlap"] = ((h>=12)&(h<16)).astype(int)
    x["ny"] = ((h>=16)&(h<21)).astype(int)

    # Volume engine - gracefully degrades when FX feed has no volume.
    vol = pd.to_numeric(x.volume, errors="coerce")
    x["volume_z"] = zscore(vol,50)
    x["volume_trend"] = vol.pct_change(5)

    # Engine interactions
    x["trend_momentum"] = x.e20_50 * x.rsi_slope
    x["trend_volatility"] = x.e20_50 * x.atr_z
    x["structure_momentum"] = (
        (x.bos_up-x.bos_down) *
        x.momentum_accel
    )
    x["breakout_quality"] = (
        (x.breakout_up-x.breakout_down) *
        x.range_expansion *
        x.consistency
    )

    return x.replace([np.inf,-np.inf],np.nan)

# ----------------------------------------------------------------
# Independent analysis engines
# ----------------------------------------------------------------

def trend_engine(x):
    z = x.iloc[-1]
    bull = bear = 0
    reasons = []

    if z.ema20 > z.ema50:
        bull += 25
        reasons.append("EMA20 above EMA50")
    elif z.ema20 < z.ema50:
        bear += 25
        reasons.append("EMA20 below EMA50")

    if z.ema50 > z.ema100:
        bull += 15
    elif z.ema50 < z.ema100:
        bear += 15

    if z.ema100 > z.ema200:
        bull += 10
    elif z.ema100 < z.ema200:
        bear += 10

    if safe_float(z.adx,0) >= 25:
        if bull > bear:
            bull += 10
        elif bear > bull:
            bear += 10

    direction = "BULLISH" if bull>bear else "BEARISH" if bear>bull else "NEUTRAL"
    return {
        "name":"Trend",
        "direction":direction,
        "score":max(bull,bear),
        "bull":bull,"bear":bear,
        "reasons":reasons
    }

def momentum_engine(x):
    z = x.iloc[-1]
    bull = bear = 0
    reasons = []

    if safe_float(z.rsi,50) > 55:
        bull += 18
        reasons.append("RSI bullish")
    elif safe_float(z.rsi,50) < 45:
        bear += 18
        reasons.append("RSI bearish")

    if safe_float(z.rsi_slope,0) > 0:
        bull += 8
    elif safe_float(z.rsi_slope,0) < 0:
        bear += 8

    if safe_float(z.macd_hist,0) > 0:
        bull += 15
        reasons.append("MACD histogram positive")
    elif safe_float(z.macd_hist,0) < 0:
        bear += 15
        reasons.append("MACD histogram negative")

    if safe_float(z.macd_slope,0) > 0:
        bull += 10
    elif safe_float(z.macd_slope,0) < 0:
        bear += 10

    if safe_float(z.roc10,0) > 0:
        bull += 10
    elif safe_float(z.roc10,0) < 0:
        bear += 10

    if safe_float(z.momentum_accel,0) > 0:
        bull += 8
    elif safe_float(z.momentum_accel,0) < 0:
        bear += 8

    if safe_float(z.impulse_score,0) > .45:
        bull += 8
    elif safe_float(z.impulse_score,0) < -.45:
        bear += 8

    direction = "BULLISH" if bull>bear else "BEARISH" if bear>bull else "NEUTRAL"
    strength = min(100, max(bull,bear))
    return {
        "name":"Momentum",
        "direction":direction,
        "score":strength,
        "bull":bull,"bear":bear,
        "reasons":reasons
    }

def volatility_engine(x):
    z = x.iloc[-1]
    atrz = safe_float(z.atr_z,0)
    bwz = safe_float(z.bbw_z,0)
    rex = safe_float(z.range_expansion,1)

    if atrz >= 1 or rex >= 1.4:
        regime = "EXPANSION"
    elif atrz <= -1 or bwz <= -1:
        regime = "COMPRESSION"
    else:
        regime = "NORMAL"

    if regime == "EXPANSION":
        score = 85
    elif regime == "COMPRESSION":
        score = 55
    else:
        score = 70

    return {
        "name":"Volatility",
        "regime":regime,
        "score":score,
        "atr_pct":safe_float(z.atr_pct),
        "atr_z":atrz,
        "range_expansion":rex,
        "bbw_z":bwz
    }

def structure_engine(x):
    z = x.iloc[-1]
    bull = bear = 0
    reasons = []

    if safe_float(z.bos_up,0) > 0:
        bull += 40
        reasons.append("Causal bullish BOS")
    if safe_float(z.bos_down,0) > 0:
        bear += 40
        reasons.append("Causal bearish BOS")

    if safe_float(z.structure_bias,0) > 0:
        bull += 20
    elif safe_float(z.structure_bias,0) < 0:
        bear += 20

    if safe_float(z.liquidity_sweep_down,0) > 0:
        bull += 20
        reasons.append("Sell-side liquidity sweep proxy")
    if safe_float(z.liquidity_sweep_up,0) > 0:
        bear += 20
        reasons.append("Buy-side liquidity sweep proxy")

    if safe_float(z.swing_pressure,.5) > .65:
        bull += 10
    elif safe_float(z.swing_pressure,.5) < .35:
        bear += 10

    direction = "BULLISH" if bull>bear else "BEARISH" if bear>bull else "NEUTRAL"
    return {
        "name":"Market Structure",
        "direction":direction,
        "score":max(bull,bear),
        "bull":bull,"bear":bear,
        "reasons":reasons
    }

def breakout_engine(x):
    z = x.iloc[-1]
    up = safe_float(z.breakout_up,0)
    dn = safe_float(z.breakout_down,0)

    if up:
        direction = "BULLISH"
        score = min(100, 60 + 20*max(0,safe_float(z.range_expansion,1)-1))
    elif dn:
        direction = "BEARISH"
        score = min(100, 60 + 20*max(0,safe_float(z.range_expansion,1)-1))
    else:
        direction = "NEUTRAL"
        score = 35

    return {
        "name":"Breakout",
        "direction":direction,
        "score":score,
        "range_expansion":safe_float(z.range_expansion)
    }

def mean_reversion_engine(x):
    z = x.iloc[-1]
    mz = safe_float(z.mean_z,0)
    bb = safe_float(z.bbpos,.5)

    if mz <= -1.5 or bb <= .05:
        direction = "BULLISH"
        score = 65
    elif mz >= 1.5 or bb >= .95:
        direction = "BEARISH"
        score = 65
    else:
        direction = "NEUTRAL"
        score = 30

    return {
        "name":"Mean Reversion",
        "direction":direction,
        "score":score,
        "mean_z":mz,
        "bbpos":bb
    }

def liquidity_engine(x):
    z = x.iloc[-1]
    bull = bear = 0

    if safe_float(z.liquidity_sweep_down,0):
        bull += 60
    if safe_float(z.liquidity_sweep_up,0):
        bear += 60

    if safe_float(z.dist_low20,1) < safe_float(z.atr_pct,0)*2:
        bull += 15
    if safe_float(z.dist_high20,1) < safe_float(z.atr_pct,0)*2:
        bear += 15

    direction = "BULLISH" if bull>bear else "BEARISH" if bear>bull else "NEUTRAL"

    return {
        "name":"Liquidity",
        "direction":direction,
        "score":max(bull,bear)
    }

def price_action_engine(x):
    z = x.iloc[-1]
    bull = bear = 0

    body = safe_float(z.body_pct,0)
    rng = safe_float(z.range_pct,0)
    cloc = safe_float(z.cloc,.5)

    if rng > 0:
        if cloc > .75 and body/rng > .55:
            bull += 50
        if cloc < .25 and body/rng > .55:
            bear += 50

    if safe_float(z.lwick_pct,0) > safe_float(z.uwick_pct,0)*1.5:
        bull += 20
    if safe_float(z.uwick_pct,0) > safe_float(z.lwick_pct,0)*1.5:
        bear += 20

    direction = "BULLISH" if bull>bear else "BEARISH" if bear>bull else "NEUTRAL"
    return {
        "name":"Price Action",
        "direction":direction,
        "score":max(bull,bear)
    }

def regime_engine(x):
    z = x.iloc[-1]
    ad = safe_float(z.adx,0)
    atrz = safe_float(z.atr_z,0)
    bwz = safe_float(z.bbw_z,0)

    if ad >= 25 and safe_float(z.e20_50,0) > 0:
        regime = "TRENDING_BULLISH"
    elif ad >= 25 and safe_float(z.e20_50,0) < 0:
        regime = "TRENDING_BEARISH"
    elif ad < 18 and bwz < 0:
        regime = "RANGE_COMPRESSION"
    elif abs(atrz) > 1:
        regime = "VOLATILITY_TRANSITION"
    else:
        regime = "TRANSITION"

    return {
        "name":"Regime",
        "regime":regime,
        "score":75 if "TRENDING" in regime else 60
    }

def session_engine(x):
    z = x.iloc[-1]
    h = int(z.datetime.hour)

    if 7 <= h < 12:
        s = "LONDON"
    elif 12 <= h < 16:
        s = "LONDON_NY_OVERLAP"
    elif 16 <= h < 21:
        s = "NEW_YORK"
    else:
        s = "ASIA/OFF-PEAK"

    return {"name":"Session","session":s}

def volume_engine(x):
    z = x.iloc[-1]
    vz = safe_float(z.volume_z, np.nan)

    if not np.isfinite(vz):
        return {
            "name":"Participation",
            "status":"UNAVAILABLE",
            "score":50
        }

    return {
        "name":"Participation",
        "status":"HIGH" if vz > 1 else "LOW" if vz < -1 else "NORMAL",
        "score":70 if vz > 0 else 50,
        "volume_z":vz
    }

def mtf_engine(higher):
    bull = bear = 0
    details = []

    for tf_name, d in higher.items():
        if d is None or len(d) < 220:
            details.append((tf_name,"INSUFFICIENT"))
            continue

        x = build_features(d)
        z = x.iloc[-1]

        if z.ema20 > z.ema50 and z.ema50 > z.ema100:
            bull += 1
            details.append((tf_name,"BULLISH"))
        elif z.ema20 < z.ema50 and z.ema50 < z.ema100:
            bear += 1
            details.append((tf_name,"BEARISH"))
        else:
            details.append((tf_name,"MIXED"))

    direction = "BULLISH" if bull>bear else "BEARISH" if bear>bull else "NEUTRAL"
    score = 0 if bull+bear == 0 else 100*max(bull,bear)/(bull+bear)

    return {
        "name":"Multi-Timeframe",
        "direction":direction,
        "score":score,
        "bull":bull,"bear":bear,
        "details":details
    }

# ----------------------------------------------------------------
# ML engine
# Event-aligned labels
# ----------------------------------------------------------------

def make_model(kind):
    if kind == "random_forest":
        m = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    elif kind == "gradient_boosting":
        m = HistGradientBoostingClassifier(
            max_iter=300,
            max_leaf_nodes=15,
            learning_rate=.04,
            l2_regularization=2,
            random_state=42
        )
    else:
        m = LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            C=.35,
            random_state=42
        )

    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scale",StandardScaler() if kind=="logistic" else "passthrough"),
        ("model",m)
    ])

def simulate_trade_from_bar(d, i, direction, rr, stop_atr, spread_pips, slippage_pips, horizon):
    """
    Event-aligned execution model.

    Signal is formed at CLOSE[i].
    Entry occurs at OPEN[i+1].
    Costs are applied at entry and exit.
    SL/TP are based on ATR at signal time.

    Same-candle ambiguity:
    if both SL and TP are touched, SL wins conservatively.
    """

    if i+1 >= len(d):
        return None

    p = pip_size(
        CURRENT_PAIR_FOR_SIM if "CURRENT_PAIR_FOR_SIM" in globals()
        else "EUR/USD"
    )

    spread = spread_pips*p
    slip = slippage_pips*p

    raw = float(d.iloc[i+1].open)

    if direction == "BUY":
        entry = raw + spread/2 + slip
    else:
        entry = raw - spread/2 - slip

    atrv = safe_float(d.iloc[i].atr)
    if not np.isfinite(atrv) or atrv <= 0:
        return None

    stop_dist = max(atrv*stop_atr, p*5)

    if direction == "BUY":
        sl = entry-stop_dist
        tp = entry+stop_dist*rr
    else:
        sl = entry+stop_dist
        tp = entry-stop_dist*rr

    end = min(len(d)-1, i+1+horizon)

    exit_i = end
    outcome = "TIME"
    exit_price = None

    for k in range(i+1,end+1):
        b = d.iloc[k]

        if direction == "BUY":
            hit_sl = b.low <= sl
            hit_tp = b.high >= tp

            if hit_sl:
                exit_price = sl-spread/2-slip
                outcome = "SL"
                exit_i = k
                break
            if hit_tp:
                exit_price = tp-spread/2-slip
                outcome = "TP"
                exit_i = k
                break

        else:
            hit_sl = b.high >= sl
            hit_tp = b.low <= tp

            if hit_sl:
                exit_price = sl+spread/2+slip
                outcome = "SL"
                exit_i = k
                break
            if hit_tp:
                exit_price = tp+spread/2+slip
                outcome = "TP"
                exit_i = k
                break

    if exit_price is None:
        close = float(d.iloc[end].close)
        if direction == "BUY":
            exit_price = close-spread/2-slip
        else:
            exit_price = close+spread/2+slip

    r = (
        (exit_price-entry)/stop_dist
        if direction=="BUY"
        else (entry-exit_price)/stop_dist
    )

    return {
        "signal_time":d.iloc[i].datetime,
        "entry_time":d.iloc[i+1].datetime,
        "exit_time":d.iloc[exit_i].datetime,
        "direction":direction,
        "entry":entry,
        "exit":exit_price,
        "sl":sl,
        "tp":tp,
        "outcome":outcome,
        "pnl_r":float(r),
        "holding_bars":int(exit_i-(i+1))
    }

def event_labels(d, rr, stop_atr, spread, slip, horizon):
    """
    Create two causal labels:
      y_buy = 1 if a BUY trade reaches TP before SL
      y_sell = 1 if a SELL trade reaches TP before SL

    The labels use the same execution assumptions as the backtest.
    """
    global CURRENT_PAIR_FOR_SIM

    # Add causal ATR once so every label uses the same signal-time
    # volatility measurement without recalculating ATR per trade.
    fd = build_features(d)
    rows_buy = []
    rows_sell = []

    for i in range(len(fd)-1):
        b = simulate_trade_from_bar(
            fd,i,"BUY",rr,stop_atr,spread,slip,horizon
        )
        s = simulate_trade_from_bar(
            fd,i,"SELL",rr,stop_atr,spread,slip,horizon
        )

        rows_buy.append(np.nan if b is None else int(b["outcome"]=="TP"))
        rows_sell.append(np.nan if s is None else int(s["outcome"]=="TP"))

    rows_buy.append(np.nan)
    rows_sell.append(np.nan)

    return (
        pd.Series(rows_buy,index=d.index),
        pd.Series(rows_sell,index=d.index)
    )

def fit_binary_model(X, y, kind):
    good = y.notna()
    X2 = X.loc[good]
    y2 = y.loc[good].astype(int)

    if len(X2) < 150 or y2.nunique() < 2:
        return None

    m = make_model(kind)
    m.fit(X2,y2)
    return m

def probability(m, X):
    if m is None:
        return np.nan
    return float(m.predict_proba(X)[0,1])

def train_pair_models(X, yb, ys, kind):
    return (
        fit_binary_model(X,yb,kind),
        fit_binary_model(X,ys,kind)
    )

# ----------------------------------------------------------------
# Composite signal engine
# ----------------------------------------------------------------

def composite_signal(
    x,
    higher,
    buy_prob,
    sell_prob,
    min_prob=70,
    min_score=68,
    min_edge=0.08,
    max_cost_r=0.35
):
    tr = trend_engine(x)
    mo = momentum_engine(x)
    vo = volatility_engine(x)
    stc = structure_engine(x)
    br = breakout_engine(x)
    mr = mean_reversion_engine(x)
    liq = liquidity_engine(x)
    pa = price_action_engine(x)
    reg = regime_engine(x)
    ses = session_engine(x)
    vol = volume_engine(x)
    mtf = mtf_engine(higher)

    engines = [tr,mo,stc,br,mr,liq,pa]

    bull = 0
    bear = 0
    weight_total = 0

    weights = {
        "Trend":1.4,
        "Momentum":1.5,
        "Market Structure":1.7,
        "Breakout":1.0,
        "Mean Reversion":.7,
        "Liquidity":1.1,
        "Price Action":.8
    }

    for e in engines:
        w = weights[e["name"]]
        if e["direction"] == "BULLISH":
            bull += e["score"]*w
            weight_total += e["score"]*w
        elif e["direction"] == "BEARISH":
            bear += e["score"]*w
            weight_total += e["score"]*w

    if mtf["direction"] == "BULLISH":
        bull += 20*mtf["score"]/100
    elif mtf["direction"] == "BEARISH":
        bear += 20*mtf["score"]/100

    if weight_total <= 0:
        analysis_score = 0
    else:
        analysis_score = 100*abs(bull-bear)/max(bull+bear,1)

    # Direction from structural/technical evidence first.
    technical_direction = (
        "BUY" if bull>bear
        else "SELL" if bear>bull
        else "NO TRADE"
    )

    # ML chooses direction only if it agrees with the technical side.
    bp = safe_float(buy_prob,0)
    sp = safe_float(sell_prob,0)

    ml_direction = "BUY" if bp>sp else "SELL" if sp>bp else "NO TRADE"
    ml_conf = max(bp,sp)*100
    ml_edge = abs(bp-sp)

    blockers = []

    if technical_direction == "NO TRADE":
        blockers.append("Technical engines have no directional edge.")

    if ml_direction != technical_direction:
        blockers.append("ML direction conflicts with technical engines.")

    if ml_conf < min_prob:
        blockers.append(f"ML confidence below {min_prob:.0f}%.")

    if analysis_score < min_score:
        blockers.append(f"Technical score below {min_score:.0f}.")

    if ml_edge < min_edge:
        blockers.append("BUY/SELL probability edge is too small.")

    # Regime filters
    if reg["regime"] == "RANGE_COMPRESSION" and br["direction"] != "NEUTRAL":
        # Breakouts in compression are allowed only with strong structure.
        if stc["score"] < 60:
            blockers.append("Compression regime without strong structure.")

    if vo["regime"] == "EXPANSION":
        # Expansion increases execution risk; require strong structure.
        if stc["score"] < 50:
            blockers.append("Volatility expansion without structure confirmation.")

    # Mean-reversion conflict
    if technical_direction == "BUY" and mr["direction"] == "BEARISH" and mr["score"] >= 65:
        blockers.append("Buy setup is extended into mean-reversion pressure.")
    if technical_direction == "SELL" and mr["direction"] == "BULLISH" and mr["score"] >= 65:
        blockers.append("Sell setup is extended into mean-reversion pressure.")

    approved = len(blockers) == 0

    if approved:
        if ml_conf >= 82 and analysis_score >= 80:
            grade = "A+"
        elif ml_conf >= 76 and analysis_score >= 74:
            grade = "A"
        else:
            grade = "B"
    else:
        grade = "NO TRADE"

    return {
        "direction": technical_direction if approved else "NO TRADE",
        "grade":grade,
        "approved":approved,
        "ml_conf":ml_conf,
        "buy_prob":bp*100,
        "sell_prob":sp*100,
        "ml_edge":ml_edge*100,
        "analysis_score":analysis_score,
        "blockers":blockers,
        "engines":{
            "Trend":tr,
            "Momentum":mo,
            "Volatility":vo,
            "Market Structure":stc,
            "Breakout":br,
            "Mean Reversion":mr,
            "Liquidity":liq,
            "Price Action":pa,
            "Regime":reg,
            "Session":ses,
            "Participation":vol,
            "Multi-Timeframe":mtf
        }
    }

# ----------------------------------------------------------------
# Walk-forward backtest
# ----------------------------------------------------------------

def walk_forward_backtest(
    d, pair, horizon, train, test, kind,
    rr, stop_atr, spread, slip,
    min_prob, min_score, min_edge,
    one_trade
):
    global CURRENT_PAIR_FOR_SIM
    CURRENT_PAIR_FOR_SIM = pair

    x = build_features(d)
    yb, ys = event_labels(
        d,rr,stop_atr,spread,slip,horizon
    )

    good = x[FEATURES].notna().sum(axis=1) >= int(len(FEATURES)*.45)
    good &= yb.notna() & ys.notna()

    x = x.loc[good].reset_index(drop=True)
    yb = yb.loc[good].reset_index(drop=True)
    ys = ys.loc[good].reset_index(drop=True)
    # x contains raw OHLC plus causal indicators, including ATR.
    dd = x.copy().reset_index(drop=True)

    if len(x) < train+test+100:
        return {
            "error":(
                f"Not enough usable history. Need roughly "
                f"{train+test+100}, got {len(x)}."
            )
        }

    rows = []
    block_stats = []

    start = train

    while start < len(x):
        end = min(start+test,len(x))

        Xtr = x.iloc[:start][FEATURES]
        ytr_b = yb.iloc[:start]
        ytr_s = ys.iloc[:start]

        Xte = x.iloc[start:end][FEATURES]

        # Calibration / threshold selection occurs strictly inside training.
        split = int(len(Xtr)*.8)

        Xfit = Xtr.iloc[:split]
        Xval = Xtr.iloc[split:]
        yfit_b = ytr_b.iloc[:split]
        yval_b = ytr_b.iloc[split:]
        yfit_s = ytr_s.iloc[:split]
        yval_s = ytr_s.iloc[split:]

        if len(Xval) < 50:
            start = end
            continue

        mb, ms = train_pair_models(
            Xfit,yfit_b,yfit_s,kind
        )
        if mb is None or ms is None:
            start = end
            continue

        # Validation probabilities.
        pbv = mb.predict_proba(Xval)[:,1]
        psv = ms.predict_proba(Xval)[:,1]

        # Determine an honest threshold from validation only.
        chosen = min_prob
        best_val = -999

        for t in np.arange(min_prob,91,1):
            conf = np.maximum(pbv,psv)*100
            edge = np.abs(pbv-psv)*100
            take = (conf>=t) & (edge>=min_edge*100)

            if take.sum() < 12:
                continue

            direction = np.where(pbv>=psv,1,0)
            actual = np.where(direction==1,yval_b.values,yval_s.values)

            wr = float(np.mean(actual[take]))
            coverage = float(take.mean())

            # Favor profitable classification without rewarding
            # extremely tiny samples.
            objective = wr - .15*(1-coverage)

            if objective > best_val:
                best_val = objective
                chosen = float(t)

        # Retrain on ALL information available before OOS block.
        mb, ms = train_pair_models(
            Xtr,ytr_b,ytr_s,kind
        )
        if mb is None or ms is None:
            start = end
            continue

        pb = mb.predict_proba(Xte)[:,1]
        ps = ms.predict_proba(Xte)[:,1]

        last_exit = -1

        for j in range(len(Xte)):
            i = start+j

            conf = max(pb[j],ps[j])*100
            edge = abs(pb[j]-ps[j])*100

            if conf < chosen:
                continue
            if edge < min_edge*100:
                continue

            direction = "BUY" if pb[j]>=ps[j] else "SELL"

            if one_trade and i <= last_exit:
                continue

            tr = simulate_trade_from_bar(
                dd,i,direction,
                rr,stop_atr,spread,slip,horizon
            )
            if tr is None:
                continue

            # Reconstruct a compact evidence score using only information
            # known at signal time.
            xx = x.iloc[:i+1]
            te = trend_engine(xx)
            me = momentum_engine(xx)
            se = structure_engine(xx)

            bscore = 0
            if direction=="BUY":
                if te["direction"]=="BULLISH": bscore += 30
                if me["direction"]=="BULLISH": bscore += 30
                if se["direction"]=="BULLISH": bscore += 40
            else:
                if te["direction"]=="BEARISH": bscore += 30
                if me["direction"]=="BEARISH": bscore += 30
                if se["direction"]=="BEARISH": bscore += 40

            if bscore < min_score:
                continue

            tr.update({
                "probability":float(conf),
                "buy_probability":float(pb[j]*100),
                "sell_probability":float(ps[j]*100),
                "probability_edge":float(edge),
                "threshold":float(chosen),
                "technical_score":float(bscore)
            })

            rows.append(tr)

            if one_trade:
                last_exit = i+horizon

        block = {
            "block_start":str(dd.iloc[start].datetime),
            "block_end":str(dd.iloc[end-1].datetime),
            "threshold":chosen,
            "oos_rows":end-start
        }
        block_stats.append(block)

        start = end

    out = pd.DataFrame(rows)

    if out.empty:
        return {
            "trades":0,
            "wins":0,
            "losses":0,
            "win_rate":0,
            "expectancy":0,
            "profit_factor":None,
            "max_drawdown":0,
            "signals":out,
            "blocks":block_stats,
            "avg_threshold":np.nan
        }

    pnl = out.pnl_r.to_numpy(float)
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    ddv = eq-peak

    wins = pnl[pnl>0]
    losses = pnl[pnl<0]

    pf = (
        wins.sum()/abs(losses.sum())
        if len(losses) and losses.sum()!=0
        else None
    )

    # OOS Brier-like statistic for realized TP/SL outcome.
    actual = (out.outcome=="TP").astype(int).to_numpy()
    pred = np.clip(out.probability/100,0,1)
    brier = float(np.mean((pred-actual)**2))

    return {
        "trades":len(out),
        "wins":int((pnl>0).sum()),
        "losses":int((pnl<=0).sum()),
        "win_rate":float((pnl>0).mean()*100),
        "expectancy":float(pnl.mean()),
        "profit_factor":None if pf is None else float(pf),
        "max_drawdown":float(ddv.min()),
        "brier":brier,
        "avg_threshold":float(
            np.mean([b["threshold"] for b in block_stats])
        ) if block_stats else np.nan,
        "signals":out,
        "blocks":block_stats
    }

# ----------------------------------------------------------------
# Charts
# ----------------------------------------------------------------

def chart(d,pair):
    x = build_features(d).tail(240)

    f = go.Figure(
        go.Candlestick(
            x=x.datetime,
            open=x.open,
            high=x.high,
            low=x.low,
            close=x.close,
            name=pair
        )
    )

    for n in [20,50,200]:
        f.add_trace(
            go.Scatter(
                x=x.datetime,
                y=x[f"ema{n}"],
                name=f"EMA{n}"
            )
        )

    f.update_layout(
        height=570,
        xaxis_rangeslider_visible=False,
        title=f"{pair} — {APP_VERSION}"
    )
    return f

# ----------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------

st.set_page_config(
    page_title="Forex AI Pro V11",
    page_icon="🤖",
    layout="wide"
)

db_init()

st.title("🤖 Forex AI Pro V11.0")
st.caption(
    "Multi-engine forex analysis + event-aligned walk-forward research "
    "• research/paper trading only"
)

API = api_key()

if not API:
    st.error(
        "TWELVE_DATA_API_KEY is not configured in Streamlit Secrets."
    )
    st.stop()

# ============================================================
# LIVE DASHBOARD
# ============================================================

with st.sidebar:
    st.header("LIVE SIGNAL ENGINE")

    pair = st.selectbox("Pair",PAIRS)
    entry_tf = st.selectbox(
        "Entry timeframe",
        ["1m","3m","5m","15m","30m","1h"],
        index=3
    )

    htf_names = st.multiselect(
        "Higher timeframes",
        ["1h","4h","1day"],
        ["1h","4h"]
    )

    size = st.slider(
        "Historical candles",
        500,5000,2500,100
    )

    model_kind = st.selectbox(
        "ML engine",
        ["ensemble","logistic","random_forest","gradient_boosting"]
    )

    # Ensemble live model is approximated by averaging three models.
    min_prob = st.slider(
        "Minimum ML probability %",
        55,90,72
    )
    min_score = st.slider(
        "Minimum technical score",
        50,95,70
    )
    min_edge = st.slider(
        "Minimum BUY/SELL probability edge %",
        3,30,8
    )

    rr = st.slider(
        "R:R",
        1.0,4.0,2.0,0.25
    )

    stop_atr = st.slider(
        "Stop ATR multiple",
        0.75,3.0,1.5,0.1
    )

    spread = st.number_input(
        "Expected spread (pips)",
        0.0,10.0,1.0,0.1
    )

    slip = st.number_input(
        "Expected slippage (pips)",
        0.0,5.0,0.2,0.1
    )

    generate = st.button(
        "🔄 GENERATE LIVE SIGNAL",
        type="primary",
        use_container_width=True
    )

if generate:
    try:
        d = candles(
            pair,TF[entry_tf],size,API
        )

        higher = {}
        for h in htf_names:
            higher[h] = candles(
                pair,
                TF[h],
                min(size,1500),
                API
            )

    except Exception as e:
        st.exception(e)
        st.stop()

    h = health(d,TF[entry_tf])

    a,b,c,e = st.columns(4)
    a.metric("Candles",len(d))
    b.metric("Freshness","OK" if h["fresh"] else "STALE")
    c.metric("Gaps","OK" if h["gaps_ok"] else "CHECK")
    e.metric("Age",f'{h["age"]:.1f}m')

    for issue in h["issues"]:
        st.warning(issue)

    if not h["valid"] or not h["fresh"] or not h["gaps_ok"]:
        st.error("🔴 NO TRADE — data gate failed.")
        st.stop()

    x = build_features(d)

    st.plotly_chart(
        chart(d,pair),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Train live event-aligned models
    # --------------------------------------------------------

    global CURRENT_PAIR_FOR_SIM
    CURRENT_PAIR_FOR_SIM = pair

    yb,ys = event_labels(
        d,rr,stop_atr,spread,slip,1
    )

    X = x[FEATURES]
    good = X.notna().sum(axis=1) >= int(len(FEATURES)*.45)

    models = []

    if model_kind == "ensemble":
        kinds = [
            "logistic",
            "random_forest",
            "gradient_boosting"
        ]
    else:
        kinds = [model_kind]

    for k in kinds:
        mb,ms = train_pair_models(
            X.loc[good],
            yb.loc[good],
            ys.loc[good],
            k
        )
        if mb is not None and ms is not None:
            models.append((mb,ms))

    if not models:
        st.error(
            "ML engine could not train enough valid event-aligned data."
        )
        st.stop()

    latest = X.iloc[-1:]

    bp = float(np.mean([
        m[0].predict_proba(latest)[0,1]
        for m in models
    ]))

    sp = float(np.mean([
        m[1].predict_proba(latest)[0,1]
        for m in models
    ]))

    cf = composite_signal(
        x,
        higher,
        bp,sp,
        min_prob=min_prob,
        min_score=min_score,
        min_edge=min_edge/100
    )

    st.subheader("🎯 LIVE FOREX SIGNAL")

    a,b,c,dcol = st.columns(4)
    a.metric("Signal",cf["direction"])
    b.metric("Grade",cf["grade"])
    c.metric("BUY probability",f'{cf["buy_prob"]:.1f}%')
    dcol.metric("SELL probability",f'{cf["sell_prob"]:.1f}%')

    a,b,c,dcol = st.columns(4)
    a.metric("Technical score",f'{cf["analysis_score"]:.1f}')
    b.metric("Probability edge",f'{cf["ml_edge"]:.1f}%')
    c.metric(
        "Volatility",
        cf["engines"]["Volatility"]["regime"]
    )
    dcol.metric(
        "Regime",
        cf["engines"]["Regime"]["regime"]
    )

    if cf["approved"]:
        z = x.iloc[-1]
        entry = float(d.close.iloc[-1])
        atrv = float(z.atr)
        p = pip_size(pair)

        stop_dist = max(atrv*stop_atr,p*5)

        if cf["direction"]=="BUY":
            sl = entry-stop_dist
            tp = entry+stop_dist*rr
        else:
            sl = entry+stop_dist
            tp = entry-stop_dist*rr

        st.success(
            f'🟢 {pair} {cf["direction"]} — '
            f'{cf["grade"]} — APPROVED'
        )

        a,b,c,e = st.columns(4)
        a.metric("Entry",f"{entry:.6f}")
        b.metric("SL",f"{sl:.6f}")
        c.metric("TP",f"{tp:.6f}")
        e.metric("R:R",f"{rr:.2f}")

        payload = {
            "pair":pair,
            "engine":"V11 MULTI-ENGINE",
            "timeframe":entry_tf,
            "direction":cf["direction"],
            "probability":cf["ml_conf"],
            "threshold":min_prob,
            "entry":entry,
            "stop_loss":sl,
            "take_profit":tp,
            "regime":cf["engines"]["Regime"]["regime"],
            "status":"APPROVED",
            "explanation":"; ".join(
                cf["engines"]["Momentum"]["reasons"] +
                cf["engines"]["Market Structure"]["reasons"]
            ),
            "analysis":cf
        }

    else:
        st.error("🔴 NO TRADE")

        if cf["blockers"]:
            for b in cf["blockers"]:
                st.write("•",b)

        payload = {
            "pair":pair,
            "engine":"V11 MULTI-ENGINE",
            "timeframe":entry_tf,
            "direction":"NO TRADE",
            "probability":cf["ml_conf"],
            "threshold":min_prob,
            "regime":cf["engines"]["Regime"]["regime"],
            "status":"BLOCKED",
            "explanation":"; ".join(cf["blockers"]),
            "analysis":cf
        }

    st.subheader("🔬 Engine Matrix")

    matrix_rows = []

    for name,e in cf["engines"].items():
        row = {"Engine":name}

        if "direction" in e:
            row["Direction"] = e["direction"]
        if "score" in e:
            row["Score"] = round(
                safe_float(e["score"],0),1
            )
        if "regime" in e:
            row["Regime"] = e["regime"]
        if "session" in e:
            row["Session"] = e["session"]
        if "status" in e:
            row["Status"] = e["status"]

        matrix_rows.append(row)

    st.dataframe(
        pd.DataFrame(matrix_rows),
        use_container_width=True,
        hide_index=True
    )

    if st.button("📝 Journal signal",key="jlive_v11"):
        journal(payload)
        st.success("Signal saved.")

# ============================================================
# RESEARCH / BACKTEST
# ============================================================

st.divider()
st.header("🧪 V11 EVENT-ALIGNED WALK-FORWARD BACKTEST")

st.caption(
    "The model is trained on the same executable trade definition "
    "used by the backtest: signal at close → next-bar open entry → "
    "spread/slippage → ATR stop/TP → horizon exit. "
    "Threshold selection is validation-only."
)

bp = st.selectbox(
    "Backtest pair",
    PAIRS,
    key="v11_bp"
)

btf = st.selectbox(
    "Backtest timeframe",
    ["1m","3m","5m","15m","30m","1h"],
    index=3,
    key="v11_btf"
)

bh = st.number_input(
    "Maximum holding horizon (candles)",
    1,30,3,1
)

btrain = st.number_input(
    "Initial training candles",
    500,3000,1000,50
)

btest = st.number_input(
    "Out-of-sample block",
    25,500,150,25
)

bmodel = st.selectbox(
    "Backtest ML",
    ["ensemble","logistic","random_forest","gradient_boosting"],
    key="v11_model"
)

brr = st.slider(
    "Backtest R:R",
    1.0,4.0,2.0,.25,
    key="v11_rr"
)

bstop = st.slider(
    "Backtest stop ATR",
    .75,3.0,1.5,.1,
    key="v11_stop"
)

bspread = st.number_input(
    "Backtest spread (pips)",
    0.0,10.0,1.0,.1,
    key="v11_spread"
)

bslip = st.number_input(
    "Backtest slippage (pips)",
    0.0,5.0,.2,.1,
    key="v11_slip"
)

bprob = st.slider(
    "Minimum ML probability %",
    55,90,72,
    key="v11_prob"
)

bscore = st.slider(
    "Minimum technical score",
    50,95,70,
    key="v11_score"
)

bedge = st.slider(
    "Minimum probability edge %",
    3,30,8,
    key="v11_edge"
)

bone = st.checkbox(
    "One trade at a time",
    True,
    key="v11_one"
)

if st.button(
    "▶ RUN V11 REALISTIC WALK-FORWARD",
    type="primary"
):
    try:
        bd = candles(
            bp,TF[btf],size,API
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    hh = health(bd,TF[btf])

    if not hh["valid"]:
        st.error("Backtest data failed OHLC validation.")
        st.stop()

    if len(bd) < int(btrain)+int(btest)+250:
        st.warning(
            f"Only {len(bd)} candles loaded. "
            f"Consider increasing historical candles."
        )

    with st.spinner(
        "Running event-aligned walk-forward test..."
    ):
        result = walk_forward_backtest(
            bd,
            bp,
            int(bh),
            int(btrain),
            int(btest),
            bmodel,
            float(brr),
            float(bstop),
            float(bspread),
            float(bslip),
            float(bprob),
            float(bscore),
            float(bedge),
            bone
        )

    if "error" in result:
        st.error(result["error"])
    else:
        a,b,c,dcol,e = st.columns(5)

        a.metric("Trades",result["trades"])
        b.metric(
            "Win rate",
            f'{result["win_rate"]:.2f}%'
        )
        c.metric(
            "Expectancy",
            f'{result["expectancy"]:.3f} R'
        )
        dcol.metric(
            "Profit factor",
            "N/A"
            if result["profit_factor"] is None
            else f'{result["profit_factor"]:.2f}'
        )
        e.metric(
            "Max DD",
            f'{result["max_drawdown"]:.2f} R'
        )

        a,b,c,dcol = st.columns(4)
        a.metric(
            "OOS Brier",
            "N/A"
            if not np.isfinite(result.get("brier",np.nan))
            else f'{result["brier"]:.4f}'
        )
        b.metric(
            "Avg locked threshold",
            "N/A"
            if not np.isfinite(result.get("avg_threshold",np.nan))
            else f'{result["avg_threshold"]:.1f}%'
        )

        sig = result["signals"]

        if not sig.empty:
            st.subheader("Trade Results")
            st.dataframe(
                sig.tail(500),
                use_container_width=True,
                hide_index=True
            )

            eq = sig.pnl_r.cumsum()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sig.exit_time,
                    y=eq,
                    mode="lines",
                    name="Cumulative R"
                )
            )
            fig.update_layout(
                height=380,
                title="Out-of-sample cumulative R"
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )

            outcomes = (
                sig.outcome.value_counts()
                .rename_axis("Outcome")
                .reset_index(name="Count")
            )

            st.subheader("Outcome Distribution")
            st.dataframe(
                outcomes,
                use_container_width=True,
                hide_index=True
            )

            if len(sig):
                st.subheader("Performance Diagnostics")

                longest_loss = 0
                current_loss = 0

                for r in sig.pnl_r:
                    if r <= 0:
                        current_loss += 1
                        longest_loss = max(
                            longest_loss,current_loss
                        )
                    else:
                        current_loss = 0

                avg_hold = sig.holding_bars.mean()

                a,b,c = st.columns(3)
                a.metric(
                    "Longest losing streak",
                    int(longest_loss)
                )
                b.metric(
                    "Average holding",
                    f"{avg_hold:.2f} bars"
                )
                c.metric(
                    "Median trade R",
                    f'{sig.pnl_r.median():.3f}'
                )

        else:
            st.warning(
                "No trades survived the complete V11 filter. "
                "That is preferable to fabricating a trade, but it "
                "means the configuration needs more research."
            )

        # Save research result.
        payload = {
            "version":APP_VERSION,
            "pair":bp,
            "timeframe":btf,
            "horizon":int(bh),
            "train":int(btrain),
            "test_block":int(btest),
            "model":bmodel,
            "rr":float(brr),
            "stop_atr":float(bstop),
            "spread":float(bspread),
            "slippage":float(bslip),
            "min_probability":float(bprob),
            "min_score":float(bscore),
            "min_edge":float(bedge),
            "one_trade":bool(bone),
            "blocks":result.get("blocks",[])
        }

        with sqlite3.connect(DB) as con:
            con.execute("""
            INSERT INTO backtests(
                created_at,pair,timeframe,engine,trades,wins,losses,
                win_rate,expectancy,profit_factor,max_drawdown,
                brier,payload
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,(
                now().isoformat(),
                bp,btf,
                "V11 MULTI-ENGINE",
                result["trades"],
                result["wins"],
                result["losses"],
                result["win_rate"],
                result["expectancy"],
                result["profit_factor"],
                result["max_drawdown"],
                result.get("brier"),
                json.dumps(payload,default=str)
            ))

        # Explicit research gate.
        if (
            result["trades"] >= 50 and
            result["expectancy"] > 0 and
            (result["profit_factor"] or 0) > 1 and
            result["max_drawdown"] > -30
        ):
            st.success(
                "🟢 RESEARCH GATE: preliminary positive OOS evidence. "
                "Still paper-trade before any real-money use."
            )
        else:
            st.error(
                "🔴 RESEARCH GATE: strategy does not currently "
                "meet the robustness criteria. DO NOT treat it as "
                "a validated trading system."
            )

        st.info(
            "A high historical win rate is not required for validity. "
            "The key tests are positive out-of-sample expectancy, "
            "profit factor above 1 after costs, controlled drawdown, "
            "adequate trade count and stability across walk-forward blocks."
        )

# ============================================================
# JOURNAL
# ============================================================

st.divider()
st.header("📒 Signal Journal")

j = load_journal()

if j.empty:
    st.info("No signals journaled yet.")
else:
    st.dataframe(
        j,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    f"Forex AI Pro {APP_VERSION} • Twelve Data • "
    "research/paper trading only • no guaranteed win rate"
)
