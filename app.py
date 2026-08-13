import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Forex AI Pro Analyzer v5", page_icon="📈", layout="wide")

TD = "https://api.twelvedata.com"
CFTC = "https://publicreporting.cftc.gov/resource/yw9f-hn96.json"
CAL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
CCY = {p: tuple(p.split("/")) for p in PAIRS}
COT = {
    "EUR": "EURO FX", "GBP": "BRITISH POUND STERLING",
    "JPY": "JAPANESE YEN", "CHF": "SWISS FRANC",
    "AUD": "AUSTRALIAN DOLLAR", "CAD": "CANADIAN DOLLAR",
    "NZD": "NEW ZEALAND DOLLAR",
}

def sec(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)

TWELVE_DATA_API_KEY = sec("TWELVE_DATA_API_KEY")
FRED_API_KEY = sec("FRED_API_KEY")

@st.cache_data(ttl=30)
def candles(symbol, timeframe):
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing from Streamlit Secrets.")
    response = requests.get(
        f"{TD}/time_series",
        params={
            "symbol": symbol, "interval": timeframe, "outputsize": 500,
            "apikey": TWELVE_DATA_API_KEY, "timezone": "UTC",
        },
        timeout=20,
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("message", response.text[:300])
        except Exception:
            detail = response.text[:300]
        raise RuntimeError(
            f"Twelve Data returned HTTP {response.status_code} for {symbol} "
            f"({timeframe}): {detail}"
        )
    data = response.json()
    if "values" not in data:
        raise RuntimeError(
            f"No Twelve Data values for {symbol} ({timeframe}). "
            f"{data.get('message', 'Unknown API response.')}"
        )
    frame = pd.DataFrame(data["values"])
    if frame.empty:
        raise RuntimeError(f"Twelve Data returned no candles for {symbol} ({timeframe}).")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = ["datetime", "open", "high", "low", "close"]
    frame = frame.dropna(subset=[c for c in required if c in frame.columns])
    return frame.sort_values("datetime").reset_index(drop=True)

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def true_range(frame):
    previous_close = frame.close.shift()
    return pd.concat([
        frame.high - frame.low,
        (frame.high - previous_close).abs(),
        (frame.low - previous_close).abs(),
    ], axis=1).max(axis=1)

def atr(frame, period=14):
    return true_range(frame).ewm(alpha=1 / period, adjust=False).mean()

def macd(series):
    macd_line = ema(series, 12) - ema(series, 26)
    signal_line = ema(macd_line, 9)
    return macd_line, signal_line, macd_line - signal_line

def adx(frame, period=14):
    up = frame.high.diff()
    down = -frame.low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=frame.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=frame.index)
    tr_avg = true_range(frame).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr_avg
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr_avg
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean(), plus_di, minus_di

def add_indicators(frame):
    frame = frame.copy()
    for period in [20, 50, 100, 200]:
        frame[f"ema{period}"] = ema(frame.close, period)
    frame["rsi"] = rsi(frame.close)
    frame["macd"], frame["macd_signal"], frame["macd_hist"] = macd(frame.close)
    frame["adx"], frame["pdi"], frame["mdi"] = adx(frame)
    low14 = frame.low.rolling(14).min()
    high14 = frame.high.rolling(14).max()
    frame["stoch_k"] = 100 * (frame.close - low14) / (high14 - low14).replace(0, np.nan)
    frame["stoch_d"] = frame.stoch_k.rolling(3).mean()
    frame["roc"] = frame.close.pct_change(12) * 100
    frame["atr"] = atr(frame)
    frame["bb_mid"] = frame.close.rolling(20).mean()
    sd = frame.close.rolling(20).std()
    frame["bb_up"] = frame.bb_mid + 2 * sd
    frame["bb_low"] = frame.bb_mid - 2 * sd
    return frame.dropna().reset_index(drop=True)

def swings(frame, window=3):
    highs, lows = [], []
    if len(frame) < window * 2 + 1:
        return highs, lows
    for i in range(window, len(frame) - window):
        if frame.high.iloc[i] >= frame.high.iloc[i-window:i+window+1].max():
            highs.append(float(frame.high.iloc[i]))
        if frame.low.iloc[i] <= frame.low.iloc[i-window:i+window+1].min():
            lows.append(float(frame.low.iloc[i]))
    return highs, lows

def market_structure(frame):
    highs, lows = swings(frame)
    trend, bos = "RANGE", "None"
    if len(highs) > 1 and len(lows) > 1:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            trend = "BULLISH"
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            trend = "BEARISH"
    if highs and frame.close.iloc[-1] > highs[-1]:
        bos = "Bullish BOS"
    elif lows and frame.close.iloc[-1] < lows[-1]:
        bos = "Bearish BOS"
    return trend, bos, highs, lows

def candlestick_patterns(frame):
    if len(frame) < 2:
        return []
    a, p = frame.iloc[-1], frame.iloc[-2]
    body = abs(a.close - a.open)
    rng = max(a.high - a.low, 1e-12)
    upper = a.high - max(a.open, a.close)
    lower = min(a.open, a.close) - a.low
    out = []
    if body / rng < 0.12:
        out.append("Doji")
    if lower > 2 * body and upper < 1.2 * body:
        out.append("Bullish pin")
    if upper > 2 * body and lower < 1.2 * body:
        out.append("Bearish pin")
    if a.close > a.open and p.close < p.open and a.close >= p.open and a.open <= p.close:
        out.append("Bullish engulfing")
    if a.close < a.open and p.close > p.open and a.open >= p.close and a.close <= p.open:
        out.append("Bearish engulfing")
    return out

def levels(frame):
    data = frame.tail(150)
    highs, lows = swings(data)
    support = min(lows, default=float(data.low.min()))
    resistance = max(highs, default=float(data.high.max()))
    high, low = float(data.high.max()), float(data.low.min())
    price_range = high - low
    fib = {str(p): high - p * price_range for p in [0, .236, .382, .5, .618, .786, 1]}
    current_atr = float(data.atr.iloc[-1])
    equal_highs = len(highs) > 1 and abs(highs[-1] - highs[-2]) <= .25 * current_atr
    equal_lows = len(lows) > 1 and abs(lows[-1] - lows[-2]) <= .25 * current_atr
    fvgs = []
    for i in range(max(2, len(data) - 80), len(data)):
        if data.low.iloc[i] > data.high.iloc[i-2]:
            fvgs.append(("Bullish FVG", float(data.high.iloc[i-2]), float(data.low.iloc[i])))
        elif data.high.iloc[i] < data.low.iloc[i-2]:
            fvgs.append(("Bearish FVG", float(data.high.iloc[i]), float(data.low.iloc[i-2])))
    order_block = "None"
    if len(data) >= 3:
        p, z = data.iloc[-2], data.iloc[-1]
        if p.close < p.open and z.close > p.high:
            order_block = "Bullish order-block proxy"
        elif p.close > p.open and z.close < p.low:
            order_block = "Bearish order-block proxy"
    return support, resistance, fib, equal_highs, equal_lows, fvgs[-3:], order_block

@st.cache_data(ttl=21600)
def cot_raw():
    r = requests.get(CFTC, params={"$limit": 5000, "$order": "report_date_as_yyyy_mm_dd DESC"}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def cot(currency):
    try:
        d = cot_raw()
        name_col = next((z for z in d.columns if z.lower() == "contract_market_name"), None)
        if not name_col:
            return None, "COT contract field unavailable"
        q = d[d[name_col].astype(str).str.upper().str.contains(COT[currency], regex=False, na=False)].copy()
        long_col = next((z for z in q.columns if "leveraged_money_long" in z.lower()), None)
        short_col = next((z for z in q.columns if "leveraged_money_short" in z.lower()), None)
        if not long_col or not short_col:
            return None, "COT position fields unavailable"
        q[long_col] = pd.to_numeric(q[long_col], errors="coerce")
        q[short_col] = pd.to_numeric(q[short_col], errors="coerce")
        q["net"] = q[long_col] - q[short_col]
        q = q.dropna(subset=["net"])
        if q.empty:
            return None, "No COT data"
        n = float(q.iloc[0].net)
        return float(np.tanh(n / 100000) * 10), f"net {n:,.0f} (weekly)"
    except Exception as e:
        return None, f"COT unavailable: {e}"

@st.cache_data(ttl=1800)
def calendar():
    r = requests.get(CAL, timeout=20)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def events(pair):
    try:
        d = calendar()
        base, quote = CCY[pair]
        cc = next((c for c in d.columns if c.lower() in ["country", "currency"]), None)
        dc = next((c for c in d.columns if c.lower() in ["date", "datetime"]), None)
        if not cc or not dc:
            return 0, "Calendar schema changed"
        d[dc] = pd.to_datetime(d[dc], utc=True, errors="coerce")
        q = d[
            d[cc].astype(str).str.upper().isin([base, quote])
            & (d[dc] >= pd.Timestamp.now(tz="UTC"))
        ].sort_values(dc)
        if q.empty:
            return 0, "No upcoming pair events"
        z = q.iloc[0]
        impact = str(z.get("impact", "")).lower()
        adjustment = {"high": -12, "medium": -5, "low": -1}.get(impact, 0)
        title = str(z.get("title", z.get("event", "Economic event")))
        return adjustment, f"{title} | {z[dc].strftime('%d %b %H:%M UTC')} | {impact or 'unknown'}"
    except Exception as e:
        return 0, f"Calendar unavailable: {e}"

@st.cache_data(ttl=21600)
def fred(series_id):
    if not FRED_API_KEY:
        return None
    r = requests.get(
        FRED_URL,
        params={
            "series_id": series_id, "api_key": FRED_API_KEY,
            "file_type": "json", "sort_order": "desc", "limit": 10,
        },
        timeout=20,
    )
    if r.status_code != 200:
        try:
            detail = r.json().get("error_message", r.text[:300])
        except Exception:
            detail = r.text[:300]
        raise RuntimeError(f"FRED HTTP {r.status_code}: {detail}")
    values = [z for z in r.json().get("observations", []) if z.get("value") not in [".", None]]
    return float(values[0]["value"]) if values else None

def rates(pair):
    base, quote = CCY[pair]
    if not FRED_API_KEY:
        return 0, "FRED_API_KEY not configured"
    try:
        usd_rate = fred("DFF")
        if usd_rate is None:
            return 0, "FRED DFF unavailable"
        if base == "USD" and quote != "USD":
            return 0, f"USD rate available ({usd_rate:.2f}%), but {quote} official rate feed is not configured"
        if quote == "USD" and base != "USD":
            return 0, f"USD rate available ({usd_rate:.2f}%), but {base} official rate feed is not configured"
        return 0, "No valid differential configured"
    except Exception as e:
        return 0, f"FRED rate unavailable: {e}"

def technical_score(frame):
    a = frame.iloc[-1]
    score, notes = 0, []
    if a.ema20 > a.ema50 > a.ema100 > a.ema200:
        score += 20; notes.append("EMA bullish")
    elif a.ema20 < a.ema50 < a.ema100 < a.ema200:
        score -= 20; notes.append("EMA bearish")
    score += 6 if a.close > a.ema200 else -6
    if a.rsi >= 55:
        score += 10; notes.append(f"RSI {a.rsi:.1f} bullish")
    elif a.rsi <= 45:
        score -= 10; notes.append(f"RSI {a.rsi:.1f} bearish")
    score += 10 if a.macd_hist > 0 else -10
    if a.adx >= 25:
        score += 8 if a.pdi > a.mdi else -8
    if a.stoch_k > a.stoch_d and a.stoch_k < 80:
        score += 4
    elif a.stoch_k < a.stoch_d and a.stoch_k > 20:
        score -= 4
    score += 4 if a.roc > 0 else -4 if a.roc < 0 else 0
    score += 4 if a.close > a.bb_mid else -4
    return score, notes

def analyze(pair, confidence_threshold):
    # Keep the slash: Twelve Data expects EUR/USD, GBP/USD, etc.
    frames = {tf: add_indicators(candles(pair, tf)) for tf in ["15min", "1h", "4h"]}
    weights = {"15min": 1, "1h": 2, "4h": 3}
    detail, parts = [], []
    for tf, weight in weights.items():
        score, notes = technical_score(frames[tf])
        parts.append(score * weight)
        detail.append(f"{tf}: {', '.join(notes) if notes else 'No strong technical bias'}")
    technical = sum(parts) / 6

    trend, bos, _, _ = market_structure(frames["1h"])
    structure_score = 10 if trend == "BULLISH" else -10 if trend == "BEARISH" else 0
    structure_score += 8 if bos == "Bullish BOS" else -8 if bos == "Bearish BOS" else 0

    patterns = candlestick_patterns(frames["15min"])
    structure_score += sum(3 if "Bullish" in p else -3 if "Bearish" in p else 0 for p in patterns)

    base, quote = CCY[pair]
    cb, nb = cot(base)
    cq, nq = cot(quote)
    cot_score = (cb or 0) - (cq or 0)
    rate_score, rate_note = rates(pair)
    event_score, event_note = events(pair)
    total = technical + structure_score + cot_score + rate_score + event_score
    confidence = int(np.clip(50 + abs(total) * .55, 50, 95))

    signal = (
        "BUY" if total >= 18 and confidence >= confidence_threshold
        else "SELL" if total <= -18 and confidence >= confidence_threshold
        else "NO TRADE"
    )

    x = frames["15min"]
    price = float(x.close.iloc[-1])
    current_atr = float(x.atr.iloc[-1])

    if signal == "BUY":
        sl = min(price - 1.5 * current_atr, float(x.low.tail(60).min()))
        risk = price - sl
        tp1, tp2 = price + 1.5 * risk, price + 2.2 * risk
    elif signal == "SELL":
        sl = max(price + 1.5 * current_atr, float(x.high.tail(60).max()))
        risk = sl - price
        tp1, tp2 = price - 1.5 * risk, price - 2.2 * risk
    else:
        sl = tp1 = tp2 = np.nan

    status = {
        "Twelve Data": True,
        "CFTC COT": cb is not None or cq is not None,
        "Economic calendar": not event_note.startswith("Calendar unavailable"),
        "Interest rates": rate_score != 0,
    }

    return locals()

st.title("📈 Forex AI Pro Analyzer v5")
st.caption("Technical + structure + COT + economic-event risk + FRED rate data")

with st.sidebar:
    pair = st.selectbox("Forex pair", PAIRS)
    threshold = st.slider("Minimum confidence", 50, 90, 65)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

if not TWELVE_DATA_API_KEY:
    st.error("Add TWELVE_DATA_API_KEY in Streamlit Secrets.")
    st.stop()

try:
    a = analyze(pair, threshold)
except Exception as e:
    st.error(f"Analysis failed: {e}")
    st.stop()

c = st.columns(4)
c[0].metric("Signal", a["signal"])
c[1].metric("Confidence", f'{a["confidence"]}/100')
c[2].metric("Price", f'{a["price"]:.5f}')
c[3].metric("Score", f'{a["total"]:.1f}')

if a["signal"] != "NO TRADE":
    c = st.columns(3)
    c[0].metric("Stop Loss", f'{a["sl"]:.5f}')
    c[1].metric("TP1", f'{a["tp1"]:.5f}')
    c[2].metric("TP2", f'{a["tp2"]:.5f}')
else:
    st.warning("NO TRADE — evidence is below the selected threshold.")

st.subheader("Data-source status")
c = st.columns(4)
for col, (name, ok) in zip(c, a["status"].items()):
    col.metric(name, "AVAILABLE" if ok else "UNAVAILABLE")

t1, t2, t3, t4 = st.tabs(["Signal breakdown", "Indicators", "Levels & liquidity", "Data notes"])

with t1:
    st.write(
        "Technical:", round(a["technical"], 1),
        "| Structure:", round(a["structure_score"], 1),
        "| COT:", round(a["cot_score"], 1),
        "| Rates:", round(a["rate_score"], 1),
        "| Event:", round(a["event_score"], 1),
    )
    st.write("Rate engine:", a["rate_note"])
    st.write("Next event:", a["event_note"])
    st.write("COT base:", a["nb"])
    st.write("COT quote:", a["nq"])
    st.write("Patterns:", ", ".join(a["patterns"]) or "None")
    for z in a["detail"]:
        st.write("•", z)

with t2:
    rows = []
    for tf, x in a["frames"].items():
        z = x.iloc[-1]
        rows.append({
            "TF": tf, "Close": z.close, "EMA20": z.ema20, "EMA50": z.ema50,
            "EMA100": z.ema100, "EMA200": z.ema200, "RSI": z.rsi,
            "MACD": z.macd_hist, "ADX": z.adx, "Stoch": z.stoch_k,
            "ROC": z.roc, "ATR": z.atr,
        })
    st.dataframe(pd.DataFrame(rows).round(5), use_container_width=True, hide_index=True)

with t3:
    sup, res, fib, eqh, eql, fvg, ob = a["lev"] if "lev" in a else levels(a["frames"]["15min"])
    st.write("Support:", sup, "| Resistance:", res, "| Equal highs:", eqh, "| Equal lows:", eql)
    st.write("FVG:", fvg)
    st.write("Order block:", ob)
    st.dataframe(
        pd.DataFrame({"Fib": list(fib), "Price": list(fib.values())}).round(5),
        use_container_width=True,
        hide_index=True,
    )

with t4:
    st.write("UTC:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    st.info("COT is weekly/delayed. Missing data is never fabricated.")
    if not FRED_API_KEY:
        st.warning("FRED_API_KEY is not configured.")
    st.warning("Research tool only. No model guarantees profit.")
