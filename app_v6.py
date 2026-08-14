import os
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forex AI Pro Analyzer v6", page_icon="📈", layout="wide")

TD = "https://api.twelvedata.com"
CFTC = "https://publicreporting.cftc.gov/resource/yw9f-hn96.json"
CAL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
CCY = {p: tuple(p.split("/")) for p in PAIRS}
COT_NAMES = {
    "EUR": "EURO FX", "GBP": "BRITISH POUND STERLING", "JPY": "JAPANESE YEN",
    "CHF": "SWISS FRANC", "AUD": "AUSTRALIAN DOLLAR", "CAD": "CANADIAN DOLLAR",
    "NZD": "NEW ZEALAND DOLLAR"
}

def sec(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)

KEY = sec("TWELVE_DATA_API_KEY")
FRED = sec("FRED_API_KEY")

@st.cache_data(ttl=30)
def candles(symbol, interval, outputsize=500):
    if not KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing.")
    r = requests.get(
        f"{TD}/time_series",
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": KEY,
            "timezone": "UTC",
        },
        timeout=20,
    )
    try:
        data = r.json()
    except Exception:
        data = {}
    if r.status_code != 200 or "values" not in data:
        raise RuntimeError(f"Twelve Data: {data.get('message', 'request failed')}")
    x = pd.DataFrame(data["values"])
    x["datetime"] = pd.to_datetime(x["datetime"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in x:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.sort_values("datetime").reset_index(drop=True)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    a = up.ewm(alpha=1 / n, adjust=False).mean()
    b = down.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + a / b.replace(0, np.nan))

def true_range(x):
    prev = x.close.shift()
    return pd.concat(
        [x.high - x.low, (x.high - prev).abs(), (x.low - prev).abs()], axis=1
    ).max(axis=1)

def atr(x, n=14):
    return true_range(x).ewm(alpha=1 / n, adjust=False).mean()

def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal

def adx(x, n=14):
    up = x.high.diff()
    down = -x.low.diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0), index=x.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0), index=x.index)
    trn = true_range(x).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / trn
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / trn
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi

def add_indicators(x):
    x = x.copy()
    for n in [20, 50, 100, 200]:
        x[f"ema{n}"] = ema(x.close, n)
    x["rsi"] = rsi(x.close)
    x["macd"], x["macd_signal"], x["macd_hist"] = macd(x.close)
    x["adx"], x["pdi"], x["mdi"] = adx(x)
    lo = x.low.rolling(14).min()
    hi = x.high.rolling(14).max()
    x["stoch_k"] = 100 * (x.close - lo) / (hi - lo).replace(0, np.nan)
    x["stoch_d"] = x.stoch_k.rolling(3).mean()
    x["roc"] = x.close.pct_change(12) * 100
    x["atr"] = atr(x)
    x["bb_mid"] = x.close.rolling(20).mean()
    sd = x.close.rolling(20).std()
    x["bb_up"] = x.bb_mid + 2 * sd
    x["bb_low"] = x.bb_mid - 2 * sd
    return x.dropna().reset_index(drop=True)

def swings(x, w=3):
    highs, lows = [], []
    for i in range(w, len(x) - w):
        if x.high.iloc[i] >= x.high.iloc[i - w : i + w + 1].max():
            highs.append(float(x.high.iloc[i]))
        if x.low.iloc[i] <= x.low.iloc[i - w : i + w + 1].min():
            lows.append(float(x.low.iloc[i]))
    return highs, lows

def structure(x):
    highs, lows = swings(x)
    trend = "RANGE"
    bos = "None"
    if len(highs) > 1 and len(lows) > 1:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            trend = "BULLISH"
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            trend = "BEARISH"
    if highs and x.close.iloc[-1] > highs[-1]:
        bos = "Bullish BOS"
    elif lows and x.close.iloc[-1] < lows[-1]:
        bos = "Bearish BOS"
    return trend, bos, highs, lows

def patterns(x):
    a, p = x.iloc[-1], x.iloc[-2]
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
    if (
        a.close > a.open and p.close < p.open
        and a.close >= p.open and a.open <= p.close
    ):
        out.append("Bullish engulfing")
    if (
        a.close < a.open and p.close > p.open
        and a.open >= p.close and a.close <= p.open
    ):
        out.append("Bearish engulfing")
    return out

def levels(x):
    d = x.tail(150)
    highs, lows = swings(d)
    support = min(lows, default=float(d.low.min()))
    resistance = max(highs, default=float(d.high.max()))
    H, L = float(d.high.max()), float(d.low.min())
    R = H - L
    fib = {str(p): H - p * R for p in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]}
    current_atr = float(d.atr.iloc[-1])
    equal_highs = len(highs) > 1 and abs(highs[-1] - highs[-2]) <= 0.25 * current_atr
    equal_lows = len(lows) > 1 and abs(lows[-1] - lows[-2]) <= 0.25 * current_atr
    fvg = []
    for i in range(max(2, len(d) - 80), len(d)):
        if d.low.iloc[i] > d.high.iloc[i - 2]:
            fvg.append(("Bullish FVG", float(d.high.iloc[i - 2]), float(d.low.iloc[i])))
        elif d.high.iloc[i] < d.low.iloc[i - 2]:
            fvg.append(("Bearish FVG", float(d.high.iloc[i]), float(d.low.iloc[i - 2])))
    order_block = "None"
    if len(d) >= 3:
        p, z = d.iloc[-2], d.iloc[-1]
        if p.close < p.open and z.close > p.high:
            order_block = "Bullish order-block proxy"
        elif p.close > p.open and z.close < p.low:
            order_block = "Bearish order-block proxy"
    return support, resistance, fib, equal_highs, equal_lows, fvg[-3:], order_block

def tech_score(x):
    a = x.iloc[-1]
    score = 0
    notes = []
    if a.ema20 > a.ema50 > a.ema100 > a.ema200:
        score += 20
        notes.append("EMA bullish")
    elif a.ema20 < a.ema50 < a.ema100 < a.ema200:
        score -= 20
        notes.append("EMA bearish")
    score += 6 if a.close > a.ema200 else -6
    if a.rsi >= 55:
        score += 10
        notes.append(f"RSI {a.rsi:.1f} bullish")
    elif a.rsi <= 45:
        score -= 10
        notes.append(f"RSI {a.rsi:.1f} bearish")
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

@st.cache_data(ttl=21600)
def cot_raw():
    r = requests.get(
        CFTC,
        params={"$limit": 5000, "$order": "report_date_as_yyyy_mm_dd DESC"},
        timeout=30,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json())

def cot(currency):
    try:
        d = cot_raw()
        name = COT_NAMES.get(currency)
        contract_col = next(
            (c for c in d.columns if c.lower() == "contract_market_name"), None
        )
        if not contract_col or not name:
            return None, "COT contract field unavailable"
        q = d[
            d[contract_col].astype(str).str.upper().str.contains(
                name, regex=False, na=False
            )
        ].copy()
        long_col = next(
            (c for c in q.columns if "leveraged_money_long" in c.lower()), None
        )
        short_col = next(
            (c for c in q.columns if "leveraged_money_short" in c.lower()), None
        )
        if not long_col or not short_col:
            return None, "COT position fields unavailable"
        q[long_col] = pd.to_numeric(q[long_col], errors="coerce")
        q[short_col] = pd.to_numeric(q[short_col], errors="coerce")
        q["net"] = q[long_col] - q[short_col]
        q = q.dropna(subset=["net"])
        if q.empty:
            return None, "No COT data"
        net = float(q.iloc[0].net)
        return float(np.tanh(net / 100000) * 10), f"net {net:,.0f} (weekly)"
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
        country_col = next(
            (c for c in d.columns if c.lower() in ["country", "currency"]), None
        )
        date_col = next(
            (c for c in d.columns if c.lower() in ["date", "datetime"]), None
        )
        if not country_col or not date_col:
            return 0, "Calendar schema changed"
        d[date_col] = pd.to_datetime(d[date_col], utc=True, errors="coerce")
        q = d[
            d[country_col].astype(str).str.upper().isin([base, quote])
            & (d[date_col] >= pd.Timestamp.now(tz="UTC"))
        ].sort_values(date_col)
        if q.empty:
            return 0, "No upcoming pair events"
        z = q.iloc[0]
        impact = str(z.get("impact", "")).lower()
        adjustment = {"high": -12, "medium": -5, "low": -1}.get(impact, 0)
        title = str(z.get("title", z.get("event", "Economic event")))
        return adjustment, f"{title} | {z[date_col].strftime('%d %b %H:%M UTC')} | {impact or 'unknown'}"
    except Exception as e:
        return 0, f"Calendar unavailable: {e}"

@st.cache_data(ttl=21600)
def fred(series):
    if not FRED:
        return None
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series,
            "api_key": FRED,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 10,
        },
        timeout=20,
    )
    r.raise_for_status()
    values = [
        z for z in r.json().get("observations", [])
        if z.get("value") not in [".", None]
    ]
    return float(values[0]["value"]) if values else None

def rates(pair):
    # Deliberately conservative: the old implementation only had a verified
    # U.S. DFF series, so this version does not invent non-US rates.
    base, quote = CCY[pair]
    if base == "USD" and quote != "USD":
        usd = fred("DFF")
        return 0, "USD rate available, but quote-country official rate feed is not configured"
    if quote == "USD" and base != "USD":
        usd = fred("DFF")
        return 0, "USD rate available, but base-country official rate feed is not configured"
    return 0, "Interest-rate differential unavailable"

def session_name():
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        return "ASIA"
    if 8 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 21:
        return "NEW YORK"
    return "LATE / QUIET"

def session_stats(x):
    now = pd.Timestamp.now(tz="UTC")
    day = now.normalize()
    d = x[x.datetime >= day]
    if d.empty:
        return None
    return {
        "high": float(d.high.max()),
        "low": float(d.low.min()),
        "open": float(d.open.iloc[0]),
    }

def daily_context(x):
    if len(x) < 2:
        return "Insufficient data"
    now = x.iloc[-1]
    prev_day = x[x.datetime.dt.date < now.datetime.date()]
    if prev_day.empty:
        return "Previous-day range unavailable"
    ph = float(prev_day.high.max())
    pl = float(prev_day.low.min())
    price = float(now.close)
    if price > ph:
        return "Above previous-day high"
    if price < pl:
        return "Below previous-day low"
    return "Inside previous-day range"

def live_quote(pair):
    # Twelve Data's time_series endpoint is used rather than assuming that a
    # separate quote endpoint is enabled on every account.
    x = candles(pair.replace("/", ""), "1min", 5)
    last = x.iloc[-1]
    price = float(last.close)
    previous = float(x.close.iloc[-2]) if len(x) > 1 else price
    change = (price / previous - 1) * 100 if previous else 0
    return price, change, last.datetime

def analyze(pair, threshold):
    frames = {
        tf: add_indicators(candles(pair.replace("/", ""), tf))
        for tf in ["15min", "1h", "4h"]
    }
    weights = {"15min": 1, "1h": 2, "4h": 3}
    details, weighted = [], []
    for tf, weight in weights.items():
        s, notes = tech_score(frames[tf])
        weighted.append(s * weight)
        details.append(f"{tf}: {', '.join(notes)}")
    technical = sum(weighted) / 6

    trend, bos, _, _ = structure(frames["1h"])
    structure_score = 10 if trend == "BULLISH" else -10 if trend == "BEARISH" else 0
    structure_score += 8 if bos == "Bullish BOS" else -8 if bos == "Bearish BOS" else 0

    pats = patterns(frames["15min"])
    structure_score += sum(
        3 if "Bullish" in p else -3 if "Bearish" in p else 0 for p in pats
    )

    base, quote = CCY[pair]
    cb, cb_note = cot(base)
    cq, cq_note = cot(quote)
    cot_score = (cb or 0) - (cq or 0)

    rate_score, rate_note = rates(pair)
    event_score, event_note = events(pair)

    total = technical + structure_score + cot_score + rate_score + event_score
    confidence = int(np.clip(50 + abs(total) * 0.55, 50, 95))

    signal = (
        "BUY" if total >= 18 and confidence >= threshold
        else "SELL" if total <= -18 and confidence >= threshold
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

    return {
        "frames": frames, "technical": technical, "structure_score": structure_score,
        "cot_score": cot_score, "rate_score": rate_score, "event_score": event_score,
        "total": total, "confidence": confidence, "signal": signal, "price": price,
        "sl": sl, "tp1": tp1, "tp2": tp2, "details": details, "patterns": pats,
        "trend": trend, "bos": bos, "cb_note": cb_note, "cq_note": cq_note,
        "rate_note": rate_note, "event_note": event_note, "status": status,
        "levels": levels(x), "daily_context": daily_context(x),
        "session_stats": session_stats(x),
    }

st.title("📈 Forex AI Pro Analyzer v6")
st.caption("Multi-timeframe technical analysis with live panel, market context and conservative data-source handling.")

with st.sidebar:
    pair = st.selectbox("Forex pair", PAIRS)
    threshold = st.slider("Minimum confidence", 50, 90, 65)
    auto_refresh = st.checkbox("Auto-refresh live panel", True)
    if st.button("Refresh analysis"):
        st.cache_data.clear()
        st.rerun()

if not KEY:
    st.error("Add TWELVE_DATA_API_KEY in Streamlit Secrets.")
    st.stop()

# Live market panel
try:
    live_price, live_change, live_time = live_quote(pair)
    cols = st.columns(5)
    cols[0].metric(pair, f"{live_price:.5f}")
    cols[1].metric("1-min change", f"{live_change:+.3f}%")
    cols[2].metric("Session", session_name())
    cols[3].metric("Market status", "OPEN")
    cols[4].metric("Last update", live_time.strftime("%H:%M:%S UTC"))
except Exception as e:
    st.warning(f"Live market panel unavailable: {e}")

if auto_refresh:
    st.caption("Live values update when the Streamlit page reruns. Use the page refresh control if automatic reruns are not enabled by your Streamlit environment.")

try:
    result = analyze(pair, threshold)
except Exception as e:
    st.error(f"Analysis failed: {e}")
    st.stop()

cols = st.columns(4)
cols[0].metric("Signal", result["signal"])
cols[1].metric("Confidence", f'{result["confidence"]}/100')
cols[2].metric("Analysis price", f'{result["price"]:.5f}')
cols[3].metric("Score", f'{result["total"]:.1f}')

if result["signal"] != "NO TRADE":
    cols = st.columns(3)
    cols[0].metric("Stop Loss", f'{result["sl"]:.5f}')
    cols[1].metric("TP1", f'{result["tp1"]:.5f}')
    cols[2].metric("TP2", f'{result["tp2"]:.5f}')
else:
    st.warning("NO TRADE — evidence is below the selected confidence threshold.")

st.subheader("Market context")
cols = st.columns(3)
cols[0].metric("Session", session_name())
cols[1].metric("Daily context", result["daily_context"])
if result["session_stats"]:
    cols[2].metric(
        "Session range",
        f'{result["session_stats"]["low"]:.5f}–{result["session_stats"]["high"]:.5f}'
    )
else:
    cols[2].metric("Session range", "Unavailable")

st.subheader("Data-source status")
cols = st.columns(4)
for col, (name, ok) in zip(cols, result["status"].items()):
    col.metric(name, "AVAILABLE" if ok else "UNAVAILABLE")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Signal breakdown", "Indicators", "Levels & liquidity", "Data notes"]
)

with tab1:
    st.write(
        "Technical:", round(result["technical"], 1),
        "| Structure:", round(result["structure_score"], 1),
        "| COT:", round(result["cot_score"], 1),
        "| Rates:", round(result["rate_score"], 1),
        "| Event:", round(result["event_score"], 1),
    )
    st.write("Rate engine:", result["rate_note"])
    st.write("Next event:", result["event_note"])
    st.write("COT base:", result["cb_note"])
    st.write("COT quote:", result["cq_note"])
    st.write("Patterns:", ", ".join(result["patterns"]) or "None")
    for note in result["details"]:
        st.write("•", note)

with tab2:
    rows = []
    for tf, x in result["frames"].items():
        z = x.iloc[-1]
        rows.append({
            "TF": tf, "Close": z.close, "EMA20": z.ema20, "EMA50": z.ema50,
            "EMA100": z.ema100, "EMA200": z.ema200, "RSI": z.rsi,
            "MACD": z.macd_hist, "ADX": z.adx, "Stoch": z.stoch_k,
            "ROC": z.roc, "ATR": z.atr,
        })
    st.dataframe(pd.DataFrame(rows).round(5), use_container_width=True, hide_index=True)

with tab3:
    support, resistance, fib, equal_highs, equal_lows, fvg, order_block = result["levels"]
    st.write(
        "Support:", support, "| Resistance:", resistance,
        "| Equal highs:", equal_highs, "| Equal lows:", equal_lows
    )
    st.write("FVG:", fvg)
    st.write("Order block:", order_block)
    st.dataframe(
        pd.DataFrame({"Fib": list(fib), "Price": list(fib.values())}).round(5),
        use_container_width=True, hide_index=True
    )

with tab4:
    st.write("UTC:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    st.info("COT is weekly/delayed. Missing data is never fabricated.")
    st.info("Interest-rate differential is intentionally marked unavailable until verified official non-US rate feeds are configured.")
    st.warning("Research tool only. No model guarantees profit.")
