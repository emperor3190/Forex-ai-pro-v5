import os
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Forex AI Pro Analyzer v6
# Corrected Twelve Data FX symbols: EUR/USD, GBP/USD, etc.
# ============================================================

st.set_page_config(
    page_title="Forex AI Pro Analyzer v6",
    page_icon="📈",
    layout="wide",
)

TWELVE_DATA_URL = "https://api.twelvedata.com"
CFTC_URL = "https://publicreporting.cftc.gov/resource/yw9f-hn96.json"
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# IMPORTANT:
# Twelve Data expects standard FX symbols such as EUR/USD.
PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD",
]

CCY = {p: tuple(p.split("/")) for p in PAIRS}

COT = {
    "EUR": "EURO FX",
    "GBP": "BRITISH POUND STERLING",
    "JPY": "JAPANESE YEN",
    "CHF": "SWISS FRANC",
    "AUD": "AUSTRALIAN DOLLAR",
    "CAD": "CANADIAN DOLLAR",
    "NZD": "NEW ZEALAND DOLLAR",
}


def get_secret(name, default=""):
    try:
        value = st.secrets.get(name, default)
        return value if value else default
    except Exception:
        return os.getenv(name, default)


TWELVE_KEY = get_secret("TWELVE_DATA_API_KEY")
FRED_KEY = get_secret("FRED_API_KEY")


def twelve_symbol(pair):
    """Return Twelve Data's standard FX symbol format."""
    if "/" in pair:
        base, quote = pair.split("/", 1)
        return f"{base.upper()}/{quote.upper()}"
    return pair.upper()


def twelve_request(endpoint, params):
    """Call Twelve Data and turn API errors into useful messages."""
    if not TWELVE_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing from Streamlit Secrets.")

    p = dict(params)
    p["apikey"] = TWELVE_KEY

    r = requests.get(
        f"{TWELVE_DATA_URL}/{endpoint}",
        params=p,
        timeout=30,
    )

    # Twelve Data may return useful JSON even when HTTP status is not 200.
    try:
        data = r.json()
    except Exception:
        data = {}

    if r.status_code >= 400:
        message = data.get("message") or data.get("code") or r.text[:300]
        raise RuntimeError(
            f"Twelve Data HTTP {r.status_code}: {message}"
        )

    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(
            f"Twelve Data: {data.get('message', 'API error')}"
        )

    return data


@st.cache_data(ttl=30)
def candles(pair, timeframe):
    """Fetch OHLCV candles using the correct Twelve Data FX symbol."""
    symbol = twelve_symbol(pair)

    data = twelve_request(
        "time_series",
        {
            "symbol": symbol,
            "interval": timeframe,
            "outputsize": 500,
            "timezone": "UTC",
        },
    )

    values = data.get("values")
    if not values:
        raise RuntimeError(
            f"No candle data returned for {symbol} ({timeframe}). "
            "Check the Twelve Data plan and symbol."
        )

    x = pd.DataFrame(values)

    if "datetime" not in x.columns:
        raise RuntimeError("Twelve Data response has no datetime field.")

    x["datetime"] = pd.to_datetime(
        x["datetime"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close", "volume"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise RuntimeError(
            f"Missing Twelve Data candle fields: {', '.join(missing)}"
        )

    x = x.dropna(subset=required)
    x = x.sort_values("datetime").reset_index(drop=True)

    if len(x) < 60:
        raise RuntimeError(
            f"Only {len(x)} candles returned for {symbol} {timeframe}; "
            "not enough data for the indicators."
        )

    return x


def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(x):
    previous = x.close.shift()
    return pd.concat(
        [
            x.high - x.low,
            (x.high - previous).abs(),
            (x.low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(x, n=14):
    return true_range(x).ewm(alpha=1 / n, adjust=False).mean()


def macd(series):
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal


def adx(x, n=14):
    up = x.high.diff()
    down = -x.low.diff()

    plus = pd.Series(
        np.where((up > down) & (up > 0), up, 0),
        index=x.index,
    )
    minus = pd.Series(
        np.where((down > up) & (down > 0), down, 0),
        index=x.index,
    )

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

    low14 = x.low.rolling(14).min()
    high14 = x.high.rolling(14).max()
    x["stoch_k"] = (
        100 * (x.close - low14) /
        (high14 - low14).replace(0, np.nan)
    )
    x["stoch_d"] = x.stoch_k.rolling(3).mean()

    x["roc"] = x.close.pct_change(12) * 100
    x["atr"] = atr(x)

    x["bb_mid"] = x.close.rolling(20).mean()
    sd = x.close.rolling(20).std()
    x["bb_up"] = x.bb_mid + 2 * sd
    x["bb_low"] = x.bb_mid - 2 * sd

    return x.dropna().reset_index(drop=True)


def swings(x, window=3):
    highs = []
    lows = []

    for i in range(window, len(x) - window):
        if x.high.iloc[i] >= x.high.iloc[i-window:i+window+1].max():
            highs.append(float(x.high.iloc[i]))

        if x.low.iloc[i] <= x.low.iloc[i-window:i+window+1].min():
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
    if len(x) < 2:
        return []

    current = x.iloc[-1]
    previous = x.iloc[-2]

    body = abs(current.close - current.open)
    rng = max(current.high - current.low, 1e-12)
    upper = current.high - max(current.open, current.close)
    lower = min(current.open, current.close) - current.low

    found = []

    if body / rng < 0.12:
        found.append("Doji")

    if lower > 2 * body and upper < 1.2 * body:
        found.append("Bullish pin")

    if upper > 2 * body and lower < 1.2 * body:
        found.append("Bearish pin")

    if (
        current.close > current.open
        and previous.close < previous.open
        and current.close >= previous.open
        and current.open <= previous.close
    ):
        found.append("Bullish engulfing")

    if (
        current.close < current.open
        and previous.close > previous.open
        and current.open >= previous.close
        and current.close <= previous.open
    ):
        found.append("Bearish engulfing")

    return found


def levels(x):
    d = x.tail(150)
    highs, lows = swings(d)

    support = min(lows, default=float(d.low.min()))
    resistance = max(highs, default=float(d.high.max()))

    high = float(d.high.max())
    low = float(d.low.min())
    rng = high - low

    fib = {
        str(level): high - level * rng
        for level in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    }

    current_atr = float(d.atr.iloc[-1])

    equal_highs = (
        len(highs) > 1
        and abs(highs[-1] - highs[-2]) <= 0.25 * current_atr
    )
    equal_lows = (
        len(lows) > 1
        and abs(lows[-1] - lows[-2]) <= 0.25 * current_atr
    )

    fvg = []

    for i in range(max(2, len(d) - 80), len(d)):
        if d.low.iloc[i] > d.high.iloc[i - 2]:
            fvg.append(
                (
                    "Bullish FVG",
                    float(d.high.iloc[i - 2]),
                    float(d.low.iloc[i]),
                )
            )
        elif d.high.iloc[i] < d.low.iloc[i - 2]:
            fvg.append(
                (
                    "Bearish FVG",
                    float(d.high.iloc[i]),
                    float(d.low.iloc[i - 2]),
                )
            )

    order_block = "None"

    if len(d) >= 3:
        previous = d.iloc[-2]
        current = d.iloc[-1]

        if previous.close < previous.open and current.close > previous.high:
            order_block = "Bullish order-block proxy"
        elif previous.close > previous.open and current.close < previous.low:
            order_block = "Bearish order-block proxy"

    return (
        support,
        resistance,
        fib,
        equal_highs,
        equal_lows,
        fvg[-3:],
        order_block,
    )


@st.cache_data(ttl=21600)
def cot_raw():
    response = requests.get(
        CFTC_URL,
        params={
            "$limit": 5000,
            "$order": "report_date_as_yyyy_mm_dd DESC",
        },
        timeout=30,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())


def cot(currency):
    try:
        data = cot_raw()

        name_col = next(
            (
                c for c in data.columns
                if c.lower() == "contract_market_name"
            ),
            None,
        )

        if not name_col:
            return None, "COT contract field unavailable"

        query = data[
            data[name_col]
            .astype(str)
            .str.upper()
            .str.contains(
                COT[currency],
                regex=False,
                na=False,
            )
        ].copy()

        long_col = next(
            (
                c for c in query.columns
                if "leveraged_money_long" in c.lower()
            ),
            None,
        )

        short_col = next(
            (
                c for c in query.columns
                if "leveraged_money_short" in c.lower()
            ),
            None,
        )

        if not long_col or not short_col:
            return None, "COT position fields unavailable"

        query[long_col] = pd.to_numeric(
            query[long_col], errors="coerce"
        )
        query[short_col] = pd.to_numeric(
            query[short_col], errors="coerce"
        )

        query["net"] = query[long_col] - query[short_col]
        query = query.dropna(subset=["net"])

        if query.empty:
            return None, "No COT data"

        net = float(query.iloc[0]["net"])

        # Small normalized contribution to the total score.
        score = float(np.tanh(net / 100000) * 10)

        return score, f"net {net:,.0f} (weekly)"

    except Exception as exc:
        return None, f"COT unavailable: {exc}"


@st.cache_data(ttl=1800)
def calendar_data():
    response = requests.get(CAL_URL, timeout=20)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def events(pair):
    try:
        data = calendar_data()
        base, quote = CCY[pair]

        country_col = next(
            (
                c for c in data.columns
                if c.lower() in ["country", "currency"]
            ),
            None,
        )

        date_col = next(
            (
                c for c in data.columns
                if c.lower() in ["date", "datetime"]
            ),
            None,
        )

        if not country_col or not date_col:
            return 0, "Calendar schema changed"

        data[date_col] = pd.to_datetime(
            data[date_col],
            utc=True,
            errors="coerce",
        )

        filtered = data[
            data[country_col]
            .astype(str)
            .str.upper()
            .isin([base, quote])
            & (data[date_col] >= pd.Timestamp.now(tz="UTC"))
        ].sort_values(date_col)

        if filtered.empty:
            return 0, "No upcoming pair events"

        row = filtered.iloc[0]
        impact = str(row.get("impact", "")).lower()

        adjustment = {
            "high": -12,
            "medium": -5,
            "low": -1,
        }.get(impact, 0)

        title = str(
            row.get(
                "title",
                row.get("event", "Economic event"),
            )
        )

        return (
            adjustment,
            f"{title} | "
            f"{row[date_col].strftime('%d %b %H:%M UTC')} | "
            f"{impact or 'unknown'}",
        )

    except Exception as exc:
        return 0, f"Calendar unavailable: {exc}"


@st.cache_data(ttl=21600)
def fred(series):
    if not FRED_KEY:
        return None

    response = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series,
            "api_key": FRED_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 10,
        },
        timeout=20,
    )

    response.raise_for_status()

    observations = [
        item for item in response.json().get("observations", [])
        if item.get("value") not in [".", None]
    ]

    if not observations:
        return None

    return float(observations[0]["value"])


def rates(pair):
    """
    Conservative rate engine.
    DFF is only the U.S. effective federal funds rate.
    It is NOT a valid rate source for non-USD currencies.
    Therefore this function reports unavailable unless both
    sides have verified feeds.
    """
    base, quote = CCY[pair]

    if base == "USD" and quote == "USD":
        return 0, "Same-currency pair"

    # Keep the existing conservative behavior rather than
    # pretending DFF is the policy rate for EUR/GBP/JPY/etc.
    return 0, "Verified rate differential unavailable"


def technical_score(x):
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

    if a.roc > 0:
        score += 4
    elif a.roc < 0:
        score -= 4

    score += 4 if a.close > a.bb_mid else -4

    return score, notes


def analyze(pair, threshold):
    timeframes = ["15min", "1h", "4h"]

    frames = {
        tf: add_indicators(candles(pair, tf))
        for tf in timeframes
    }

    weights = {
        "15min": 1,
        "1h": 2,
        "4h": 3,
    }

    details = []
    parts = []

    for tf in weights:
        score, notes = technical_score(frames[tf])
        parts.append(score * weights[tf])
        details.append(
            f"{tf}: {', '.join(notes) if notes else 'Neutral'}"
        )

    technical = sum(parts) / 6

    trend, bos, _, _ = structure(frames["1h"])

    structure_score = (
        10 if trend == "BULLISH"
        else -10 if trend == "BEARISH"
        else 0
    )

    structure_score += (
        8 if bos == "Bullish BOS"
        else -8 if bos == "Bearish BOS"
        else 0
    )

    pats = patterns(frames["15min"])

    structure_score += sum(
        3 if "Bullish" in p
        else -3 if "Bearish" in p
        else 0
        for p in pats
    )

    base, quote = CCY[pair]

    base_cot, base_cot_note = cot(base)
    quote_cot, quote_cot_note = cot(quote)

    cot_score = (base_cot or 0) - (quote_cot or 0)

    rate_score, rate_note = rates(pair)
    event_score, event_note = events(pair)

    total = (
        technical
        + structure_score
        + cot_score
        + rate_score
        + event_score
    )

    confidence = int(
        np.clip(50 + abs(total) * 0.55, 50, 95)
    )

    if total >= 18 and confidence >= threshold:
        signal = "BUY"
    elif total <= -18 and confidence >= threshold:
        signal = "SELL"
    else:
        signal = "NO TRADE"

    current = frames["15min"]
    price = float(current.close.iloc[-1])
    current_atr = float(current.atr.iloc[-1])

    if signal == "BUY":
        stop_loss = min(
            price - 1.5 * current_atr,
            float(current.low.tail(60).min()),
        )
        risk = price - stop_loss
        tp1 = price + 1.5 * risk
        tp2 = price + 2.2 * risk

    elif signal == "SELL":
        stop_loss = max(
            price + 1.5 * current_atr,
            float(current.high.tail(60).max()),
        )
        risk = stop_loss - price
        tp1 = price - 1.5 * risk
        tp2 = price - 2.2 * risk

    else:
        stop_loss = np.nan
        tp1 = np.nan
        tp2 = np.nan

    level_data = levels(current)

    status = {
        "Twelve Data": True,
        "CFTC COT": base_cot is not None or quote_cot is not None,
        "Economic calendar": not event_note.startswith(
            "Calendar unavailable"
        ),
        "Interest rates": rate_score != 0,
    }

    return {
        "frames": frames,
        "technical": technical,
        "structure_score": structure_score,
        "cot_score": cot_score,
        "rate_score": rate_score,
        "event_score": event_score,
        "total": total,
        "confidence": confidence,
        "signal": signal,
        "price": price,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "trend": trend,
        "bos": bos,
        "patterns": pats,
        "details": details,
        "base_cot_note": base_cot_note,
        "quote_cot_note": quote_cot_note,
        "rate_note": rate_note,
        "event_note": event_note,
        "levels": level_data,
        "status": status,
    }


# ============================================================
# Dashboard
# ============================================================

st.title("📈 Forex AI Pro Analyzer v6")
st.caption(
    "Multi-timeframe technical analysis with live market data, "
    "structure, COT, economic-event risk and conservative rate handling."
)

with st.sidebar:
    st.header("Analysis settings")

    pair = st.selectbox(
        "Forex pair",
        PAIRS,
        index=0,
    )

    threshold = st.slider(
        "Minimum confidence",
        50,
        90,
        65,
    )

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.info(
        "Twelve Data FX symbols are sent as EUR/USD, GBP/USD, "
        "USD/JPY, etc."
    )

if not TWELVE_KEY:
    st.error(
        "Add TWELVE_DATA_API_KEY to Streamlit Secrets before running analysis."
    )
    st.stop()

# Live panel: this uses the same corrected symbol format.
st.subheader("📡 Live market panel")

try:
    live_data = twelve_request(
        "quote",
        {"symbol": twelve_symbol(pair)},
    )

    if live_data.get("status") == "error":
        raise RuntimeError(
            live_data.get("message", "Quote unavailable")
        )

    live_cols = st.columns(4)
    live_cols[0].metric(
        "Symbol",
        live_data.get("symbol", twelve_symbol(pair)),
    )
    live_cols[1].metric(
        "Price",
        live_data.get("close", live_data.get("price", "—")),
    )
    live_cols[2].metric(
        "Open",
        live_data.get("open", "—"),
    )
    live_cols[3].metric(
        "Previous close",
        live_data.get("previous_close", "—"),
    )

except Exception as exc:
    st.warning(f"Live market panel unavailable: {exc}")

try:
    result = analyze(pair, threshold)

except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.stop()


cols = st.columns(4)

cols[0].metric(
    "Signal",
    result["signal"],
)

cols[1].metric(
    "Confidence",
    f'{result["confidence"]}/100',
)

cols[2].metric(
    "Price",
    f'{result["price"]:.5f}',
)

cols[3].metric(
    "Score",
    f'{result["total"]:.1f}',
)

if result["signal"] != "NO TRADE":
    cols = st.columns(3)

    cols[0].metric(
        "Stop Loss",
        f'{result["stop_loss"]:.5f}',
    )

    cols[1].metric(
        "TP1",
        f'{result["tp1"]:.5f}',
    )

    cols[2].metric(
        "TP2",
        f'{result["tp2"]:.5f}',
    )

else:
    st.warning(
        "NO TRADE — evidence is below the selected confidence threshold."
    )


st.subheader("Data-source status")

status_cols = st.columns(4)

for col, (name, available) in zip(
    status_cols,
    result["status"].items(),
):
    col.metric(
        name,
        "AVAILABLE" if available else "UNAVAILABLE",
    )


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Signal breakdown",
        "Indicators",
        "Levels & liquidity",
        "Data notes",
    ]
)


with tab1:
    st.write(
        "Technical:",
        round(result["technical"], 1),
        "| Structure:",
        round(result["structure_score"], 1),
        "| COT:",
        round(result["cot_score"], 1),
        "| Rates:",
        round(result["rate_score"], 1),
        "| Event:",
        round(result["event_score"], 1),
    )

    st.write(
        "Rate engine:",
        result["rate_note"],
    )

    st.write(
        "Next event:",
        result["event_note"],
    )

    st.write(
        "COT base:",
        result["base_cot_note"],
    )

    st.write(
        "COT quote:",
        result["quote_cot_note"],
    )

    st.write(
        "1H structure:",
        result["trend"],
        "|",
        result["bos"],
    )

    st.write(
        "Patterns:",
        ", ".join(result["patterns"]) or "None",
    )

    for detail in result["details"]:
        st.write("•", detail)


with tab2:
    rows = []

    for tf, frame in result["frames"].items():
        row = frame.iloc[-1]

        rows.append(
            {
                "TF": tf,
                "Close": row.close,
                "EMA20": row.ema20,
                "EMA50": row.ema50,
                "EMA100": row.ema100,
                "EMA200": row.ema200,
                "RSI": row.rsi,
                "MACD": row.macd_hist,
                "ADX": row.adx,
                "Stoch": row.stoch_k,
                "ROC": row.roc,
                "ATR": row.atr,
            }
        )

    st.dataframe(
        pd.DataFrame(rows).round(5),
        use_container_width=True,
        hide_index=True,
    )


with tab3:
    (
        support,
        resistance,
        fib,
        equal_highs,
        equal_lows,
        fvg,
        order_block,
    ) = result["levels"]

    st.write(
        "Support:",
        support,
        "| Resistance:",
        resistance,
        "| Equal highs:",
        equal_highs,
        "| Equal lows:",
        equal_lows,
    )

    st.write("FVG:", fvg)
    st.write("Order block:", order_block)

    st.dataframe(
        pd.DataFrame(
            {
                "Fib": list(fib),
                "Price": list(fib.values()),
            }
        ).round(5),
        use_container_width=True,
        hide_index=True,
    )


with tab4:
    st.write(
        "UTC:",
        datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    st.info(
        "COT is weekly/delayed. Missing data is never fabricated."
    )

    st.info(
        "Interest-rate scoring remains conservative until "
        "verified policy-rate feeds are configured for each currency."
    )

    st.success(
        f"Twelve Data symbol used: {twelve_symbol(pair)}"
    )

    st.warning(
        "Research and analysis tool only. It does not guarantee "
        "profit or trading accuracy."
    )
