import os
from datetime import datetime, timezone
import requests, numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Forex AI Pro Analyzer V7", page_icon="📈", layout="wide")

TD = "https://api.twelvedata.com"
CFTC = "https://publicreporting.cftc.gov/resource/yw9f-hn96.json"
CAL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

PAIRS = ["EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","USD/CAD","NZD/USD"]
CCY = {p: tuple(p.split("/")) for p in PAIRS}
COT = {"EUR":"EURO FX","GBP":"BRITISH POUND STERLING","JPY":"JAPANESE YEN",
       "CHF":"SWISS FRANC","AUD":"AUSTRALIAN DOLLAR","CAD":"CANADIAN DOLLAR",
       "NZD":"NEW ZEALAND DOLLAR"}

def secret(name, default=""):
    try:
        v = st.secrets.get(name, default)
        return v if v else default
    except Exception:
        return os.getenv(name, default)

TD_KEY = secret("TWELVE_DATA_API_KEY")
FRED_KEY = secret("FRED_API_KEY")

def td_symbol(pair):
    a,b = pair.split("/")
    return f"{a.upper()}/{b.upper()}"

def td(endpoint, params):
    if not TD_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing from Streamlit Secrets.")
    p = dict(params); p["apikey"] = TD_KEY
    r = requests.get(f"{TD}/{endpoint}", params=p, timeout=30)
    try: j = r.json()
    except Exception: j = {}
    if r.status_code >= 400:
        raise RuntimeError(f"Twelve Data HTTP {r.status_code}: {j.get('message', r.text[:250])}")
    if isinstance(j,dict) and j.get("status") == "error":
        raise RuntimeError(f"Twelve Data: {j.get('message','API error')}")
    return j

@st.cache_data(ttl=30)
def candles(pair, tf):
    j = td("time_series", {"symbol":td_symbol(pair),"interval":tf,"outputsize":500,"timezone":"UTC"})
    v = j.get("values")
    if not v: raise RuntimeError(f"No candle data for {td_symbol(pair)} ({tf}).")
    x = pd.DataFrame(v)
    x["datetime"] = pd.to_datetime(x["datetime"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in x: x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["open","high","low","close"]).sort_values("datetime").reset_index(drop=True)
    if len(x) < 60: raise RuntimeError(f"Only {len(x)} candles returned for {td_symbol(pair)} {tf}.")
    return x

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); u=d.clip(lower=0); dn=-d.clip(upper=0)
    a=u.ewm(alpha=1/n,adjust=False).mean(); b=dn.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+a/b.replace(0,np.nan))
def tr(x):
    p=x.close.shift()
    return pd.concat([x.high-x.low,(x.high-p).abs(),(x.low-p).abs()],axis=1).max(axis=1)
def atr(x,n=14): return tr(x).ewm(alpha=1/n,adjust=False).mean()
def macd(s):
    m=ema(s,12)-ema(s,26); q=ema(m,9); return m,q,m-q
def adx(x,n=14):
    up=x.high.diff(); dn=-x.low.diff()
    p=pd.Series(np.where((up>dn)&(up>0),up,0),index=x.index)
    m=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=x.index)
    a=tr(x).ewm(alpha=1/n,adjust=False).mean()
    pdi=100*p.ewm(alpha=1/n,adjust=False).mean()/a
    mdi=100*m.ewm(alpha=1/n,adjust=False).mean()/a
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean(),pdi,mdi

def add(x):
    x=x.copy()
    for n in [20,50,100,200]: x[f"ema{n}"]=ema(x.close,n)
    x["rsi"]=rsi(x.close)
    x["macd"],x["macd_signal"],x["macd_hist"]=macd(x.close)
    x["adx"],x["pdi"],x["mdi"]=adx(x)
    lo=x.low.rolling(14).min(); hi=x.high.rolling(14).max()
    x["stoch_k"]=100*(x.close-lo)/(hi-lo).replace(0,np.nan)
    x["stoch_d"]=x.stoch_k.rolling(3).mean()
    x["roc"]=x.close.pct_change(12)*100; x["atr"]=atr(x)
    x["bb_mid"]=x.close.rolling(20).mean(); sd=x.close.rolling(20).std()
    x["bb_up"]=x.bb_mid+2*sd; x["bb_low"]=x.bb_mid-2*sd
    return x.dropna().reset_index(drop=True)

def swings(x,w=3):
    h=[]; l=[]
    for i in range(w,len(x)-w):
        if x.high.iloc[i]>=x.high.iloc[i-w:i+w+1].max(): h.append(float(x.high.iloc[i]))
        if x.low.iloc[i]<=x.low.iloc[i-w:i+w+1].min(): l.append(float(x.low.iloc[i]))
    return h,l

def structure(x):
    h,l=swings(x); t="RANGE"; b="None"
    if len(h)>1 and len(l)>1:
        if h[-1]>h[-2] and l[-1]>l[-2]: t="BULLISH"
        elif h[-1]<h[-2] and l[-1]<l[-2]: t="BEARISH"
    if h and x.close.iloc[-1]>h[-1]: b="Bullish BOS"
    elif l and x.close.iloc[-1]<l[-1]: b="Bearish BOS"
    return t,b,h,l

def patterns(x):
    a,p=x.iloc[-1],x.iloc[-2]; body=abs(a.close-a.open); rng=max(a.high-a.low,1e-12)
    u=a.high-max(a.open,a.close); lo=min(a.open,a.close)-a.low; out=[]
    if body/rng<.12: out.append("Doji")
    if lo>2*body and u<1.2*body: out.append("Bullish pin")
    if u>2*body and lo<1.2*body: out.append("Bearish pin")
    if a.close>a.open and p.close<p.open and a.close>=p.open and a.open<=p.close: out.append("Bullish engulfing")
    if a.close<a.open and p.close>p.open and a.open>=p.close and a.close<=p.open: out.append("Bearish engulfing")
    return out

def levels(x):
    d=x.tail(150); h,l=swings(d)
    support=min(l,default=float(d.low.min())); resistance=max(h,default=float(d.high.max()))
    H=float(d.high.max()); L=float(d.low.min()); R=H-L
    fib={str(p):H-p*R for p in [0,.236,.382,.5,.618,.786,1]}
    a=float(d.atr.iloc[-1])
    eqh=len(h)>1 and abs(h[-1]-h[-2])<=.25*a
    eql=len(l)>1 and abs(l[-1]-l[-2])<=.25*a
    fvg=[]
    for i in range(max(2,len(d)-80),len(d)):
        if d.low.iloc[i]>d.high.iloc[i-2]:
            fvg.append(("Bullish FVG",float(d.high.iloc[i-2]),float(d.low.iloc[i])))
        elif d.high.iloc[i]<d.low.iloc[i-2]:
            fvg.append(("Bearish FVG",float(d.high.iloc[i]),float(d.low.iloc[i-2])))
    ob="None"
    if len(d)>=3:
        p,z=d.iloc[-2],d.iloc[-1]
        if p.close<p.open and z.close>p.high: ob="Bullish order-block proxy"
        elif p.close>p.open and z.close<p.low: ob="Bearish order-block proxy"
    return support,resistance,fib,eqh,eql,fvg[-3:],ob

@st.cache_data(ttl=21600)
def cot_raw():
    r=requests.get(CFTC,params={"$limit":5000,"$order":"report_date_as_yyyy_mm_dd DESC"},timeout=30)
    r.raise_for_status(); return pd.DataFrame(r.json())

def cot(c):
    try:
        d=cot_raw()
        nc=next((z for z in d.columns if z.lower()=="contract_market_name"),None)
        if not nc: return None,"COT contract field unavailable"
        q=d[d[nc].astype(str).str.upper().str.contains(COT[c],regex=False,na=False)].copy()
        lc=next((z for z in q.columns if "leveraged_money_long" in z.lower()),None)
        sc=next((z for z in q.columns if "leveraged_money_short" in z.lower()),None)
        if not lc or not sc: return None,"COT position fields unavailable"
        q[lc]=pd.to_numeric(q[lc],errors="coerce"); q[sc]=pd.to_numeric(q[sc],errors="coerce")
        q=q.assign(net=q[lc]-q[sc]).dropna(subset=["net"])
        if q.empty: return None,"No COT data"
        n=float(q.iloc[0].net); return float(np.tanh(n/100000)*10),f"net {n:,.0f} (weekly)"
    except Exception as e: return None,f"COT unavailable: {e}"

@st.cache_data(ttl=1800)
def calendar_data():
    r=requests.get(CAL,timeout=20); r.raise_for_status(); return pd.DataFrame(r.json())

def events(pair):
    try:
        d=calendar_data(); base,quote=CCY[pair]
        cc=next((c for c in d.columns if c.lower() in ["country","currency"]),None)
        dc=next((c for c in d.columns if c.lower() in ["date","datetime"]),None)
        if not cc or not dc: return 0,"Calendar schema changed"
        d[dc]=pd.to_datetime(d[dc],utc=True,errors="coerce")
        q=d[d[cc].astype(str).str.upper().isin([base,quote])&(d[dc]>=pd.Timestamp.now(tz="UTC"))].sort_values(dc)
        if q.empty:return 0,"No upcoming pair events"
        z=q.iloc[0]; impact=str(z.get("impact","")).lower()
        adj={"high":-12,"medium":-5,"low":-1}.get(impact,0)
        title=str(z.get("title",z.get("event","Economic event")))
        return adj,f"{title} | {z[dc].strftime('%d %b %H:%M UTC')} | {impact or 'unknown'}"
    except Exception as e:return 0,f"Calendar unavailable: {e}"

def rates(pair):
    return 0,"Verified policy-rate differential unavailable"

def tech(x):
    a=x.iloc[-1]; s=0; notes=[]
    if a.ema20>a.ema50>a.ema100>a.ema200: s+=20; notes.append("EMA bullish")
    elif a.ema20<a.ema50<a.ema100<a.ema200: s-=20; notes.append("EMA bearish")
    s+=6 if a.close>a.ema200 else -6
    if a.rsi>=55: s+=10; notes.append(f"RSI {a.rsi:.1f} bullish")
    elif a.rsi<=45: s-=10; notes.append(f"RSI {a.rsi:.1f} bearish")
    s+=10 if a.macd_hist>0 else -10
    if a.adx>=25: s+=8 if a.pdi>a.mdi else -8
    s+=4 if a.stoch_k>a.stoch_d and a.stoch_k<80 else -4 if a.stoch_k<a.stoch_d and a.stoch_k>20 else 0
    s+=4 if a.roc>0 else -4 if a.roc<0 else 0
    s+=4 if a.close>a.bb_mid else -4
    return s,notes

def analyze(pair,threshold):
    fs={tf:add(candles(pair,tf)) for tf in ["15min","1h","4h"]}
    w={"15min":1,"1h":2,"4h":3}; details=[]; parts=[]
    for tf in w:
        s,n=tech(fs[tf]); parts.append(s*w[tf]); details.append((tf,s,n))
    technical=sum(parts)/6
    trend,bos,_,_=structure(fs["1h"])
    stc=10 if trend=="BULLISH" else -10 if trend=="BEARISH" else 0
    stc+=8 if bos=="Bullish BOS" else -8 if bos=="Bearish BOS" else 0
    pats=patterns(fs["15min"])
    stc+=sum(3 if "Bullish" in p else -3 if "Bearish" in p else 0 for p in pats)
    base,quote=CCY[pair]; cb,nb=cot(base); cq,nq=cot(quote)
    cs=(cb or 0)-(cq or 0); rs,rn=rates(pair); ev,en=events(pair)
    total=technical+stc+cs+rs+ev
    conf=int(np.clip(50+abs(total)*.55,50,95))
    sig="BUY" if total>=18 and conf>=threshold else "SELL" if total<=-18 and conf>=threshold else "NO TRADE"
    x=fs["15min"]; price=float(x.close.iloc[-1]); aa=float(x.atr.iloc[-1])
    if sig=="BUY":
        sl=min(price-1.5*aa,float(x.low.tail(60).min())); risk=price-sl; tp1=price+1.5*risk; tp2=price+2.2*risk
    elif sig=="SELL":
        sl=max(price+1.5*aa,float(x.high.tail(60).max())); risk=sl-price; tp1=price-1.5*risk; tp2=price-2.2*risk
    else: sl=tp1=tp2=np.nan
    lev=levels(x)
    status={"Twelve Data":True,"CFTC COT":cb is not None or cq is not None,
            "Economic calendar":not en.startswith("Calendar unavailable"),"Interest rates":False}
    return locals()

st.title("📈 Forex AI Pro Analyzer V7")
st.caption("Readable multi-timeframe Forex analysis — technicals, structure, liquidity, FVG, news and data-source status.")

with st.sidebar:
    st.header("⚙️ Settings")
    pair=st.selectbox("Forex pair",PAIRS)
    threshold=st.slider("Minimum confidence",50,90,65)
    if st.button("🔄 Refresh data"):
        st.cache_data.clear(); st.rerun()

if not TD_KEY:
    st.error("Add TWELVE_DATA_API_KEY to Streamlit Secrets."); st.stop()

try:
    a=analyze(pair,threshold)
except Exception as e:
    st.error(f"Analysis failed: {e}"); st.stop()

try:
    q=td("quote",{"symbol":td_symbol(pair)})
except Exception:
    q={}

st.subheader(f"📡 {pair} — Live Market")
c=st.columns(5)
c[0].metric("Price",q.get("close",f'{a["price"]:.5f}'))
c[1].metric("Open",q.get("open","—"))
c[2].metric("Previous close",q.get("previous_close","—"))
c[3].metric("Signal",a["sig"])
c[4].metric("Confidence",f'{a["conf"]}/100')

st.subheader("🎯 Final Market Decision")
if a["sig"]=="BUY": st.success("🟢 BUY")
elif a["sig"]=="SELL": st.error("🔴 SELL")
else: st.warning("🟡 NO TRADE")
cc=st.columns(4)
cc[0].metric("Overall score",f'{a["total"]:.1f}')
cc[1].metric("Technical",f'{a["technical"]:.1f}')
cc[2].metric("Structure",f'{a["stc"]:.1f}')
cc[3].metric("Event risk",f'{a["ev"]:.1f}')

st.subheader("🕒 Multi-Timeframe Bias")
rows=[]
for tf,s,n in a["details"]:
    direction="BULLISH" if s>5 else "BEARISH" if s<-5 else "NEUTRAL"
    rows.append({"Timeframe":tf,"Technical score":round(s,1),"Bias":direction,"Key evidence":", ".join(n) or "Mixed"})
st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

st.subheader("📊 Indicator Dashboard")
for tf in ["15min","1h","4h"]:
    x=a["fs"][tf].iloc[-1]
    st.markdown(f"### {tf}")
    c=st.columns(4)
    c[0].markdown(f"**Trend**\n\nEMA20 `{x.ema20:.5f}`  \nEMA50 `{x.ema50:.5f}`  \nEMA100 `{x.ema100:.5f}`  \nEMA200 `{x.ema200:.5f}`")
    c[1].markdown(f"**Momentum**\n\nRSI `{x.rsi:.1f}`  \nMACD `{x.macd:.5f}`  \nSignal `{x.macd_signal:.5f}`  \nHistogram `{x.macd_hist:.5f}`  \nStoch K/D `{x.stoch_k:.1f}/{x.stoch_d:.1f}`  \nROC `{x.roc:.2f}%`")
    c[2].markdown(f"**Strength / Volatility**\n\nADX `{x.adx:.1f}`  \n+DI/-DI `{x.pdi:.1f}/{x.mdi:.1f}`  \nATR `{x.atr:.5f}`")
    c[3].markdown(f"**Bollinger Bands**\n\nUpper `{x.bb_up:.5f}`  \nMiddle `{x.bb_mid:.5f}`  \nLower `{x.bb_low:.5f}`")

st.subheader("🧠 Smart Money & Price Action")
c=st.columns(4)
c[0].metric("1H Structure",a["trend"])
c[1].metric("BOS",a["bos"])
c[2].metric("Equal highs",str(a["lev"][3]))
c[3].metric("Equal lows",str(a["lev"][4]))
st.write("**Candlestick patterns:**",", ".join(a["pats"]) or "None")
st.write("**Order block:**",a["lev"][6])
st.write("**FVG:**",a["lev"][5] or "None detected")
st.write("**Support:**",a["lev"][0],"| **Resistance:**",a["lev"][1])

st.subheader("🌍 Fundamental & Sentiment")
c=st.columns(4)
for col,(name,ok) in zip(c,a["status"].items()):
    col.metric(name,"AVAILABLE" if ok else "UNAVAILABLE")
st.write("**Next economic event:**",a["en"])
st.write("**COT base:**",a["nb"])
st.write("**COT quote:**",a["nq"])
st.write("**Interest rates:**",a["rn"])

st.subheader("🎯 Trade Planning")
if a["sig"]!="NO TRADE":
    c=st.columns(4)
    c[0].metric("Entry / current price",f'{a["price"]:.5f}')
    c[1].metric("Stop Loss",f'{a["sl"]:.5f}')
    c[2].metric("TP1",f'{a["tp1"]:.5f}')
    c[3].metric("TP2",f'{a["tp2"]:.5f}')
else:
    st.info("No trade plan generated because the setup did not meet the confidence threshold.")

with st.expander("🔎 Data & methodology notes"):
    st.write("Timeframes: 15min, 1H, 4H. Technical weighting: 15min=1, 1H=2, 4H=3.")
    st.write("Indicators: EMA20/50/100/200, RSI, MACD + signal + histogram, ADX, +DI/-DI, Stochastic K/D, ROC, ATR and Bollinger Bands.")
    st.write("Price action: swing structure, BOS, candlestick patterns, FVG, equal highs/lows, support/resistance and Fibonacci levels.")
    st.write("COT is weekly/delayed and may be unavailable depending on the public dataset schema.")
    st.write("Interest-rate scoring is deliberately disabled until verified feeds for each currency are configured.")
    st.warning("Research/analysis tool only. No system guarantees profitable trades.")

st.caption(f"Last dashboard time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
