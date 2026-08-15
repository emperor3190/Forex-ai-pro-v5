import os
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forex AI Pro Analyzer V8", page_icon="📈", layout="wide")

TD_URL = "https://api.twelvedata.com"
CFTC_URL = "https://publicreporting.cftc.gov/resource/yw9f-hn96.json"
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

PAIRS = ["EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","USD/CAD","NZD/USD"]
CCY = {p: tuple(p.split("/")) for p in PAIRS}
COT = {
    "EUR":"EURO FX","GBP":"BRITISH POUND STERLING","JPY":"JAPANESE YEN",
    "CHF":"SWISS FRANC","AUD":"AUSTRALIAN DOLLAR","CAD":"CANADIAN DOLLAR",
    "NZD":"NEW ZEALAND DOLLAR"
}
FOREX_TFS = ["15min","1h","4h"]
TF_WEIGHT = {"15min":1,"1h":2,"4h":3}


def secret(name, default=""):
    try:
        value = st.secrets.get(name, default)
        return value if value else default
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
    r = requests.get(f"{TD_URL}/{endpoint}", params=p, timeout=30)
    try:
        data = r.json()
    except Exception:
        data = {}
    if r.status_code >= 400:
        raise RuntimeError(f"Twelve Data HTTP {r.status_code}: {data.get('message', r.text[:250])}")
    if isinstance(data,dict) and data.get("status") == "error":
        raise RuntimeError(f"Twelve Data: {data.get('message','API error')}")
    return data


@st.cache_data(ttl=30)
def candles(pair, timeframe, outputsize=500):
    data = td("time_series", {
        "symbol":td_symbol(pair),"interval":timeframe,
        "outputsize":int(outputsize),"timezone":"UTC"
    })
    values = data.get("values")
    if not values:
        raise RuntimeError(f"No candle data returned for {td_symbol(pair)} ({timeframe}).")
    x = pd.DataFrame(values)
    if "datetime" not in x.columns:
        raise RuntimeError("Twelve Data response has no datetime field.")
    x["datetime"] = pd.to_datetime(x["datetime"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in x.columns: x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["open","high","low","close"]).sort_values("datetime").reset_index(drop=True)
    if len(x) < 60:
        raise RuntimeError(f"Only {len(x)} candles returned for {td_symbol(pair)} {timeframe}; not enough data.")
    return x


def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def rsi(s,n=14):
    d=s.diff(); u=d.clip(lower=0); dn=-d.clip(upper=0)
    au=u.ewm(alpha=1/n,adjust=False).mean()
    ad=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=au/ad.replace(0,np.nan)
    return 100-100/(1+rs)

def tr(x):
    p=x.close.shift()
    return pd.concat([x.high-x.low,(x.high-p).abs(),(x.low-p).abs()],axis=1).max(axis=1)

def atr(x,n=14): return tr(x).ewm(alpha=1/n,adjust=False).mean()

def macd(s):
    m=ema(s,12)-ema(s,26); q=ema(m,9)
    return m,q,m-q

def adx(x,n=14):
    up=x.high.diff(); dn=-x.low.diff()
    p=pd.Series(np.where((up>dn)&(up>0),up,0),index=x.index)
    m=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=x.index)
    a=tr(x).ewm(alpha=1/n,adjust=False).mean()
    pdi=100*p.ewm(alpha=1/n,adjust=False).mean()/a
    mdi=100*m.ewm(alpha=1/n,adjust=False).mean()/a
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean(),pdi,mdi

def add_indicators(x):
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
    x["atr_pct"]=100*x.atr/x.close.replace(0,np.nan)
    x["range"]=x.high-x.low
    x["body"]=(x.close-x.open).abs()
    x["body_ratio"]=x.body/x.range.replace(0,np.nan)
    x["close_location"]=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    x["roc_delta"]=x.roc.diff()
    x["macd_hist_delta"]=x.macd_hist.diff()
    if "volume" in x.columns:
        x["volume_ma20"]=x.volume.rolling(20).mean()
        x["volume_ratio"]=x.volume/x.volume_ma20.replace(0,np.nan)
    else:
        x["volume_ma20"]=np.nan; x["volume_ratio"]=np.nan
    req=["ema200","rsi","macd_hist","adx","stoch_k","stoch_d","atr","bb_mid","bb_up","bb_low"]
    return x.dropna(subset=req).reset_index(drop=True)


def swings(x,w=3):
    h=[]; l=[]
    for i in range(w,len(x)-w):
        if x.high.iloc[i]>=x.high.iloc[i-w:i+w+1].max(): h.append((i,float(x.high.iloc[i])))
        if x.low.iloc[i]<=x.low.iloc[i-w:i+w+1].min(): l.append((i,float(x.low.iloc[i])))
    return h,l


def structure(x):
    h,l=swings(x); trend="RANGE"; bos="None"
    if len(h)>1 and len(l)>1:
        if h[-1][1]>h[-2][1] and l[-1][1]>l[-2][1]: trend="BULLISH"
        elif h[-1][1]<h[-2][1] and l[-1][1]<l[-2][1]: trend="BEARISH"
    close=float(x.close.iloc[-1])
    if h and close>h[-1][1]: bos="Bullish BOS"
    elif l and close<l[-1][1]: bos="Bearish BOS"
    return trend,bos,h,l


def choch_mss(x):
    h,l=swings(x)
    if len(h)<2 or len(l)<2: return "NONE",0
    close=float(x.close.iloc[-1])
    if close>h[-2][1] and h[-1][1]<=h[-2][1]: return "BULLISH CHOCH/MSS",8
    if close<l[-2][1] and l[-1][1]>=l[-2][1]: return "BEARISH CHOCH/MSS",-8
    return "NONE",0


def patterns(x):
    if len(x)<2: return []
    a,p=x.iloc[-1],x.iloc[-2]; body=abs(a.close-a.open); rng=max(a.high-a.low,1e-12)
    u=a.high-max(a.open,a.close); lo=min(a.open,a.close)-a.low; out=[]
    if body/rng<.12: out.append("Doji")
    if lo>2*body and u<1.2*body: out.append("Bullish pin")
    if u>2*body and lo<1.2*body: out.append("Bearish pin")
    if a.close>a.open and p.close<p.open and a.close>=p.open and a.open<=p.close: out.append("Bullish engulfing")
    if a.close<a.open and p.close>p.open and a.open>=p.close and a.close<=p.open: out.append("Bearish engulfing")
    return out


def liquidity_features(x):
    d=x.tail(180); h,l=swings(d); a=float(d.atr.iloc[-1])
    eqh=len(h)>1 and abs(h[-1][1]-h[-2][1])<=.25*a
    eql=len(l)>1 and abs(l[-1][1]-l[-2][1])<=.25*a
    price=float(d.close.iloc[-1]); sweep="NONE"; score=0
    if h and price<h[-1][1] and float(d.high.iloc[-1])>h[-1][1]:
        sweep="BEARISH LIQUIDITY SWEEP"; score-=7
    if l and price>l[-1][1] and float(d.low.iloc[-1])<l[-1][1]:
        sweep="BULLISH LIQUIDITY SWEEP"; score+=7
    return eqh,eql,sweep,score


def levels(x):
    d=x.tail(180); h,l=swings(d)
    support=min([v for _,v in l],default=float(d.low.min()))
    resistance=max([v for _,v in h],default=float(d.high.max()))
    H=float(d.high.max()); L=float(d.low.min()); R=H-L
    fib={str(p):H-p*R for p in [0,.236,.382,.5,.618,.786,1]}
    a=float(d.atr.iloc[-1])
    eqh=len(h)>1 and abs(h[-1][1]-h[-2][1])<=.25*a
    eql=len(l)>1 and abs(l[-1][1]-l[-2][1])<=.25*a
    fvg=[]
    for i in range(max(2,len(d)-100),len(d)):
        if d.low.iloc[i]>d.high.iloc[i-2]:
            fvg.append(("Bullish FVG",float(d.high.iloc[i-2]),float(d.low.iloc[i]),str(d.datetime.iloc[i])))
        elif d.high.iloc[i]<d.low.iloc[i-2]:
            fvg.append(("Bearish FVG",float(d.high.iloc[i]),float(d.low.iloc[i-2]),str(d.datetime.iloc[i])))
    ob="None"
    if len(d)>=3:
        p,z=d.iloc[-2],d.iloc[-1]
        if p.close<p.open and z.close>p.high: ob="Bullish order-block proxy"
        elif p.close>p.open and z.close<p.low: ob="Bearish order-block proxy"
    return support,resistance,fib,eqh,eql,fvg[-5:],ob


def displacement_score(x):
    a=x.iloc[-1]; body_atr=float(a.body)/max(float(a.atr),1e-12)
    score=7 if body_atr>=.8 and a.body_ratio>=.60 and a.close>a.open else -7 if body_atr>=.8 and a.body_ratio>=.60 else 0
    return score,f"body/ATR={body_atr:.2f}"


def momentum_engine(x):
    a,prev=x.iloc[-1],x.iloc[-2]; score=0; notes=[]
    if a.close>a.ema20>a.ema50: score+=8; notes.append("price above EMA20/50")
    elif a.close<a.ema20<a.ema50: score-=8; notes.append("price below EMA20/50")
    if a.macd_hist>0 and a.macd_hist_delta>0: score+=10; notes.append("bullish MACD acceleration")
    elif a.macd_hist<0 and a.macd_hist_delta<0: score-=10; notes.append("bearish MACD acceleration")
    if a.roc>0 and a.roc_delta>0: score+=7; notes.append("bullish ROC acceleration")
    elif a.roc<0 and a.roc_delta<0: score-=7; notes.append("bearish ROC acceleration")
    if a.adx>=20: score+=6 if a.pdi>a.mdi else -6
    if a.body_ratio>=.60: score+=5 if a.close>a.open else -5
    if a.close_location>=.75: score+=4
    elif a.close_location<=.25: score-=4
    if not np.isnan(a.volume_ratio) and a.volume_ratio>=1.3:
        score+=4 if a.close>a.open else -4; notes.append("volume expansion")
    prev_dir=np.sign(prev.close-prev.open); cur_dir=np.sign(a.close-a.open)
    if prev_dir<=0<cur_dir: score+=4; notes.append("bullish candle transition")
    elif prev_dir>=0>cur_dir: score-=4; notes.append("bearish candle transition")
    return score,notes


@st.cache_data(ttl=21600)
def cot_raw():
    r=requests.get(CFTC_URL,params={"$limit":5000,"$order":"report_date_as_yyyy_mm_dd DESC"},timeout=30)
    r.raise_for_status(); return pd.DataFrame(r.json())


def cot(currency):
    try:
        d=cot_raw()
        nc=next((z for z in d.columns if z.lower()=="contract_market_name"),None)
        if not nc: return None,"COT contract field unavailable"
        q=d[d[nc].astype(str).str.upper().str.contains(COT[currency],regex=False,na=False)].copy()
        lc=next((z for z in q.columns if "leveraged_money_long" in z.lower()),None)
        sc=next((z for z in q.columns if "leveraged_money_short" in z.lower()),None)
        if not lc or not sc: return None,"COT position fields unavailable"
        q[lc]=pd.to_numeric(q[lc],errors="coerce"); q[sc]=pd.to_numeric(q[sc],errors="coerce")
        q["net"]=q[lc]-q[sc]; q=q.dropna(subset=["net"])
        if q.empty: return None,"No COT data"
        n=float(q.iloc[0].net); return float(np.tanh(n/100000)*10),f"net {n:,.0f} (weekly)"
    except Exception as e: return None,f"COT unavailable: {e}"


@st.cache_data(ttl=1800)
def calendar_data():
    r=requests.get(CAL_URL,timeout=20); r.raise_for_status(); return pd.DataFrame(r.json())


def events(pair):
    try:
        d=calendar_data(); base,quote=CCY[pair]
        cc=next((c for c in d.columns if c.lower() in ["country","currency"]),None)
        dc=next((c for c in d.columns if c.lower() in ["date","datetime"]),None)
        if not cc or not dc: return 0,"Calendar schema changed"
        d[dc]=pd.to_datetime(d[dc],utc=True,errors="coerce")
        q=d[d[cc].astype(str).str.upper().isin([base,quote])&(d[dc]>=pd.Timestamp.now(tz="UTC"))].sort_values(dc)
        if q.empty: return 0,"No upcoming pair events"
        z=q.iloc[0]; impact=str(z.get("impact","")).lower()
        adj={"high":-12,"medium":-5,"low":-1}.get(impact,0)
        title=str(z.get("title",z.get("event","Economic event")))
        return adj,f"{title} | {z[dc].strftime('%d %b %H:%M UTC')} | {impact or 'unknown'}"
    except Exception as e: return 0,f"Calendar unavailable: {e}"


@st.cache_data(ttl=21600)
def fred(series):
    if not FRED_KEY: return None
    r=requests.get("https://api.stlouisfed.org/fred/series/observations",params={
        "series_id":series,"api_key":FRED_KEY,"file_type":"json",
        "sort_order":"desc","limit":10},timeout=20)
    r.raise_for_status()
    obs=[z for z in r.json().get("observations",[]) if z.get("value") not in [".",None]]
    return float(obs[0]["value"]) if obs else None


def rates(pair):
    return 0,"Verified policy-rate differential unavailable"


def technical_score(x):
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


def analyze_forex(pair,threshold):
    frames={tf:add_indicators(candles(pair,tf)) for tf in FOREX_TFS}
    details=[]; parts=[]
    for tf in FOREX_TFS:
        s,n=technical_score(frames[tf]); parts.append(s*TF_WEIGHT[tf]); details.append((tf,s,n))
    technical=sum(parts)/sum(TF_WEIGHT.values())
    trend,bos,_,_=structure(frames["1h"])
    stc=10 if trend=="BULLISH" else -10 if trend=="BEARISH" else 0
    stc+=8 if bos=="Bullish BOS" else -8 if bos=="Bearish BOS" else 0
    pats=patterns(frames["15min"])
    stc+=sum(3 if "Bullish" in p else -3 if "Bearish" in p else 0 for p in pats)
    shift,shift_score=choch_mss(frames["15min"])
    liq=liquidity_features(frames["15min"])
    disp,disp_note=displacement_score(frames["15min"])
    try:
        fast5=add_indicators(candles(pair,"5min"))
        mom,mom_notes=momentum_engine(fast5)
    except Exception:
        mom,mom_notes=0,["5min momentum unavailable"]
    base,quote=CCY[pair]; cb,nb=cot(base); cq,nq=cot(quote)
    cs=(cb or 0)-(cq or 0); rs,rn=rates(pair); ev,en=events(pair)
    total=technical+stc+shift_score+liq[3]+disp+mom+cs+rs+ev
    conf=int(np.clip(50+abs(total)*.48,50,95))
    sig="BUY" if total>=22 and conf>=threshold else "SELL" if total<=-22 and conf>=threshold else "NO TRADE"
    x=frames["15min"]; price=float(x.close.iloc[-1]); aa=float(x.atr.iloc[-1])
    if sig=="BUY":
        sl=min(price-1.5*aa,float(x.low.tail(60).min())); risk=max(price-sl,aa); tp1=price+1.5*risk; tp2=price+2.2*risk
    elif sig=="SELL":
        sl=max(price+1.5*aa,float(x.high.tail(60).max())); risk=max(sl-price,aa); tp1=price-1.5*risk; tp2=price-2.2*risk
    else: sl=tp1=tp2=np.nan
    lev=levels(x)
    signal_time=pd.to_datetime(x.datetime.iloc[-1],utc=True).to_pydatetime()
    age=max(0,int((datetime.now(timezone.utc)-signal_time).total_seconds()))
    return {
        "frames":frames,"technical":technical,"structure_score":stc,
        "shift":shift,"shift_score":shift_score,"liquidity":liq,
        "displacement":disp,"displacement_note":disp_note,
        "momentum":mom,"momentum_notes":mom_notes,"cot_score":cs,
        "rate_score":rs,"event_score":ev,"total":total,"confidence":conf,
        "signal":sig,"price":price,"stop_loss":sl,"tp1":tp1,"tp2":tp2,
        "trend":trend,"bos":bos,"patterns":pats,"details":details,
        "base_cot_note":nb,"quote_cot_note":nq,"rate_note":rn,
        "event_note":en,"levels":lev,"signal_age_sec":age,
        "status":{"Twelve Data":True,"CFTC COT":cb is not None or cq is not None,
                  "Economic calendar":not en.startswith("Calendar unavailable"),
                  "Interest rates":rs!=0}
    }


def binary_analysis(pair,expiry_minutes,threshold):
    fast=add_indicators(candles(pair,"1min"))
    mid=add_indicators(candles(pair,"5min"))
    bias=add_indicators(candles(pair,"15min"))
    fs,_=technical_score(fast); ms,_=technical_score(mid); bs,_=technical_score(bias)
    mom,mn=momentum_engine(fast); disp,dn=displacement_score(fast)
    shift,shift_score=choch_mss(mid); liq=liquidity_features(fast)
    ev,en=events(pair)
    score=.45*fs+.30*ms+.15*bs+mom+.70*disp+.70*shift_score+.50*liq[3]+.35*ev
    chop=0; a=fast.iloc[-1]
    if a.adx<16: chop=8; score*=.75
    if a.rsi>82 and score>0: score-=6
    if a.rsi<18 and score<0: score+=6
    conf=int(np.clip(50+abs(score)*.8,50,95))
    direction="CALL" if score>=18 and conf>=threshold else "PUT" if score<=-18 and conf>=threshold else "NO TRADE"
    signal_time=pd.to_datetime(a.datetime,utc=True).to_pydatetime()
    age=max(0,int((datetime.now(timezone.utc)-signal_time).total_seconds()))
    freshness="FRESH" if age<=60 else "AGING" if age<=180 else "STALE"
    if freshness=="STALE" and direction!="NO TRADE": direction="WAIT"
    return {"direction":direction,"score":score,"confidence":conf,"entry":float(a.close),
            "signal_time":signal_time,"age_sec":age,"freshness":freshness,
            "expiry_minutes":expiry_minutes,"fast_score":fs,"mid_score":ms,
            "bias_score":bs,"momentum":mom,"momentum_notes":mn,
            "displacement":disp,"displacement_note":dn,"shift":shift,
            "liquidity":liq,"event_score":ev,"event_note":en,"chop_penalty":chop}


def backtest_fast(df,threshold=65,horizon=3,max_rows=300):
    x=add_indicators(df.copy())
    if len(x)<120: return pd.DataFrame()
    rows=[]; start=max(100,len(x)-max_rows)
    for i in range(start,len(x)-horizon):
        a=x.iloc[i]; score=0
        if a.ema20>a.ema50>a.ema100: score+=18
        elif a.ema20<a.ema50<a.ema100: score-=18
        score+=8 if a.close>a.ema200 else -8
        score+=8 if a.macd_hist>0 else -8
        score+=7 if a.rsi>=55 else -7 if a.rsi<=45 else 0
        score+=6 if a.roc>0 else -6 if a.roc<0 else 0
        if a.adx>=20: score+=6 if a.pdi>a.mdi else -6
        conf=int(np.clip(50+abs(score)*.8,50,95))
        sig="BUY" if score>=20 and conf>=threshold else "SELL" if score<=-20 and conf>=threshold else "NO TRADE"
        entry=float(a.close); future=float(x.close.iloc[i+horizon]); ret=(future-entry)/entry*100
        outcome="WIN" if sig=="BUY" and future>entry else "WIN" if sig=="SELL" and future<entry else "LOSS" if sig in ["BUY","SELL"] else "SKIP"
        rows.append({"time":a.datetime,"signal":sig,"confidence":conf,"entry":entry,
                     "future_price":future,"return_pct":ret,"outcome":outcome})
    return pd.DataFrame(rows)


st.title("📈 Forex AI Pro Analyzer V8")
st.caption("V6 + V7 feature-preserving upgrade for Forex and binary-options analysis.")

with st.sidebar:
    st.header("⚙️ Settings")
    pair=st.selectbox("Forex pair",PAIRS)
    mode=st.radio("Analysis mode",["Forex","Binary Options"])
    threshold=st.slider("Minimum confidence",50,90,65)
    expiry=st.selectbox("Expiry guidance (minutes)",[1,2,3,5,10,15],index=2) if mode=="Binary Options" else 5
    if st.button("🔄 Refresh data"):
        st.cache_data.clear(); st.rerun()

if not TD_KEY:
    st.error("Add TWELVE_DATA_API_KEY to Streamlit Secrets."); st.stop()

st.subheader(f"📡 {pair} — Live Market")
try: quote=td("quote",{"symbol":td_symbol(pair)})
except Exception: quote={}

live=st.columns(5)
live[0].metric("Price",quote.get("close","—"))
live[1].metric("Open",quote.get("open","—"))
live[2].metric("Previous close",quote.get("previous_close","—"))

if mode=="Forex":
    try: result=analyze_forex(pair,threshold)
    except Exception as e: st.error(f"Forex analysis failed: {e}"); st.stop()
    live[3].metric("Signal",result["signal"]); live[4].metric("Confidence",f'{result["confidence"]}/100')
    st.subheader("🎯 Final Forex Decision")
    st.success("🟢 BUY") if result["signal"]=="BUY" else st.error("🔴 SELL") if result["signal"]=="SELL" else st.warning("🟡 NO TRADE")
    c=st.columns(6)
    for col,(label,val) in zip(c,[("Overall score",result["total"]),("Technical",result["technical"]),("Structure",result["structure_score"]),("Momentum",result["momentum"]),("Liquidity",result["liquidity"][3]),("Event risk",result["event_score"])]): col.metric(label,f"{val:.1f}")
    st.subheader("🕒 Multi-Timeframe Bias")
    rows=[]
    for tf,s,n in result["details"]:
        rows.append({"Timeframe":tf,"Technical score":round(s,1),"Bias":"BULLISH" if s>5 else "BEARISH" if s<-5 else "NEUTRAL","Key evidence":", ".join(n) or "Mixed"})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.subheader("⚡ Early Momentum & Signal Timing")
    c=st.columns(4); c[0].metric("Momentum score",f'{result["momentum"]:.1f}'); c[1].metric("Displacement",f'{result["displacement"]:.1f}'); c[2].metric("Structure shift",result["shift"]); c[3].metric("Signal age",f'{result["signal_age_sec"]}s')
    st.write("**Momentum evidence:**",", ".join(result["momentum_notes"]) or "Mixed"); st.write("**Displacement:**",result["displacement_note"])
    st.subheader("📊 Indicator Dashboard")
    for tf in FOREX_TFS:
        x=result["frames"][tf].iloc[-1]; st.markdown(f"### {tf}"); c=st.columns(4)
        c[0].markdown(f"**Trend**\n\nEMA20 `{x.ema20:.5f}`  \nEMA50 `{x.ema50:.5f}`  \nEMA100 `{x.ema100:.5f}`  \nEMA200 `{x.ema200:.5f}`")
        c[1].markdown(f"**Momentum**\n\nRSI `{x.rsi:.1f}`  \nMACD `{x.macd:.5f}`  \nHistogram `{x.macd_hist:.5f}`  \nStoch K/D `{x.stoch_k:.1f}/{x.stoch_d:.1f}`  \nROC `{x.roc:.2f}%`")
        c[2].markdown(f"**Strength / Volatility**\n\nADX `{x.adx:.1f}`  \n+DI/-DI `{x.pdi:.1f}/{x.mdi:.1f}`  \nATR `{x.atr:.5f}`  \nATR% `{x.atr_pct:.3f}%`")
        c[3].markdown(f"**Bollinger**\n\nUpper `{x.bb_up:.5f}`  \nMiddle `{x.bb_mid:.5f}`  \nLower `{x.bb_low:.5f}`  \nBody ratio `{x.body_ratio:.2f}`")
    st.subheader("🧠 Smart Money & Price Action")
    c=st.columns(5); c[0].metric("1H Structure",result["trend"]); c[1].metric("BOS",result["bos"]); c[2].metric("CHOCH/MSS",result["shift"]); c[3].metric("Equal highs",str(result["liquidity"][0])); c[4].metric("Equal lows",str(result["liquidity"][1]))
    st.write("**Candlestick patterns:**",", ".join(result["patterns"]) or "None"); st.write("**Liquidity sweep:**",result["liquidity"][2]); st.write("**Order block:**",result["levels"][6]); st.write("**FVG:**",result["levels"][5] or "None detected"); st.write("**Support:**",result["levels"][0],"| **Resistance:**",result["levels"][1])
    st.subheader("🌍 Fundamental & Sentiment")
    c=st.columns(4)
    for col,(name,ok) in zip(c,result["status"].items()): col.metric(name,"AVAILABLE" if ok else "UNAVAILABLE")
    st.write("**Next economic event:**",result["event_note"]); st.write("**COT base:**",result["base_cot_note"]); st.write("**COT quote:**",result["quote_cot_note"]); st.write("**Interest rates:**",result["rate_note"])
    st.subheader("🎯 Forex Trade Planning")
    if result["signal"]!="NO TRADE":
        c=st.columns(5); c[0].metric("Entry / current price",f'{result["price"]:.5f}'); c[1].metric("Stop Loss",f'{result["stop_loss"]:.5f}'); c[2].metric("TP1",f'{result["tp1"]:.5f}'); c[3].metric("TP2",f'{result["tp2"]:.5f}'); c[4].metric("Entry freshness",f'{result["signal_age_sec"]}s')
    else: st.info("No trade plan generated because the evidence threshold was not met.")
else:
    try: binary=binary_analysis(pair,expiry,threshold)
    except Exception as e: st.error(f"Binary-options analysis failed: {e}"); st.stop()
    live[3].metric("Direction",binary["direction"]); live[4].metric("Confidence",f'{binary["confidence"]}/100')
    st.subheader("🎯 Binary Options Decision")
    st.success("🟢 CALL") if binary["direction"]=="CALL" else st.error("🔴 PUT") if binary["direction"]=="PUT" else st.warning("⏳ WAIT" if binary["direction"]=="WAIT" else "🟡 NO TRADE")
    c=st.columns(6)
    for col,label,val in zip(c,["Binary score","1m technical","5m technical","15m bias","Momentum","Signal age"],[binary["score"],binary["fast_score"],binary["mid_score"],binary["bias_score"],binary["momentum"],binary["age_sec"]]): col.metric(label,f"{val:.1f}" if isinstance(val,float) else str(val))
    st.subheader("⚡ Entry Timing Engine")
    c=st.columns(4); c[0].metric("Entry price",f'{binary["entry"]:.5f}'); c[1].metric("Freshness",binary["freshness"]); c[2].metric("Expiry guidance",f'{expiry} min'); c[3].metric("Chop penalty",f'{binary["chop_penalty"]:.1f}')
    st.write("**Momentum evidence:**",", ".join(binary["momentum_notes"]) or "Mixed"); st.write("**Displacement:**",binary["displacement_note"]); st.write("**Structure shift:**",binary["shift"]); st.write("**Liquidity sweep:**",binary["liquidity"][2]); st.write("**Event filter:**",binary["event_note"])
    st.info("CALL/PUT is directional analysis, not a guarantee. Use the current platform entry price and reject stale signals.")

st.divider()
st.subheader("🧪 Historical Signal Validation")
bt_tf=st.selectbox("Backtest timeframe",["1min","5min","15min","1h"],index=2)
bt_horizon=st.slider("Forward bars used for outcome",1,15,3)
bt_rows=st.slider("Historical rows",100,500,300,step=50)
if st.button("▶ Run backtest"):
    try:
        raw=candles(pair,bt_tf,min(500,bt_rows+250)); bt=backtest_fast(raw,threshold=threshold,horizon=bt_horizon,max_rows=bt_rows)
        if bt.empty: st.warning("Not enough historical candles for the selected test.")
        else:
            traded=bt[bt.outcome.isin(["WIN","LOSS"])]; wins=int((traded.outcome=="WIN").sum()); losses=int((traded.outcome=="LOSS").sum()); wr=100*wins/len(traded) if len(traded) else 0
            c=st.columns(4); c[0].metric("Signals tested",str(len(traded))); c[1].metric("Wins",str(wins)); c[2].metric("Losses",str(losses)); c[3].metric("Win rate",f"{wr:.1f}%")
            st.dataframe(bt.tail(100),use_container_width=True,hide_index=True)
    except Exception as e: st.error(f"Backtest failed: {e}")

with st.expander("🔎 V8 data, methodology and safeguards"):
    st.write("Preserved V6/V7 indicators: EMA20/50/100/200, RSI, MACD, ADX/+DI/-DI, Stochastic K/D, ROC, ATR, Bollinger Bands.")
    st.write("Preserved price-action tools: swing structure, BOS, candlestick patterns, support/resistance, Fibonacci, equal highs/lows, FVG and order-block proxy.")
    st.write("V8 additions: CHOCH/MSS proxy, liquidity-sweep proxy, displacement, momentum acceleration, signal-age/freshness, binary-options layer and historical validation.")
    st.write("COT is weekly/delayed. Interest-rate scoring remains conservative until verified feeds for each currency are configured.")
    st.warning("No trading model can guarantee perfect or 100% accurate signals. Binary options are especially sensitive to timing, expiry and execution.")

st.caption("Forex AI Pro Analyzer V8 • Last dashboard time (UTC): "+datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
