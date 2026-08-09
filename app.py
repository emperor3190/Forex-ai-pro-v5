import os
from datetime import datetime, timezone
import requests, numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Forex AI Pro Analyzer v5", page_icon="📈", layout="wide")
TD="https://api.twelvedata.com"; CFTC="https://publicreporting.cftc.gov/resource/yw9f-hn96.json"
CAL="https://nfs.faireconomy.media/ff_calendar_thisweek.json"
PAIRS=["EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","USD/CAD","NZD/USD"]
CCY={p:tuple(p.split("/")) for p in PAIRS}
COT={"EUR":"EURO FX","GBP":"BRITISH POUND STERLING","JPY":"JAPANESE YEN","CHF":"SWISS FRANC","AUD":"AUSTRALIAN DOLLAR","CAD":"CANADIAN DOLLAR","NZD":"NEW ZEALAND DOLLAR"}

def sec(k,d=""):
    try:return st.secrets.get(k,d)
    except:return os.getenv(k,d)
KEY=sec("TWELVE_DATA_API_KEY"); FRED=sec("FRED_API_KEY")

@st.cache_data(ttl=30)
def candles(symbol,tf):
    if not KEY: raise RuntimeError("TWELVE_DATA_API_KEY is missing.")
    r=requests.get(f"{TD}/time_series",params={"symbol":symbol,"interval":tf,"outputsize":500,"apikey":KEY,"timezone":"UTC"},timeout=20)
    r.raise_for_status(); j=r.json()
    if "values" not in j: raise RuntimeError(j.get("message","No Twelve Data values."))
    x=pd.DataFrame(j["values"]); x["datetime"]=pd.to_datetime(x["datetime"],utc=True)
    for c in ["open","high","low","close","volume"]:
        if c in x:x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.sort_values("datetime").reset_index(drop=True)

def ema(s,n):return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff();u=d.clip(lower=0);v=-d.clip(upper=0);a=u.ewm(alpha=1/n,adjust=False).mean();b=v.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+a/b.replace(0,np.nan))
def tr(x):
    p=x.close.shift();return pd.concat([x.high-x.low,(x.high-p).abs(),(x.low-p).abs()],axis=1).max(axis=1)
def atr(x,n=14):return tr(x).ewm(alpha=1/n,adjust=False).mean()
def macd(s):
    m=ema(s,12)-ema(s,26);q=ema(m,9);return m,q,m-q
def adx(x,n=14):
    up=x.high.diff();dn=-x.low.diff();p=pd.Series(np.where((up>dn)&(up>0),up,0),index=x.index);m=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=x.index)
    a=tr(x).ewm(alpha=1/n,adjust=False).mean();pdi=100*p.ewm(alpha=1/n,adjust=False).mean()/a;mdi=100*m.ewm(alpha=1/n,adjust=False).mean()/a
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan);return dx.ewm(alpha=1/n,adjust=False).mean(),pdi,mdi
def add(x):
    x=x.copy()
    for n in [20,50,100,200]:x[f"ema{n}"]=ema(x.close,n)
    x["rsi"]=rsi(x.close);x["macd"],x["macd_signal"],x["macd_hist"]=macd(x.close);x["adx"],x["pdi"],x["mdi"]=adx(x)
    lo=x.low.rolling(14).min();hi=x.high.rolling(14).max();x["stoch_k"]=100*(x.close-lo)/(hi-lo).replace(0,np.nan);x["stoch_d"]=x.stoch_k.rolling(3).mean()
    x["roc"]=x.close.pct_change(12)*100;x["atr"]=atr(x);x["bb_mid"]=x.close.rolling(20).mean();sd=x.close.rolling(20).std();x["bb_up"]=x.bb_mid+2*sd;x["bb_low"]=x.bb_mid-2*sd
    return x.dropna().reset_index(drop=True)

def swings(x,w=3):
    h=[];l=[]
    for i in range(w,len(x)-w):
        if x.high.iloc[i]>=x.high.iloc[i-w:i+w+1].max():h.append(float(x.high.iloc[i]))
        if x.low.iloc[i]<=x.low.iloc[i-w:i+w+1].min():l.append(float(x.low.iloc[i]))
    return h,l
def structure(x):
    h,l=swings(x);t="RANGE";b="None"
    if len(h)>1 and len(l)>1:
        if h[-1]>h[-2] and l[-1]>l[-2]:t="BULLISH"
        elif h[-1]<h[-2] and l[-1]<l[-2]:t="BEARISH"
    if h and x.close.iloc[-1]>h[-1]:b="Bullish BOS"
    elif l and x.close.iloc[-1]<l[-1]:b="Bearish BOS"
    return t,b,h,l
def patterns(x):
    a,p=x.iloc[-1],x.iloc[-2];body=abs(a.close-a.open);rng=max(a.high-a.low,1e-12);u=a.high-max(a.open,a.close);lo=min(a.open,a.close)-a.low;o=[]
    if body/rng<.12:o.append("Doji")
    if lo>2*body and u<1.2*body:o.append("Bullish pin")
    if u>2*body and lo<1.2*body:o.append("Bearish pin")
    if a.close>a.open and p.close<p.open and a.close>=p.open and a.open<=p.close:o.append("Bullish engulfing")
    if a.close<a.open and p.close>p.open and a.open>=p.close and a.close<=p.open:o.append("Bearish engulfing")
    return o
def levels(x):
    d=x.tail(150);h,l=swings(d);sup=min(l,default=float(d.low.min()));res=max(h,default=float(d.high.max()));H=float(d.high.max());L=float(d.low.min());R=H-L
    fib={str(p):H-p*R for p in [0,.236,.382,.5,.618,.786,1]};a=float(d.atr.iloc[-1]);eqh=len(h)>1 and abs(h[-1]-h[-2])<=.25*a;eql=len(l)>1 and abs(l[-1]-l[-2])<=.25*a
    f=[] 
    for i in range(max(2,len(d)-80),len(d)):
        if d.low.iloc[i]>d.high.iloc[i-2]:f.append(("Bullish FVG",float(d.high.iloc[i-2]),float(d.low.iloc[i])))
        elif d.high.iloc[i]<d.low.iloc[i-2]:f.append(("Bearish FVG",float(d.high.iloc[i]),float(d.low.iloc[i-2])))
    ob="None"
    if len(d)>=3:
        p,z=d.iloc[-2],d.iloc[-1]
        if p.close<p.open and z.close>p.high:ob="Bullish order-block proxy"
        elif p.close>p.open and z.close<p.low:ob="Bearish order-block proxy"
    return sup,res,fib,eqh,eql,f[-3:],ob

@st.cache_data(ttl=21600)
def cot_raw():
    r=requests.get(CFTC,params={"$limit":5000,"$order":"report_date_as_yyyy_mm_dd DESC"},timeout=30);r.raise_for_status();return pd.DataFrame(r.json())
def cot(c):
    try:
        d=cot_raw();nc=next((z for z in d.columns if z.lower()=="contract_market_name"),None)
        q=d[d[nc].astype(str).str.upper().str.contains(COT[c],regex=False,na=False)]
        lc=next((z for z in q.columns if "leveraged_money_long" in z.lower()),None);sc=next((z for z in q.columns if "leveraged_money_short" in z.lower()),None)
        if not lc or not sc:return None,"COT position fields unavailable"
        q[lc]=pd.to_numeric(q[lc],errors="coerce");q[sc]=pd.to_numeric(q[sc],errors="coerce");q=q.assign(net=q[lc]-q[sc]).dropna(subset=["net"])
        if q.empty:return None,"No COT data"
        n=float(q.iloc[0].net);return float(np.tanh(n/100000)*10),f"net {n:,.0f} (weekly)"
    except Exception as e:return None,f"COT unavailable: {e}"

@st.cache_data(ttl=1800)
def calendar():
    r=requests.get(CAL,timeout=20);r.raise_for_status();return pd.DataFrame(r.json())
def events(pair):
    try:
        d=calendar();base,quote=CCY[pair];cc=next((c for c in d.columns if c.lower() in ["country","currency"]),None);dc=next((c for c in d.columns if c.lower() in ["date","datetime"]),None)
        if not cc or not dc:return 0,"Calendar schema changed"
        d[dc]=pd.to_datetime(d[dc],utc=True,errors="coerce");q=d[d[cc].astype(str).str.upper().isin([base,quote])&(d[dc]>=pd.Timestamp.now(tz="UTC"))].sort_values(dc)
        if q.empty:return 0,"No upcoming pair events"
        z=q.iloc[0];impact=str(z.get("impact","")).lower();adj={"high":-12,"medium":-5,"low":-1}.get(impact,0);title=str(z.get("title",z.get("event","Economic event")))
        return adj,f"{title} | {z[dc].strftime('%d %b %H:%M UTC')} | {impact or 'unknown'}"
    except Exception as e:return 0,f"Calendar unavailable: {e}"

@st.cache_data(ttl=21600)
def fred(series):
    if not FRED:return None
    r=requests.get("https://api.stlouisfed.org/fred/series/observations",params={"series_id":series,"api_key":FRED,"file_type":"json","sort_order":"desc","limit":10},timeout=20);r.raise_for_status()
    v=[z for z in r.json().get("observations",[]) if z.get("value") not in [".",None]];return float(v[0]["value"]) if v else None
def rates(pair):
    base,quote=CCY[pair]
    # Only USD is scored here until an official verified feed for the other currency is configured.
    if base=="USD":br=fred("DFF")
    else:br=None
    if quote=="USD":qr=fred("DFF")
    else:qr=None
    if br is None or qr is None:return 0,"Rate differential unavailable for both currencies"
    d=br-qr;return float(np.clip(d*3,-12,12)),f"{base} {br:.2f}% vs {quote} {qr:.2f}%"

def tech(x):
    a=x.iloc[-1];s=0;n=[]
    if a.ema20>a.ema50>a.ema100>a.ema200:s+=20;n.append("EMA bullish")
    elif a.ema20<a.ema50<a.ema100<a.ema200:s-=20;n.append("EMA bearish")
    s+=6 if a.close>a.ema200 else -6
    if a.rsi>=55:s+=10;n.append(f"RSI {a.rsi:.1f} bullish")
    elif a.rsi<=45:s-=10;n.append(f"RSI {a.rsi:.1f} bearish")
    s+=10 if a.macd_hist>0 else -10;s+=(8 if a.pdi>a.mdi else -8) if a.adx>=25 else 0
    s+=4 if a.stoch_k>a.stoch_d and a.stoch_k<80 else -4 if a.stoch_k<a.stoch_d and a.stoch_k>20 else 0
    s+=4 if a.roc>0 else -4 if a.roc<0 else 0;s+=4 if a.close>a.bb_mid else -4
    return s,n

def analyze(pair,threshold):
    fs={tf:add(candles(pair.replace("/",""),tf)) for tf in ["15min","1h","4h"]};w={"15min":1,"1h":2,"4h":3};detail=[];parts=[]
    for tf in w:
        s,n=tech(fs[tf]);parts.append(s*w[tf]);detail.append(f"{tf}: {', '.join(n)}")
    technical=sum(parts)/6;t,b,_,_=structure(fs["1h"]);stc=10 if t=="BULLISH" else -10 if t=="BEARISH" else 0;stc+=8 if b=="Bullish BOS" else -8 if b=="Bearish BOS" else 0
    pats=patterns(fs["15min"]);stc+=sum(3 if "Bullish" in p else -3 if "Bearish" in p else 0 for p in pats)
    base,quote=CCY[pair];cb,nb=cot(base);cq,nq=cot(quote);cs=(cb or 0)-(cq or 0);rs,rn=rates(pair);ev,en=events(pair);total=technical+stc+cs+rs+ev;conf=int(np.clip(50+abs(total)*.55,50,95))
    sig="BUY" if total>=18 and conf>=threshold else "SELL" if total<=-18 and conf>=threshold else "NO TRADE"
    x=fs["15min"];price=float(x.close.iloc[-1]);a=float(x.atr.iloc[-1])
    if sig=="BUY":sl=min(price-1.5*a,float(x.low.tail(60).min()));risk=price-sl;tp1=price+1.5*risk;tp2=price+2.2*risk
    elif sig=="SELL":sl=max(price+1.5*a,float(x.high.tail(60).max()));risk=sl-price;tp1=price-1.5*risk;tp2=price-2.2*risk
    else:sl=tp1=tp2=np.nan
    lev=levels(x);status={"Twelve Data":True,"CFTC COT":cb is not None or cq is not None,"Economic calendar":not en.startswith("Calendar unavailable"),"Interest rates":rs!=0}
    return locals()

st.title("📈 Forex AI Pro Analyzer v5")
st.caption("Technical + structure + COT + economic-event risk + verified-rate scoring")
with st.sidebar:
    pair=st.selectbox("Forex pair",PAIRS);threshold=st.slider("Minimum confidence",50,90,65)
    if st.button("Refresh data"):st.cache_data.clear();st.rerun()
if not KEY:st.error("Add TWELVE_DATA_API_KEY in Streamlit Secrets.");st.stop()
try:a=analyze(pair,threshold)
except Exception as e:st.error(f"Analysis failed: {e}");st.stop()
c=st.columns(4);c[0].metric("Signal",a["sig"]);c[1].metric("Confidence",f'{a["conf"]}/100');c[2].metric("Price",f'{a["price"]:.5f}');c[3].metric("Score",f'{a["total"]:.1f}')
if a["sig"]!="NO TRADE":
    c=st.columns(3);c[0].metric("Stop Loss",f'{a["sl"]:.5f}');c[1].metric("TP1",f'{a["tp1"]:.5f}');c[2].metric("TP2",f'{a["tp2"]:.5f}')
else:st.warning("NO TRADE — evidence is below the selected threshold.")
st.subheader("Data-source status");c=st.columns(4)
for col,(name,ok) in zip(c,a["status"].items()):col.metric(name,"AVAILABLE" if ok else "UNAVAILABLE")
t1,t2,t3,t4=st.tabs(["Signal breakdown","Indicators","Levels & liquidity","Data notes"])
with t1:
    st.write("Technical:",round(a["technical"],1),"| Structure:",round(a["stc"],1),"| COT:",round(a["cs"],1),"| Rates:",round(a["rs"],1),"| Event:",round(a["ev"],1))
    st.write("Rate engine:",a["rn"]);st.write("Next event:",a["en"]);st.write("COT base:",a["nb"]);st.write("COT quote:",a["nq"]);st.write("Patterns:",", ".join(a["pats"]) or "None")
    for z in a["detail"]:st.write("•",z)
with t2:
    rows=[]
    for tf,x in a["fs"].items():
        z=x.iloc[-1];rows.append({"TF":tf,"Close":z.close,"EMA20":z.ema20,"EMA50":z.ema50,"EMA100":z.ema100,"EMA200":z.ema200,"RSI":z.rsi,"MACD":z.macd_hist,"ADX":z.adx,"Stoch":z.stoch_k,"ROC":z.roc,"ATR":z.atr})
    st.dataframe(pd.DataFrame(rows).round(5),use_container_width=True,hide_index=True)
with t3:
    sup,res,fib,eqh,eql,fvg,ob=a["lev"];st.write("Support:",sup,"| Resistance:",res,"| Equal highs:",eqh,"| Equal lows:",eql);st.write("FVG:",fvg);st.write("Order block:",ob);st.dataframe(pd.DataFrame({"Fib":list(fib),"Price":list(fib.values())}).round(5),use_container_width=True,hide_index=True)
with t4:
    st.write("UTC:",datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"));st.info("COT is weekly/delayed. Missing data is never fabricated.");st.warning("Research tool only. No model guarantees profit.")
