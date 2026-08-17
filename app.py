
import os, json, sqlite3
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

APP_VERSION="V10.1"
BASE="https://api.twelvedata.com"
DATA_DIR=".v10_data"; DB=os.path.join(DATA_DIR,"forex_ai_pro_v10.sqlite3")
PAIRS=["EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","USD/CAD","NZD/USD","EUR/GBP"]
TF={"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1h","4h":"4h","1day":"1day"}
FEATURES=["ret1","ret3","ret5","ret10","range","body","uwick","lwick","atrp","rsi","adx","macdh",
"e9","e20","e50","e20_50","bbpos","vol20","accel","trend","dh20","dl20","hsin","hcos","dsin","dcos",
"asia","london","overlap","ny","e100","e200","e50_100","e100_200","rsis","adxs","atrs","macds",
"bbw","rangez","volz","cloc","bdirection","consistency","retz"]

def now(): return pd.Timestamp.now(tz="UTC")
def key():
    try: k=st.secrets.get("TWELVE_DATA_API_KEY","")
    except Exception: k=""
    return str(k or os.getenv("TWELVE_DATA_API_KEY",""))
def db():
    os.makedirs(DATA_DIR,exist_ok=True)
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS signals(
        id INTEGER PRIMARY KEY,created_at TEXT,pair TEXT,engine TEXT,timeframe TEXT,direction TEXT,
        probability REAL,threshold REAL,entry REAL,strike REAL,expiry_minutes REAL,stop_loss REAL,
        take_profit REAL,regime TEXT,status TEXT,explanation TEXT,payload TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS backtests(
        id INTEGER PRIMARY KEY,created_at TEXT,pair TEXT,timeframe TEXT,engine TEXT,trades INTEGER,
        wins INTEGER,losses INTEGER,win_rate REAL,expectancy REAL,profit_factor REAL,
        max_drawdown REAL,brier REAL,payload TEXT)""")
def journal(p):
    db()
    with sqlite3.connect(DB) as c:
        c.execute("""INSERT INTO signals(created_at,pair,engine,timeframe,direction,probability,threshold,
        entry,strike,expiry_minutes,stop_loss,take_profit,regime,status,explanation,payload)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(now().isoformat(),p.get("pair",""),p.get("engine",""),
        p.get("timeframe",""),p.get("direction","NO TRADE"),p.get("probability"),p.get("threshold"),
        p.get("entry"),p.get("strike"),p.get("expiry_minutes"),p.get("stop_loss"),p.get("take_profit"),
        p.get("regime"),p.get("status"),p.get("explanation",""),json.dumps(p,default=str)))
def load_journal(n=200):
    db()
    with sqlite3.connect(DB) as c:return pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT ?",c,params=(n,))

@st.cache_data(ttl=20,show_spinner=False)
def candles(symbol,interval,size,apikey):
    if not apikey: raise ValueError("TWELVE_DATA_API_KEY is not configured.")
    r=requests.get(f"{BASE}/time_series",params={"symbol":symbol,"interval":interval,"outputsize":size,"apikey":apikey,"format":"JSON"},timeout=20)
    r.raise_for_status(); p=r.json()
    if p.get("status")=="error": raise RuntimeError(p.get("message","Twelve Data error."))
    d=pd.DataFrame(p.get("values",[]))
    if d.empty: raise RuntimeError("Twelve Data returned no candles.")
    for c in ["open","high","low","close","volume"]:
        if c in d:d[c]=pd.to_numeric(d[c],errors="coerce")
    if "volume" not in d:d["volume"]=np.nan
    d["datetime"]=pd.to_datetime(d.datetime,utc=True,errors="coerce")
    return d.dropna(subset=["datetime","open","high","low","close"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

def health(d,interval):
    out={"valid":True,"fresh":True,"gaps_ok":True,"issues":[]}
    if d.empty:return {"valid":False,"fresh":False,"gaps_ok":False,"issues":["No data."]}
    if d[["open","high","low","close"]].isna().any().any():out["valid"]=False;out["issues"].append("Missing OHLC.")
    if (d.high<d.low).any():out["valid"]=False;out["issues"].append("High below low.")
    if (d.high<d.open).any() or (d.high<d.close).any():out["valid"]=False;out["issues"].append("Close/open above high.")
    if (d.low>d.open).any() or (d.low>d.close).any():out["valid"]=False;out["issues"].append("Close/open below low.")
    mins={"1min":1,"3min":3,"5min":5,"15min":15,"30min":30,"1h":60,"4h":240,"1day":1440}.get(interval,5)
    age=max(0,(now()-d.datetime.iloc[-1]).total_seconds()/60);out["age"]=age
    if age>mins*2+3:out["fresh"]=False;out["issues"].append(f"Data stale ({age:.1f} min).")
    diff=d.datetime.diff().dt.total_seconds().div(60).dropna()
    if not diff.empty and (diff>mins*2.5).any():out["gaps_ok"]=False;out["issues"].append("Large candle gap.")
    return out

def ema(s,n):return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff();g=d.clip(lower=0);l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,min_periods=n,adjust=False).mean();al=l.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,np.nan))
def atr(d,n=14):
    pc=d.close.shift();tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
def adx(d,n=14):
    up=d.high.diff();dn=-d.low.diff();plus=pd.Series(np.where((up>dn)&(up>0),up,0.),index=d.index);minus=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=d.index)
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    av=tr.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    p=100*plus.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/av;m=100*minus.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/av
    dx=100*(p-m).abs()/(p+m).replace(0,np.nan)
    return dx.ewm(alpha=1/n,min_periods=n,adjust=False).mean()

def feats(d):
    x=d.copy()
    for n in [9,20,50,100,200]:x[f"ema{n}"]=ema(x.close,n)
    x["rsi"]=rsi(x.close);x["atr"]=atr(x);x["adx"]=adx(x);macd=ema(x.close,12)-ema(x.close,26);x["macdh"]=macd-ema(macd,9)
    mid=x.close.rolling(20).mean();sd=x.close.rolling(20).std();x["bbu"]=mid+2*sd;x["bbl"]=mid-2*sd;x["bbm"]=mid
    x["ret1"]=x.close.pct_change();x["ret3"]=x.close.pct_change(3);x["ret5"]=x.close.pct_change(5);x["ret10"]=x.close.pct_change(10)
    x["range"]=(x.high-x.low)/x.close;x["body"]=(x.close-x.open).abs()/x.close
    x["uwick"]=(x.high-x[["open","close"]].max(axis=1))/x.close;x["lwick"]=(x[["open","close"]].min(axis=1)-x.low)/x.close;x["atrp"]=x.atr/x.close
    x["e9"]=(x.close-x.ema9)/x.close;x["e20"]=(x.close-x.ema20)/x.close;x["e50"]=(x.close-x.ema50)/x.close
    x["e100"]=(x.close-x.ema100)/x.close;x["e200"]=(x.close-x.ema200)/x.close;x["e20_50"]=(x.ema20-x.ema50)/x.close
    x["e50_100"]=(x.ema50-x.ema100)/x.close;x["e100_200"]=(x.ema100-x.ema200)/x.close
    x["bbpos"]=(x.close-x.bbl)/(x.bbu-x.bbl).replace(0,np.nan);x["bbw"]=(x.bbu-x.bbl)/x.bbm.replace(0,np.nan)
    x["vol20"]=x.ret1.rolling(20).std();x["accel"]=x.ret3-x.ret3.shift(3);x["trend"]=x.e20_50*x.adx
    x["dh20"]=(x.high.rolling(20).max()-x.close)/x.close;x["dl20"]=(x.close-x.low.rolling(20).min())/x.close
    x["rsis"]=x.rsi.diff(3);x["adxs"]=x.adx.diff(3);x["atrs"]=x.atrp.diff(3);x["macds"]=x.macdh.diff(3)
    x["rangez"]=(x["range"]-x["range"].rolling(50).mean())/x["range"].rolling(50).std().replace(0,np.nan)
    x["volz"]=(x.vol20-x.vol20.rolling(50).mean())/x.vol20.rolling(50).std().replace(0,np.nan)
    x["cloc"]=(x.close-x.low)/(x.high-x.low).replace(0,np.nan);x["bdirection"]=np.sign(x.close-x.open)
    x["consistency"]=x.ret1.rolling(10).apply(lambda z:abs(np.sign(z).sum())/len(z),raw=True)
    x["retz"]=(x.ret1-x.ret1.rolling(20).mean())/x.ret1.rolling(20).std().replace(0,np.nan)
    h=x.datetime.dt.hour;w=x.datetime.dt.dayofweek
    x["hsin"]=np.sin(2*np.pi*h/24);x["hcos"]=np.cos(2*np.pi*h/24);x["dsin"]=np.sin(2*np.pi*w/7);x["dcos"]=np.cos(2*np.pi*w/7)
    x["asia"]=((h>=0)&(h<7)).astype(int);x["london"]=((h>=7)&(h<12)).astype(int);x["overlap"]=((h>=12)&(h<16)).astype(int);x["ny"]=((h>=16)&(h<21)).astype(int)
    return x.replace([np.inf,-np.inf],np.nan)

def model(kind):
    if kind=="random_forest":m=RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=8,max_features="sqrt",class_weight="balanced_subsample",random_state=42,n_jobs=-1)
    elif kind=="gradient_boosting":m=HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,random_state=42)
    else:m=LogisticRegression(max_iter=2000,class_weight="balanced",C=.5,random_state=42)
    return Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler() if kind=="logistic" else "passthrough"),("model",m)])

def supervised(d,h):
    x=feats(d);f=x.close.shift(-h);ok=f.notna()
    return x.loc[ok,FEATURES],(f.loc[ok]>x.loc[ok,"close"]).astype(int)

def fit_models(Xtr,ytr,Xv,yv,kind):
    ks=["logistic","random_forest","gradient_boosting"] if kind=="ensemble" else [kind];ms=[];ps=[]
    for k in ks:
        try:m=model(k);m.fit(Xtr,ytr);ms.append(m);ps.append(m.predict_proba(Xv)[:,1])
        except Exception:pass
    if not ms:return None
    mat=np.column_stack(ps);w=np.ones(len(ms))/len(ms);best=np.inf
    grid=((a,b,1-a-b) for a in np.linspace(0,1,11) for b in np.linspace(0,1-a,11)) if len(ms)==3 else ((a,1-a) for a in np.linspace(0,1,21)) if len(ms)==2 else [(1,)]
    for z in grid:
        z=np.asarray(z);s=brier_score_loss(yv,np.clip(mat@z,.001,.999))
        if s<best:best=s;w=z
    raw=np.clip(mat@w,.001,.999);cal=LogisticRegression(max_iter=1000);cal.fit(raw.reshape(-1,1),yv)
    return {"models":ms,"weights":w,"cal":cal}

def predict(f,X):
    mat=np.column_stack([m.predict_proba(X)[:,1] for m in f["models"]]);raw=np.clip(mat@f["weights"],.001,.999)
    return f["cal"].predict_proba(raw.reshape(-1,1))[:,1]

def probability_engine(d,h=1,kind="ensemble"):
    X,y=supervised(d,h)
    if len(X)<500 or y.nunique()<2:return None
    a=int(len(X)*.6);b=int(len(X)*.8)
    f=fit_models(X.iloc[:a],y.iloc[:a],X.iloc[a:b],y.iloc[a:b],kind)
    if not f:return None
    test=predict(f,X.iloc[b:]);yt=y.iloc[b:]
    return f,{"brier":brier_score_loss(yt,test),"accuracy":accuracy_score(yt,test>=.5),"logloss":log_loss(yt,np.clip(test,.001,.999))}

def confluence(d,higher):
    x=feats(d);z=x.iloc[-1];bull=bear=0;reasons=[]
    if z.ema20>z.ema50:bull+=15;reasons.append("EMA20 > EMA50")
    elif z.ema20<z.ema50:bear+=15;reasons.append("EMA20 < EMA50")
    if z.ema50>z.ema100:bull+=7
    elif z.ema50<z.ema100:bear+=7
    if z.ema100>z.ema200:bull+=5
    elif z.ema100<z.ema200:bear+=5
    if z.macdh>0:bull+=8
    elif z.macdh<0:bear+=8
    if pd.notna(z.rsi):
        if 52<=z.rsi<=70:bull+=8
        elif 30<=z.rsi<=48:bear+=8
    if pd.notna(z.adx) and z.adx>=25:
        if bull>bear:bull+=6
        elif bear>bull:bear+=6
    hb=hr=0
    for h in higher:
        q=feats(h).iloc[-1]
        if q.ema20>q.ema50:hb+=1
        elif q.ema20<q.ema50:hr+=1
    if hb>hr:bull+=16;reasons.append(f"Higher TF bullish {hb}/{len(higher)}")
    elif hr>hb:bear+=16;reasons.append(f"Higher TF bearish {hr}/{len(higher)}")
    direction="BUY" if bull>bear else "SELL" if bear>bull else "NO TRADE"
    score=50+45*abs(bull-bear)/max(bull+bear,1)
    regime="UNKNOWN" if len(x)<100 else "TRENDING_BULLISH" if z.adx>=25 and z.ema20>z.ema50 else "TRENDING_BEARISH" if z.adx>=25 else "RANGING" if z.adx<18 else "TRANSITION"
    blockers=[]
    if len(d)<250:blockers.append("Need >=250 candles.")
    if hb and hr:blockers.append("Higher-timeframe conflict.")
    if direction=="NO TRADE":blockers.append("No directional edge.")
    if regime=="UNKNOWN":blockers.append("Unknown regime.")
    return {"direction":direction,"score":score,"regime":regime,"rsi":z.rsi,"adx":z.adx,"atr":z.atr,"reasons":reasons,"blockers":blockers}

def grade(p,c,a,s):
    if p>=85 and c>=82 and a>=1 and s<=.10:return"A+"
    if p>=80 and c>=78 and a>=.66 and s<=.15:return"A"
    if p>=75 and c>=72 and a>=.66:return"B"
    if p>=70 and c>=68:return"C"
    return"D"

def pip(pair):return .01 if pair.endswith("/JPY") else .0001

def realistic_fx(d,i,direction,atrv,pair,rr,stop_atr,spread,slip,horizon):
    if i+1>=len(d):return None
    p=pip(pair);sp=spread*p;slp=slip*p;raw=float(d.iloc[i+1].open)
    entry=raw+sp/2+slp if direction=="BUY" else raw-sp/2-slp
    sd=max(float(atrv)*stop_atr,p*5);sl=entry-sd if direction=="BUY" else entry+sd;tp=entry+sd*rr if direction=="BUY" else entry-sd*rr
    end=min(len(d)-1,i+1+horizon);exit_i=end;outcome="TIME";ex=None
    for k in range(i+1,end+1):
        b=d.iloc[k]
        if direction=="BUY":
            if b.low<=sl:ex=sl-sp/2-slp;outcome="SL";exit_i=k;break
            if b.high>=tp:ex=tp-sp/2-slp;outcome="TP";exit_i=k;break
        else:
            if b.high>=sl:ex=sl+sp/2+slp;outcome="SL";exit_i=k;break
            if b.low<=tp:ex=tp+sp/2+slp;outcome="TP";exit_i=k;break
    if ex is None:
        close=float(d.iloc[end].close);ex=close-sp/2-slp if direction=="BUY" else close+sp/2+slp
    r=(ex-entry)/sd if direction=="BUY" else (entry-ex)/sd
    return {"signal_time":d.iloc[i].datetime,"entry_time":d.iloc[i+1].datetime,"exit_time":d.iloc[exit_i].datetime,
            "direction":direction,"entry":entry,"exit":ex,"sl":sl,"tp":tp,"outcome":outcome,"pnl_r":float(r)}

def robust_bt(d,pair,horizon,train,test,minp,kind,target,rr,stop_atr,spread,slip,one):
    x=feats(d).reset_index(drop=True);y=(x.close.shift(-horizon)>x.close).astype(int);ok=y.notna()&x[FEATURES].notna().any(axis=1)
    x=x.loc[ok].reset_index(drop=True);y=y.loc[ok].astype(int).reset_index(drop=True);X=x[FEATURES]
    if len(X)<train+test+100:return {"error":"Not enough history for selected windows."}
    rows=[];ths=[];start=train;last_exit=-1
    while start<len(X):
        end=min(start+test,len(X));Xtr,ytr=X.iloc[:start],y.iloc[:start];Xte=X.iloc[start:end]
        cut=int(len(Xtr)*.8);tvx,tvy=Xtr.iloc[:cut],ytr.iloc[:cut];vvx,vvy=Xtr.iloc[cut:],ytr.iloc[cut:]
        if len(vvx)<50 or tvy.nunique()<2 or vvy.nunique()<2:start=end;continue
        vf=fit_models(tvx,tvy,vvx,vvy,kind)
        if not vf:start=end;continue
        pv=predict(vf,vvx);th=float(minp);bestn=0;bestwr=0
        for t in np.arange(minp,91):
            take=np.maximum(pv,1-pv)*100>=t
            if take.sum()<10:continue
            wr=((pv[take]>=.5).astype(int)==vvy.values[take]).mean()*100
            if wr>=target and (t>th or bestn==0):th=float(t);bestwr=wr;bestn=int(take.sum())
            elif bestn==0 and wr>bestwr:th=float(t);bestwr=wr;bestn=int(take.sum())
        ths.append(th)
        kinds=["logistic","random_forest","gradient_boosting"] if kind=="ensemble" else [kind];ms=[]
        for k in kinds:
            try:m=model(k);m.fit(Xtr,ytr);ms.append(m)
            except Exception:pass
        if not ms:start=end;continue
        vm=np.column_stack([m.predict_proba(vvx)[:,1] for m in ms]);w=np.ones(len(ms))/len(ms);bestsc=np.inf
        grid=((a,b,1-a-b) for a in np.linspace(0,1,11) for b in np.linspace(0,1-a,11)) if len(ms)==3 else ((a,1-a) for a in np.linspace(0,1,21)) if len(ms)==2 else [(1,)]
        for z in grid:
            z=np.asarray(z);sc=brier_score_loss(vvy,np.clip(vm@z,.001,.999))
            if sc<bestsc:bestsc=sc;w=z
        raw=np.clip(vm@w,.001,.999);cal=LogisticRegression(max_iter=1000);cal.fit(raw.reshape(-1,1),vvy)
        tm=np.column_stack([m.predict_proba(Xte)[:,1] for m in ms]);pt=cal.predict_proba(np.clip(tm@w,.001,.999).reshape(-1,1))[:,1]
        for j,p in enumerate(pt):
            conf=max(p,1-p)*100
            if conf<th:continue
            i=start+j
            if one and i<=last_exit:continue
            direction="BUY" if p>=.5 else "SELL"
            tr=realistic_fx(x,i,direction,float(x.iloc[i].atr),pair,rr,stop_atr,spread,slip,horizon)
            if tr is None:continue
            tr.update(probability=float(conf),threshold=th,regime="BULL" if x.iloc[i].ema20>x.iloc[i].ema50 else "BEAR")
            rows.append(tr);last_exit=i+(tr["exit_time"]-tr["entry_time"])/pd.Timedelta(minutes=1) if False else i+tr.get("holding_bars",0)+1
            # one-trade-at-a-time is conservatively enforced to the signal's max holding window.
            if one:last_exit=i+1+horizon
        start=end
    out=pd.DataFrame(rows)
    if out.empty:return {"trades":0,"wins":0,"losses":0,"win_rate":0,"expectancy":0,"profit_factor":None,"max_drawdown":0,"signals":out,"threshold":np.mean(ths) if ths else minp}
    pnl=out.pnl_r.to_numpy(float);eq=pd.Series(pnl).cumsum();wins=pnl[pnl>0];loss=pnl[pnl<0]
    return {"trades":len(out),"wins":int((pnl>0).sum()),"losses":int((pnl<=0).sum()),"win_rate":float((pnl>0).mean()*100),
            "expectancy":float(pnl.mean()),"profit_factor":float(wins.sum()/abs(loss.sum())) if loss.sum()!=0 else None,
            "max_drawdown":float((eq-eq.cummax()).min()),"signals":out,"threshold":float(np.mean(ths)) if ths else minp}

def chart(d,pair):
    x=feats(d).tail(220);f=go.Figure(go.Candlestick(x=x.datetime,open=x.open,high=x.high,low=x.low,close=x.close,name=pair))
    for n in [20,50,200]:f.add_trace(go.Scatter(x=x.datetime,y=x[f"ema{n}"],name=f"EMA{n}"))
    f.update_layout(height=560,xaxis_rangeslider_visible=False,title=f"{pair} — V10.1")
    return f

st.set_page_config(page_title="Forex AI Pro V10.1",page_icon="🤖",layout="wide");db()
st.title("🤖 Forex AI Pro V10.1")
st.caption("Live signal generation + realistic walk-forward research • paper/manual trading only")
API=key()
if not API:st.error("TWELVE_DATA_API_KEY is not configured in Streamlit Secrets.");st.stop()

with st.sidebar:
    st.header("LIVE SIGNAL")
    pair=st.selectbox("Pair",PAIRS);entry_tf=st.selectbox("Entry timeframe",["1m","3m","5m","15m","30m","1h"])
    htf=st.multiselect("Higher timeframes",["1h","4h","1day"],["1h","4h"]);size=st.slider("Historical candles",500,5000,2000,100)
    kind=st.selectbox("ML model",["ensemble","logistic","random_forest","gradient_boosting"])
    minp=st.slider("Minimum probability %",50,90,70);target=st.slider("Validation target win rate %",70,95,80)
    minconf=st.slider("Minimum confluence",50,95,75);minagree=st.slider("Minimum model agreement",.5,1.,.66,.01);maxspread=st.slider("Maximum model spread",.02,.3,.15,.01)
    qmode=st.selectbox("Signal quality",["A+","A"]);risk=st.slider("Forex risk %",.1,3.,1.,.1);rr=st.slider("Forex R:R",1.,5.,2.,.25)
    balance=st.number_input("Research balance",100.,1000000.,1000.);run=st.button("🔄 GENERATE LIVE SIGNAL",type="primary",use_container_width=True)

if run:
    try:
        d=candles(pair,TF[entry_tf],size,API);higher=[candles(pair,TF[h],min(size,1500),API) for h in htf]
    except Exception as e:st.exception(e);st.stop()
    h=health(d,TF[entry_tf]);a,b,c,e=st.columns(4);a.metric("Candles",len(d));b.metric("Freshness","OK" if h["fresh"] else "STALE");c.metric("Gaps","OK" if h["gaps_ok"] else "CHECK");e.metric("Age",f'{h["age"]:.1f}m')
    for issue in h["issues"]:st.warning(issue)
    if not h["valid"] or not h["fresh"] or not h["gaps_ok"]:st.error("🔴 NO TRADE — data gate failed.");st.stop()
    cf=confluence(d,higher);st.plotly_chart(chart(d,pair),use_container_width=True)
    a,b,c,e=st.columns(4);a.metric("Confluence",f'{cf["score"]:.1f}');b.metric("Direction",cf["direction"]);c.metric("Regime",cf["regime"]);e.metric("ADX",f'{cf["adx"]:.1f}')
    eng=probability_engine(d,1,kind)
    payload={"pair":pair,"engine":"FOREX","timeframe":entry_tf,"direction":"NO TRADE","status":"BLOCKED","regime":cf["regime"]}
    st.subheader("🎯 LIVE FOREX SIGNAL")
    if eng and cf["direction"]!="NO TRADE":
        f,test=eng;pc=predict(f,feats(d)[FEATURES].iloc[-1:])[0];pdir=pc*100 if cf["direction"]=="BUY" else (1-pc)*100
        x=feats(d).iloc[-1:][FEATURES];probs=[m.predict_proba(x)[0,1] for m in f["models"]];ag=sum((v>=.5)==(pc>=.5) for v in probs)/len(probs);spread=max(probs)-min(probs);g=grade(pdir,cf["score"],ag,spread)
        threshold=max(minp,65)
        blocks=list(cf["blockers"])
        if pdir<threshold:blocks.append(f"Probability below {threshold:.0f}%.")
        if cf["score"]<minconf:blocks.append("Confluence below threshold.")
        if ag<minagree:blocks.append("Model disagreement.")
        if spread>maxspread:blocks.append("Model spread too wide.")
        if g not in (["A+"] if qmode=="A+" else ["A+","A"]):blocks.append("Quality grade below requirement.")
        if blocks:
            st.error("🔴 NO TRADE");st.write(" • ".join(blocks));payload.update(probability=pdir,threshold=threshold,explanation="; ".join(blocks))
        else:
            av=float(cf["atr"]);entry=float(d.close.iloc[-1]);p=pip(pair);sd=max(av*1.5,p*5);sl=entry-sd if cf["direction"]=="BUY" else entry+sd;tp=entry+sd*rr if cf["direction"]=="BUY" else entry-sd*rr
            st.success(f"🟢 {pair} {cf['direction']} — {g} — {pdir:.2f}% APPROVED")
            a,b,c,e=st.columns(4);a.metric("Entry",f"{entry:.6f}");b.metric("SL",f"{sl:.6f}");c.metric("TP",f"{tp:.6f}");e.metric("R:R",f"{rr:.2f}")
            payload.update(direction=cf["direction"],probability=pdir,threshold=threshold,entry=entry,stop_loss=sl,take_profit=tp,regime=cf["regime"],status="APPROVED",explanation="; ".join(cf["reasons"]))
    else:st.warning("No validated live directional model or no directional edge.")
    if st.button("📝 Journal signal",key="jlive"):journal(payload);st.success("Signal saved.")
    st.subheader("Evidence")
    for r in cf["reasons"]:st.write("•",r)
    if cf["blockers"]:st.warning("Blockers: "+" | ".join(cf["blockers"]))

st.divider();st.header("🧪 V10.1 REALISTIC WALK-FORWARD BACKTEST")
st.caption("Next-bar entry • spread/slippage • locked validation threshold • true ensemble • conservative same-candle SL/TP rule • optional one-trade-at-a-time.")
bp=st.selectbox("Backtest pair",PAIRS,key="bp");btf=st.selectbox("Backtest timeframe",["1m","3m","5m","15m","30m","1h"],2,key="btf")
bh=st.number_input("Maximum holding horizon (candles)",1,30,1);btrain=st.number_input("Initial training candles",300,3000,800,50);btest=st.number_input("OOS test block",25,500,100,25)
bmin=st.slider("Minimum probability",50,90,70,key="bmin");btar=st.slider("Validation target win rate",70,95,80,key="btar");bmodel=st.selectbox("Backtest model",["ensemble","logistic","random_forest","gradient_boosting"],key="bmodel")
brr=st.slider("Forex R:R",1.,5.,2.,.25);bstop=st.slider("Stop ATR multiple",.5,3.,1.5,.1);bspread=st.number_input("Spread (pips)",0.,10.,1.,.1);bslip=st.number_input("Slippage (pips)",0.,5.,.2,.1);bone=st.checkbox("One trade at a time",True)
if st.button("▶ RUN REALISTIC V10.1 BACKTEST",type="primary"):
    try:bd=candles(bp,TF[btf],size,API)
    except Exception as e:st.exception(e);st.stop()
    hh=health(bd,TF[btf])
    if not hh["valid"]:st.error("Backtest data failed validation.");st.stop()
    r=robust_bt(bd,bp,int(bh),int(btrain),int(btest),float(bmin),bmodel,float(btar),float(brr),float(bstop),float(bspread),float(bslip),bone)
    if "error" in r:st.error(r["error"])
    else:
        a,b,c,e,f=st.columns(5);a.metric("Trades",r["trades"]);b.metric("Win rate",f'{r["win_rate"]:.2f}%');c.metric("Expectancy",f'{r["expectancy"]:.3f} R');e.metric("Profit factor","N/A" if r["profit_factor"] is None else f'{r["profit_factor"]:.2f}');f.metric("Max DD",f'{r["max_drawdown"]:.2f} R')
        if not r["signals"].empty:st.dataframe(r["signals"].tail(300),use_container_width=True)
        st.caption(f"Average locked validation threshold: {r['threshold']:.1f}% • spread={bspread:.1f} pips • slippage={bslip:.1f} pips • next-bar entry • one-trade-at-a-time={bone}")
        with sqlite3.connect(DB) as con:con.execute("""INSERT INTO backtests(created_at,pair,timeframe,engine,trades,wins,losses,win_rate,expectancy,profit_factor,max_drawdown,brier,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",(now().isoformat(),bp,btf,"FOREX",r["trades"],r["wins"],r["losses"],r["win_rate"],r["expectancy"],r["profit_factor"],r["max_drawdown"],None,json.dumps(r,default=str)))
        st.success("Realistic V10.1 walk-forward completed.")

st.divider();st.header("📒 Signal Journal");j=load_journal()
if j.empty:st.info("No signals journaled yet.")
else:st.dataframe(j,use_container_width=True)
st.caption(f"Forex AI Pro {APP_VERSION} • Twelve Data • Research/paper-trading only • No guaranteed win rate")
