
import os, json, sqlite3, math, warnings
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
warnings.filterwarnings("ignore")

APP_VERSION = "V12.0"
BASE = "https://api.twelvedata.com"
DATA_DIR = ".v12_data"
DB = os.path.join(DATA_DIR, "forex_ai_pro_v12.sqlite3")

PAIRS = ["EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","USD/CAD","NZD/USD","EUR/GBP"]
TF = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1h","4h":"4h","1day":"1day"}

FEATURES = [
    "ret1","ret3","ret6","ret12","range","body","uwick","lwick","cloc",
    "atrp","atrz","rsi","rsis","adx","adxs","macdh","macds","roc",
    "ema20dist","ema50dist","ema20_50","ema50_100","ema100_200",
    "slope20","slope50","slope200","bbpos","bbw","rv20","rvz","rangez",
    "impulse","consistency","retz","dist_hi20","dist_lo20",
    "structure","bos","sweep","trend_score","momentum_score","vol_score",
    "breakout_score","reversion_score","liquidity_score","price_score",
    "session_score","regime_code","engine_edge","engine_conf"
]

def now():
    return pd.Timestamp.now(tz="UTC")

def api_key():
    try:
        k = st.secrets.get("TWELVE_DATA_API_KEY","")
    except Exception:
        k = ""
    return str(k or os.getenv("TWELVE_DATA_API_KEY",""))

def db_init():
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS signals(
            id INTEGER PRIMARY KEY, created_at TEXT, pair TEXT, timeframe TEXT,
            direction TEXT, probability REAL, expected_value REAL, grade TEXT,
            entry REAL, stop_loss REAL, take_profit REAL, risk_pct REAL,
            regime TEXT, status TEXT, explanation TEXT, payload TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS backtests(
            id INTEGER PRIMARY KEY, created_at TEXT, pair TEXT, timeframe TEXT,
            trades INTEGER, wins INTEGER, losses INTEGER, win_rate REAL,
            expectancy REAL, profit_factor REAL, max_drawdown REAL,
            brier REAL, logloss REAL, payload TEXT)""")

def journal(p):
    db_init()
    with sqlite3.connect(DB) as c:
        c.execute("""INSERT INTO signals(
            created_at,pair,timeframe,direction,probability,expected_value,grade,
            entry,stop_loss,take_profit,risk_pct,regime,status,explanation,payload)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now().isoformat(),p.get("pair"),p.get("timeframe"),p.get("direction"),
             p.get("probability"),p.get("expected_value"),p.get("grade"),
             p.get("entry"),p.get("stop_loss"),p.get("take_profit"),p.get("risk_pct"),
             p.get("regime"),p.get("status"),p.get("explanation",""),
             json.dumps(p,default=str)))

def load_journal(n=200):
    db_init()
    with sqlite3.connect(DB) as c:
        return pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT ?",c,params=(n,))

@st.cache_data(ttl=20, show_spinner=False)
def candles(symbol, interval, size, key):
    if not key:
        raise ValueError("TWELVE_DATA_API_KEY is not configured.")
    r = requests.get(f"{BASE}/time_series",
                     params={"symbol":symbol,"interval":interval,"outputsize":size,
                             "apikey":key,"format":"JSON"}, timeout=25)
    r.raise_for_status()
    j = r.json()
    if j.get("status") == "error":
        raise RuntimeError(j.get("message","Twelve Data error"))
    d = pd.DataFrame(j.get("values",[]))
    if d.empty:
        raise RuntimeError("No candles returned.")
    for c in ["open","high","low","close","volume"]:
        if c in d:
            d[c] = pd.to_numeric(d[c],errors="coerce")
    if "volume" not in d:
        d["volume"] = np.nan
    d["datetime"] = pd.to_datetime(d["datetime"],utc=True,errors="coerce")
    return (d.dropna(subset=["datetime","open","high","low","close"])
             .sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True))

def data_health(d, interval):
    mins = {"1min":1,"3min":3,"5min":5,"15min":15,"30min":30,"1h":60,"4h":240,"1day":1440}.get(interval,5)
    issues=[]
    valid=not d.empty
    if valid:
        valid &= not d[["open","high","low","close"]].isna().any().any()
        valid &= not (d.high<d.low).any()
        valid &= not (d.high<d[["open","close"]].max(axis=1)).any()
        valid &= not (d.low>d[["open","close"]].min(axis=1)).any()
    if not valid:
        return {"valid":False,"fresh":False,"gaps_ok":False,"age":999999,"issues":["Invalid OHLC data."]}
    age=max(0,(now()-d.datetime.iloc[-1]).total_seconds()/60)
    fresh=age <= mins*2+3
    if not fresh: issues.append(f"Data stale ({age:.1f} min).")
    diff=d.datetime.diff().dt.total_seconds().div(60).dropna()
    gaps_ok=diff.empty or not (diff > mins*2.5).any()
    if not gaps_ok: issues.append("Large candle gaps detected.")
    return {"valid":True,"fresh":fresh,"gaps_ok":gaps_ok,"age":age,"issues":issues}

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def atr(d,n=14):
    pc=d.close.shift()
    tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,min_periods=n,adjust=False).mean()

def rsi(s,n=14):
    ch=s.diff()
    up=ch.clip(lower=0); dn=-ch.clip(upper=0)
    au=up.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    ad=dn.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    rs=au/ad.replace(0,np.nan)
    return 100-100/(1+rs)

def adx(d,n=14):
    up=d.high.diff(); dn=-d.low.diff()
    plus=pd.Series(np.where((up>dn)&(up>0),up,0.),index=d.index)
    minus=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=d.index)
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),
                  (d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    av=tr.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    p=100*plus.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/av
    m=100*minus.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/av
    dx=100*(p-m).abs()/(p+m).replace(0,np.nan)
    return dx.ewm(alpha=1/n,min_periods=n,adjust=False).mean()

def zscore(s,n=50):
    mu=s.rolling(n).mean(); sd=s.rolling(n).std()
    return (s-mu)/sd.replace(0,np.nan)

def build_features(d):
    x=d.copy()
    x["atr"]=atr(x)
    x["rsi"]=rsi(x.close)
    x["adx"]=adx(x)
    for n in [20,50,100,200]:
        x[f"ema{n}"]=ema(x.close,n)
    macd=ema(x.close,12)-ema(x.close,26)
    x["macdh"]=macd-ema(macd,9)
    x["roc"]=x.close.pct_change(6)
    x["ret1"]=x.close.pct_change()
    x["ret3"]=x.close.pct_change(3)
    x["ret6"]=x.close.pct_change(6)
    x["ret12"]=x.close.pct_change(12)
    x["range"]=(x.high-x.low)/x.close
    x["body"]=(x.close-x.open).abs()/x.close
    x["uwick"]=(x.high-x[["open","close"]].max(axis=1))/x.close
    x["lwick"]=(x[["open","close"]].min(axis=1)-x.low)/x.close
    x["cloc"]=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    x["atrp"]=x.atr/x.close
    x["atrz"]=zscore(x.atrp)
    x["rsis"]=x.rsi.diff(3)
    x["adxs"]=x.adx.diff(3)
    x["macds"]=x.macdh.diff(3)
    x["ema20dist"]=(x.close-x.ema20)/x.close
    x["ema50dist"]=(x.close-x.ema50)/x.close
    x["ema20_50"]=(x.ema20-x.ema50)/x.close
    x["ema50_100"]=(x.ema50-x.ema100)/x.close
    x["ema100_200"]=(x.ema100-x.ema200)/x.close
    x["slope20"]=x.ema20.pct_change(5)
    x["slope50"]=x.ema50.pct_change(8)
    x["slope200"]=x.ema200.pct_change(12)
    mid=x.close.rolling(20).mean(); sd=x.close.rolling(20).std()
    x["bbpos"]=(x.close-(mid-2*sd))/(4*sd).replace(0,np.nan)
    x["bbw"]=(4*sd)/mid.replace(0,np.nan)
    x["rv20"]=x.ret1.rolling(20).std()
    x["rvz"]=zscore(x.rv20)
    x["rangez"]=zscore(x["range"])
    x["impulse"]=((x.close-x.open)/x.close)/(x.atrp.replace(0,np.nan))
    x["consistency"]=x.ret1.rolling(12).apply(lambda z:abs(np.sign(z).sum())/len(z),raw=True)
    x["retz"]=zscore(x.ret1,20)

    # Causal market structure proxies: all rolling windows contain only current/past bars.
    hi20=x.high.rolling(20).max().shift(1)
    lo20=x.low.rolling(20).min().shift(1)
    x["dist_hi20"]=(hi20-x.close)/x.close
    x["dist_lo20"]=(x.close-lo20)/x.close
    x["bos"]=np.where(x.close>hi20,1,np.where(x.close<lo20,-1,0))
    prev_hi=x.high.rolling(8).max().shift(1)
    prev_lo=x.low.rolling(8).min().shift(1)
    x["sweep"]=np.where((x.high>prev_hi)&(x.close<prev_hi),-1,
                 np.where((x.low<prev_lo)&(x.close>prev_lo),1,0))
    x["structure"]=(
        np.sign(x.ema20_50)*0.35 +
        np.sign(x.slope50)*0.20 +
        x.bos*0.30 + x.sweep*0.15
    )

    # Engine scores, normalized to -100..100.
    trend = (np.tanh(x.ema20_50*400)*45 +
             np.tanh(x.ema50_100*250)*25 +
             np.tanh(x.slope50*2500)*20 +
             np.where(x.adx>=25,np.sign(x.ema20_50)*10,0))
    momentum = (np.tanh(x.rsi.sub(50).div(12))*35 +
                np.tanh(x.macdh.div(x.atrp.replace(0,np.nan))*2)*30 +
                np.tanh(x.roc.div(x.rv20.replace(0,np.nan))*1.5)*20 +
                np.sign(x.rsis)*15)
    vol = (np.tanh(x.atrz)*35 + np.tanh(x.rvz)*25 +
           np.tanh(x.rangez)*25 + np.where((x.atrz>0)&(x.rangez>0),15,0))
    breakout = (x.bos*50 + np.tanh(x.rangez)*25 +
                np.sign(x.ema20_50)*15 + np.where(x.bbw> x.bbw.rolling(20).mean(),10,0))
    reversion = (-np.tanh((x.bbpos-.5)*4)*45 -
                 np.tanh((x.rsi-50)/12)*30 -
                 np.tanh(x.impulse)*15 +
                 np.where(x.sweep!=0,x.sweep*10,0))
    liquidity = (x.sweep*65 +
                 np.sign(-x.dist_hi20+x.dist_lo20)*20 +
                 np.sign(x.cloc-.5)*15)
    price = (np.sign(x.close-x.open)*25 +
             np.tanh((x.cloc-.5)*4)*30 +
             np.tanh((x.lwick-x.uwick)*100)*20 +
             np.tanh(x.impulse)*25)

    h=x.datetime.dt.hour
    london=((h>=7)&(h<12)); overlap=((h>=12)&(h<16)); ny=((h>=16)&(h<21))
    session=np.where(overlap,100,np.where(london,75,np.where(ny,65,np.where((h<7),35,20))))
    x["trend_score"]=trend.clip(-100,100)
    x["momentum_score"]=momentum.clip(-100,100)
    x["vol_score"]=vol.clip(-100,100)
    x["breakout_score"]=breakout.clip(-100,100)
    x["reversion_score"]=reversion.clip(-100,100)
    x["liquidity_score"]=liquidity.clip(-100,100)
    x["price_score"]=price.clip(-100,100)
    x["session_score"]=session
    # Regime: 0 range, 1 bullish trend, 2 bearish trend, 3 breakout/expansion, 4 transition.
    x["regime_code"]=np.where((x.adx<18)&(x.atrz<.5),0,
                       np.where((x.adx>=25)&(x.ema20_50>0),1,
                       np.where((x.adx>=25)&(x.ema20_50<0),2,
                       np.where((x.atrz>1)&(x.rangez>1),3,4))))
    # Candidate edge is deliberately moderate; ML supplies the final filter.
    x["engine_edge"]=(.24*x.trend_score + .22*x.momentum_score +
                      .18*x.structure*100 + .14*x.breakout_score +
                      .10*x.liquidity_score + .07*x.price_score +
                      .05*np.sign(x.reversion_score)*np.minimum(abs(x.reversion_score),50))
    x["engine_conf"]=np.minimum(100,abs(x.engine_edge))
    return x.replace([np.inf,-np.inf],np.nan)

def candidate_direction(row):
    e=float(row.engine_edge)
    # Regime-aware setup selection. Not every engine must agree.
    reg=int(row.regime_code)
    if reg in (1,2):
        threshold=22
    elif reg==3:
        threshold=20
    elif reg==0:
        threshold=18
    else:
        threshold=24
    if e>=threshold: return "BUY"
    if e<=-threshold: return "SELL"
    return "NONE"

def pip(pair):
    return .01 if pair.endswith("/JPY") else .0001

def simulate_trade(d,i,direction,rr,stop_atr,spread,slip,horizon):
    if i+1>=len(d): return None
    atrv=float(d.iloc[i].atr) if pd.notna(d.iloc[i].atr) else np.nan
    if not np.isfinite(atrv): return None
    p=pip(d.attrs.get("pair",""))
    spread_px=spread*p; slip_px=slip*p
    raw=float(d.iloc[i+1].open)
    if direction=="BUY":
        entry=raw+spread_px/2+slip_px
        sd=max(atrv*stop_atr,p*4)
        sl=entry-sd; tp=entry+sd*rr
    else:
        entry=raw-spread_px/2-slip_px
        sd=max(atrv*stop_atr,p*4)
        sl=entry+sd; tp=entry-sd*rr
    end=min(len(d)-1,i+1+horizon)
    exit_i=end; outcome="TIME"; ex=None
    for k in range(i+1,end+1):
        b=d.iloc[k]
        if direction=="BUY":
            hit_sl=b.low<=sl; hit_tp=b.high>=tp
            if hit_sl and hit_tp:
                ex=sl-spread_px/2-slip_px; outcome="SL"; exit_i=k; break
            if hit_sl:
                ex=sl-spread_px/2-slip_px; outcome="SL"; exit_i=k; break
            if hit_tp:
                ex=tp-spread_px/2-slip_px; outcome="TP"; exit_i=k; break
        else:
            hit_sl=b.high>=sl; hit_tp=b.low<=tp
            if hit_sl and hit_tp:
                ex=sl+spread_px/2+slip_px; outcome="SL"; exit_i=k; break
            if hit_sl:
                ex=sl+spread_px/2+slip_px; outcome="SL"; exit_i=k; break
            if hit_tp:
                ex=tp+spread_px/2+slip_px; outcome="TP"; exit_i=k; break
    if ex is None:
        close=float(d.iloc[end].close)
        ex=close-spread_px/2-slip_px if direction=="BUY" else close+spread_px/2+slip_px
    r=(ex-entry)/sd if direction=="BUY" else (entry-ex)/sd
    return {
        "signal_time":d.iloc[i].datetime,"entry_time":d.iloc[i+1].datetime,
        "exit_time":d.iloc[exit_i].datetime,"direction":direction,
        "entry":entry,"exit":ex,"sl":sl,"tp":tp,"outcome":outcome,
        "pnl_r":float(r),"holding_bars":int(exit_i-(i+1))
    }

def make_event_labels(x,pair,rr,stop_atr,spread,slip,horizon):
    x=x.copy(); x.attrs["pair"]=pair
    y=pd.Series(np.nan,index=x.index,dtype=float)
    dirs=pd.Series("NONE",index=x.index,dtype=object)
    for i in range(len(x)-1):
        direction=candidate_direction(x.iloc[i])
        if direction=="NONE": continue
        tr=simulate_trade(x,i,direction,rr,stop_atr,spread,slip,horizon)
        if tr is None: continue
        dirs.iloc[i]=direction
        y.iloc[i]=1.0 if tr["pnl_r"]>0 else 0.0
    return y,dirs

def base_model(kind):
    if kind=="logistic":
        m=LogisticRegression(max_iter=2000,C=.35,class_weight="balanced")
        return Pipeline([("imp",SimpleImputer(strategy="median")),
                         ("scale",StandardScaler()),("model",m)])
    if kind=="random_forest":
        m=RandomForestClassifier(n_estimators=350,max_depth=7,min_samples_leaf=8,
                                 max_features="sqrt",class_weight="balanced_subsample",
                                 random_state=42,n_jobs=-1)
        return Pipeline([("imp",SimpleImputer(strategy="median")),("model",m)])
    return HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=15,
                                           learning_rate=.05,l2_regularization=1.5,
                                           random_state=42)

def fit_probability(X,y,kind):
    ks=["logistic","random_forest","gradient"] if kind=="ensemble" else [kind]
    models=[]
    for k in ks:
        try:
            m=base_model(k); m.fit(X,y); models.append(m)
        except Exception:
            pass
    return models

def predict_probability(models,X):
    if not models: return np.full(len(X),.5)
    ps=np.column_stack([m.predict_proba(X)[:,1] for m in models])
    # Equal weighting is intentionally fixed in live/OOS prediction; no tuning on OOS.
    return np.mean(ps,axis=1)

def calibration_fit(models,Xv,yv):
    p=predict_probability(models,Xv)
    if len(np.unique(yv))<2: return None
    cal=LogisticRegression(C=1,max_iter=1000)
    cal.fit(p.reshape(-1,1),yv)
    return cal

def calibrated_predict(models,cal,X):
    p=predict_probability(models,X)
    if cal is None: return np.clip(p,.02,.98)
    return np.clip(cal.predict_proba(p.reshape(-1,1))[:,1],.02,.98)

def grade(prob,ev,conf):
    if prob>=.68 and ev>=.25 and conf>=60: return "A+"
    if prob>=.63 and ev>=.18 and conf>=52: return "A"
    if prob>=.58 and ev>=.10 and conf>=45: return "B"
    return "C"

def walk_forward(d,pair,horizon,train,test,rr,stop_atr,spread,slip,kind,
                 min_prob,one_trade):
    x=build_features(d); x.attrs["pair"]=pair
    y,dirs=make_event_labels(x,pair,rr,stop_atr,spread,slip,horizon)
    valid=x[FEATURES].notna().sum(axis=1)>=int(len(FEATURES)*.40)
    x=x.loc[valid].reset_index(drop=True); y=y.loc[valid].reset_index(drop=True)
    dirs=dirs.loc[valid].reset_index(drop=True)
    x.attrs["pair"]=pair
    n=len(x)
    rows=[]; diagnostics=[]; oos_probs=[]; oos_y=[]; start=train; last_exit=-1

    while start < n:
        end=min(start+test,n)
        # Labels for training must be fully known before the OOS boundary.
        tr_end=max(0,start-horizon)
        train_mask=y.iloc[:tr_end].notna() & dirs.iloc[:tr_end].ne("NONE")
        if train_mask.sum()<180:
            start=end; continue
        Xtr=x.loc[:tr_end-1,FEATURES].loc[train_mask]
        ytr=y.iloc[:tr_end].loc[train_mask].astype(int)

        # Validation is inside the training period only.
        cut=int(len(Xtr)*.80)
        if cut<120 or len(Xtr)-cut<50 or ytr.nunique()<2:
            start=end; continue
        Xfit=Xtr.iloc[:cut]; yfit=ytr.iloc[:cut]
        Xval=Xtr.iloc[cut:]; yval=ytr.iloc[cut:]
        if yval.nunique()<2:
            start=end; continue

        models=fit_probability(Xfit,yfit,kind)
        if not models:
            start=end; continue
        cal=calibration_fit(models,Xval,yval)
        pv=calibrated_predict(models,cal,Xval)
        # Threshold is chosen only from validation, with a minimum sample count.
        # Choose the threshold from REALIZED validation outcomes only.
        # This is deliberately inside the training window; OOS data never
        # participates in threshold selection.
        chosen=float(min_prob)
        best_ev=-999.0
        for th in np.arange(max(0.52,min_prob),.81,.01):
            take=pv>=th
            if take.sum()<12: continue
            wr=float(yval.to_numpy()[take].mean())
            ev=wr*rr-(1-wr)
            if ev>best_ev:
                best_ev=ev; chosen=float(th)

        Xte=x.iloc[start:end][FEATURES]
        pte=calibrated_predict(models,cal,Xte)
        # OOS probability diagnostics are recorded for every candidate setup,
        # before applying the trading threshold.
        for j,pwin in enumerate(pte):
            i=start+j
            if dirs.iloc[i]!="NONE" and pd.notna(y.iloc[i]):
                oos_probs.append(float(pwin)); oos_y.append(int(y.iloc[i]))
        for j,pwin in enumerate(pte):
            i=start+j
            direction=dirs.iloc[i]
            if direction=="NONE" or pwin<chosen: continue
            if one_trade and i<=last_exit: continue
            ev=float(pwin*rr-(1-pwin))
            conf=float(x.iloc[i].engine_conf)
            g=grade(float(pwin),ev,conf)
            if ev<=0: continue
            tr=simulate_trade(x,i,direction,rr,stop_atr,spread,slip,horizon)
            if tr is None: continue
            tr.update(probability=float(pwin),expected_value=ev,threshold=chosen,
                      grade=g,regime=int(x.iloc[i].regime_code),
                      engine_edge=float(x.iloc[i].engine_edge))
            rows.append(tr)
            last_exit=i+1+horizon

        diagnostics.append({"oos_start":x.iloc[start].datetime.iloc[0],
                            "oos_end":x.iloc[end-1].datetime.iloc[-1] if end>start else None,
                            "train_events":int(train_mask.sum()),
                            "validation_events":int(len(Xval)),
                            "threshold":chosen})
        start=end

    out=pd.DataFrame(rows)
    brier = brier_score_loss(oos_y,oos_probs) if len(oos_y)>=20 and len(set(oos_y))==2 else None
    ll = log_loss(oos_y,np.clip(oos_probs,.02,.98)) if len(oos_y)>=20 and len(set(oos_y))==2 else None
    if out.empty:
        return {"trades":0,"wins":0,"losses":0,"win_rate":0.0,"expectancy":0.0,
                "profit_factor":None,"max_drawdown":0.0,"brier":brier,"logloss":ll,
                "signals":out,"diagnostics":pd.DataFrame(diagnostics)}
    pnl=out.pnl_r.to_numpy(float)
    eq=pd.Series(pnl).cumsum()
    dd=eq-eq.cummax()
    wins=pnl[pnl>0]; losses=pnl[pnl<=0]
    return {"trades":len(out),"wins":int((pnl>0).sum()),"losses":int((pnl<=0).sum()),
            "win_rate":float((pnl>0).mean()*100),"expectancy":float(pnl.mean()),
            "profit_factor":float(wins.sum()/abs(losses.sum())) if losses.sum()!=0 else None,
            "max_drawdown":float(dd.min()),"brier":brier,"logloss":ll,
            "signals":out,"diagnostics":pd.DataFrame(diagnostics)}

def chart(d,pair):
    x=build_features(d).tail(240)
    f=go.Figure(go.Candlestick(x=x.datetime,open=x.open,high=x.high,
                                low=x.low,close=x.close,name=pair))
    for n in [20,50,200]:
        f.add_trace(go.Scatter(x=x.datetime,y=x[f"ema{n}"],name=f"EMA{n}"))
    f.update_layout(height=560,xaxis_rangeslider_visible=False,title=f"{pair} — V12.0")
    return f

def engine_table(row):
    names=["Trend","Momentum","Volatility","Market structure","Breakout",
           "Mean reversion","Liquidity","Price action"]
    vals=[row.trend_score,row.momentum_score,row.vol_score,row.structure*100,
          row.breakout_score,row.reversion_score,row.liquidity_score,row.price_score]
    return pd.DataFrame({"Engine":names,"Score":[round(float(v),1) for v in vals],
                         "Bias":["BUY" if v>12 else "SELL" if v<-12 else "NEUTRAL" for v in vals]})

st.set_page_config(page_title="Forex AI Pro V12",page_icon="🤖",layout="wide")
db_init()
st.title("🤖 Forex AI Pro V12")
st.caption("Regime-aware multi-engine AI • event-aligned walk-forward research • paper/manual trading only")

API=api_key()
if not API:
    st.error("TWELVE_DATA_API_KEY is not configured in Streamlit Secrets.")
    st.stop()

with st.sidebar:
    st.header("LIVE SIGNAL")
    pair=st.selectbox("Pair",PAIRS)
    entry_tf=st.selectbox("Entry timeframe",["1m","3m","5m","15m","30m","1h"])
    htf=st.multiselect("Higher timeframes",["1h","4h","1day"],["1h","4h"])
    size=st.slider("Historical candles",600,5000,2500,100)
    model_kind=st.selectbox("Probability model",["ensemble","gradient","random_forest","logistic"])
    min_prob=st.slider("Minimum calibrated trade probability",.50,.80,.58,.01)
    rr=st.slider("R:R",1.0,4.0,2.0,.25)
    stop_atr=st.slider("Stop ATR multiple",.75,3.0,1.5,.05)
    spread=st.number_input("Spread (pips)",0.0,10.0,1.0,.1)
    slip=st.number_input("Slippage (pips)",0.0,5.0,.2,.1)
    risk=st.slider("Risk per trade %",.1,3.0,.5,.1)
    run=st.button("⚡ GENERATE LIVE SIGNAL",type="primary",use_container_width=True)

if run:
    try:
        d=candles(pair,TF[entry_tf],size,API)
        higher=[candles(pair,TF[h],min(size,1800),API) for h in htf]
    except Exception as e:
        st.exception(e); st.stop()
    h=data_health(d,TF[entry_tf])
    a,b,c,e=st.columns(4)
    a.metric("Candles",len(d)); b.metric("Fresh","OK" if h["fresh"] else "STALE")
    c.metric("Gaps","OK" if h["gaps_ok"] else "CHECK"); e.metric("Age",f"{h['age']:.1f}m")
    for issue in h["issues"]: st.warning(issue)
    if not (h["valid"] and h["fresh"] and h["gaps_ok"]):
        st.error("🔴 NO TRADE — data gate failed."); st.stop()

    x=build_features(d); x.attrs["pair"]=pair
    row=x.iloc[-1]
    direction=candidate_direction(row)
    st.plotly_chart(chart(d,pair),use_container_width=True)
    st.subheader("Engine matrix")
    st.dataframe(engine_table(row),use_container_width=True,hide_index=True)

    # Higher-TF direction confirmation is informative, not an absolute gate.
    htf_bias=[]
    for hd in higher:
        hx=build_features(hd).iloc[-1]
        htf_bias.append("BUY" if hx.engine_edge>18 else "SELL" if hx.engine_edge<-18 else "NEUTRAL")

    y,dirs=make_event_labels(x,pair,rr,stop_atr,spread,slip,1)
    train_mask=y.notna() & dirs.ne("NONE")
    if train_mask.sum()<220:
        st.error("Not enough historical setup events for a reliable live model.")
        st.stop()

    cut=int(train_mask.sum()*.80)
    idx=np.where(train_mask.to_numpy())[0]
    fit_idx=idx[:cut]; val_idx=idx[cut:]
    models=fit_probability(x.iloc[fit_idx][FEATURES],y.iloc[fit_idx].astype(int),model_kind)
    cal=calibration_fit(models,x.iloc[val_idx][FEATURES],y.iloc[val_idx].astype(int))
    p=float(calibrated_predict(models,cal,x.iloc[[-1]][FEATURES])[0])

    ev=float(p*rr-(1-p))
    conf=float(row.engine_conf)
    grade_now=grade(p,ev,conf)
    regime={0:"RANGE",1:"TRENDING BULL",2:"TRENDING BEAR",3:"BREAKOUT/EXPANSION",4:"TRANSITION"}[int(row.regime_code)]

    a,b,c,e,f=st.columns(5)
    a.metric("Setup",direction); b.metric("Probability",f"{p*100:.1f}%")
    c.metric("Expected value",f"{ev:.3f} R"); e.metric("Regime",regime); f.metric("Grade",grade_now)

    blockers=[]
    if direction=="NONE": blockers.append("No setup edge.")
    if p<min_prob: blockers.append(f"Calibrated probability below {min_prob*100:.0f}%.")
    if ev<=0: blockers.append("Expected value is not positive.")
    if conf<35: blockers.append("Engine confidence is weak.")
    if regime=="TRANSITION" and conf<55: blockers.append("Unstable regime.")
    if len(htf_bias)>=2 and all(z!="NEUTRAL" for z in htf_bias):
        if direction=="BUY" and sum(z=="BUY" for z in htf_bias)==0: blockers.append("Higher-timeframe conflict.")
        if direction=="SELL" and sum(z=="SELL" for z in htf_bias)==0: blockers.append("Higher-timeframe conflict.")

    payload={"pair":pair,"timeframe":entry_tf,"direction":"NO TRADE","probability":p,
             "expected_value":ev,"grade":grade_now,"regime":regime,"risk_pct":risk,
             "status":"BLOCKED"}
    if blockers:
        st.error("🔴 NO TRADE")
        st.write(" • ".join(blockers))
        payload["explanation"]="; ".join(blockers)
    else:
        av=float(row.atr); px=float(row.close); pp=pip(pair)
        sd=max(av*stop_atr,pp*4)
        entry=px
        sl=entry-sd if direction=="BUY" else entry+sd
        tp=entry+sd*rr if direction=="BUY" else entry-sd*rr
        st.success(f"🟢 {pair} {direction} — {grade_now} — calibrated edge approved")
        a,b,c,e=st.columns(4)
        a.metric("Entry",f"{entry:.6f}"); b.metric("Stop",f"{sl:.6f}")
        c.metric("Target",f"{tp:.6f}"); e.metric("R:R",f"1:{rr:.2f}")
        payload.update(direction=direction,entry=entry,stop_loss=sl,take_profit=tp,status="APPROVED",
                       explanation=f"Engines + calibrated probability + positive EV; HTF={htf_bias}")
    if st.button("📝 JOURNAL SIGNAL",key="j12"):
        journal(payload); st.success("Signal journaled.")

st.divider()
st.header("🧪 V12 REALISTIC WALK-FORWARD BACKTEST")
st.caption("The ML predicts whether the actual candidate setup wins. Training labels never extend into the OOS block. Entry is next-bar with spread/slippage; ambiguous same-bar SL+TP is treated conservatively as SL.")

bp=st.selectbox("Backtest pair",PAIRS,key="bp12")
btf=st.selectbox("Backtest timeframe",["1m","3m","5m","15m","30m","1h"],2,key="btf12")
bsize=st.slider("Backtest candles",1000,5000,3000,100,key="bs12")
horizon=st.number_input("Maximum holding bars",1,30,6)
train=st.number_input("Initial training bars",500,3000,1200,100)
test=st.number_input("OOS block bars",50,500,150,25)
brr=st.slider("Backtest R:R",1.0,4.0,2.0,.25,key="brr12")
bstop=st.slider("Backtest stop ATR",.75,3.0,1.5,.05,key="bst12")
bspread=st.number_input("Backtest spread pips",0.,10.,1.,.1,key="bsp12")
bslip=st.number_input("Backtest slippage pips",0.,5.,.2,.1,key="bsl12")
bone=st.checkbox("One trade at a time",True,key="bone12")
bmodel=st.selectbox("Backtest probability model",["ensemble","gradient","random_forest","logistic"],key="bm12")
bmin=st.slider("Minimum probability",.50,.80,.56,.01,key="bmin12")

if st.button("▶ RUN V12 REALISTIC WALK-FORWARD",type="primary",key="runbt12"):
    try:
        bd=candles(bp,TF[btf],bsize,API)
    except Exception as e:
        st.exception(e); st.stop()
    hh=data_health(bd,TF[btf])
    if not hh["valid"]:
        st.error("Backtest data failed validation."); st.stop()
    r=walk_forward(bd,bp,int(horizon),int(train),int(test),brr,bstop,bspread,bslip,bmodel,bmin,bone)
    a,b,c,e,f,g=st.columns(6)
    a.metric("Trades",r["trades"]); b.metric("Win rate",f"{r['win_rate']:.2f}%")
    c.metric("Expectancy",f"{r['expectancy']:.3f} R")
    e.metric("Profit factor","N/A" if r["profit_factor"] is None else f"{r['profit_factor']:.2f}")
    f.metric("Max DD",f"{r['max_drawdown']:.2f} R")
    g.metric("OOS blocks",len(r["diagnostics"]))
    st.caption(
        f"OOS probability Brier: {'N/A' if r['brier'] is None else f'{r["brier"]:.4f}'} • "
        f"OOS log loss: {'N/A' if r['logloss'] is None else f'{r["logloss"]:.4f}'}"
    )
    if r["signals"].empty:
        st.warning("No trades passed the calibrated probability + positive-EV filter. This is a diagnostic result, not a forced win rate.")
    else:
        st.dataframe(r["signals"].tail(500),use_container_width=True)
        eq=r["signals"].pnl_r.cumsum()
        fig=go.Figure(go.Scatter(x=np.arange(len(eq)),y=eq,mode="lines",name="Cumulative R"))
        fig.update_layout(height=350,title="Out-of-sample equity curve")
        st.plotly_chart(fig,use_container_width=True)
    if not r["diagnostics"].empty:
        st.subheader("Walk-forward diagnostics")
        st.dataframe(r["diagnostics"],use_container_width=True,hide_index=True)
    with sqlite3.connect(DB) as con:
        con.execute("""INSERT INTO backtests(created_at,pair,timeframe,trades,wins,losses,win_rate,
                       expectancy,profit_factor,max_drawdown,brier,logloss,payload)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (now().isoformat(),bp,btf,r["trades"],r["wins"],r["losses"],r["win_rate"],
                     r["expectancy"],r["profit_factor"],r["max_drawdown"],r["brier"],r["logloss"],
                     json.dumps(r,default=str)))
    st.success("V12 walk-forward completed.")

st.divider()
st.header("📒 Signal Journal")
j=load_journal()
if j.empty: st.info("No signals journaled yet.")
else: st.dataframe(j,use_container_width=True)
st.caption(f"Forex AI Pro {APP_VERSION} • Twelve Data • Research/paper-trading only • No guaranteed win rate")
