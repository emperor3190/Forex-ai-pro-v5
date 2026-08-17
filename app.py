import os
import json
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss


APP_VERSION = "V9.1"
TWELVE_DATA_BASE = "https://api.twelvedata.com"
DATA_DIR = ".v9_data"
DB_PATH = os.path.join(DATA_DIR, "forex_ai_pro_v9_1.sqlite3")

DEFAULT_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
]

TIMEFRAMES = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1h", "4h": "4h", "1day": "1day",
}

FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "range_pct", "body_pct",
    "upper_wick_pct", "lower_wick_pct", "atr_pct", "rsi", "adx",
    "macd_hist", "ema9_dist", "ema20_dist", "ema50_dist", "ema20_50",
    "bb_pos", "volatility_20", "momentum_accel", "trend_strength",
    "distance_high_20", "distance_low_20", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "session_asia", "session_london",
    "session_overlap", "session_ny",
]


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def get_api_key() -> str:
    try:
        key = st.secrets.get("TWELVE_DATA_API_KEY", "")
    except Exception:
        key = ""
    return str(key or os.getenv("TWELVE_DATA_API_KEY", ""))


def ensure_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            pair TEXT NOT NULL, engine TEXT NOT NULL, timeframe TEXT NOT NULL,
            direction TEXT NOT NULL, probability REAL, threshold REAL,
            entry REAL, strike REAL, expiry_minutes REAL, stop_loss REAL,
            take_profit REAL, regime TEXT, status TEXT, explanation TEXT,
            payload TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            pair TEXT NOT NULL, timeframe TEXT NOT NULL, engine TEXT NOT NULL,
            trades INTEGER, wins INTEGER, losses INTEGER, win_rate REAL,
            expectancy REAL, profit_factor REAL, max_drawdown REAL,
            brier REAL, payload TEXT)""")


def journal_signal(payload: Dict) -> None:
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""INSERT INTO signals
        (created_at,pair,engine,timeframe,direction,probability,threshold,
         entry,strike,expiry_minutes,stop_loss,take_profit,regime,status,
         explanation,payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (utc_now().isoformat(), payload.get("pair",""),
         payload.get("engine",""), payload.get("timeframe",""),
         payload.get("direction","NO TRADE"), payload.get("probability"),
         payload.get("threshold"), payload.get("entry"), payload.get("strike"),
         payload.get("expiry_minutes"), payload.get("stop_loss"),
         payload.get("take_profit"), payload.get("regime"),
         payload.get("status"), payload.get("explanation",""),
         json.dumps(payload, default=str)))


def journal_backtest(pair: str, timeframe: str, engine: str, metrics: Dict) -> None:
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""INSERT INTO backtests
        (created_at,pair,timeframe,engine,trades,wins,losses,win_rate,
         expectancy,profit_factor,max_drawdown,brier,payload)
         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (utc_now().isoformat(), pair, timeframe, engine,
         metrics.get("trades",0), metrics.get("wins",0),
         metrics.get("losses",0), metrics.get("win_rate",0),
         metrics.get("expectancy",0), metrics.get("profit_factor"),
         metrics.get("max_drawdown",0), metrics.get("brier"),
         json.dumps(metrics, default=str)))


def load_journal(limit=300) -> pd.DataFrame:
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
            conn, params=(int(limit),))


@st.cache_data(ttl=20, show_spinner=False)
def twelve_time_series(symbol: str, interval: str, outputsize: int, api_key: str) -> pd.DataFrame:
    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY is not configured.")
    r = requests.get(f"{TWELVE_DATA_BASE}/time_series",
                     params={"symbol":symbol,"interval":interval,
                             "outputsize":int(outputsize),"apikey":api_key,"format":"JSON"},
                     timeout=20)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") == "error":
        raise RuntimeError(payload.get("message","Twelve Data error."))
    values = payload.get("values")
    if not values:
        raise RuntimeError("Twelve Data returned no candles.")
    df = pd.DataFrame(values)
    required = ["datetime","open","high","low","close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing fields: {missing}")
    for c in ["open","high","low","close","volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df:
        df["volume"] = np.nan
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    return (df.dropna(subset=required).sort_values("datetime")
              .drop_duplicates("datetime").reset_index(drop=True))


@st.cache_data(ttl=10, show_spinner=False)
def twelve_quote(symbol: str, api_key: str) -> Dict:
    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY is not configured.")
    r = requests.get(f"{TWELVE_DATA_BASE}/quote",
                     params={"symbol":symbol,"apikey":api_key}, timeout=15)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") == "error":
        raise RuntimeError(payload.get("message","Twelve Data quote error."))
    return payload


def validate_market_data(df: pd.DataFrame, timeframe: str) -> Dict:
    result = {"valid":True,"fresh":True,"gaps_ok":True,"issues":[],
              "rows":len(df),"age_minutes":None}
    if df.empty:
        result.update(valid=False,fresh=False,gaps_ok=False)
        result["issues"].append("No market data.")
        return result
    if df[["open","high","low","close"]].isna().any().any():
        result["valid"] = False; result["issues"].append("Missing OHLC values.")
    if (df["high"] < df["low"]).any():
        result["valid"] = False; result["issues"].append("High below low.")
    if (df["high"] < df["open"]).any() or (df["high"] < df["close"]).any():
        result["valid"] = False; result["issues"].append("Open/close above high.")
    if (df["low"] > df["open"]).any() or (df["low"] > df["close"]).any():
        result["valid"] = False; result["issues"].append("Open/close below low.")
    if df["datetime"].duplicated().any():
        result["valid"] = False; result["issues"].append("Duplicate timestamps.")
    minutes = {"1min":1,"3min":3,"5min":5,"15min":15,"30min":30,
               "1h":60,"4h":240,"1day":1440}.get(timeframe,5)
    age = max(0,(utc_now()-pd.Timestamp(df["datetime"].iloc[-1])).total_seconds()/60)
    result["age_minutes"] = round(age,2)
    if age > minutes*2+3:
        result["fresh"] = False
        result["issues"].append(f"Data appears stale ({age:.1f} min old).")
    diffs = df["datetime"].diff().dt.total_seconds().div(60).dropna()
    if not diffs.empty and (diffs > minutes*2.5).any():
        result["gaps_ok"] = False
        result["issues"].append("Large candle gap detected.")
    return result


def ema(s,n): return s.ewm(span=n,adjust=False).mean()


def rsi(s,n=14):
    d=s.diff(); gain=d.clip(lower=0); loss=-d.clip(upper=0)
    ag=gain.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    al=loss.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return 100-100/(1+rs)


def atr(df,n=14):
    pc=df["close"].shift(1)
    tr=pd.concat([df["high"]-df["low"],(df["high"]-pc).abs(),
                  (df["low"]-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,min_periods=n,adjust=False).mean()


def adx(df,n=14):
    up=df["high"].diff(); down=-df["low"].diff()
    plus=pd.Series(np.where((up>down)&(up>0),up,0.),index=df.index)
    minus=pd.Series(np.where((down>up)&(down>0),down,0.),index=df.index)
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift()).abs(),
                  (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
    av=tr.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    pdi=100*plus.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/av
    mdi=100*minus.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/av
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,min_periods=n,adjust=False).mean()


def add_indicators(df):
    x=df.copy()
    for n in [9,20,50,100,200]: x[f"ema{n}"]=ema(x.close,n)
    x["rsi"]=rsi(x.close); x["atr"]=atr(x); x["adx"]=adx(x)
    macd=ema(x.close,12)-ema(x.close,26)
    x["macd_hist"]=macd-ema(macd,9)
    mid=x.close.rolling(20).mean(); std=x.close.rolling(20).std()
    x["bb_mid"]=mid; x["bb_upper"]=mid+2*std; x["bb_lower"]=mid-2*std
    x["ret_1"]=x.close.pct_change(1); x["ret_3"]=x.close.pct_change(3)
    x["ret_5"]=x.close.pct_change(5); x["ret_10"]=x.close.pct_change(10)
    x["range_pct"]=(x.high-x.low)/x.close
    x["body_pct"]=(x.close-x.open).abs()/x.close
    x["upper_wick_pct"]=(x.high-x[["open","close"]].max(axis=1))/x.close
    x["lower_wick_pct"]=(x[["open","close"]].min(axis=1)-x.low)/x.close
    x["atr_pct"]=x.atr/x.close
    x["ema9_dist"]=(x.close-x.ema9)/x.close
    x["ema20_dist"]=(x.close-x.ema20)/x.close
    x["ema50_dist"]=(x.close-x.ema50)/x.close
    x["ema20_50"]=(x.ema20-x.ema50)/x.close
    x["bb_pos"]=(x.close-x.bb_lower)/(x.bb_upper-x.bb_lower).replace(0,np.nan)
    x["volatility_20"]=x.ret_1.rolling(20).std()
    x["momentum_accel"]=x.ret_3-x.ret_3.shift(3)
    x["trend_strength"]=x.ema20_50*x.adx
    x["distance_high_20"]=(x.high.rolling(20).max()-x.close)/x.close
    x["distance_low_20"]=(x.close-x.low.rolling(20).min())/x.close
    hours=x.datetime.dt.hour; dow=x.datetime.dt.dayofweek
    x["hour_sin"]=np.sin(2*np.pi*hours/24); x["hour_cos"]=np.cos(2*np.pi*hours/24)
    x["dow_sin"]=np.sin(2*np.pi*dow/7); x["dow_cos"]=np.cos(2*np.pi*dow/7)
    x["session_asia"]=((hours>=0)&(hours<7)).astype(int)
    x["session_london"]=((hours>=7)&(hours<12)).astype(int)
    x["session_overlap"]=((hours>=12)&(hours<16)).astype(int)
    x["session_ny"]=((hours>=16)&(hours<21)).astype(int)
    return x.replace([np.inf,-np.inf],np.nan)


def structure_snapshot(df,n=40):
    if len(df)<n: return {"trend":"UNKNOWN","structure":"UNKNOWN","sweep":"NONE"}
    x=df.tail(n); hh=x.high.rolling(5,center=True).max()
    ll=x.low.rolling(5,center=True).min()
    sh=x.loc[x.high.eq(hh),"high"].dropna(); sl=x.loc[x.low.eq(ll),"low"].dropna()
    trend="RANGE"
    if len(sh)>=2 and len(sl)>=2:
        if sh.iloc[-1]>sh.iloc[-2] and sl.iloc[-1]>sl.iloc[-2]: trend="BULLISH"
        elif sh.iloc[-1]<sh.iloc[-2] and sl.iloc[-1]<sl.iloc[-2]: trend="BEARISH"
    prev_high=x.high.iloc[:-1].max(); prev_low=x.low.iloc[:-1].min(); last=x.iloc[-1]
    sweep="NONE"
    if last.high>prev_high and last.close<prev_high: sweep="BUY_SIDE_SWEEP"
    elif last.low<prev_low and last.close>prev_low: sweep="SELL_SIDE_SWEEP"
    return {"trend":trend,"structure":trend,"sweep":sweep,
            "prev_high":float(prev_high),"prev_low":float(prev_low)}


def detect_regime(x):
    if len(x)<100: return "UNKNOWN"
    last=x.iloc[-1]; adxv=float(last.adx) if pd.notna(last.adx) else 0
    atrpct=float(last.atr_pct) if pd.notna(last.atr_pct) else 0
    q=x.atr_pct.rolling(100).quantile(.80).iloc[-1]
    if adxv>=25:
        if last.ema20>last.ema50: return "TRENDING_BULLISH"
        if last.ema20<last.ema50: return "TRENDING_BEARISH"
    if pd.notna(q) and atrpct>=q: return "HIGH_VOLATILITY"
    if adxv<18: return "RANGING"
    return "TRANSITION"


def fvg_snapshot(x):
    if len(x)<3: return "NONE"
    a,c=x.iloc[-3],x.iloc[-1]
    if a.high<c.low: return "BULLISH_FVG"
    if a.low>c.high: return "BEARISH_FVG"
    return "NONE"


def make_supervised(df,horizon):
    x=add_indicators(df); future=x.close.shift(-horizon); valid=future.notna()
    y=(future.loc[valid]>x.loc[valid,"close"]).astype(int)
    return x.loc[valid,FEATURES].copy(),y


def build_model(model_type="logistic"):
    if model_type=="random_forest":
        return Pipeline([("imputer",SimpleImputer(strategy="median")),
            ("model",RandomForestClassifier(n_estimators=350,max_depth=7,
             min_samples_leaf=8,class_weight="balanced_subsample",
             random_state=42,n_jobs=-1))])
    return Pipeline([("imputer",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("model",LogisticRegression(max_iter=1500,class_weight="balanced",
                                    random_state=42))])


def chronological_split(X,y,train_ratio=.70,validation_ratio=.15):
    n=len(X); a=int(n*train_ratio); b=int(n*(train_ratio+validation_ratio))
    return X.iloc[:a],X.iloc[a:b],X.iloc[b:],y.iloc[:a],y.iloc[a:b],y.iloc[b:]


def fit_probability_model(df,horizon,model_type="logistic"):
    X,y=make_supervised(df,horizon)
    if len(X)<350 or y.nunique()<2: return {"ok":False,"message":"Not enough diverse historical samples."}
    Xtr,Xv,Xte,ytr,yv,yte=chronological_split(X,y)
    if ytr.nunique()<2 or yte.nunique()<2:
        return {"ok":False,"message":"Chronological split lacks class diversity."}
    model=build_model(model_type); model.fit(Xtr,ytr)
    p_val=model.predict_proba(Xv)[:,1]; p_test=model.predict_proba(Xte)[:,1]
    cal=None
    if yv.nunique()>=2:
        cal=LogisticRegression(max_iter=1000)
        cal.fit(np.clip(p_val,.001,.999).reshape(-1,1),yv)
        calibrated=cal.predict_proba(np.clip(p_test,.001,.999).reshape(-1,1))[:,1]
    else: calibrated=p_test
    return {"ok":True,"model":model,"calibrator":cal,"features":FEATURES,
            "horizon":horizon,"validation_brier":float(brier_score_loss(yv,p_val)),
            "test_brier":float(brier_score_loss(yte,calibrated)),
            "test_accuracy":float(accuracy_score(yte,calibrated>=.5)),
            "test_logloss":float(log_loss(yte,np.clip(calibrated,.001,.999))),
            "test_size":len(yte)}


def live_probability(fitted,df):
    if not fitted.get("ok"): return None
    x=add_indicators(df).iloc[-1:][FEATURES]
    raw=float(fitted["model"].predict_proba(x)[0,1])
    if fitted.get("calibrator") is None: return raw
    return float(fitted["calibrator"].predict_proba(
        np.array([[np.clip(raw,.001,.999)]]))[0,1])


def confluence_score(df,higher_frames):
    x=add_indicators(df); last=x.iloc[-1]; structure=structure_snapshot(x)
    regime=detect_regime(x); fvg=fvg_snapshot(x)
    bull=bear=0; reasons=[]
    if last.ema20>last.ema50: bull+=15; reasons.append("EMA20 above EMA50")
    elif last.ema20<last.ema50: bear+=15; reasons.append("EMA20 below EMA50")
    if last.macd_hist>0: bull+=8
    elif last.macd_hist<0: bear+=8
    if pd.notna(last.rsi):
        if 52<=last.rsi<=70: bull+=8
        elif 30<=last.rsi<=48: bear+=8
    if pd.notna(last.adx) and last.adx>=25:
        if bull>bear: bull+=6
        elif bear>bull: bear+=6
    if structure["trend"]=="BULLISH": bull+=12; reasons.append("Bullish structure")
    elif structure["trend"]=="BEARISH": bear+=12; reasons.append("Bearish structure")
    if structure["sweep"]=="SELL_SIDE_SWEEP": bull+=10; reasons.append("Sell-side liquidity sweep")
    elif structure["sweep"]=="BUY_SIDE_SWEEP": bear+=10; reasons.append("Buy-side liquidity sweep")
    if fvg=="BULLISH_FVG": bull+=5; reasons.append("Bullish FVG")
    elif fvg=="BEARISH_FVG": bear+=5; reasons.append("Bearish FVG")
    ht_bull=ht_bear=0
    for h in higher_frames:
        if len(h)<80: continue
        hh=add_indicators(h).iloc[-1]
        if hh.ema20>hh.ema50: ht_bull+=1
        elif hh.ema20<hh.ema50: ht_bear+=1
    if ht_bull>ht_bear: bull+=16; reasons.append(f"Higher-timeframe bullish ({ht_bull}/{len(higher_frames)})")
    elif ht_bear>ht_bull: bear+=16; reasons.append(f"Higher-timeframe bearish ({ht_bear}/{len(higher_frames)})")
    direction="BUY" if bull>bear else "SELL" if bear>bull else "NO TRADE"
    total=max(bull+bear,1); score=50+45*abs(bull-bear)/total
    blockers=[]
    if len(df)<250: blockers.append("At least 250 candles recommended.")
    if regime=="UNKNOWN": blockers.append("Unknown market regime.")
    if ht_bull and ht_bear: blockers.append("Higher-timeframe conflict.")
    if direction=="NO TRADE": blockers.append("No directional edge.")
    if regime=="HIGH_VOLATILITY": blockers.append("Extreme volatility: wait for stabilization.")
    return {"direction":direction,"score":round(float(score),2),"bull_score":bull,
            "bear_score":bear,"regime":regime,"trend":structure["trend"],
            "structure":structure["structure"],"liquidity":structure["sweep"],
            "fvg":fvg,"rsi":float(last.rsi) if pd.notna(last.rsi) else None,
            "adx":float(last.adx) if pd.notna(last.adx) else None,
            "atr":float(last.atr) if pd.notna(last.atr) else None,
            "reasons":reasons,"blockers":blockers}


def binary_break_even(payout_pct):
    payout=payout_pct/100
    return 1/(1+payout)*100 if payout>0 else 100


def binary_ev(probability_pct,payout_pct):
    p=probability_pct/100; payout=payout_pct/100
    return p*payout-(1-p)


def binary_threshold(payout_pct,safety_margin_pct):
    return min(99.,binary_break_even(payout_pct)+safety_margin_pct)


def pip_size(pair): return .01 if pair.endswith("/JPY") else .0001


def forex_plan(pair,direction,entry,atr_value,balance,risk_pct,rr):
    p=pip_size(pair); stop_distance=max(float(atr_value)*1.5,p*5)
    risk_amount=balance*risk_pct/100
    if direction=="BUY": sl,tp=entry-stop_distance,entry+stop_distance*rr
    else: sl,tp=entry+stop_distance,entry-stop_distance*rr
    units=risk_amount/max(stop_distance,1e-12)
    return {"entry":float(entry),"stop_loss":float(sl),"take_profit":float(tp),
            "risk_amount":float(risk_amount),"units_estimate":float(units),
            "pip_distance":float(stop_distance/p),"rr":float(rr)}


def binary_signal(pair,df,higher_frames,fitted_by_expiry,payout_pct,
                  safety_margin_pct,strike_override=None):
    confluence=confluence_score(df,higher_frames); price=float(df.close.iloc[-1])
    strike=float(strike_override) if strike_override is not None else price
    threshold=binary_threshold(payout_pct,safety_margin_pct); scored=[]
    for minutes,fitted in fitted_by_expiry.items():
        p=live_probability(fitted,df) if fitted.get("ok") else None
        if p is None: continue
        call_p=p*100
        if confluence["direction"]=="BUY": direction,prob="CALL",call_p
        elif confluence["direction"]=="SELL": direction,prob="PUT",100-call_p
        else: continue
        scored.append({"expiry_minutes":minutes,"direction":direction,
                       "probability":prob,"ev":binary_ev(prob,payout_pct)})
    if not scored:
        return {"approved":False,"direction":"NO TRADE",
                "reason":"No validated probability model is available.",
                "strike":strike,"price":price,"candidates":[]}
    best=max(scored,key=lambda z:(z["probability"],z["ev"]))
    approved=(best["probability"]>=threshold and best["ev"]>0 and
              not confluence["blockers"])
    return {"approved":approved,"direction":best["direction"] if approved else "NO TRADE",
            "probability":best["probability"],"expiry_minutes":best["expiry_minutes"],
            "strike":strike,"reference_price":price,"payout_pct":payout_pct,
            "break_even_pct":binary_break_even(payout_pct),"threshold_pct":threshold,
            "expected_value_per_unit":best["ev"],"candidates":scored,
            "confluence":confluence,
            "reason":"Approved" if approved else ("; ".join(confluence["blockers"]) or
                                                    "Probability/EV below threshold.")}


def make_direction_labels(df,horizon):
    future=df.close.shift(-horizon); y=pd.Series(np.nan,index=df.index,dtype=float)
    valid=future.notna(); y.loc[valid]=(future.loc[valid]>df.loc[valid,"close"]).astype(int)
    return y


def walk_forward(df,horizon,train_size,test_size,min_probability,
                 binary=False,payout_pct=80,safety_margin_pct=5):
    x=add_indicators(df); y=make_direction_labels(x,horizon); X=x[FEATURES].copy()
    valid=y.notna() & X.notna().any(axis=1); X,y,x=X.loc[valid],y.loc[valid].astype(int),x.loc[valid]
    if len(X)<train_size+test_size+50:
        return {"error":"Not enough history for selected walk-forward window."}
    rows=[]; start=train_size
    while start<len(X):
        end=min(start+test_size,len(X)); Xtr,ytr=X.iloc[:start],y.iloc[:start]
        Xte,yte=X.iloc[start:end],y.iloc[start:end]
        if ytr.nunique()<2: start=end; continue
        model=build_model("logistic"); model.fit(Xtr,ytr)
        p_call=model.predict_proba(Xte)[:,1]*100
        threshold=max(binary_break_even(payout_pct)+safety_margin_pct,min_probability) if binary else min_probability
        for j,pc in enumerate(p_call):
            direction=("CALL" if pc>=50 else "PUT") if binary else ("BUY" if pc>=50 else "SELL")
            p=pc if pc>=50 else 100-pc
            if p<threshold: continue
            actual=int(yte.iloc[j]); win=actual==(direction in ("CALL","BUY"))
            rows.append({"timestamp":x.datetime.iloc[start+j],"direction":direction,
                         "probability":p,"win":int(win),
                         "ev":binary_ev(p,payout_pct) if binary else None})
        start=end
    out=pd.DataFrame(rows)
    if out.empty:
        return {"trades":0,"wins":0,"losses":0,"win_rate":0,"expectancy":0,
                "profit_factor":None,"max_drawdown":0,"signals":out,
                "message":"No trades passed the probability threshold."}
    pnl=np.where(out.win.eq(1),payout_pct/100 if binary else 1.,-1.)
    equity=pd.Series(pnl).cumsum(); dd=equity-equity.cummax()
    wins=pnl[pnl>0]; losses=pnl[pnl<0]
    return {"trades":len(out),"wins":int(out.win.sum()),
            "losses":int((1-out.win).sum()),"win_rate":float(out.win.mean()*100),
            "expectancy":float(np.mean(pnl)),
            "profit_factor":float(wins.sum()/abs(losses.sum())) if losses.sum() else None,
            "max_drawdown":float(dd.min()),"signals":out,"threshold":threshold,
            "payout_pct":payout_pct if binary else None}


def chart(df,pair):
    x=add_indicators(df).tail(220); fig=go.Figure()
    fig.add_trace(go.Candlestick(x=x.datetime,open=x.open,high=x.high,low=x.low,close=x.close,name=pair))
    fig.add_trace(go.Scatter(x=x.datetime,y=x.ema20,name="EMA20"))
    fig.add_trace(go.Scatter(x=x.datetime,y=x.ema50,name="EMA50"))
    fig.update_layout(height=560,xaxis_rangeslider_visible=False,
                      margin=dict(l=10,r=10,t=35,b=10),title=f"{pair} — V9.1 validated market view")
    return fig


st.set_page_config(page_title="Forex AI Pro V9.1",page_icon="🤖",layout="wide")
ensure_db()
st.title("🤖 Forex AI Pro V9.1")
st.caption("Leakage-aware probability • regime detection • multi-timeframe confluence • Forex + Binary research")

api_key=get_api_key()
if not api_key:
    st.error("TWELVE_DATA_API_KEY is not configured in Streamlit Secrets.")
    st.stop()

with st.sidebar:
    st.header("V9.1 Controls")
    pair=st.selectbox("Forex pair",DEFAULT_PAIRS)
    entry_tf_label=st.selectbox("Entry timeframe",["1m","3m","5m","15m","30m","1h"])
    higher_labels=st.multiselect("Higher timeframes",["1h","4h"],default=["1h","4h"])
    outputsize=st.slider("Historical candles",400,5000,1500,100)
    model_type=st.selectbox("ML model",["logistic","random_forest"])
    st.divider()
    min_probability=st.slider("Minimum model probability %",50,90,65)
    risk_pct=st.slider("Forex risk %",.1,3.,1.,.1)
    rr=st.slider("Forex R:R",1.,5.,2.,.25)
    balance=st.number_input("Research account balance",100.,1000000.,1000.)
    st.divider()
    payout=st.slider("Binary payout %",50,95,80)
    safety_margin=st.slider("Binary probability safety margin %",0,15,5)
    strike_override_text=st.text_input("Broker strike/reference (optional)")
    expiry_candidates=st.multiselect("Binary expiry minutes",[1,3,5,10,15,30],default=[1,3,5,10,15])
    run=st.button("🔄 Analyze V9.1",type="primary",use_container_width=True)

@st.cache_data(ttl=30,show_spinner=False)
def load_frames(pair,entry_interval,higher_labels,outputsize,api_key):
    frames={"entry":twelve_time_series(pair,entry_interval,outputsize,api_key)}
    for label in higher_labels:
        frames[label]=twelve_time_series(pair,TIMEFRAMES[label],min(outputsize,1500),api_key)
    return frames


if run:
    try: frames=load_frames(pair,TIMEFRAMES[entry_tf_label],tuple(higher_labels),outputsize,api_key)
    except Exception as exc: st.exception(exc); st.stop()
    entry=frames["entry"]; validation=validate_market_data(entry,TIMEFRAMES[entry_tf_label])
    st.subheader("1. Data health")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Candles",len(entry)); c2.metric("Freshness","OK" if validation["fresh"] else "STALE")
    c3.metric("Gaps","OK" if validation["gaps_ok"] else "CHECK"); c4.metric("Issues",len(validation["issues"]))
    for issue in validation["issues"]: st.warning(issue)
    if not validation["valid"] or not validation["fresh"] or not validation["gaps_ok"]:
        st.error("NO TRADE: data-quality gate failed."); st.stop()

    higher_frames=[frames[h] for h in higher_labels]
    st.subheader("2. Market intelligence")
    conf=confluence_score(entry,higher_frames)
    cols=st.columns(7)
    cols[0].metric("Confluence",f'{conf["score"]:.1f}/100')
    cols[1].metric("Direction",conf["direction"]); cols[2].metric("Regime",conf["regime"])
    cols[3].metric("Structure",conf["structure"]); cols[4].metric("Liquidity",conf["liquidity"])
    cols[5].metric("RSI",f'{conf["rsi"]:.1f}' if conf["rsi"] is not None else "N/A")
    cols[6].metric("ADX",f'{conf["adx"]:.1f}' if conf["adx"] is not None else "N/A")
    st.plotly_chart(chart(entry,pair),use_container_width=True)

    st.subheader("3. Calibrated probability engine")
    forex_fit=fit_probability_model(entry,1,model_type)
    binary_fits={}
    if forex_fit["ok"]:
        p_call=live_probability(forex_fit,entry)
        p_dir=p_call*100 if conf["direction"]=="BUY" else (1-p_call)*100
        st.metric("Calibrated directional probability",f"{p_dir:.2f}%")
        st.caption(f"Unseen-test Brier {forex_fit['test_brier']:.4f} • accuracy {forex_fit['test_accuracy']*100:.1f}% • log-loss {forex_fit['test_logloss']:.4f}")
    else: st.warning(forex_fit["message"])

    tf_minutes={"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60}[entry_tf_label]
    for mins in expiry_candidates:
        horizon=max(1,int(round(mins/tf_minutes)))
        fit=fit_probability_model(entry,horizon,model_type)
        if fit["ok"]: binary_fits[mins]=fit
    if binary_fits:
        table=[]
        for mins,fit in binary_fits.items():
            pc=live_probability(fit,entry)
            table.append({"Expiry (min)":mins,"CALL probability":round(pc*100,2),
                          "Test Brier":round(fit["test_brier"],4),
                          "Test accuracy %":round(fit["test_accuracy"]*100,2),
                          "Samples":fit["test_size"]})
        st.dataframe(pd.DataFrame(table),use_container_width=True)

    st.subheader("4. Forex decision engine")
    forex_payload={"pair":pair,"engine":"FOREX","timeframe":entry_tf_label,
                   "direction":"NO TRADE","status":"BLOCKED","regime":conf["regime"]}
    if forex_fit["ok"] and conf["direction"] in ("BUY","SELL"):
        p_call=live_probability(forex_fit,entry); p_dir=p_call*100 if conf["direction"]=="BUY" else (1-p_call)*100
        blockers=list(conf["blockers"])
        if p_dir<min_probability: blockers.append("Calibrated probability below threshold.")
        if blockers:
            st.warning("NO TRADE — "+" ".join(blockers)); forex_payload.update(probability=p_dir,threshold=min_probability,explanation="; ".join(blockers))
        else:
            plan=forex_plan(pair,conf["direction"],float(entry.close.iloc[-1]),float(conf["atr"]),balance,risk_pct,rr)
            st.success(f'FOREX {conf["direction"]} APPROVED — {p_dir:.2f}%')
            st.json({"pair":pair,"direction":conf["direction"],"probability":round(p_dir,2),**plan,"regime":conf["regime"]})
            forex_payload.update(direction=conf["direction"],probability=p_dir,threshold=min_probability,
                                 entry=plan["entry"],stop_loss=plan["stop_loss"],take_profit=plan["take_profit"],
                                 regime=conf["regime"],status="APPROVED",explanation="; ".join(conf["reasons"]))
    if st.button("📝 Journal Forex result",key="journal_forex"):
        journal_signal(forex_payload); st.success("Forex signal saved.")

    st.subheader("5. Binary options research engine")
    strike_override=None
    if strike_override_text.strip():
        try: strike_override=float(strike_override_text.strip())
        except ValueError: st.error("Broker strike/reference must be numeric.")
    if binary_fits:
        binary=binary_signal(pair,entry,higher_frames,binary_fits,payout,safety_margin,strike_override)
        bc=st.columns(6)
        bc[0].metric("Decision",binary["direction"]); bc[1].metric("Probability",f'{binary.get("probability",0):.2f}%')
        bc[2].metric("Expiry",f'{binary.get("expiry_minutes","N/A")} min')
        bc[3].metric("Strike",f'{binary.get("strike",0):.6f}')
        bc[4].metric("Break-even",f'{binary.get("break_even_pct",binary_break_even(payout)):.2f}%')
        bc[5].metric("EV/unit",f'{binary.get("expected_value_per_unit",0):.3f}')
        (st.success if binary["approved"] else st.warning)(
            f'{binary["direction"]} {"APPROVED" if binary["approved"] else "— NO TRADE"} • {binary.get("probability",0):.2f}% • {binary.get("expiry_minutes","N/A")} min'
        )
        if binary.get("candidates"):
            st.dataframe(pd.DataFrame(binary["candidates"]).sort_values("probability",ascending=False),use_container_width=True)
        binary_payload={"pair":pair,"engine":"BINARY","timeframe":entry_tf_label,
                        "direction":binary["direction"],"probability":binary.get("probability"),
                        "threshold":binary.get("threshold_pct"),"entry":binary.get("reference_price"),
                        "strike":binary.get("strike"),"expiry_minutes":binary.get("expiry_minutes"),
                        "regime":conf["regime"],"status":"APPROVED" if binary["approved"] else "BLOCKED",
                        "explanation":binary["reason"]}
        if st.button("📝 Journal Binary result",key="journal_binary"):
            journal_signal(binary_payload); st.success("Binary signal saved.")

    st.subheader("6. Explainable evidence")
    for reason in conf["reasons"]: st.write("•",reason)
    if conf["blockers"]: st.warning("Hard blockers: "+" | ".join(conf["blockers"]))


st.divider()
st.header("🧪 V9.1 Strict Walk-forward Backtesting")
bt_pair=st.selectbox("Backtest pair",DEFAULT_PAIRS,key="bt_pair")
bt_tf_label=st.selectbox("Backtest timeframe",["1m","3m","5m","15m","30m","1h"],index=2,key="bt_tf")
bt_horizon=st.number_input("Outcome horizon (candles)",1,30,1)
bt_train=st.number_input("Initial training candles",250,3000,600,50)
bt_test=st.number_input("Test block candles",25,500,100,25)
bt_min_prob=st.slider("Backtest minimum probability",50,90,65,key="bt_prob")
bt_payout=st.slider("Backtest binary payout %",50,95,80,key="bt_payout")

if st.button("▶ Run Forex + Binary walk-forward"):
    try: bt_df=twelve_time_series(bt_pair,TIMEFRAMES[bt_tf_label],int(outputsize),api_key)
    except Exception as exc: st.exception(exc); st.stop()
    health=validate_market_data(bt_df,TIMEFRAMES[bt_tf_label])
    if not health["valid"]: st.error("Backtest data failed validation."); st.stop()
    forex_bt=walk_forward(bt_df,int(bt_horizon),int(bt_train),int(bt_test),float(bt_min_prob),binary=False)
    binary_bt=walk_forward(bt_df,int(bt_horizon),int(bt_train),int(bt_test),float(bt_min_prob),binary=True,payout_pct=float(bt_payout),safety_margin_pct=5)
    for title,res in [("Forex",forex_bt),("Binary",binary_bt)]:
        st.subheader(title)
        if "error" in res: st.error(res["error"]); continue
        a,b,c,d,e=st.columns(5)
        a.metric("Trades",res["trades"]); b.metric("Win rate",f'{res["win_rate"]:.2f}%')
        c.metric("Expectancy",f'{res["expectancy"]:.3f}')
        d.metric("Profit factor","N/A" if res["profit_factor"] is None else f'{res["profit_factor"]:.2f}')
        e.metric("Max DD",f'{res["max_drawdown"]:.2f}')
        if title=="Binary": st.caption(f"Break-even at {bt_payout}% payout: {binary_break_even(bt_payout):.2f}%")
        if not res["signals"].empty: st.dataframe(res["signals"].tail(200),use_container_width=True)
        journal_backtest(bt_pair,bt_tf_label,title.upper(),res)
    st.success("Walk-forward test completed. Research statistics only.")


st.divider()
st.header("📒 Signal journal")
journal=load_journal(200)
if journal.empty: st.info("No signals have been journaled yet.")
else: st.dataframe(journal,use_container_width=True)
st.caption(f"Forex AI Pro {APP_VERSION} • Twelve Data • Research/paper-trading only")
