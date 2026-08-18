"""V12 AI Trading Platform - single-file modular Forex + Binary framework.
Research/paper trading only; no profit guarantee. Live adapters require official broker APIs.
Run: streamlit run v12app.py
"""
from __future__ import annotations
import uuid, random, math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
import numpy as np, pandas as pd

@dataclass
class Config:
    balance: float=10000.; risk=.005; max_daily_loss=.03; max_drawdown=.15
    max_open=3; min_score=72.; min_binary_conf=72.; binary_payout=.70
    ema_fast=9; ema20=20; ema50=50; ema100=100; ema200=200; rsi_n=14; atr_n=14
    breakout_n=20; sr_n=100; max_spread=30.; binary_expiry=15
    paper=True; session_filter=True; news_filter=False

@dataclass
class EntryDecision:
    approved: bool; market: str; symbol: str; direction: str; entry: float
    zone_low: float; zone_high: float; quality: float; confidence: float
    sl: Optional[float]=None; tp: Optional[float]=None; rr: Optional[float]=None
    expiry_minutes: Optional[int]=None; expiry_time: Optional[datetime]=None
    reasons: List[str]=field(default_factory=list); vetoes: List[str]=field(default_factory=list)

@dataclass
class Trade:
    id:str; symbol:str; market:str; direction:str; entry:float; size:float
    opened_at:datetime; sl:Optional[float]=None; tp:Optional[float]=None
    expiry:Optional[datetime]=None; score:float=0.; confidence:float=0.; pnl:float=0.; result:str="OPEN"

# ---------- indicators ----------
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))
def atr(df,n=14):
    tr=pd.concat([df.high-df.low,(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def macd(s):
    m=ema(s,12)-ema(s,26); sig=ema(m,9); return m,sig,m-sig
def stoch(df,n=14):
    lo=df.low.rolling(n).min(); hi=df.high.rolling(n).max(); return 100*(df.close-lo)/(hi-lo).replace(0,np.nan)
def roc(s,n=10): return s.pct_change(n)*100
def adx(df,n=14):
    up=df.high.diff(); dn=-df.low.diff(); p=up.where((up>dn)&(up>0),0); m=dn.where((dn>up)&(dn>0),0); a=atr(df,n)
    pdi=100*p.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan); mdi=100*m.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    return (100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)).ewm(alpha=1/n,adjust=False).mean(),pdi,mdi

# ---------- data ----------
class DataEngine:
    def __init__(self): self.cache={}; self.connected=True
    def clean(self,df):
        x=df.copy(); x.columns=[str(c).lower().strip() for c in x.columns]
        need={'open','high','low','close'}
        if not need.issubset(x): raise ValueError(f"CSV needs {sorted(need)}")
        for c in need:x[c]=pd.to_numeric(x[c],errors='coerce')
        x=x.dropna(subset=list(need)); x=x[~x.index.duplicated(keep='last')]
        if not isinstance(x.index,pd.DatetimeIndex): x.index=pd.RangeIndex(len(x))
        return x
    def ingest(self,symbol,tf,df): self.cache[(symbol,tf)]=self.clean(df); return self.cache[(symbol,tf)]

# ---------- analysis engines ----------
class TrendEngine:
    def run(self,df,c):
        p=df.close.iloc[-1]; e=[ema(df.close,n).iloc[-1] for n in (c.ema20,c.ema50,c.ema100,c.ema200)]; ax=adx(df,c.atr_n)[0].iloc[-1]
        bull=sum(p>x for x in e); bear=sum(p<x for x in e)
        state='STRONG BULLISH' if bull==4 and ax>=25 else 'STRONG BEARISH' if bear==4 and ax>=25 else 'WEAK BULLISH' if bull>bear else 'WEAK BEARISH' if bear>bull else 'SIDEWAYS'
        return {'state':state,'ema9':float(ema(df.close,c.ema_fast).iloc[-1]),'ema20':float(e[0]),'ema50':float(e[1]),'ema100':float(e[2]),'ema200':float(e[3]),'adx':float(ax)}
class MomentumEngine:
    def run(self,df,c):
        r=float(rsi(df.close,c.rsi_n).iloc[-1]); s=float(stoch(df).iloc[-1]); _,_,mh=macd(df.close); h=float(mh.iloc[-1]); ro=float(roc(df.close).iloc[-1]);
        state='BULLISH' if (r>50)+(s>50)+(h>0)+(ro>0)>=3 else 'BEARISH' if (r<50)+(s<50)+(h<0)+(ro<0)>=3 else 'NEUTRAL'
        div='NONE'
        if len(df)>12:
            pr=df.close.iloc[-1]-df.close.iloc[-10]; rr=rsi(df.close,c.rsi_n); rd=rr.iloc[-1]-rr.iloc[-10]
            div='BEARISH_DIVERGENCE' if pr>0 and rd<0 else 'BULLISH_DIVERGENCE' if pr<0 and rd>0 else 'NONE'
        return {'state':state,'rsi':r,'stoch':s,'macd_hist':h,'roc':ro,'divergence':div,'exhaustion':r>75 or r<25,'acceleration':float(h-mh.iloc[-2]) if len(df)>1 else 0}
class VolatilityEngine:
    def run(self,df,c):
        a=atr(df,c.atr_n); ap=float((a/df.close*100).iloc[-1]); base=float((a/df.close*100).rolling(100).median().iloc[-1]) if len(df)>100 else ap; ratio=ap/max(base,1e-9)
        state='VERY LOW' if ratio<.55 else 'LOW' if ratio<.8 else 'NORMAL' if ratio<1.35 else 'HIGH' if ratio<2 else 'EXTREME'
        mid=df.close.rolling(20).mean(); sd=df.close.rolling(20).std(); bw=float((4*sd/mid).iloc[-1]*100)
        return {'atr':float(a.iloc[-1]),'atr_pct':ap,'ratio':ratio,'bb_width':bw,'state':state}
class StructureEngine:
    def run(self,df,c):
        p=df.close.iloc[-1]; hi=df.high.rolling(c.breakout_n).max().shift(1).iloc[-1]; lo=df.low.rolling(c.breakout_n).min().shift(1).iloc[-1]
        st='HH_HL' if p>df.close.iloc[-6] and df.low.iloc[-1]>df.low.iloc[-6] else 'LH_LL' if p<df.close.iloc[-6] and df.high.iloc[-1]<df.high.iloc[-6] else 'RANGE'
        bos='BULLISH_BOS' if p>hi else 'BEARISH_BOS' if p<lo else 'NONE'
        return {'structure':st,'bos':bos,'swing_high':float(df.high.tail(c.sr_n).max()),'swing_low':float(df.low.tail(c.sr_n).min())}
class PriceActionEngine:
    def run(self,df):
        o,h,l,cl=df.open,df.high,df.low,df.close; body=(cl-o).abs(); rng=(h-l).replace(0,np.nan); u=h-np.maximum(o,cl); lo=np.minimum(o,cl)-l; p=[]
        if cl.iloc[-1]>o.iloc[-1] and cl.iloc[-2]<o.iloc[-2] and cl.iloc[-1]>=o.iloc[-2] and o.iloc[-1]<=cl.iloc[-2]:p.append('BULLISH_ENGULFING')
        if cl.iloc[-1]<o.iloc[-1] and cl.iloc[-2]>o.iloc[-2] and cl.iloc[-1]<=o.iloc[-2] and o.iloc[-1]>=cl.iloc[-2]:p.append('BEARISH_ENGULFING')
        if lo.iloc[-1]>2*body.iloc[-1] and u.iloc[-1]<body.iloc[-1]:p.append('BULLISH_PIN')
        if u.iloc[-1]>2*body.iloc[-1] and lo.iloc[-1]<body.iloc[-1]:p.append('BEARISH_PIN')
        if body.iloc[-1]/rng.iloc[-1]<.15:p.append('DOJI')
        if h.iloc[-1]<h.iloc[-2] and l.iloc[-1]>l.iloc[-2]:p.append('INSIDE_BAR')
        return {'patterns':p,'bull':sum(x.startswith('BULLISH') for x in p),'bear':sum(x.startswith('BEARISH') for x in p)}
class SREngine:
    def run(self,df,c):
        p=float(df.close.iloc[-1]); look=df.tail(c.sr_n); sup=float(look.low.min()); res=float(look.high.max());
        return {'support':sup,'resistance':res,'distance_support':p-sup,'distance_resistance':res-p}
class BreakoutEngine:
    def run(self,df,c):
        hi=df.high.rolling(c.breakout_n).max().shift(1).iloc[-1]; lo=df.low.rolling(c.breakout_n).min().shift(1).iloc[-1]; p=df.close.iloc[-1]; state='BREAKOUT_UP' if p>hi else 'BREAKOUT_DOWN' if p<lo else 'NONE'
        if len(df)>22:
            oldhi=df.high.rolling(c.breakout_n).max().shift(1).iloc[-2]; oldlo=df.low.rolling(c.breakout_n).min().shift(1).iloc[-2]
            if df.close.iloc[-2]>oldhi and p<hi: state='FAKEOUT_UP'
            if df.close.iloc[-2]<oldlo and p>lo: state='FAKEOUT_DOWN'
        return {'state':state,'upper':float(hi),'lower':float(lo)}
class LiquidityEngine:
    def run(self,df):
        a=float(atr(df,14).iloc[-1] or 1); return {'equal_highs':abs(df.high.iloc[-1]-df.high.tail(10).max())<.15*a,'equal_lows':abs(df.low.iloc[-1]-df.low.tail(10).min())<.15*a,'sweep_high':df.high.iloc[-1]>df.high.iloc[-2] and df.close.iloc[-1]<df.high.iloc[-2],'sweep_low':df.low.iloc[-1]<df.low.iloc[-2] and df.close.iloc[-1]>df.low.iloc[-2],'fvg_up':len(df)>3 and df.low.iloc[-1]>df.high.iloc[-3],'fvg_down':len(df)>3 and df.high.iloc[-1]<df.low.iloc[-3]}
class SessionEngine:
    def run(self,dt=None):
        h=(dt or datetime.now(timezone.utc)).hour; s='ASIAN' if h<8 else 'LONDON' if h<13 else 'LONDON/NEW YORK OVERLAP' if h<17 else 'NEW YORK' if h<22 else 'ROLLOVER'; return {'session':s,'allowed':s!='ROLLOVER'}
class RegimeEngine:
    def run(self,t,v,s,b):
        if v['state']=='EXTREME':return 'ABNORMAL'
        if v['state']=='VERY LOW':return 'LOW VOLATILITY'
        if b['state'].startswith('BREAKOUT'):return 'BREAKOUT'
        if s['structure']=='RANGE':return 'RANGING'
        if t['state'] in ('STRONG BULLISH','STRONG BEARISH'):return 'TRENDING'
        return 'TRANSITION'
class MTFEngine:
    def run(self,frames,c):
        states={}; bull=bear=0
        for tf,df in frames.items():
            if len(df)<c.ema50+5:continue
            q=TrendEngine().run(df,c);states[tf]=q['state'];bull+=int('BULLISH' in q['state']);bear+=int('BEARISH' in q['state'])
        n=max(1,len(states)); return {'states':states,'direction':'BULLISH' if bull>bear else 'BEARISH' if bear>bull else 'NEUTRAL','alignment':max(bull,bear)/n*100}
class CurrencyStrengthEngine:
    def run(self,data,period=10):
        mapping={'EURUSD':('EUR','USD'),'GBPUSD':('GBP','USD'),'USDJPY':('USD','JPY'),'USDCHF':('USD','CHF'),'AUDUSD':('AUD','USD'),'USDCAD':('USD','CAD'),'NZDUSD':('NZD','USD')}; out={}
        for sym,(b,q) in mapping.items():
            if sym in data and len(data[sym])>period:
                r=float(data[sym].close.pct_change(period).iloc[-1])*100;out.setdefault(b,[]).append(r);out.setdefault(q,[]).append(-r)
        return {k:float(np.mean(v)) for k,v in out.items()}
class CorrelationEngine:
    def matrix(self,data): return pd.DataFrame({k:v.close.pct_change() for k,v in data.items()}).corr()
class NewsEngine:
    def __init__(self,enabled=False):self.enabled=enabled
    def allowed(self):return True if not self.enabled else True

# ---------- intelligence / confluence ----------
class Intelligence:
    def __init__(self,c):self.c=c; self.t=TrendEngine();self.m=MomentumEngine();self.v=VolatilityEngine();self.s=StructureEngine();self.pa=PriceActionEngine();self.sr=SREngine();self.bo=BreakoutEngine();self.lq=LiquidityEngine();self.reg=RegimeEngine();self.se=SessionEngine()
    def analyze(self,df,mtf=None):
        a={'trend':self.t.run(df,self.c),'momentum':self.m.run(df,self.c),'volatility':self.v.run(df,self.c),'structure':self.s.run(df,self.c),'price_action':self.pa.run(df),'support_resistance':self.sr.run(df,self.c),'breakout':self.bo.run(df,self.c),'liquidity':self.lq.run(df),'mtf':mtf or {}}
        a['regime']=self.reg.run(a['trend'],a['volatility'],a['structure'],a['breakout']);a['session']=self.se.run();return a
class ConfluenceEngine:
    def score(self,a):
        b=s=0.; d={}
        def add(k,x,y):
            nonlocal b,s;b+=x;s+=y;d[k]=(x,y)
        t=a['trend']['state'];add('trend',20 if 'BULLISH' in t else 0,20 if 'BEARISH' in t else 0)
        m=a['momentum'];add('momentum',15 if m['state']=='BULLISH' else 0,15 if m['state']=='BEARISH' else 0)
        st=a['structure'];add('structure',15 if st['structure']=='HH_HL' else 0,15 if st['structure']=='LH_LL' else 0)
        pa=a['price_action'];add('price_action',10 if pa['bull'] else 0,10 if pa['bear'] else 0)
        p=a['_price'];sr=a['support_resistance'];av=a['_atr'];add('S/R',7 if sr['distance_support']<.6*av else 0,7 if sr['distance_resistance']<.6*av else 0)
        bo=a['breakout']['state'];add('breakout',10 if bo=='BREAKOUT_UP' else 0,10 if bo=='BREAKOUT_DOWN' else 0)
        add('volatility',5 if a['volatility']['state'] in ('NORMAL','HIGH') else 0,5 if a['volatility']['state'] in ('NORMAL','HIGH') else 0)
        add('regime',5 if a['regime'] in ('TRENDING','BREAKOUT') and 'BULLISH' in t else 0,5 if a['regime'] in ('TRENDING','BREAKOUT') and 'BEARISH' in t else 0)
        add('session',3 if a['session']['allowed'] else 0,3 if a['session']['allowed'] else 0)
        l=a['liquidity'];add('liquidity',2 if l['sweep_low'] or l['fvg_up'] else 0,2 if l['sweep_high'] or l['fvg_down'] else 0)
        mt=a['mtf'];add('MTF',5*mt.get('alignment',0)/100 if mt.get('direction')=='BULLISH' else 0,5*mt.get('alignment',0)/100 if mt.get('direction')=='BEARISH' else 0)
        score=min(100,max(b,s));return {'direction':'CALL' if b>=s else 'PUT','score':score,'bull':b,'bear':s,'detail':d}

# ---------- separate entry engines ----------
class ForexEntryPriceEngine:
    def calculate(self,symbol,df,a,sc,c):
        p=float(df.close.iloc[-1]);av=a['volatility']['atr'];direction=sc['direction'];e20=a['trend']['ema20'];sr=a['support_resistance']
        if direction=='CALL': fair=.5*p+.3*e20+.2*max(sr['support'],min(p,e20)); low=fair-.35*av;high=fair+.2*av;sl=min(sr['support']-.2*av,p-1.5*av);tp=max(p+2.2*av,sr['resistance']-.1*av)
        else: fair=.5*p+.3*e20+.2*min(sr['resistance'],max(p,e20));low=fair-.2*av;high=fair+.35*av;sl=max(sr['resistance']+.2*av,p+1.5*av);tp=min(p-2.2*av,sr['support']+.1*av)
        rr=abs(tp-p)/max(abs(p-sl),1e-9);q=max(0,sc['score']-12*abs(p-fair)/max(av,1e-9));v=[]
        if not low<=p<=high:v.append('PRICE OUTSIDE ENTRY ZONE')
        if rr<1.4:v.append('R:R BELOW 1.4')
        if a['regime'] in ('ABNORMAL','LOW VOLATILITY'):v.append('REGIME VETO')
        return EntryDecision(not v,'FOREX',symbol,direction,p,low,high,q,sc['score'],sl,tp,rr,reasons=[f'fair={fair}',f'zone={low}..{high}',f'RR={rr:.2f}'],vetoes=v)
class BinaryEntryPriceEngine:
    def calculate(self,symbol,df,a,sc,c,payout=None):
        payout=c.binary_payout if payout is None else payout;p=float(df.close.iloc[-1]);av=a['volatility']['atr'];direction=sc['direction'];e9=a['trend']['ema9'];e20=a['trend']['ema20'];fair=.45*p+.35*e9+.2*e20;hz=max(.25*av,p*.0001);lo=fair-hz;hi=fair+hz;conf=sc['score']*.72+20;r=a['momentum']['rsi'];acc=a['momentum']['acceleration']
        if direction=='CALL' and (r>72 or acc<0):conf-=10
        if direction=='PUT' and (r<28 or acc>0):conf-=10
        if a['volatility']['state'] in ('EXTREME','VERY LOW'):conf-=15
        conf=float(np.clip(conf,0,99));ev=(conf/100)*payout-(1-conf/100);expiry=15 if conf>=86 else 10 if conf>=78 else 5;v=[]
        if not lo<=p<=hi:v.append('PRICE OUTSIDE BINARY ENTRY ZONE')
        if conf<c.min_binary_conf:v.append('CONFIDENCE BELOW THRESHOLD')
        if ev<=0:v.append('NEGATIVE EXPECTED VALUE')
        if a['regime'] in ('ABNORMAL','LOW VOLATILITY'):v.append('REGIME VETO')
        if a['momentum']['exhaustion']:v.append('MOMENTUM EXHAUSTION')
        return EntryDecision(not v,'BINARY',symbol,direction,p,lo,hi,conf,conf,expiry_minutes=expiry,expiry_time=datetime.now(timezone.utc)+timedelta(minutes=expiry),reasons=[f'fair_strike={fair}',f'payout={payout:.2%}',f'EV={ev:.4f}',f'expiry={expiry}m'],vetoes=v)

# ---------- risk/execution/management ----------
class RiskEngine:
    def __init__(self,c):self.c=c;self.equity=c.balance;self.peak=c.balance;self.paused=False;self.emergency=False;self.loss_streak=0
    def update(self,e):self.equity=e;self.peak=max(self.peak,e)
    def veto(self,open_trades,spread=0,data_ok=True,session_ok=True,news_ok=True,correlation_ok=True):
        v=[]
        if self.paused:v.append('PAUSED')
        if self.emergency:v.append('EMERGENCY STOP')
        if len(open_trades)>=self.c.max_open:v.append('MAX OPEN POSITIONS')
        if 1-self.equity/max(self.peak,1e-9)>=self.c.max_drawdown:v.append('MAX DRAWDOWN')
        if spread>self.c.max_spread:v.append('SPREAD')
        if not data_ok:v.append('DATA QUALITY')
        if not session_ok:v.append('SESSION')
        if not news_ok:v.append('NEWS')
        if not correlation_ok:v.append('CORRELATION')
        if self.loss_streak>=5:v.append('LOSS STREAK')
        return not v,v
    def forex_size(self,entry,sl):return (self.equity*self.c.risk)/max(abs(entry-sl)*100000,1e-9)
    def binary_stake(self):return self.equity*self.c.risk
class ExecutionEngine:
    def __init__(self,paper=True):self.paper=paper;self.orders={};self.connected=True
    def place(self,t):
        if not self.paper:return {'ok':False,'error':'Official broker adapter required'}
        self.orders[t.id]=t;return {'ok':True,'mode':'PAPER','id':t.id}
    def close(self,tid):return {'ok':self.orders.pop(tid,None) is not None}
class ForexExecutionAdapter(ExecutionEngine):pass
class BinaryExecutionAdapter(ExecutionEngine):pass
class TradeManager:
    def check(self,t,p):
        if t.market=='BINARY':return 'EXPIRE' if t.expiry and datetime.now(timezone.utc)>=t.expiry else 'HOLD'
        if t.direction=='CALL' and ((t.sl and p<=t.sl) or (t.tp and p>=t.tp)):return 'CLOSE'
        if t.direction=='PUT' and ((t.sl and p>=t.sl) or (t.tp and p<=t.tp)):return 'CLOSE'
        return 'HOLD'

# ---------- backtest / walk-forward / monte-carlo / optimizer ----------
class BacktestEngine:
    def __init__(self,c):self.c=c
    def run(self,symbol,df,market='FOREX',payout=None):
        df=DataEngine().clean(df);eq=self.c.balance;peak=eq;rows=[];start=max(self.c.ema200,self.c.sr_n)+5; strat=Strategy(self.c)
        for i in range(start,len(df)-1):
            r=strat.evaluate(symbol,df.iloc[:i+1],market,payout)
            if not r['approved']:continue
            e=r['entry']; j=min(len(df)-1,i+max(1,(e.expiry_minutes or 5)//5));future=float(df.close.iloc[j]);win=future>e.entry if e.direction=='CALL' else future<e.entry
            if market=='BINARY':pnl=eq*self.c.risk*(payout if win else -1) if win else -eq*self.c.risk
            else:pnl=eq*self.c.risk*(2 if win else -1)
            eq+=pnl;peak=max(peak,eq);rows.append({'i':i,'direction':e.direction,'entry':e.entry,'pnl':pnl,'equity':eq,'win':win})
        pn=[x['pnl'] for x in rows];w=sum(x>0 for x in pn);dd=(1-eq/peak)*100 if peak else 0
        return {'symbol':symbol,'market':market,'final_balance':eq,'net_profit':eq-self.c.balance,'return_pct':(eq/self.c.balance-1)*100,'trades':len(rows),'win_rate':w/len(rows)*100 if rows else 0,'max_drawdown_pct':dd,'trades_detail':rows}
class WalkForwardEngine:
    def run(self,symbol,df,market='FOREX',parts=4):
        out=[];n=len(df);block=max(100,n//parts)
        for k in range(parts):
            x=df.iloc[k*block:min(n,(k+1)*block)];split=int(len(x)*.7)
            if len(x)-split<20:continue
            out.append({'window':k+1,'train_bars':split,'test':BacktestEngine(self.c).run(symbol,x.iloc[split:],market)})
        return out
class MonteCarloEngine:
    def run(self,pnls,runs=500):
        if not pnls:return {}
        finals=[]
        for _ in range(runs):finals.append(self.c.balance+sum(random.choices(pnls,k=len(pnls))))
        return {'p05':float(np.percentile(finals,5)),'median':float(np.percentile(finals,50)),'p95':float(np.percentile(finals,95))}
    def __init__(self,c):self.c=c
class Optimizer:
    def __init__(self,c):self.c=c
    def run(self,symbol,df,market):
        old=self.c.min_score;out=[]
        for x in (68,72,75,78,80):
            self.c.min_score=x;r=BacktestEngine(self.c).run(symbol,df,market);out.append({'threshold':x,'return':r['return_pct'],'drawdown':r['max_drawdown_pct'],'trades':r['trades']})
        self.c.min_score=old;return pd.DataFrame(out)

# ---------- strategy / tracker / journal ----------
class Strategy:
    def __init__(self,c):self.c=c;self.i=Intelligence(c);self.cf=ConfluenceEngine();self.fe=ForexEntryPriceEngine();self.be=BinaryEntryPriceEngine()
    def evaluate(self,symbol,df,market='FOREX',payout=None,mtf=None):
        a=self.i.analyze(df,mtf);a['_price']=float(df.close.iloc[-1]);a['_atr']=float(a['volatility']['atr']);sc=self.cf.score(a)
        if sc['score']<self.c.min_score:return {'approved':False,'analysis':a,'confluence':sc,'entry':None}
        e=self.be.calculate(symbol,df,a,sc,self.c,payout) if market=='BINARY' else self.fe.calculate(symbol,df,a,sc,self.c)
        return {'approved':e.approved,'analysis':a,'confluence':sc,'entry':e}
class Tracker:
    def __init__(self,c):self.s=Strategy(c)
    def scan(self,data):
        rows=[]
        for sym,df in data.items():
            try:
                r=self.s.evaluate(sym,df);a=r['analysis'];rows.append({'PAIR':sym,'DIRECTION':r['confluence']['direction'],'SCORE':r['confluence']['score'],'REGIME':a['regime'],'VOLATILITY':a['volatility']['state'],'SESSION':a['session']['session'],'TREND':a['trend']['state'],'MOMENTUM':a['momentum']['state'],'STRUCTURE':a['structure']['structure'],'ENTRY QUALITY':r['entry'].quality if r['entry'] else 0})
            except Exception as e:rows.append({'PAIR':sym,'ERROR':str(e)})
        return pd.DataFrame(rows).sort_values('SCORE',ascending=False) if rows else pd.DataFrame()
class Journal:
    def __init__(self):self.rows=[]
    def add(self,t,e):self.rows.append({**asdict(t),'entry_decision':asdict(e)})
    def df(self):return pd.DataFrame(self.rows)

# ---------- application + dashboard ----------
class App:
    def __init__(self,c=None):
        self.c=c or Config();self.data=DataEngine();self.risk=RiskEngine(self.c);self.exec=ExecutionEngine(self.c.paper);self.open=[];self.journal=Journal();self.strategy=Strategy(self.c);self.tracker=Tracker(self.c)
    def execute(self,symbol,result):
        e=result.get('entry');ok,v=self.risk.veto(self.open)
        if not e or not e.approved:return {'ok':False,'error':'Entry engine rejected trade','vetoes':e.vetoes if e else []}
        if not ok:return {'ok':False,'error':'Risk engine veto','vetoes':v}
        size=self.risk.binary_stake() if e.market=='BINARY' else self.risk.forex_size(e.entry,e.sl)
        t=Trade(str(uuid.uuid4()),symbol,e.market,e.direction,e.entry,size,datetime.now(timezone.utc),e.sl,e.tp,e.expiry,e.quality,e.confidence)
        x=self.exec.place(t)
        if x['ok']:self.open.append(t);self.journal.add(t,e)
        return x

def dashboard():
    try:import streamlit as st
    except ImportError:print('pip install streamlit pandas numpy');return
    st.set_page_config(page_title='V12 AI Trading Platform',layout='wide');st.title('V12 AI Trading Platform')
    if 'app' not in st.session_state:st.session_state.app=App()
    app=st.session_state.app;c=app.c
    with st.sidebar:
        c.risk=st.number_input('Risk/trade',.001,.05,c.risk,.001);c.min_score=st.slider('Confluence threshold',0,100,int(c.min_score));c.min_binary_conf=st.slider('Binary confidence',50,99,int(c.min_binary_conf));c.binary_payout=st.slider('Binary payout',.5,.95,c.binary_payout,.01)
        if st.button('PAUSE NEW TRADES'):app.risk.paused=True
        if st.button('RESUME'):app.risk.paused=False
        if st.button('EMERGENCY STOP'):app.risk.emergency=True;app.risk.paused=True
    a,b,d,e=st.columns(4);a.metric('Balance',f'${app.risk.equity:,.2f}');b.metric('Drawdown',f'{(1-app.risk.equity/max(app.risk.peak,1))*100:.2f}%');d.metric('Open',len(app.open));e.metric('Status','EMERGENCY' if app.risk.emergency else 'PAUSED' if app.risk.paused else 'READY')
    upload=st.file_uploader('Upload OHLCV CSV',type=['csv']);symbol=st.text_input('Symbol','EURUSD');market=st.selectbox('Market',['FOREX','BINARY'])
    if upload:
        df=app.data.ingest(symbol,'M5',pd.read_csv(upload));result=app.strategy.evaluate(symbol,df,market,c.binary_payout);a=result['analysis'];sc=result['confluence'];en=result['entry']
        st.subheader('Market Intelligence');st.json({'Trend':a['trend']['state'],'Momentum':a['momentum']['state'],'Volatility':a['volatility']['state'],'Structure':a['structure']['structure'],'BOS':a['structure']['bos'],'Breakout':a['breakout']['state'],'Regime':a['regime'],'Session':a['session'],'MTF':a['mtf'],'Liquidity':a['liquidity']})
        st.metric('Signal',f"{sc['direction']} {sc['score']:.1f}/100")
        st.subheader('Entry Price Engine');st.json(asdict(en) if en else {'approved':False})
        if en and en.approved and st.button('EXECUTE PAPER TRADE'):st.json(app.execute(symbol,result))
        st.subheader('Backtest');
        if st.button('RUN BACKTEST'):
            r=BacktestEngine(c).run(symbol,df,market,c.binary_payout);st.json({k:v for k,v in r.items() if k!='trades_detail'});st.dataframe(pd.DataFrame(r['trades_detail']),use_container_width=True)
        x,y,z=st.columns(3)
        if x.button('WALK FORWARD'):st.json(WalkForwardEngine(c).run(symbol,df,market))
        if y.button('OPTIMIZE'):st.dataframe(Optimizer(c).run(symbol,df,market),use_container_width=True)
        if z.button('MONTE CARLO'):
            r=BacktestEngine(c).run(symbol,df,market,c.binary_payout);st.json(MonteCarloEngine(c).run([x['pnl'] for x in r['trades_detail']]))
    st.subheader('Open Positions');st.dataframe(pd.DataFrame([asdict(x) for x in app.open]),use_container_width=True)
    st.subheader('Journal');st.dataframe(app.journal.df(),use_container_width=True)

if __name__=='__main__':
    print('V12 AI Trading Platform');print('Run: streamlit run v12app.py');print('Separate Forex and Binary Entry Price Engines included.')
