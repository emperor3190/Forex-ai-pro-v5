
"""
V14 AI Forex Trading Bot — Accuracy, Robustness & Unified Real-Forex Data V8
Single-file Streamlit trading research / paper-trading terminal.

Pipeline:
Market Data -> Validated Specialist Evidence -> Central Assessment -> Signal Intelligence
-> Entry Validation -> Signal Timing -> Trade Quality -> Risk -> Final Decision
-> Paper Execution -> Trade Monitor -> Journal -> V14 Backtest/Forward Test/Monte Carlo
-> Optimizer -> Dashboard.

This V14 build implements the agreed frozen architecture plus an additive accuracy/robustness validation layer with strict pair/timeframe
synchronization so scanners and analysis cannot silently use candles belonging to a different currency pair. The Deriv real-Forex source uses the
proven authenticated PAT -> Options account -> OTP -> WebSocket workflow and remains read-only. Live scanner pairs are fetched from
the configured live source; synthetic data is never substituted for a missing live pair.

Research/paper-trading only. Live execution requires an official broker/data API.
"""

from __future__ import annotations

import math
import os
import uuid
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time as _time
from dataclasses import dataclass, replace
from datetime import datetime, timezone, time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo
import streamlit as st


# ============================================================
# V14 FROZEN ARCHITECTURE — CORE CONTRACTS / ROUTING
# ============================================================
# This layer is additive. Existing dashboard/features remain intact.
# Specialist analysis engines produce evidence only. Their results are
# wrapped and routed through Central Assessment before intelligence/decision.
# FinalDecisionEngine is the sole authority for BUY/SELL/NO-TRADE.
# ============================================================

from dataclasses import asdict, field

def utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()

def _safe_timestamp(value=None) -> Optional[str]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC").isoformat()
    except Exception:
        return None

@dataclass
class TimestampEnvelope:
    data_timestamp: Optional[str] = None
    calculation_timestamp: str = field(default_factory=utc_now_iso)
    output_timestamp: Optional[str] = None
    data_age_seconds: Optional[float] = None
    timestamp_validity: str = "UNKNOWN"

@dataclass
class EngineEvidencePackage:
    engine_id: str
    engine_name: str
    symbol: str
    timeframe: str
    evidence: Dict[str, Any]
    timestamps: TimestampEnvelope
    data_provenance: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    final_signal_authorized: bool = False

    def finalize(self) -> Dict[str, Any]:
        self.timestamps.output_timestamp = utc_now_iso()
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "evidence": self.evidence,
            "timestamps": asdict(self.timestamps),
            "data_provenance": self.data_provenance,
            "quality": self.quality,
            "final_signal_authorized": False,
        }

class ArchitectureEventBus:
    """In-process routing bus. No trading decision is made here."""
    def __init__(self):
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def publish(self, package: Dict[str, Any]) -> None:
        key = f'{package.get("symbol","")}|{package.get("timeframe","")}|{package.get("engine_name","")}'
        with self._lock:
            self._latest[key] = package

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._latest)

ARCHITECTURE_BUS = ArchitectureEventBus()

class CentralAssessmentDataQualityEngine:
    """Central trust/readiness gate for validated market + engine evidence."""
    @staticmethod
    def assess(
        symbol: str,
        timeframe: str,
        market_quality: Dict[str, Any],
        evidence_packages: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        calc_ts = utc_now_iso()
        qualities = []
        conflicts = []
        invalid = []
        for name, pkg in evidence_packages.items():
            q = pkg.get("quality", {}) if isinstance(pkg, dict) else {}
            score = q.get("score")
            if score is not None:
                try:
                    qualities.append(float(score))
                except (TypeError, ValueError):
                    pass
            if str(q.get("status", "")).upper() in {"FAIL", "FAILED", "INVALID", "STALE"}:
                invalid.append(name)
            if pkg.get("evidence", {}).get("conflict_status") not in (None, "", "NONE"):
                conflicts.append({
                    "engine": name,
                    "status": pkg["evidence"].get("conflict_status"),
                })

        market_score = float(market_quality.get("score", 0.0) or 0.0)
        engine_score = float(np.mean(qualities)) if qualities else market_score
        overall = float(np.clip(0.60 * market_score + 0.40 * engine_score, 0, 100))
        ready = (
            overall >= 60
            and not invalid
            and bool(market_quality.get("signal_allowed", False))
        )
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_timestamp": market_quality.get("last_timestamp"),
            "calculation_timestamp": calc_ts,
            "output_timestamp": utc_now_iso(),
            "data_age_seconds": market_quality.get("age_seconds"),
            "timestamp_validity": "VALID" if market_quality.get("last_timestamp") else "UNKNOWN",
            "data_quality_score": market_score,
            "engine_quality_score": engine_score,
            "overall_data_quality_score": overall,
            "engine_count": len(evidence_packages),
            "invalid_engines": invalid,
            "conflict_map": conflicts,
            "quality_gate_status": "PASS" if ready else "BLOCK",
            "assessment_readiness": ready,
            "quality_state": "READY" if ready else "BLOCKED",
        }

class SignalIntelligenceLayer:
    """V14 integration layer: consumes Central-accepted evidence only.

    It creates directional context and readiness evidence, never BUY/SELL/NO-TRADE.
    FinalDecisionEngine is the sole final-decision authority.
    """
    @staticmethod
    def assess(central, evidence_packages):
        calc_ts=utc_now_iso()
        base={"symbol":central.get("symbol"),"timeframe":central.get("timeframe"),
              "engine_conflicts":list(central.get("conflict_map",[]) or []),
              "calculation_timestamp":calc_ts,"output_timestamp":utc_now_iso()}
        if not central.get("assessment_readiness",False):
            return {**base,"status":"BLOCKED","decision_readiness":False,"market_context":"NEUTRAL",
                    "directional_context":"NEUTRAL","overall_intelligence_score":0.0,"confidence_score":0.0,
                    "integrated_probability":None,"engine_confluence":0.0,"mtf_alignment":0.0,"mtf_coverage":0.0,
                    "signal_stability":0.0,"freshness_score":0.0,"candidate_setup":None,
                    "entry_zone_context":"NOT_REACHED","entry_timing_context":"NOT_REACHED",
                    "trade_quality_context":"NOT_REACHED","invalidation_conditions":["CENTRAL_ASSESSMENT_BLOCKED"],
                    "explanation":"CENTRAL_ASSESSMENT_BLOCKED","reason":"CENTRAL_ASSESSMENT_BLOCKED"}
        weighted=[]; scores=[]; freshness=[]; mtf_alignment=0.0; mtf_coverage=0.0
        for name,pkg in (evidence_packages or {}).items():
            if not isinstance(pkg,dict): continue
            ev=pkg.get("evidence",{}) or {}; q=pkg.get("quality",{}) or {}
            d=str(ev.get("direction",ev.get("bias",""))).upper()
            if d in {"BULLISH","BEARISH"}:
                try: w=float(q.get("weight",1.0))
                except Exception: w=1.0
                weighted.append((d,max(w,0.01),name))
            for k in ("score","strength","confidence","quality_score","alignment"):
                try:
                    if k in ev and np.isfinite(float(ev[k])):
                        scores.append(float(np.clip(float(ev[k]),0,100))); break
                except (TypeError,ValueError): pass
            age=(pkg.get("timestamps") or {}).get("data_age_seconds")
            if isinstance(age,(int,float)) and np.isfinite(age): freshness.append(max(0.0,100.0-min(float(age)/60.0,100.0)))
            if name in {"MTFConfirmation","MultiTimeframeEngine"}:
                try: mtf_alignment=float(ev.get("alignment",0) or 0); mtf_coverage=float(ev.get("coverage_percent",0) or 0)
                except Exception: pass
        bw=sum(w for d,w,_ in weighted if d=="BULLISH"); sw=sum(w for d,w,_ in weighted if d=="BEARISH")
        context="BULLISH" if bw>sw else "BEARISH" if sw>bw else "NEUTRAL"
        agreement=100.0*max(bw,sw)/max(bw+sw,1e-9)
        evidence_score=float(np.mean(scores)) if scores else float(central.get("engine_quality_score",0) or 0)
        data_score=float(central.get("overall_data_quality_score",0) or 0)
        fresh=float(np.mean(freshness)) if freshness else (100.0 if central.get("data_timestamp") else 0.0)
        stability=float(np.clip(100.0-5.0*len(base["engine_conflicts"]),0,100))
        overall=float(np.clip(0.35*agreement+0.30*evidence_score+0.15*data_score+0.10*fresh+0.10*stability,0,100))
        invalid=[]
        if data_score<70: invalid.append("DATA_QUALITY_BELOW_70")
        if agreement<50: invalid.append("DIRECTIONAL_CONFLUENCE_BELOW_50")
        if base["engine_conflicts"]: invalid.append("ENGINE_CONFLICT_PRESENT")
        ready=bool(context in {"BULLISH","BEARISH"} and overall>=60 and central.get("assessment_readiness"))
        return {**base,"status":"READY" if ready else "BLOCKED","market_context":context,"directional_context":context,
                "evidence_summary":{"bullish":sum(d=="BULLISH" for d,_,_ in weighted),"bearish":sum(d=="BEARISH" for d,_,_ in weighted),"neutral":max(0,len(evidence_packages)-len(weighted))},
                "engine_confluence":round(agreement,1),"engine_conflicts":base["engine_conflicts"],
                "integrated_probability":round(overall/100.0,4),"confidence_score":round(overall,1),
                "overall_intelligence_score":round(overall,1),"mtf_alignment":round(mtf_alignment,1),"mtf_coverage":round(mtf_coverage,1),
                "freshness_score":round(fresh,1),"signal_stability":round(stability,1),"invalidation_conditions":invalid,
                "candidate_setup":{"directional_context":context,"confluence":round(agreement,1),"status":"CANDIDATE_CONTEXT" if ready else "NOT_READY"},
                "entry_zone_context":"PENDING_DOWNSTREAM_ENTRY_ASSESSMENT","entry_timing_context":"PENDING_DOWNSTREAM_TIMING",
                "trade_quality_context":"PENDING_DOWNSTREAM_QUALITY","decision_readiness":ready,
                "explanation":"READY_FOR_DOWNSTREAM_GATES" if ready else "SIGNAL_INTELLIGENCE_NOT_READY"}


class ExecutionQualityEngine:
    """Execution-condition evidence; never selects trade direction."""
    @staticmethod
    def assess(current_price=None, bid=None, ask=None, spread_pips=None, latency_ms=None) -> Dict[str, Any]:
        calc_ts = utc_now_iso()
        spread = None
        if spread_pips is not None:
            try: spread = float(spread_pips)
            except (TypeError, ValueError): pass
        valid_price = current_price is not None or (bid is not None and ask is not None)
        return {
            "status": "PASS" if valid_price else "BLOCK",
            "price_validity": valid_price,
            "spread_pips": spread,
            "latency_ms": latency_ms,
            "slippage_status": "UNKNOWN",
            "fill_quality": "UNKNOWN",
            "calculation_timestamp": calc_ts,
            "output_timestamp": utc_now_iso(),
        }

class AccountPortfolioStateEngine:
    """Authoritative account/portfolio state adapter for risk and execution."""
    @staticmethod
    def snapshot(balance=0.0, equity=0.0, margin=0.0, positions=None, orders=None) -> Dict[str, Any]:
        return {
            "balance": float(balance or 0),
            "equity": float(equity or 0),
            "margin": float(margin or 0),
            "free_margin": float((equity or 0) - (margin or 0)),
            "positions": list(positions or []),
            "orders": list(orders or []),
            "exposure": 0.0,
            "pnl": 0.0,
            "drawdown": 0.0,
            "calculation_timestamp": utc_now_iso(),
            "output_timestamp": utc_now_iso(),
        }

class TradePositionLifecycleEngine:
    """Tracks trade state; does not create new signals."""
    STATES = ("CANDIDATE","VALIDATED","DECIDED","ORDERED","OPEN","MANAGING","BREAK_EVEN","EXITED","CLOSED")
    @staticmethod
    def transition(trade: Dict[str, Any], state: str) -> Dict[str, Any]:
        if state not in TradePositionLifecycleEngine.STATES:
            raise ValueError(f"Invalid lifecycle state: {state}")
        out = dict(trade or {})
        out["lifecycle_state"] = state
        out["state_timestamp"] = utc_now_iso()
        return out

class ConfigurationRulesGovernanceLayer:
    """Single controlled view of active configuration/version metadata."""
    VERSION = "V14-FROZEN-ARCH"
    @staticmethod
    def snapshot(cfg) -> Dict[str, Any]:
        values = {}
        try:
            values = asdict(cfg)
        except Exception:
            values = dict(getattr(cfg, "__dict__", {}))
        return {
            "version": ConfigurationRulesGovernanceLayer.VERSION,
            "configuration": values,
            "timestamp": utc_now_iso(),
        }

class SystemCircuitBreaker:
    """System-wide emergency halt. It never chooses BUY/SELL."""
    def __init__(self):
        self._halted = False
        self._reason = ""
        self._timestamp = None

    def halt(self, reason: str) -> None:
        self._halted = True
        self._reason = str(reason)
        self._timestamp = utc_now_iso()

    def release(self) -> None:
        self._halted = False
        self._reason = ""
        self._timestamp = utc_now_iso()

    def status(self) -> Dict[str, Any]:
        return {
            "halted": self._halted,
            "reason": self._reason,
            "timestamp": self._timestamp,
        }

SYSTEM_CIRCUIT_BREAKER = SystemCircuitBreaker()

class TradeJournalDecisionReplayEngine:
    """Decision reconstruction; distinct from low-level audit logging."""
    @staticmethod
    def record(decision: Dict[str, Any], central=None, intelligence=None) -> Dict[str, Any]:
        return {
            "replay_id": str(uuid.uuid4()),
            "decision": dict(decision or {}),
            "central_assessment": dict(central or {}),
            "signal_intelligence": dict(intelligence or {}),
            "recorded_at": utc_now_iso(),
        }

class FinalDecisionEngine:
    """ONLY authority for final BUY/SELL/NO-TRADE."""
    FINAL_STATES = {"BUY", "SELL", "NO-TRADE"}

    @staticmethod
    def resolve(
        intelligence: Dict[str, Any],
        risk: Dict[str, Any],
        entry_valid: bool,
        timing_valid: bool,
        execution_valid: bool,
        circuit_status: Dict[str, Any],
        trade_quality_valid: bool = True,
        engine_health_valid: bool = True,
    ) -> Dict[str, Any]:
        calc_ts = utc_now_iso()
        if circuit_status.get("halted"):
            decision = "NO-TRADE"
            reason = "SYSTEM_CIRCUIT_BREAKER"
        elif intelligence.get("status") != "READY":
            decision = "NO-TRADE"
            reason = "CENTRAL_ASSESSMENT_OR_INTELLIGENCE_BLOCKED"
        elif not entry_valid:
            decision = "NO-TRADE"
            reason = "ENTRY_VALIDATION_FAILED"
        elif not timing_valid:
            decision = "NO-TRADE"
            reason = "ENTRY_TIMING_FAILED"
        elif not execution_valid:
            decision = "NO-TRADE"
            reason = "EXECUTION_QUALITY_FAILED"
        elif not trade_quality_valid:
            decision = "NO-TRADE"
            reason = "TRADE_QUALITY_FILTER_FAILED"
        elif not engine_health_valid:
            decision = "NO-TRADE"
            reason = "ENGINE_HEALTH_FAILED"
        elif not bool(risk.get("approved", risk.get("status") == "APPROVED")):
            decision = "NO-TRADE"
            reason = "RISK_GATE_FAILED"
        else:
            direction = str(intelligence.get("market_context", "NEUTRAL")).upper()
            decision = direction if direction in {"BUY","SELL"} else (
                "BUY" if direction == "BULLISH" else "SELL" if direction == "BEARISH" else "NO-TRADE"
            )
            reason = "FINAL_VALIDATED_DECISION" if decision != "NO-TRADE" else "NO_DIRECTION"
        return {
            "final_decision": decision,
            "direction": decision if decision in {"BUY","SELL"} else "NONE",
            "decision_timestamp": utc_now_iso(),
            "calculation_timestamp": calc_ts,
            "output_timestamp": utc_now_iso(),
            "decision_explanation": reason,
            "decision_id": str(uuid.uuid4()),
        }

V14_EVIDENCE_WEIGHTS={"TrendEngine":1.0,"EMAEngine":0.9,"MACDEngine":0.9,"MomentumEngine":1.0,"VolatilityEngine":0.9,"StructureEngine":1.0,"PriceActionEngine":1.0,"SupportResistanceEngine":0.8,"BreakoutEngine":0.9,"LiquidityEngine":0.8,"FVGEngine":0.8,"MarketMicrostructureEngine":0.7,"OrderFlowVolumeFlowEngine":0.9,"ReversalDetectionEngine":0.7,"CurrencyStrengthEngine":0.9,"CorrelationEngine":0.6,"MTFConfirmation":1.2,"AIProbability":1.0,"AdvancedMomentumEngine":1.0,"VolumeFlowEngine":0.9,"CandleTimingEngine":0.6,"MarketRegionEngine":0.6,"DirectProbabilityEngine":0.9,"EconomicCalendarEventEngine":0.7,"NewsEventFilterEngine":0.8,"SessionEngine":0.7,"COTEngine":0.4,"ConfluenceEngine":1.0,"EnsembleDecisionEngine":1.1,"RiskControlEngine":1.0,"SignalTimingEngine":0.7,"MarketTrackerEngine":0.5,"ReversalContext":0.7}
def _evidence_weight(name): return float(V14_EVIDENCE_WEIGHTS.get(name,1.0))
def _evidence_quality_status(evidence,market_quality):
    s=str((evidence or {}).get("status","")).upper()
    if s in {"FAIL","FAILED","INVALID","STALE","ERROR"}: return "FAIL"
    if s in {"NO DATA","UNAVAILABLE","INSUFFICIENT"}: return "UNAVAILABLE"
    return "PASS" if market_quality.get("signal_allowed",False) else "BLOCK"
def _evidence_quality_score(evidence,market_quality):
    ev=evidence or {}
    for k in ("quality_score","confidence","score","strength","alignment"):
        try:
            v=float(ev.get(k))
            if np.isfinite(v): return float(np.clip(v,0,100))
        except (TypeError,ValueError): pass
    return float(np.clip(market_quality.get("score",0) or 0,0,100))
def _evidence_input_sufficient(evidence):
    return str((evidence or {}).get("status","OK")).upper() not in {"NO DATA","UNAVAILABLE","INSUFFICIENT","ERROR","FAIL","FAILED","INVALID"}

def package_engine_evidence(
    engine_id: str,
    engine_name: str,
    symbol: str,
    timeframe: str,
    evidence: Dict[str, Any],
    data_timestamp=None,
    quality=None,
    provenance=None,
) -> Dict[str, Any]:
    """Standard wrapper: specialist result -> Central Assessment."""
    data_ts = _safe_timestamp(data_timestamp)
    calc_ts = utc_now_iso()
    age = None
    if data_ts:
        try:
            age = max(0.0, (pd.Timestamp.now(tz="UTC") - pd.Timestamp(data_ts)).total_seconds())
        except Exception:
            pass
    pkg = EngineEvidencePackage(
        engine_id=engine_id,
        engine_name=engine_name,
        symbol=symbol,
        timeframe=str(timeframe),
        evidence=dict(evidence or {}),
        timestamps=TimestampEnvelope(
            data_timestamp=data_ts,
            calculation_timestamp=calc_ts,
            data_age_seconds=age,
            timestamp_validity="VALID" if data_ts else "UNKNOWN",
        ),
        data_provenance=dict(provenance or {}),
        quality=dict(quality or {}),
    )
    result = pkg.finalize()
    ARCHITECTURE_BUS.publish(result)
    return result


# Optional WebSocket client for Deriv authenticated market-data streaming. READ-ONLY.
try:
    import websocket
except ImportError:
    websocket = None



class V14ArchitectureAuditEngine:
    """Static V14 architecture verification; never creates a trading signal."""
    REQUIRED=["MarketDataEngine","HistoricalDataEngine","LiveMarketDataEngine","TickNormalizerEngine","OHLCVBuilderEngine","DataValidationEngine","TrendEngine","EMAEngine","MACDEngine","MomentumEngine","VolatilityEngine","StructureEngine","PriceActionEngine","SupportResistanceEngine","FVGEngine","BreakoutEngine","LiquidityEngine","CurrencyStrengthEngine","CorrelationEngine","MarketMicrostructureEngine","OrderFlowVolumeFlowEngine","ReversalDetectionEngine","MultiTimeframeEngine","MTFConfirmationEngine","AIProbabilityEngine","AIMLPredictionEngine","CentralAssessmentDataQualityEngine","SignalIntelligenceLayer","EntryZoneAssessmentEngine","SignalTimingEngine","StrikeEntryValidationEngine","TradeQualityFilterLayer","RiskControlEngine","EngineHealthVerificationEngine","FinalDecisionEngine","EconomicCalendarEventEngine","NewsEventFilterEngine","APIConnectionManager","TimeSynchronizationEngine","AutoRefreshEngine","BreakEvenEngine","LeakageFreeTrainingEngine","V14BacktestEngine","V14ForwardTestingEngine","MonteCarloEngine","OptimizerEngine","RealForexTradeEngine","MarketTrackerEngine"]
    @classmethod
    def run(cls,source_text):
        # Build prohibited legacy patterns without embedding those feature names
        # verbatim in the source, so the audit itself cannot create false positives.
        forbidden=[
            "daily_"+"locked_signal", "daily_signal_"+"locked", "PersistentDaily"+"LockStore",
            "CurrentDaily"+"MarketSentimentEngine", "get_"+"locked_daily_signal", "get_"+"forming_daily_signal",
            "class Backtest"+"Engine", "class WalkForward"+"Engine", "Retained " + "Backtest", "Run " + "Walk-Forward"
        ]
        missing=[x for x in cls.REQUIRED if re.search(r"^class\s+"+re.escape(x)+r"\b",source_text,re.M) is None]
        forbidden_found=[x for x in forbidden if x in source_text]
        calls=len(re.findall(r"FinalDecisionEngine\.resolve\(",source_text))
        return {"status":"PASS" if not missing and not forbidden_found and calls>=1 else "FAIL",
                "required_engine_classes":len(cls.REQUIRED),"missing_engines":missing,
                "forbidden_legacy_features_found":forbidden_found,"final_decision_call_sites":calls,
                "final_decision_authority":"PASS" if calls>=1 else "FAIL"}


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    initial_balance: float = 10000.0
    current_equity: Optional[float] = None
    risk_per_trade: float = 0.005
    max_daily_loss: float = 0.03
    max_weekly_loss: float = 0.07
    max_drawdown: float = 0.15
    max_open_positions: int = 3
    max_symbol_exposure: float = 0.02
    min_score: float = 72.0
    max_spread_pips: float = 2.0
    slippage_pips: float = 0.3
    commission_per_lot: float = 0.0
    news_blackout_before: int = 30
    news_blackout_after: int = 15
    demo_rows: int = 1200
    recommended_history_rows: int = 10000
    historical_target_rows: int = 12000
    historical_max_requests: int = 4
    minimum_history_candles: int = 1000
    minimum_statistically_valid_trades: int = 30
    minimum_statistically_valid_oos_trades: int = 20
    data_source: str = "DEMO"
    twelve_data_api_key: str = ""
    twelve_data_outputsize: int = 5000
    allow_live_source_failover: bool = True
    live_refresh_seconds: int = 300
    data_max_age_seconds: int = 420
    ml_min_probability: float = 0.60
    signal_max_age_seconds: int = 60
    no_trade_conflict_threshold: float = 18.0
    min_signal_confidence: float = 72.0
    ai_soft_floor: float = 25.0
    min_engine_agreement: float = 60.0
    # Twelve Data Basic currently provides 8 API credits/minute. V14 keeps
    # one safety margin and never performs duplicate symbol requests.
    twelve_data_request_budget_per_minute: int = 7
    twelve_data_daily_budget: int = 760
    twelve_data_rate_guard_seconds: int = 60
    twelve_data_retry_after_429: bool = False
    twelve_data_cache_stale_seconds: int = 420
    scanner_refresh_seconds: int = 300
    # Additive Deriv authenticated real-Forex market-data stream settings.
    # Authentication is used only to obtain the short-lived OTP WebSocket URL.
    # No buy/sell/order endpoint is called by this adapter.
    deriv_api_base: str = "https://api.derivws.com"
    # Provider endpoint is resolved through the authenticated connection manager.
    deriv_stream_endpoint: str = "AUTO (OTP)"
    deriv_stream_enabled: bool = False
    deriv_stream_outputsize: int = 12000
    deriv_stream_reconnect_seconds: int = 5
    deriv_stream_stale_seconds: int = 15
    deriv_options_account_id: str = ""
    # V14 frozen architecture mode: Forex research/paper-trading only.
    architecture_version: str = "V14-FROZEN-ARCH"


# ============================================================
# MARKET DATA ENGINE
# ============================================================

class MarketDataEngine:
    TIMEFRAMES = ["M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1"]

    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x.columns = [str(c).strip().lower().replace(" ", "_") for c in x.columns]
        aliases = {
            "datetime": "time",
            "date": "time",
            "timestamp": "time",
            "vol": "volume",
            "tick_volume": "volume",
        }
        x = x.rename(columns={c: aliases.get(c, c) for c in x.columns})

        for c in ["open", "high", "low", "close"]:
            if c not in x.columns:
                raise ValueError(f"Missing required OHLC column: {c}")
            x[c] = pd.to_numeric(x[c], errors="coerce")

        if "volume" not in x.columns:
            x["volume"] = 0.0
        else:
            x["volume"] = pd.to_numeric(x["volume"], errors="coerce").fillna(0)

        if "spread_pips" in x.columns:
            x["spread_pips"] = pd.to_numeric(x["spread_pips"], errors="coerce")
        else:
            x["spread_pips"] = np.nan

        if "time" in x.columns:
            x["time"] = pd.to_datetime(x["time"], errors="coerce", utc=True)
        else:
            x["time"] = pd.date_range(
                end=pd.Timestamp.now(tz="UTC"), periods=len(x), freq="5min"
            )

        before = len(x)
        x = (
            x.dropna(subset=["open", "high", "low", "close"])
            .drop_duplicates(subset=["time"])
            .sort_values("time")
            .reset_index(drop=True)
        )
        x.attrs["duplicates_removed"] = before - len(x)
        return x

    @staticmethod
    def synthetic(symbol="EURUSD", rows=1200, seed=7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        t = pd.date_range(
            end=pd.Timestamp.now(tz="UTC"), periods=rows, freq="5min"
        )
        regimes = np.where(
            (np.arange(rows) // 180) % 3 == 0,
            0.00008,
            np.where((np.arange(rows) // 180) % 3 == 1, -0.00003, 0.0),
        )
        noise = rng.normal(0, 0.00045, rows)
        close = 1.08 + np.cumsum(regimes + noise)
        close = np.maximum(close, 0.5)
        op = np.r_[close[0], close[:-1]]
        spread = np.abs(rng.normal(0.00008, 0.000025, rows))
        high = np.maximum(op, close) + spread
        low = np.minimum(op, close) - spread
        vol = rng.integers(80, 1200, rows)
        return pd.DataFrame(
            {
                "time": t,
                "open": op,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        )

    @staticmethod
    def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        rule = {
            "M1": "1min",
            "M3": "3min",
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H4": "4h",
            "D1": "1D",
        }[timeframe]
        x = df.copy().set_index("time")
        out = (
            x.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        return out.reset_index()

    @staticmethod
    def validate(df: pd.DataFrame) -> Dict[str, Any]:
        x = MarketDataEngine.normalize(df)
        gaps = x["time"].diff().dt.total_seconds().div(60).dropna()
        return {
            "rows": len(x),
            "duplicates_removed": int(df.shape[0] - x.shape[0]),
            "missing_ohlc": int(
                x[["open", "high", "low", "close"]].isna().sum().sum()
            ),
            "large_gaps": int((gaps > 60).sum()) if len(gaps) else 0,
            "timezone": "UTC",
            "data_ok": len(x) >= 100,
        }


# ============================================================
# INDICATOR UTILITIES
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(close):
    line = ema(close, 12) - ema(close, 26)
    sig = ema(line, 9)
    return line, sig, line - sig


def stochastic(df, n=14):
    lo, hi = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    d = k.rolling(3).mean()
    return k.fillna(50), d.fillna(50)


def roc(close, n=12):
    return close.pct_change(n) * 100


def cci(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mean = tp.rolling(n).mean()
    md = (tp - mean).abs().rolling(n).mean()
    return ((tp - mean) / (0.015 * md.replace(0, np.nan))).fillna(0)


def williams_r(df, n=14):
    hi = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    return (-100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)).fillna(-50)


def mfi(df, n=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"].replace(0, 1)
    pos = mf.where(tp.diff() > 0, 0).rolling(n).sum()
    neg = mf.where(tp.diff() < 0, 0).rolling(n).sum().abs()
    ratio = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50)


def adx(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    tr = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / tr.replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / tr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return (
        dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0),
        pdi.fillna(0),
        mdi.fillna(0),
    )


def supertrend_bias(df, n=10, mult=3.0):
    a = atr(df, n)
    mid = (df["high"] + df["low"]) / 2
    upper = mid + mult * a
    lower = mid - mult * a
    return np.where(
        df["close"] > upper.shift(1),
        "BULLISH",
        np.where(df["close"] < lower.shift(1), "BEARISH", "NEUTRAL"),
    )


def ichimoku_bias(df):
    ten = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    kij = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    return np.where(ten > kij, "BULLISH", np.where(ten < kij, "BEARISH", "NEUTRAL"))


# ============================================================
# V14 DATA / TECHNICAL EVIDENCE ENGINES
# ============================================================

class HistoricalDataEngine:
    """Historical OHLCV ingestion/normalization. No signal generation."""
    @staticmethod
    def load(data: pd.DataFrame, symbol: str = "", timeframe: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        x = MarketDataEngine.normalize(data)
        validation = MarketDataEngine.validate(x) if not x.empty else {"rows": 0, "data_ok": False}
        return x, {"source": "HISTORICAL_DATA", "symbol": canonical_symbol(symbol) if symbol else symbol,
                   "timeframe": str(timeframe).upper(), "validation": validation,
                   "historical": True, "live_api_called": False}

class LiveMarketDataEngine:
    """Normalized live market-data facade. Providers remain outside analysis engines."""
    @staticmethod
    def normalize_tick(tick: Dict[str, Any]) -> Dict[str, Any]:
        return TickNormalizerEngine.normalize(tick)
    @staticmethod
    def validate(df: pd.DataFrame) -> Dict[str, Any]:
        return DataValidationEngine.validate(df)

class TickNormalizerEngine:
    """Converts heterogeneous ticks into a standard observable tick package."""
    @staticmethod
    def normalize(tick: Dict[str, Any]) -> Dict[str, Any]:
        t = dict(tick or {})
        bid = t.get("bid", t.get("price"))
        ask = t.get("ask", t.get("price"))
        try: bid = float(bid) if bid is not None else None
        except Exception: bid = None
        try: ask = float(ask) if ask is not None else None
        except Exception: ask = None
        mid = ((bid + ask) / 2.0) if bid is not None and ask is not None else (bid if bid is not None else ask)
        spread = (ask - bid) if bid is not None and ask is not None else None
        return {"symbol": t.get("symbol"), "bid": bid, "ask": ask, "mid": mid,
                "price": mid, "spread": spread, "timestamp": _safe_timestamp(t.get("timestamp", t.get("time"))),
                "source": t.get("source", "UNKNOWN"), "sequence": t.get("sequence"),
                "observable": True, "derived_fields": ["mid", "spread"] if bid is not None and ask is not None else [],
                "validated": bid is not None or ask is not None}

class OHLCVBuilderEngine:
    """Builds OHLCV from the selected provider's normalized market data.

    Provider rule: no synthetic prices are created here. If raw ticks are supplied,
    candles are constructed strictly from those observed ticks. If the provider has
    already supplied historical OHLC candles, those candles are normalized/validated
    rather than replaced with another source.
    """
    @staticmethod
    def build(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        x = MarketDataEngine.normalize(data)
        return MarketDataEngine.resample(x, str(timeframe).upper()) if str(timeframe).upper() != "RAW" else x

    @staticmethod
    def build_from_ticks(ticks, timeframe: str, symbol: str = "", source: str = "") -> pd.DataFrame:
        rows = []
        for raw in (ticks or []):
            t = TickNormalizerEngine.normalize(raw)
            if not t.get("validated") or t.get("mid") is None or t.get("timestamp") is None:
                continue
            rows.append({
                "time": t["timestamp"], "open": float(t["mid"]),
                "high": float(t["mid"]), "low": float(t["mid"]),
                "close": float(t["mid"]), "volume": 0.0,
            })
        if not rows:
            return _empty_market_data()
        x = pd.DataFrame(rows)
        x["time"] = pd.to_datetime(x["time"], utc=True, errors="coerce")
        x = x.dropna(subset=["time", "open", "high", "low", "close"])
        out = OHLCVBuilderEngine.build(x, timeframe)
        out["symbol"] = canonical_symbol(symbol) if symbol else out.get("symbol", "")
        out["source"] = source or "UNKNOWN"
        out["volume_type"] = "TICK_DERIVED_PRICE_BARS"
        return MarketDataEngine.normalize(out)

    @staticmethod
    def _merge_provider_and_tick_bars(provider_df: pd.DataFrame, tick_df: pd.DataFrame) -> pd.DataFrame:
        """Merge same-provider historical candles with observed live tick-built bars.

        A live tick-built bar replaces only the matching/in-progress candle; older
        provider history remains intact. No values are sourced from another provider.
        """
        a = MarketDataEngine.normalize(provider_df)
        b = MarketDataEngine.normalize(tick_df)
        if a.empty:
            return b
        if b.empty:
            return a
        combined = pd.concat([a, b], ignore_index=True)
        combined = combined.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        return MarketDataEngine.normalize(combined)

class DataValidationEngine:
    """Dedicated V14 data validation contract. It never creates market values."""
    @staticmethod
    def validate(df: pd.DataFrame) -> Dict[str, Any]:
        try:
            x = MarketDataEngine.normalize(df)
        except Exception as exc:
            return {"status": "FAIL", "data_ok": False, "score": 0.0, "reasons": [str(exc)]}
        base = MarketDataEngine.validate(x)
        reasons = []
        if not base.get("data_ok"): reasons.append("INSUFFICIENT DATA")
        if base.get("missing_ohlc", 0): reasons.append("MISSING OHLC")
        if base.get("large_gaps", 0): reasons.append("LARGE GAPS")
        score = 100.0 - min(50.0, base.get("large_gaps", 0) * 2.0) - (30.0 if base.get("missing_ohlc", 0) else 0.0)
        score = float(np.clip(score, 0, 100))
        return {**base, "status": "PASS" if base.get("data_ok") and score >= 70 else "WARNING" if score >= 50 else "FAIL",
                "score": score, "reasons": reasons, "signal_allowed": bool(base.get("data_ok") and score >= 70),
                "timestamp_validity": "VALID" if base.get("timezone") == "UTC" else "UNKNOWN"}

class EMAEngine:
    """Dedicated EMA evidence engine; no trading signal authority."""
    PERIODS = (9, 20, 50, 100, 200)
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        x = MarketDataEngine.normalize(df); close = x["close"]
        values = {f"EMA{n}": float(ema(close, n).iloc[-1]) for n in EMAEngine.PERIODS if len(x) >= n}
        last = float(close.iloc[-1])
        ordered = [values[f"EMA{n}"] for n in EMAEngine.PERIODS if f"EMA{n}" in values]
        direction = "BULLISH" if ordered and last > ordered[0] and all(ordered[i] >= ordered[i+1] for i in range(len(ordered)-1)) else "BEARISH" if ordered and last < ordered[0] and all(ordered[i] <= ordered[i+1] for i in range(len(ordered)-1)) else "NEUTRAL"
        return {"periods": list(EMAEngine.PERIODS), "values": values, "price": last, "direction": direction,
                "alignment": float(sum(last > v for v in ordered) / max(len(ordered),1) * 100), "status": "OK" if values else "INSUFFICIENT"}

class MACDEngine:
    """Dedicated MACD evidence engine; no trading signal authority."""
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        x = MarketDataEngine.normalize(df); line, sig, hist = macd(x["close"])
        l, s, h = map(float, (line.iloc[-1], sig.iloc[-1], hist.iloc[-1]))
        hp = float(hist.iloc[-2]) if len(hist) > 1 else h
        return {"fast": 12, "slow": 26, "signal_period": 9, "macd": l, "signal": s, "histogram": h,
                "histogram_change": h-hp, "direction": "BULLISH" if l > s else "BEARISH" if l < s else "NEUTRAL",
                "status": "OK" if len(x) >= 35 else "INSUFFICIENT"}

class FVGEngine:
    """Three-candle imbalance evidence engine; no trading signal authority."""
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        x = MarketDataEngine.normalize(df)
        if len(x) < 3: return {"detected": False, "direction": "NEUTRAL", "status": "INSUFFICIENT"}
        a,b,c=x.iloc[-3],x.iloc[-2],x.iloc[-1]
        bull = float(c.low) > float(a.high)
        bear = float(c.high) < float(a.low)
        if bull: low,high,direction=float(a.high),float(c.low),"BULLISH"
        elif bear: low,high,direction=float(c.high),float(a.low),"BEARISH"
        else: low=high=float(c.close); direction="NEUTRAL"
        return {"detected": bool(bull or bear), "direction": direction, "lower": low, "upper": high,
                "size": abs(high-low), "status": "OK"}

class MarketMicrostructureEngine:
    """Observable tick/bid/ask/spread microstructure. Never fabricates depth/order-book data."""
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        x=MarketDataEngine.normalize(df)
        spread=x.get("spread_pips", pd.Series(np.nan,index=x.index)).tail(50)
        returns=x["close"].pct_change().tail(50)
        return {"spread_mean": float(spread.mean()) if spread.notna().any() else None,
                "spread_current": float(spread.iloc[-1]) if spread.notna().any() else None,
                "tick_direction_ratio": float((returns>0).mean()*100) if returns.notna().any() else 50.0,
                "short_term_volatility": float(returns.std()*100) if returns.notna().any() else 0.0,
                "order_book_available": False, "depth_data": None,
                "data_classification": "OBSERVABLE + DERIVED", "status": "OK"}

class OrderFlowVolumeFlowEngine:
    """Formal V14 order/volume-flow contract backed by available volume only."""
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        return {**VolumeFlowEngine.analyze(df), "volume_provenance": "PROVIDER_OR_INPUT_DATA", "order_book_used": False,
                "delta_type": "PROXY" if "volume" in df.columns else "UNAVAILABLE"}

class ReversalDetectionEngine:
    """Reversal evidence from structure, momentum, rejection and divergence proxies; no signal authority."""
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        x=MarketDataEngine.normalize(df); pa=PriceActionEngine.analyze(x); st=StructureEngine.analyze(x); mo=MomentumEngine.analyze(x)
        reversal=[]
        if st.get("CHOCH") not in (None,"NONE"): reversal.append(st.get("CHOCH"))
        if any("PIN" in str(p) or "ENGULFING" in str(p) for p in pa.get("patterns",[])): reversal.append("REJECTION_PATTERN")
        if mo.get("direction") in {"BULLISH","BEARISH"}: reversal.append("MOMENTUM_SHIFT_CONTEXT")
        return {"detected": bool(reversal), "evidence": reversal, "structure_change": st.get("CHOCH","NONE"),
                "momentum_direction": mo.get("direction","NEUTRAL"), "price_action": pa.get("patterns",[]), "status":"OK"}

# ============================================================
# ANALYSIS ENGINES
# ============================================================

class TrendEngine:
    @staticmethod
    def analyze(df):
        x = df.copy()
        for n in [9, 20, 50, 100, 200]:
            x[f"ema{n}"] = ema(x.close, n)
        x["sma50"] = sma(x.close, 50)
        x["macd"], x["macd_signal"], x["macd_hist"] = macd(x.close)
        x["adx"], x["pdi"], x["mdi"] = adx(x)
        x["ichimoku"] = ichimoku_bias(x)
        x["supertrend"] = supertrend_bias(x)
        last = x.iloc[-1]

        votes = 0
        for n in [9, 20, 50, 100, 200]:
            votes += 1 if last.close > last[f"ema{n}"] else -1
        votes += 1 if last.macd_hist > 0 else -1
        votes += 1 if last.pdi > last.mdi else -1
        votes += 1 if last.ichimoku == "BULLISH" else (-1 if last.ichimoku == "BEARISH" else 0)
        votes += 1 if last.supertrend == "BULLISH" else (-1 if last.supertrend == "BEARISH" else 0)

        direction = (
            "BULLISH" if votes >= 3 else "BEARISH" if votes <= -3 else "SIDEWAYS"
        )
        strength = min(100, 50 + abs(votes) * 6 + max(0, float(last.adx) - 20))
        label = (
            "STRONG BULLISH"
            if direction == "BULLISH" and strength >= 75
            else "WEAK BULLISH"
            if direction == "BULLISH"
            else "STRONG BEARISH"
            if direction == "BEARISH" and strength >= 75
            else "WEAK BEARISH"
            if direction == "BEARISH"
            else "SIDEWAYS"
        )
        return {
            "direction": direction,
            "strength": float(strength),
            "label": label,
            "adx": float(last.adx),
            "ema9": float(last.ema9),
            "ema20": float(last.ema20),
            "ema50": float(last.ema50),
            "ema100": float(last.ema100),
            "ema200": float(last.ema200),
        }


class MomentumEngine:
    @staticmethod
    def analyze(df):
        rr = rsi(df.close)
        k, d = stochastic(df)
        ml, ms, mh = macd(df.close)
        ro = roc(df.close)
        cc = cci(df)
        wr = williams_r(df)
        mf = mfi(df)
        a, _, _ = adx(df)

        score = (
            50
            + np.clip((rr.iloc[-1] - 50) * 0.45, -22, 22)
            + np.clip(
                mh.iloc[-1] / max(abs(mh.iloc[-50:]).mean(), 1e-9) * 8,
                -10,
                10,
            )
        )
        direction = "BULLISH" if score >= 58 else "BEARISH" if score <= 42 else "NEUTRAL"
        accel = float(mh.iloc[-1] - mh.iloc[-2]) if len(mh) > 1 else 0

        div = "NONE"
        if len(df) >= 30:
            price_change = df.close.iloc[-1] - df.close.iloc[-15]
            r_change = rr.iloc[-1] - rr.iloc[-15]
            if price_change > 0 and r_change < 0:
                div = "BEARISH"
            elif price_change < 0 and r_change > 0:
                div = "BULLISH"

        state = (
            "OVERBOUGHT"
            if rr.iloc[-1] >= 70
            else "OVERSOLD"
            if rr.iloc[-1] <= 30
            else "INCREASING"
            if accel > 0
            else "DECREASING"
            if accel < 0
            else "NORMAL"
        )
        return {
            "direction": direction,
            "score": float(np.clip(score, 0, 100)),
            "rsi": float(rr.iloc[-1]),
            "stochastic": float(k.iloc[-1]),
            "macd_hist": float(mh.iloc[-1]),
            "roc": float(ro.iloc[-1]),
            "cci": float(cc.iloc[-1]),
            "williams_r": float(wr.iloc[-1]),
            "mfi": float(mf.iloc[-1]),
            "adx": float(a.iloc[-1]),
            "divergence": div,
            "state": state,
        }


class VolatilityEngine:
    @staticmethod
    def analyze(df):
        a = atr(df)
        mid = df.close.rolling(20).mean()
        sd = df.close.rolling(20).std()
        bw = (4 * sd / mid.replace(0, np.nan)) * 100
        hv = df.close.pct_change().rolling(50).std() * math.sqrt(252 * 24 * 12) * 100
        ar = (df.high - df.low) / df.close * 100
        last = len(df) - 1

        atr_pct = float(a.iloc[last] / df.close.iloc[last] * 100)
        bwv = float(bw.iloc[last])
        hvv = float(hv.iloc[last]) if np.isfinite(hv.iloc[last]) else 0
        roll_mean = ar.rolling(50).mean().iloc[last]
        roll_std = ar.rolling(50).std().iloc[last]
        z = float((ar.iloc[last] - roll_mean) / (roll_std if roll_std not in (0, np.nan) else 1))
        composite = atr_pct * 0.5 + bwv * 0.3 + hvv * 0.2

        regime = (
            "EXTREME" if z > 3 else
            "HIGH" if z > 1.5 else
            "VERY LOW" if z < -1.5 else
            "LOW" if z < -0.5 else "NORMAL"
        )
        return {
            "atr": float(a.iloc[last]),
            "atr_pct": atr_pct,
            "bb_width": bwv,
            "historical_vol": hvv,
            "range_z": z,
            "composite": float(composite),
            "regime": regime,
            "expansion": bool(z > 0.5),
            "compression": bool(z < -0.5),
        }


class StructureEngine:
    @staticmethod
    def analyze(df, lookback=5):
        x = df.tail(max(100, lookback * 10)).copy()
        highs = x.high[
            (x.high.shift(lookback) < x.high) & (x.high.shift(-lookback) < x.high)
        ]
        lows = x.low[
            (x.low.shift(lookback) > x.low) & (x.low.shift(-lookback) > x.low)
        ]

        shs = highs.dropna()
        sls = lows.dropna()
        sh = float(shs.iloc[-1]) if len(shs) else float(x.high.iloc[-1])
        sl = float(sls.iloc[-1]) if len(sls) else float(x.low.iloc[-1])
        prev_h = float(shs.iloc[-2]) if len(shs) > 1 else sh
        prev_l = float(sls.iloc[-2]) if len(sls) > 1 else sl

        hh, hl = sh > prev_h, sl > prev_l
        lh, ll = sh < prev_h, sl < prev_l
        direction = "BULLISH" if hh and hl else "BEARISH" if lh and ll else "TRANSITION"

        last = float(x.close.iloc[-1])
        bos = "BULLISH_BOS" if last > sh else "BEARISH_BOS" if last < sl else "NONE"
        choch = (
            "BULLISH_CHOCH"
            if direction == "BEARISH" and last > sh
            else "BEARISH_CHOCH"
            if direction == "BULLISH" and last < sl
            else "NONE"
        )
        return {
            "direction": direction,
            "swing_high": sh,
            "swing_low": sl,
            "HH": hh,
            "HL": hl,
            "LH": lh,
            "LL": ll,
            "BOS": bos,
            "CHOCH": choch,
            "breakout": bos != "NONE",
            "failure": False,
        }


class PriceActionEngine:
    @staticmethod
    def analyze(df):
        x = df.iloc[-3:].copy()
        o, h, l, c = x.open.iloc[-1], x.high.iloc[-1], x.low.iloc[-1], x.close.iloc[-1]
        body = abs(c - o)
        rng = max(h - l, 1e-12)
        upper = h - max(o, c)
        lower = min(o, c) - l

        bull_eng = (
            c > o and x.close.iloc[-2] < x.open.iloc[-2]
            and c >= x.open.iloc[-2] and o <= x.close.iloc[-2]
        )
        bear_eng = (
            c < o and x.close.iloc[-2] > x.open.iloc[-2]
            and c <= x.open.iloc[-2] and o >= x.close.iloc[-2]
        )
        pin_bull = lower > body * 2 and upper < body
        pin_bear = upper > body * 2 and lower < body
        doji = body / rng < 0.12
        inside = h < x.high.iloc[-2] and l > x.low.iloc[-2]
        outside = h > x.high.iloc[-2] and l < x.low.iloc[-2]
        strong = body / rng > 0.7

        patterns = []
        if bull_eng: patterns.append("BULLISH_ENGULFING")
        if bear_eng: patterns.append("BEARISH_ENGULFING")
        if pin_bull: patterns.append("BULLISH_PIN")
        if pin_bear: patterns.append("BEARISH_PIN")
        if doji: patterns.append("DOJI")
        if inside: patterns.append("INSIDE_BAR")
        if outside: patterns.append("OUTSIDE_BAR")
        if strong: patterns.append("STRONG_MOMENTUM_CANDLE")

        direction = (
            "BULLISH" if any("BULL" in p for p in patterns)
            else "BEARISH" if any("BEAR" in p for p in patterns)
            else "NEUTRAL"
        )
        return {
            "patterns": patterns,
            "direction": direction,
            "body_ratio": body / rng,
            "upper_wick": upper / rng,
            "lower_wick": lower / rng,
        }


class SupportResistanceEngine:
    @staticmethod
    def analyze(df):
        x = df.copy()
        last = float(x.close.iloc[-1])
        levels = []

        for n, label in [(20, "SWING"), (50, "MEDIUM"), (100, "LONG")]:
            if len(x) >= n:
                levels += [
                    (float(x.high.tail(n).max()), "RESISTANCE_" + label),
                    (float(x.low.tail(n).min()), "SUPPORT_" + label),
                ]

        if not levels:
            return {
                "support": last,
                "resistance": last,
                "distance_support": 0,
                "distance_resistance": 0,
                "strength": 0,
                "touches": 0,
                "break_probability": 50,
                "retest_probability": 50,
            }

        supports = [v for v, _ in levels if v <= last]
        resistances = [v for v, _ in levels if v >= last]
        support = max(supports) if supports else min(v for v, _ in levels)
        resistance = min(resistances) if resistances else max(v for v, _ in levels)
        touches = sum(1 for v, _ in levels if abs(v - last) / max(last, 1e-9) < 0.002)

        return {
            "support": float(support),
            "resistance": float(resistance),
            "distance_support": float((last - support) / last * 100),
            "distance_resistance": float((resistance - last) / last * 100),
            "strength": float(min(100, 30 + touches * 12)),
            "touches": touches,
            "break_probability": float(
                np.clip(50 + (last - support) / (resistance - support + 1e-12) * 20, 0, 100)
            ),
            "retest_probability": float(np.clip(60 - touches * 4, 10, 90)),
        }


class BreakoutEngine:
    @staticmethod
    def analyze(df, n=20):
        hi = df.high.shift(1).rolling(n).max()
        lo = df.low.shift(1).rolling(n).min()
        last = df.iloc[-1]
        up = bool(last.close > hi.iloc[-1])
        dn = bool(last.close < lo.iloc[-1])

        mean_v = df.volume.rolling(50).mean().iloc[-1]
        std_v = df.volume.rolling(50).std().iloc[-1]
        volz = (last.volume - mean_v) / (std_v if std_v not in (0, np.nan) else 1)

        breakout = "BULLISH_BREAKOUT" if up else "BEARISH_BREAKOUT" if dn else "NONE"
        confirmed = (up or dn) and volz > 0.5
        fakeout = (up or dn) and volz < -0.5

        return {
            "state": "FAKEOUT" if fakeout else "BREAKOUT" if breakout != "NONE" else "RANGE",
            "direction": "BULLISH" if up else "BEARISH" if dn else "NONE",
            "volume_z": float(volz),
            "confirmed": bool(confirmed),
            "retest": bool(up or dn),
        }


class LiquidityEngine:
    @staticmethod
    def analyze(df):
        x = df.tail(80)
        eqh = float(x.high.quantile(0.95))
        eql = float(x.low.quantile(0.05))
        last = float(x.close.iloc[-1])
        rng = float((x.high - x.low).tail(20).mean())
        sweep_up = bool(last < eqh and x.high.iloc[-1] >= eqh)
        sweep_dn = bool(last > eql and x.low.iloc[-1] <= eql)

        fvg = False
        if len(x) >= 3:
            fvg = bool(
                x.low.iloc[-1] > x.high.iloc[-3]
                or x.high.iloc[-1] < x.low.iloc[-3]
            )

        return {
            "liquidity_high": eqh,
            "liquidity_low": eql,
            "equal_highs": eqh,
            "equal_lows": eql,
            "sweep_up": sweep_up,
            "sweep_down": sweep_dn,
            "FVG": fvg,
            "liquidity_void": bool(rng > 2 * x.close.iloc[-1] * 0.001),
        }




def derive_market_context(trend, volatility, structure, breakout):
    """Context classification retained for compatibility, not a standalone engine."""
    if volatility.get("regime") == "EXTREME":
        return "ABNORMAL"
    if breakout.get("state") == "BREAKOUT":
        return "BREAKOUT"
    if volatility.get("regime") in ("VERY LOW", "LOW") and trend.get("direction") == "SIDEWAYS":
        return "LOW VOLATILITY"
    if trend.get("direction") == "SIDEWAYS":
        return "RANGING"
    if trend.get("strength", 0) >= 72 and structure.get("direction") in ("BULLISH", "BEARISH"):
        return "TRENDING"
    if structure.get("direction") == "TRANSITION":
        return "TRANSITION"
    return "EXTENDED" if trend.get("strength", 0) >= 88 else "NO-TRADE"

class SessionEngine:
    """Authoritative live Forex session/market-calendar engine.

    Rules:
      * Live mode ALWAYS uses the current UTC clock.
      * Historical/backtest mode may use an explicit candle timestamp.
      * Weekly FX open/close is evaluated in New York local time so DST is
        handled correctly (Sunday 17:00 NY -> Friday 17:00 NY).
      * Individual session windows are evaluated in their own local timezone.
      * A session label can never imply that the weekly FX market is open.
    """

    MARKET_ZONES = {
        "SYDNEY": "Australia/Sydney",
        "TOKYO": "Asia/Tokyo",
        "LONDON": "Europe/London",
        "NEW YORK": "America/New_York",
    }

    SESSION_HOURS = {
        "SYDNEY": (8, 17),
        "TOKYO": (9, 18),
        "LONDON": (8, 17),
        "NEW YORK": (8, 17),
    }

    @staticmethod
    def _utc(ts=None, live=False):
        if live or ts is None:
            return pd.Timestamp.now(tz="UTC")
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.tz_convert("UTC")

    @staticmethod
    def _weekly_market_open(utc_ts):
        ny = utc_ts.to_pydatetime().astimezone(ZoneInfo("America/New_York"))
        wd = ny.weekday()  # Mon=0 ... Sun=6
        minute = ny.hour * 60 + ny.minute
        # Standard retail FX weekly boundary in New York local time.
        if wd == 5:  # Saturday
            return False, "WEEKEND / MARKET CLOSED"
        if wd == 6 and minute < 17 * 60:  # Sunday before 17:00 NY
            return False, "WEEKEND / MARKET CLOSED"
        if wd == 4 and minute >= 17 * 60:  # Friday from 17:00 NY
            return False, "WEEKEND / MARKET CLOSED"
        return True, "MARKET OPEN"

    @staticmethod
    def _session_window(name, utc_ts):
        zone = ZoneInfo(SessionEngine.MARKET_ZONES[name])
        local = utc_ts.to_pydatetime().astimezone(zone)
        start_h, end_h = SessionEngine.SESSION_HOURS[name]
        mins = local.hour * 60 + local.minute
        return start_h * 60 <= mins < end_h * 60, local

    @classmethod
    def analyze(cls, ts=None, *, live=False):
        utc_ts = cls._utc(ts, live=live)
        market_open, market_status = cls._weekly_market_open(utc_ts)

        active = []
        local_times = {}
        if market_open:
            for name in cls.MARKET_ZONES:
                is_active, local = cls._session_window(name, utc_ts)
                local_times[name] = local.isoformat()
                if is_active:
                    active.append(name)
        else:
            for name, zone in cls.MARKET_ZONES.items():
                local_times[name] = utc_ts.to_pydatetime().astimezone(ZoneInfo(zone)).isoformat()

        if not market_open:
            session = "MARKET CLOSED"
        elif "LONDON" in active and "NEW YORK" in active:
            session = "LONDON/NEW YORK OVERLAP"
        elif "LONDON" in active:
            session = "LONDON"
        elif "NEW YORK" in active:
            session = "NEW YORK"
        elif "TOKYO" in active:
            session = "TOKYO"
        elif "SYDNEY" in active:
            session = "SYDNEY"
        else:
            session = "OFF-HOURS"

        # Determine the next major session opening for dashboard transparency.
        next_open = None
        if market_open:
            candidates = []
            for name, zone_name in cls.MARKET_ZONES.items():
                local = utc_ts.to_pydatetime().astimezone(ZoneInfo(zone_name))
                start_h, _ = cls.SESSION_HOURS[name]
                candidate = local.replace(hour=start_h, minute=0, second=0, microsecond=0)
                if candidate <= local:
                    candidate = candidate.replace(day=local.day) + pd.Timedelta(days=1).to_pytimedelta()
                candidates.append((candidate.astimezone(ZoneInfo("UTC")), name))
            if candidates:
                nxt, nm = min(candidates, key=lambda z: z[0])
                next_open = {"session": nm, "utc": nxt.isoformat()}
        else:
            # Weekly reopen is Sunday 17:00 New York from any closed period.
            ny = utc_ts.to_pydatetime().astimezone(ZoneInfo("America/New_York"))
            days_ahead = (6 - ny.weekday()) % 7
            if days_ahead == 0 and (ny.hour > 17 or (ny.hour == 17 and ny.minute >= 0)):
                days_ahead = 7
            sunday = (ny + pd.Timedelta(days=days_ahead).to_pytimedelta()).replace(
                hour=17, minute=0, second=0, microsecond=0
            )
            next_open = {"session": "WEEKLY FX OPEN", "utc": sunday.astimezone(ZoneInfo("UTC")).isoformat()}

        return {
            "session": session,
            "current_session": session,
            "active_sessions": active,
            "hour": int(utc_ts.hour),
            "minute": int(utc_ts.minute),
            "weekday": utc_ts.day_name(),
            "timestamp_utc": utc_ts.isoformat(),
            "session_tradeable": bool(market_open and active),
            "market_open": bool(market_open),
            "market_status": market_status,
            "clock_source": "CURRENT_UTC" if (live or ts is None) else "CANDLE_TIMESTAMP",
            "timezone_basis": "UTC clock + local session zones + America/New_York weekly boundary",
            "local_times": local_times,
            "next_open": next_open,
            "session_health": "HEALTHY",
        }


class CurrencyStrengthEngine:
    CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

    @staticmethod
    def analyze(df, symbol="EURUSD"):
        ret = float(df.close.pct_change(20).iloc[-1] * 100) if len(df) > 20 else 0
        base, quote = (symbol[:3], symbol[3:]) if len(symbol) >= 6 else ("EUR", "USD")
        out = {c: 0.0 for c in CurrencyStrengthEngine.CURRENCIES}
        if base in out:
            out[base] = float(np.clip(ret * 10, -5, 5))
        if quote in out:
            out[quote] = float(np.clip(-ret * 10, -5, 5))
        return {
            "matrix": out,
            "base": base,
            "quote": quote,
            "spread": out.get(base, 0) - out.get(quote, 0),
            "strongest": max(out, key=out.get),
            "weakest": min(out, key=out.get),
        }


class CorrelationEngine:
    @staticmethod
    def analyze(history: Dict[str, pd.DataFrame], symbol):
        if symbol not in history or len(history) < 2:
            return {"average_abs_corr": 0.0, "risk": "LOW", "pairs": []}

        s = history[symbol].close.pct_change().tail(200)
        rows = []
        for k, v in history.items():
            if k == symbol:
                continue
            a = v.close.pct_change().tail(200)
            n = min(len(s), len(a))
            if n > 30:
                corr = float(s.tail(n).corr(a.tail(n)))
                if np.isfinite(corr):
                    rows.append((k, corr))

        avg = float(np.mean([abs(c) for _, c in rows])) if rows else 0
        return {
            "average_abs_corr": avg,
            "risk": "HIGH" if avg > 0.75 else "MEDIUM" if avg > 0.5 else "LOW",
            "pairs": sorted(rows, key=lambda z: -abs(z[1])),
        }


class MarketTrackerEngine:
    @staticmethod
    def rank(symbols, analyses):
        rows = []
        for sym in symbols:
            a = analyses[sym]
            c = a["confluence"]
            rows.append(
                {
                    "SYMBOL": sym,
                    "DIRECTION": c["direction"],
                    "SCORE": round(c["score"], 1),
                    "REGIME": a["regime"],
                    "VOLATILITY": a["volatility"]["regime"],
                    "SESSION": a["session"]["session"],
                    "TREND": a["trend"]["label"],
                    "MOMENTUM": a["momentum"]["direction"],
                    "STRUCTURE": a["structure"]["direction"],
                    "ENTRY QUALITY": round(c["quality"], 1),
                    "RISK": a["risk"]["status"],
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("SCORE", ascending=False)
            .reset_index(drop=True)
        )


class MultiTimeframeEngine:
    """
    Current-market MTF engine.

    Live/dashboard mode:
      * Fetches each required timeframe directly from the selected live source.
      * Does NOT try to manufacture H1/H4/D1 from a small M5 window.
      * D1 is read directly so the current D1 market can participate.
      * Each timeframe is evaluated through the core technical engines.
      * Data quality is checked independently for every timeframe.
      * A weak/failed timeframe cannot be silently converted to a bullish/bearish vote.

    Backtest mode:
      * When no symbol/cfg is supplied, the historical dataframe is resampled,
        preserving the original backtest behavior.
    """

    TIMEFRAMES = ("M5", "M15", "M30", "H1", "H4", "D1")

    @staticmethod
    def _engine_direction(x):
        """Use several existing technical engines to determine one TF direction."""
        if x is None or x.empty:
            return "INSUFFICIENT", {"votes": {}, "agreement": 0.0}

        try:
            t = TrendEngine.analyze(x)
            m = MomentumEngine.analyze(x)
            s = StructureEngine.analyze(x)
            pa = PriceActionEngine.analyze(x)
            bo = BreakoutEngine.analyze(x)
            v = VolatilityEngine.analyze(x)
            re = derive_market_context(t, v, s, bo)
        except Exception as exc:
            return "ERROR", {"votes": {}, "agreement": 0.0, "error": str(exc)}

        outputs = {
            "Trend": str(t.get("direction", "")).upper(),
            "Momentum": str(m.get("direction", "")).upper(),
            "Structure": str(s.get("direction", "")).upper(),
            "Price Action": str(pa.get("direction", "")).upper(),
            "Breakout": str(bo.get("direction", "")).upper(),
        }

        valid = [v for v in outputs.values() if v in ("BULLISH", "BEARISH")]
        if not valid:
            return "NEUTRAL", {
                "votes": outputs,
                "agreement": 0.0,
                "regime": re,
                "volatility": v.get("regime", "UNKNOWN"),
            }

        bull = valid.count("BULLISH")
        bear = valid.count("BEARISH")
        total = len(valid)

        # Require a real majority. A 2/5 split must not be advertised as alignment.
        if bull > bear:
            direction = "BULLISH"
            agreement = 100.0 * bull / total
        elif bear > bull:
            direction = "BEARISH"
            agreement = 100.0 * bear / total
        else:
            direction = "MIXED"
            agreement = 50.0

        return direction, {
            "votes": outputs,
            "agreement": float(agreement),
            "regime": re,
            "volatility": v.get("regime", "UNKNOWN"),
        }

    @staticmethod
    def _direct_live_data(df, symbol, timeframe, cfg):
        """
        Get the exact timeframe needed by MTF.

        The currently loaded dataframe is reused only when its identity exactly
        matches the requested pair/timeframe/source. Otherwise the live loader
        fetches that timeframe directly.
        """
        requested_source = str(getattr(cfg, "data_source", "")).upper()

        # DERIV REAL FOREX: one authenticated stream per currency pair already
        # carries M1/M5/M15/M30/H1/H4/D1 candles. Reuse that stream here so the
        # full MTF engine reads the exact same authenticated Deriv feed instead
        # of opening a second connection for every timeframe.
        if requested_source == "DERIV REAL FOREX":
            app_id, token, configured_account = DerivRealForexStream.resolve_credentials(
                getattr(cfg, "deriv_options_account_id", "")
            )
            stream = DerivRealForexStream.get(
                canonical_symbol(symbol),
                cfg.deriv_api_base,
                app_id,
                token,
                configured_account,
                cfg.deriv_stream_outputsize,
                cfg.deriv_stream_reconnect_seconds,
            )
            status = stream.start(wait_seconds=12)
            frame = stream.frame(str(timeframe).upper())
            if frame is None or frame.empty:
                raise RuntimeError(
                    f"Deriv authenticated stream has no {canonical_symbol(symbol)}/{str(timeframe).upper()} frame. "
                    f"Diagnostic: {status}"
                )
            if (
                not status.get("healthy")
                or (
                    status.get("tick_age_seconds") is not None
                    and status.get("tick_age_seconds") > cfg.deriv_stream_stale_seconds
                )
            ):
                raise RuntimeError(
                    f"Deriv authenticated stream is stale/unhealthy for "
                    f"{canonical_symbol(symbol)}/{str(timeframe).upper()}: {status}"
                )
            return MarketDataEngine.normalize(frame), {
                "source": "DERIV REAL FOREX",
                "direct": True,
                "authenticated": True,
                "stream_reused": True,
                "provider_symbol": stream.deriv_symbol,
            }

        if (
            df is not None
            and not df.empty
            and canonical_symbol(symbol) == canonical_symbol(
                st.session_state.get("data_symbol") or symbol
            )
            and str(timeframe).upper() == str(
                st.session_state.get("data_timeframe") or ""
            ).upper()
            and str(st.session_state.get("data_source_loaded") or "").upper()
            == requested_source
        ):
            return MarketDataEngine.normalize(df), {
                "source": requested_source,
                "direct": True,
                "reused_selected_dataset": True,
            }

        # get_live_pair_data is defined later in the module but is available when
        # this method is called by the running Streamlit application.
        x, meta = get_live_pair_data(
            canonical_symbol(symbol), timeframe, cfg, force=False
        )
        return MarketDataEngine.normalize(x), dict(meta or {})

    @staticmethod
    def analyze(df, symbol=None, cfg=None):
        # ---------------- Historical/backtest compatibility ----------------
        if symbol is None or cfg is None:
            states = {}
            details = {}
            # Historical MTF must adapt to the amount of history actually
            # available. Requiring 100 candles on every resampled timeframe
            # makes a normal 250-500 bar M5 test unable to use M15/M30/H1 at
            # all (and therefore creates an artificial zero-trade condition).
            # We never fabricate a timeframe: a timeframe is simply marked
            # unavailable when its resampled history is below this minimum.
            historical_min_bars = 20
            for tf in MultiTimeframeEngine.TIMEFRAMES:
                x = MarketDataEngine.resample(df, tf)
                if len(x) < historical_min_bars:
                    states[tf] = "INSUFFICIENT"
                    details[tf] = {
                        "status": "INSUFFICIENT",
                        "rows": int(len(x)),
                        "source": "HISTORICAL_RESAMPLE",
                        "current": False,
                    }
                    continue
                direction, info = MultiTimeframeEngine._engine_direction(x)
                states[tf] = direction
                details[tf] = {
                    **info,
                    # Preserve the actual timeframe direction explicitly.
                    # The historical trade-quality adapter consumes this field
                    # to determine genuinely usable MTF evidence.
                    "direction": direction,
                    "status": "OK",
                    "rows": int(len(x)),
                    "source": "HISTORICAL_RESAMPLE",
                    "current": False,
                }

            valid = [v for v in states.values() if v in ("BULLISH", "BEARISH")]
            available = [v for v in states.values() if v != "INSUFFICIENT"]
            bull = valid.count("BULLISH")
            bear = valid.count("BEARISH")
            direction = (
                "BULLISH" if bull > bear
                else "BEARISH" if bear > bull
                else "MIXED"
            )
            align = 100.0 * max(bull, bear) / max(len(valid), 1)
            return {
                "states": states,
                "direction": direction,
                "alignment": float(align),
                "coverage_percent": float(100.0 * len(valid) / max(len(available), 1)),
                "available_timeframes": [tf for tf, state in states.items() if state != "INSUFFICIENT"],
                "usable_directional_timeframes": [tf for tf, state in states.items() if state in ("BULLISH", "BEARISH")],
                "strength": (
                    "VERY STRONG" if align >= 85
                    else "STRONG" if align >= 70
                    else "MODERATE" if align >= 55
                    else "WEAK"
                ),
                "details": details,
                "live_current_market": False,
            }

        # ---------------- Current/live market mode ----------------
        states = {}
        details = {}
        valid = []

        for tf in MultiTimeframeEngine.TIMEFRAMES:
            try:
                x, meta = MultiTimeframeEngine._direct_live_data(
                    df, symbol, tf, cfg
                )
                if x.empty or len(x) < 100:
                    states[tf] = "INSUFFICIENT"
                    details[tf] = {
                        "status": "INSUFFICIENT",
                        "rows": int(len(x)),
                        "source": str(meta.get("source", "UNKNOWN")),
                        "current": tf == "D1",
                    }
                    continue

                dq_age = {
                    "M5": getattr(cfg, "data_max_age_seconds", 420),
                    "M15": 1200,
                    "M30": 2400,
                    "H1": 4800,
                    "H4": 18000,
                    # D1 must be current enough to represent today's market.
                    # This is deliberately tighter than "one calendar day".
                    "D1": 7200,
                }.get(tf, getattr(cfg, "data_max_age_seconds", 420))

                dq = DataIntegrityEngine.assess(x, tf, dq_age)
                direction, info = MultiTimeframeEngine._engine_direction(x)

                # A failed/stale timeframe is never allowed to vote.
                usable = bool(dq.get("signal_allowed", False)) and direction in (
                    "BULLISH", "BEARISH"
                )
                if usable:
                    valid.append(direction)
                    states[tf] = direction
                    status = "OK"
                else:
                    states[tf] = "UNAVAILABLE"
                    status = "DATA QUALITY BLOCKED"

                last_time = (
                    str(x["time"].iloc[-1])
                    if "time" in x.columns and not x.empty
                    else "-"
                )

                details[tf] = {
                    **info,
                    "status": status,
                    "rows": int(len(x)),
                    "source": str(meta.get("source", "UNKNOWN")),
                    "last_timestamp": last_time,
                    "age_seconds": float(dq.get("age_seconds", 0.0)),
                    "data_quality": dq.get("status", "UNKNOWN"),
                    "data_quality_score": float(dq.get("score", 0.0)),
                    "current": True,
                    "usable_vote": usable,
                }
            except Exception as exc:
                states[tf] = "UNAVAILABLE"
                details[tf] = {
                    "status": "FETCH/ENGINE ERROR",
                    "rows": 0,
                    "source": str(getattr(cfg, "data_source", "UNKNOWN")),
                    "current": True,
                    "usable_vote": False,
                    "error": str(exc),
                }

        bull = valid.count("BULLISH")
        bear = valid.count("BEARISH")

        # MTF must have enough independent live timeframes to make a reliable
        # alignment claim. One valid timeframe must NEVER become "100% aligned".
        min_valid_timeframes = 3
        if len(valid) < min_valid_timeframes:
            direction = "INSUFFICIENT"
            align = 0.0
        else:
            direction = (
                "BULLISH" if bull > bear
                else "BEARISH" if bear > bull
                else "MIXED"
            )
            # Coverage is part of the alignment score: unavailable timeframes
            # count against alignment rather than disappearing from the denominator.
            align = 100.0 * max(bull, bear) / len(MultiTimeframeEngine.TIMEFRAMES)

        # Explicit D1 status: this tells the user whether today's current
        # daily market was actually read, rather than inferred from M5.
        d1 = details.get("D1", {})
        d1_current = bool(d1.get("usable_vote", False))
        d1_direction = states.get("D1", "UNAVAILABLE")

        # Current D1 is a mandatory anchor for live MTF. If today's D1 cannot
        # be read and quality-checked, the system must not pretend to have a
        # complete higher-timeframe/intraday alignment.
        if not d1_current:
            direction = "INSUFFICIENT"
            align = 0.0

        return {
            "states": states,
            "direction": direction if valid else "MIXED",
            "alignment": float(align if valid else 0.0),
            "strength": (
                "VERY STRONG" if align >= 85
                else "STRONG" if align >= 70
                else "MODERATE" if align >= 55
                else "WEAK"
            ),
            "coverage_percent": 100.0 * len(valid) / len(MultiTimeframeEngine.TIMEFRAMES),
            "details": details,
            "live_current_market": True,
            "d1_current_available": d1_current,
            "d1_current_direction": d1_direction,
            "usable_timeframes": len(valid),
            "total_timeframes": len(MultiTimeframeEngine.TIMEFRAMES),
        }


class EconomicEngine:
    @staticmethod
    def analyze(events: pd.DataFrame, symbol: str, now=None, before=30, after=15):
        now = now or pd.Timestamp.now(tz="UTC")
        if events is None or events.empty:
            return {
                "bias": "NEUTRAL",
                "score": 50.0,
                "risk": "UNKNOWN",
                "blocked": False,
                "next_event": "No event data",
                "events": [],
            }

        e = events.copy()
        e.columns = [str(c).lower().strip().replace(" ", "_") for c in e.columns]
        if "time" not in e:
            return {
                "bias": "NEUTRAL",
                "score": 50.0,
                "risk": "UNKNOWN",
                "blocked": False,
                "next_event": "Invalid event data",
                "events": [],
            }

        e["time"] = pd.to_datetime(e["time"], errors="coerce", utc=True)
        e = e.dropna(subset=["time"]).sort_values("time")
        currencies = [symbol[:3], symbol[3:]] if len(symbol) >= 6 else []

        if "currency" not in e:
            e["currency"] = ""
        if "importance" not in e:
            e["importance"] = "LOW"

        e["currency"] = e["currency"].astype(str).str.upper()
        e["importance"] = e["importance"].astype(str).str.upper()
        rel = e[e["currency"].isin(currencies)].copy()

        bias = 0
        blocked = False
        nearest = None

        for _, r in rel.iterrows():
            mins = (r["time"] - now).total_seconds() / 60
            imp = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(
                r["importance"], 1
            )
            if -after <= mins <= before:
                blocked = True
            if mins >= 0 and nearest is None:
                nearest = r
            d = str(r.get("bias", "NEUTRAL")).upper()
            bias += (1 if d == "BULLISH" else -1 if d == "BEARISH" else 0) * imp

        score = float(np.clip(50 + bias * 8, 0, 100))
        direction = "BULLISH" if score >= 58 else "BEARISH" if score <= 42 else "NEUTRAL"
        risk = (
            "CRITICAL" if blocked
            else "HIGH" if any(rel.importance.isin(["HIGH", "CRITICAL"]))
            else "NORMAL"
        )

        return {
            "bias": direction,
            "score": score,
            "risk": risk,
            "blocked": blocked,
            "next_event": nearest.get("event", "Event") if nearest is not None else "No upcoming relevant event",
            "events": rel.head(10).to_dict("records"),
        }


class COTEngine:
    @staticmethod
    def analyze(cot: pd.DataFrame, symbol="EURUSD"):
        if cot is None or cot.empty:
            return {
                "bias": "NEUTRAL",
                "score": 50.0,
                "status": "NO DATA",
                "net": 0,
                "weekly_change": 0,
                "percentile": 50,
            }

        x = cot.copy()
        x.columns = [str(c).lower().strip().replace(" ", "_") for c in x.columns]
        base = symbol[:3] if len(symbol) >= 6 else "EUR"

        if "currency" in x:
            x = x[x.currency.astype(str).str.upper() == base]

        if x.empty:
            return {
                "bias": "NEUTRAL",
                "score": 50.0,
                "status": "NO DATA",
                "net": 0,
                "weekly_change": 0,
                "percentile": 50,
            }

        for c in [
            "commercial_long",
            "commercial_short",
            "noncommercial_long",
            "noncommercial_short",
        ]:
            if c not in x:
                x[c] = 0.0
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)

        x["net_spec"] = x.noncommercial_long - x.noncommercial_short
        net = float(x.net_spec.iloc[-1])
        change = float(x.net_spec.diff().iloc[-1]) if len(x) > 1 else 0
        pct = float(x.net_spec.rank(pct=True).iloc[-1] * 100) if len(x) > 1 else 50

        mean_abs = abs(x.net_spec).mean() + 1e-9
        std = x.net_spec.std() + 1e-9
        score = float(
            np.clip(
                50
                + np.sign(net) * min(35, abs(net) / mean_abs * 20)
                + np.sign(change) * min(15, abs(change) / std * 5),
                0,
                100,
            )
        )
        bias = "BULLISH" if score >= 58 else "BEARISH" if score <= 42 else "NEUTRAL"

        return {
            "bias": bias,
            "score": score,
            "status": "OK",
            "net": net,
            "weekly_change": change,
            "percentile": pct,
            "commercial_long": float(x.commercial_long.iloc[-1]),
            "commercial_short": float(x.commercial_short.iloc[-1]),
            "spec_long": float(x.noncommercial_long.iloc[-1]),
            "spec_short": float(x.noncommercial_short.iloc[-1]),
        }


class ConfluenceEngine:
    @staticmethod
    def score(
        trend,
        momentum,
        volatility,
        structure,
        price_action,
        sr,
        breakout,
        liquidity,
        regime,
        session,
        mtf,
        economic,
        cot,
    ):
        direction_votes = []
        for d in [
            trend["direction"],
            momentum["direction"],
            structure["direction"],
            price_action["direction"],
            mtf["direction"],
            economic["bias"],
            cot["bias"],
        ]:
            if d in ("BULLISH", "BEARISH"):
                direction_votes.append(d)

        bull = direction_votes.count("BULLISH")
        bear = direction_votes.count("BEARISH")
        direction = "BULLISH" if bull > bear else "BEARISH" if bear > bull else "WAIT"

        parts = {
            "Trend": np.clip(trend["strength"] * 0.20, 0, 20),
            "Momentum": np.clip(momentum["score"] * 0.15, 0, 15),
            "Structure": 20 if structure["direction"] == direction else 10 if structure["direction"] == "TRANSITION" else 4,
            "Price Action": 12 if price_action["direction"] == direction else 6 if price_action["direction"] == "NEUTRAL" else 3,
            "S/R": np.clip(sr["strength"] * 0.10, 0, 10),
            "Breakout": 10 if breakout["state"] == "BREAKOUT" and breakout["direction"] == direction else 7 if breakout["state"] == "RANGE" else 2,
            "Volatility": 5 if volatility["regime"] in ("NORMAL", "LOW") else 2 if volatility["regime"] == "HIGH" else 0,
            "Regime": 5 if regime in ("TRENDING", "BREAKOUT", "RANGING") else 1,
            "Session": 3 if session["session_tradeable"] else 0,
            "Liquidity": 2 if (liquidity["sweep_up"] or liquidity["sweep_down"] or liquidity["FVG"]) else 1,
            "MTF": 10 * (mtf["alignment"] / 100),
            "Economic": 5 * (economic["score"] / 100),
            "COT": 3 * (cot["score"] / 100),
        }

        total = float(np.clip(sum(parts.values()), 0, 100))
        if economic["blocked"] or volatility["regime"] == "EXTREME":
            total = min(total, 55)

        quality = float(
            np.clip(
                total - (20 if direction == "WAIT" else 0)
                + (5 if mtf["alignment"] >= 80 else 0),
                0,
                100,
            )
        )
        grade = (
            "HIGH QUALITY" if total >= 85
            else "GOOD" if total >= 72
            else "WATCH" if total >= 60
            else "NO-TRADE"
        )

        return {
            "score": total,
            "direction": direction,
            "quality": quality,
            "entry_quality": quality,  # compatibility alias
            "components": parts,
            "grade": grade,
        }


class RiskEngine:
    @staticmethod
    def evaluate(account, cfg: Config, confluence, volatility, economic, correlation, spread=0.8):
        reasons = []
        if account["daily_loss_pct"] >= cfg.max_daily_loss * 100:
            reasons.append("DAILY LOSS LIMIT")
        if account["drawdown_pct"] >= cfg.max_drawdown * 100:
            reasons.append("MAX DRAWDOWN")
        if account["open_positions"] >= cfg.max_open_positions:
            reasons.append("MAX OPEN POSITIONS")
        if spread > cfg.max_spread_pips:
            reasons.append("SPREAD TOO WIDE")
        if volatility["regime"] == "EXTREME":
            reasons.append("ABNORMAL VOLATILITY")
        if economic["blocked"]:
            reasons.append("ECONOMIC EVENT BLACKOUT")
        if correlation["risk"] == "HIGH":
            reasons.append("CORRELATED EXPOSURE")
        if confluence["score"] < cfg.min_score:
            reasons.append("SCORE BELOW THRESHOLD")

        ok = not reasons
        return {
            "approved": ok,
            "status": "APPROVED" if ok else "VETO",
            "reasons": reasons,
            "risk_pct": cfg.risk_per_trade * 100,
        }


# ============================================================
# ENTRY / EXECUTION / MONITORING
# ============================================================

class ForexEntryEngine:
    @staticmethod
    def calculate(df, direction, confluence, cfg, symbol="EURUSD"):
        last = float(df.close.iloc[-1])
        a = float(atr(df).iloc[-1])
        s = SupportResistanceEngine.analyze(df)

        if direction == "BULLISH":
            entry = last
            sl = min(s["support"], last - a * 1.5)
            tp = max(s["resistance"], last + a * 3)
            side = "BUY"
        elif direction == "BEARISH":
            entry = last
            sl = max(s["resistance"], last + a * 1.5)
            tp = min(s["support"], last - a * 3)
            side = "SELL"
        else:
            return {
                "approved": False,
                "direction": "WAIT",
                "entry": last,
                "sl": None,
                "tp": None,
                "rr": 0,
                "lot": 0,
                "zone_low": last - a * 0.25,
                "zone_high": last + a * 0.25,
            }

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / max(risk, 1e-9)
        # Position sizing uses the supplied current equity when available.
        # This keeps live/paper/backtest risk budgets consistent as equity changes.
        current_equity = getattr(cfg, "current_equity", None)
        try:
            current_equity = float(current_equity) if current_equity is not None else float(cfg.initial_balance)
        except (TypeError, ValueError):
            current_equity = float(cfg.initial_balance)
        current_equity = max(current_equity, 0.0)
        risk_money = current_equity * cfg.risk_per_trade
        pip = 0.01 if symbol.endswith("JPY") else 0.0001
        pip_value_per_lot = 9.0 if symbol.endswith("JPY") else 10.0
        lot = risk_money / max((risk / pip) * pip_value_per_lot, 1e-9)

        return {
            "approved": confluence["score"] >= cfg.min_score,
            "direction": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "lot": float(np.clip(lot, 0.01, 100)),
            "zone_low": min(entry, entry - a * 0.25),
            "zone_high": max(entry, entry + a * 0.25),
            "quality": confluence["quality"],
            "risk_money": risk_money,
        }



class ExecutionEngine:
    @staticmethod
    def paper_order(market, symbol, direction, entry, sl=None, tp=None, expiry=None, lot=0):
        return {
            "id": uuid.uuid4().hex[:10],
            "time": datetime.now(timezone.utc).isoformat(),
            "market": market,
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "expiry": expiry,
            "lot": lot,
            "status": "OPEN",
            "paper": True,
        }


class TradeMonitorEngine:
    """Paper-trade monitor: marks open trades SL/TP/expiry status from latest candle."""

    @staticmethod
    def monitor_trade(trade: Dict[str, Any], latest_price: float) -> Dict[str, Any]:
        if trade.get("status") != "OPEN":
            return trade

        direction = trade.get("direction", "")
        sl, tp = trade.get("sl"), trade.get("tp")

        if trade.get("market") == "FOREX":
            if direction == "BUY":
                if sl is not None and latest_price <= sl:
                    trade["status"] = "STOPPED"
                elif tp is not None and latest_price >= tp:
                    trade["status"] = "TARGET"
            elif direction == "SELL":
                if sl is not None and latest_price >= sl:
                    trade["status"] = "STOPPED"
                elif tp is not None and latest_price <= tp:
                    trade["status"] = "TARGET"

        trade["last_price"] = float(latest_price)
        return trade

    @staticmethod
    def monitor_all(journal, latest_price):
        return [TradeMonitorEngine.monitor_trade(t, latest_price) for t in journal]


class TradeJournal:
    @staticmethod
    def append(trade):
        st.session_state.journal.append(trade)


# ============================================================
# V14 ARCHITECTURE-LEVEL TESTING
# ============================================================
# IMPORTANT:
#   * These V14 test engines replay the existing specialist engines in
#     chronological order and route their evidence through the V14 central
#     assessment/intelligence/final-decision path.
#   * No live API calls, no future candles, and no synthetic candles are created
#     by these test engines.
#   * The V14 forward test fixes the old split bug: the 250-candle minimum is
#     applied to the COMPLETE dataset, not independently to train and OOS.
# ============================================================

def _v14_historical_data_quality(df: pd.DataFrame, timeframe: str = "M5") -> Dict[str, Any]:
    """Historical-only data-quality gate; never compares old candles to wall clock."""
    if df is None or df.empty:
        return {
            "score": 0.0, "status": "BAD", "reasons": ["NO DATA"],
            "stale": False, "age_seconds": None, "last_timestamp": None,
            "rows": 0, "signal_allowed": False, "historical": True,
        }
    x = MarketDataEngine.normalize(df)
    reasons = []
    score = 100.0
    if len(x) < 100:
        score -= 30
        reasons.append("INSUFFICIENT HISTORY")
    bad_ohlc = int((
        (x.high < x.low) | (x.high < x.open) | (x.high < x.close) |
        (x.low > x.open) | (x.low > x.close)
    ).sum())
    if bad_ohlc:
        score -= min(30, bad_ohlc * 5)
        reasons.append("INVALID OHLC")
    dup = int(x.time.duplicated().sum())
    if dup:
        score -= 10
        reasons.append("DUPLICATE TIMESTAMPS")
    diffs = x.time.diff().dt.total_seconds().dropna()
    if len(diffs):
        expected = {
            "M1": 60, "M3": 180, "M5": 300, "M15": 900,
            "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400,
        }.get(str(timeframe).upper(), 300)
        gap_multiplier = 4.5 if str(timeframe).upper() == "D1" else 1.8
        missing = int((diffs > expected * gap_multiplier).sum())
        if missing:
            score -= min(20, missing * 2)
            reasons.append(f"MISSING/GAPPED CANDLES: {missing}")
    if len(x) >= 8 and x.close.tail(8).nunique() == 1:
        score -= 20
        reasons.append("FROZEN PRICE")
    score = float(np.clip(score, 0, 100))
    status = "EXCELLENT" if score >= 90 else "GOOD" if score >= 75 else "DEGRADED" if score >= 55 else "BAD"
    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "stale": False,
        "age_seconds": None,
        "last_timestamp": str(x.time.iloc[-1]),
        "rows": int(len(x)),
        "signal_allowed": score >= 55 and not any("INVALID" in r for r in reasons),
        "historical": True,
    }


def _v14_historical_candle_timing(window: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
    """Historical candle-timing evidence without wall-clock dependence."""
    if window is None or window.empty:
        return {"direction": "NO DATA", "score": 0.0, "progress_pct": 0.0, "status": "NO DATA"}
    row = window.iloc[-1]
    rng = max(float(row.high - row.low), 1e-12)
    body = float(row.close - row.open) / rng
    direction = "BULLISH" if body > 0.10 else "BEARISH" if body < -0.10 else "NEUTRAL"
    score = float(np.clip(50 + body * 50, 0, 100))
    return {
        "direction": direction,
        "score": score,
        "progress_pct": 100.0,
        "forming": False,
        "timeframe": str(timeframe).upper(),
        "status": "CLOSED_HISTORICAL_CANDLE",
    }




def _v14_historical_trade_quality(a: Dict[str, Any], ensemble: Dict[str, Any],
                                  no_trade: Dict[str, Any], dq: Dict[str, Any],
                                  direction: str, cfg: Config) -> Dict[str, Any]:
    """Historical-test adapter for the Trade Quality responsibility.

    Historical replay uses only evidence available inside the replay window.
    It never invents live state or future information.
    """
    mtf = a.get("mtf", {}) or {}
    details = mtf.get("details", {}) or {}
    usable = [
        tf for tf, info in details.items()
        if str(info.get("status", "")).upper() == "OK"
        and info.get("direction") in {"BULLISH", "BEARISH"}
    ]
    available = [
        tf for tf, info in details.items()
        if str(info.get("status", "")).upper() == "OK"
    ]
    coverage = 100.0 * len(usable) / max(len(available), 1)
    mtf_alignment = float(mtf.get("alignment", 0) or 0)
    agreement = float(ensemble.get("agreement", 0) or 0)
    ai_conf = float((a.get("ai") or {}).get("confidence", 0) or 0)
    score = float(np.clip(
        agreement * 0.35
        + float((a.get("confluence") or {}).get("score", 0) or 0) * 0.20
        + ai_conf * 0.15
        + mtf_alignment * 0.20
        + float(dq.get("score", 0) or 0) * 0.10,
        0, 100,
    ))
    vetoes = []
    if not dq.get("signal_allowed", False):
        vetoes.append("DATA QUALITY BLOCK")
    if direction not in {"BULLISH", "BEARISH"}:
        vetoes.append("NO VALID DIRECTION")
    if not ensemble.get("approved", False):
        vetoes.extend(ensemble.get("reasons", []))
    if not no_trade.get("trade_allowed", False):
        vetoes.extend(no_trade.get("reasons", []))
    if mtf_alignment < 60:
        vetoes.append("HISTORICAL MTF ALIGNMENT BELOW 60")
    # Historical tests may legitimately have only M5/M15/M30 (or another
    # subset) available after resampling. Require two genuinely usable
    # timeframes, but do not require unavailable higher timeframes such as H4/D1
    # to exist in a short historical dataset. No data is invented.
    if len(usable) < 2:
        vetoes.append("INSUFFICIENT HISTORICAL MTF COVERAGE")
    # Use the V14 configured minimum signal confidence rather than introducing
    # an undocumented hard-coded historical-only threshold.
    quality_threshold = float(getattr(cfg, "min_signal_confidence", 72.0))
    if score < quality_threshold:
        vetoes.append(f"TRADE QUALITY BELOW {quality_threshold:.0f}")
    vetoes = list(dict.fromkeys(str(x) for x in vetoes if x))
    qualified = not vetoes
    return {
        "status": "QUALIFIED" if qualified else "REJECTED",
        "qualified": qualified,
        "score": round(score, 1),
        "direction": direction,
        "directional_agreement": round(agreement, 1),
        "mtf_alignment": round(mtf_alignment, 1),
        "mtf_coverage": round(coverage, 1),
        "usable_timeframes": usable,
        "reasons": vetoes,
        "historical_mode": True,
        "quality_threshold": quality_threshold,
        "available_timeframes": available,
    }




class V14ReplayDecisionEngine:
    """Chronological decision replay using the existing V14 specialist engines."""

    @staticmethod
    def evaluate(window: pd.DataFrame, symbol: str, timeframe: str,
                 cfg: Config, equity: float, ai_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        x = MarketDataEngine.normalize(window)
        dq = _v14_historical_data_quality(x, timeframe)
        if not dq.get("signal_allowed", False):
            return {"eligible": False, "reason": "HISTORICAL_DATA_QUALITY_BLOCK", "data_quality": dq}

        t = TrendEngine.analyze(x)
        m = MomentumEngine.analyze(x)
        v = VolatilityEngine.analyze(x)
        s = StructureEngine.analyze(x)
        pa = PriceActionEngine.analyze(x)
        sr = SupportResistanceEngine.analyze(x)
        bo = BreakoutEngine.analyze(x)
        li = LiquidityEngine.analyze(x)
        ema_evidence = EMAEngine.analyze(x)
        macd_evidence = MACDEngine.analyze(x)
        fvg = FVGEngine.analyze(x)
        microstructure = MarketMicrostructureEngine.analyze(x)
        volume_order_flow = OrderFlowVolumeFlowEngine.analyze(x)
        reversal = ReversalDetectionEngine.analyze(x)
        re = derive_market_context(t, v, s, bo)
        se = SessionEngine.analyze(x.time.iloc[-1])
        cs = CurrencyStrengthEngine.analyze(x, symbol)
        mtf = MultiTimeframeEngine.analyze(x)
        volume_flow = VolumeFlowEngine.analyze(x)
        candle_timing = _v14_historical_candle_timing(x, timeframe)
        market_region = MarketRegionEngine.analyze(x)
        direct_probability = DirectProbabilityEngine.predict(x)
        eco = {"bias": "NEUTRAL", "score": 50.0, "blocked": False, "risk": "LOW", "status": "HISTORICAL_EVENT_DATA_NOT_SUPPLIED"}
        cot = {"bias": "NEUTRAL", "score": 50.0, "status": "HISTORICAL_COT_DATA_NOT_SUPPLIED"}
        corr = CorrelationEngine.analyze({symbol: x}, symbol)
        c = ConfluenceEngine.score(t, m, v, s, pa, sr, bo, li, re, se, mtf, eco, cot)
        adv_m = MomentumDirectionEngine.analyze(x)
        ai = dict(ai_override) if isinstance(ai_override, dict) else AIProbabilityEngine.predict(x)
        market_tracker = {"direction": c.get("direction", "WAIT"), "score": c.get("score", 0), "status": "HISTORICAL_REPLAY"}

        risk = RiskControlEngine.evaluate(
            {
                "daily_loss_pct": 0.0,
                "drawdown_pct": max(0.0, (cfg.initial_balance - equity) / max(cfg.initial_balance, 1e-9) * 100.0),
                "open_positions": 0,
            },
            cfg, c, v, eco, corr,
        )
        advisory = {
            "trend": t, "momentum": m, "volatility": v, "structure": s,
            "price_action": pa, "sr": sr, "breakout": bo, "liquidity": li,
            "ema": ema_evidence, "macd": macd_evidence, "fvg": fvg,
            "microstructure": microstructure, "order_flow": volume_order_flow, "reversal": reversal,
            "regime": re, "session": se, "currency_strength": cs, "mtf": mtf,
            "economic": eco, "cot": cot, "correlation": corr, "confluence": c,
            "risk": risk, "data_quality": dq, "advanced_momentum": adv_m,
            "ai": ai, "volume_flow": volume_flow, "candle_timing": candle_timing,
            "market_region": market_region, "direct_probability": direct_probability,
            "market_tracker": market_tracker,
            "signal_timing": {"fresh": True, "age_seconds": 0.0, "status": "FRESH_HISTORICAL"},
        }
        ensemble = EnsembleDecisionEngine.decide(advisory, ai, dq, cfg)
        no_trade = NoTradeEngine.evaluate(advisory, ensemble, dq, cfg, signal_age=0)
        advisory["ensemble"] = ensemble
        advisory["no_trade"] = no_trade

        # Package specialist evidence exactly as the live V14 architecture does.
        evidence_inputs = {
            "TrendEngine": t, "MomentumEngine": m, "VolatilityEngine": v,
            "StructureEngine": s, "PriceActionEngine": pa,
            "EMAEngine": ema_evidence, "MACDEngine": macd_evidence, "FVGEngine": fvg,
            "MarketMicrostructureEngine": microstructure, "OrderFlowVolumeFlowEngine": volume_order_flow,
            "ReversalDetectionEngine": reversal,
            "SupportResistanceEngine": sr, "BreakoutEngine": bo,
            "LiquidityEngine": li, "CurrencyStrengthEngine": cs,
            "CorrelationEngine": corr, "VolumeFlowEngine": volume_flow,
            "MTFConfirmation": mtf, "AIProbability": ai,
            "SessionEngine": se, "EconomicCalendarEventEngine": eco,
            "NewsEventFilterEngine": {"filter_status":"NOT_SUPPLIED", "status":"HISTORICAL_EVENT_DATA_NOT_SUPPLIED", "blocked":False},
            "COTEngine": cot, "CurrencyStrengthEngine": cs, "CorrelationEngine": corr,
            "CandleTimingEngine": candle_timing, "MarketRegionEngine": market_region,
            "DirectProbabilityEngine": direct_probability, "AdvancedMomentumEngine": adv_m,
            "MarketTrackerEngine": market_tracker, "ConfluenceEngine": c, "EnsembleDecisionEngine": ensemble,
            "RiskControlEngine": risk, "SignalTimingEngine": {"fresh":True,"status":"FRESH_HISTORICAL","score":100.0},
            "ReversalContext": {"direction": re if re in {"BULLISH", "BEARISH"} else "NEUTRAL", "context": re},
        }
        data_ts = x.time.iloc[-1]
        evidence_packages = {
            name: package_engine_evidence(
                engine_id=name.upper(), engine_name=name, symbol=symbol,
                timeframe=timeframe, evidence=evidence, data_timestamp=data_ts,
                quality={"score": dq.get("score", 0.0), "status": "PASS" if dq.get("signal_allowed") else "BLOCK"},
                provenance={"source": "HISTORICAL_REPLAY", "live_api_called": False},
            )
            for name, evidence in evidence_inputs.items()
        }
        central = CentralAssessmentDataQualityEngine.assess(symbol, timeframe, dq, evidence_packages)
        intelligence = SignalIntelligenceLayer.assess(central, evidence_packages)
        direction = str(intelligence.get("market_context", "NEUTRAL")).upper()
        entry_cfg = replace(cfg, current_equity=float(equity))
        entry = ForexEntryEngine.calculate(x, direction, c, entry_cfg, symbol=symbol)
        timing = {"fresh": True, "age_seconds": 0.0, "status": "FRESH_HISTORICAL"}
        execution = ExecutionQualityEngine.assess(current_price=float(x.close.iloc[-1]))
        historical_tq = _v14_historical_trade_quality(advisory, ensemble, no_trade, dq, direction, cfg)
        # Risk is evaluated again after the quality stage so the final decision
        # always receives the current account/equity risk gate.
        final_risk = dict(risk)
        if not historical_tq.get("qualified", False):
            final_risk["approved"] = False
            final_risk["status"] = "VETO"
            final_risk["reasons"] = list(dict.fromkeys(final_risk.get("reasons", []) + ["TRADE QUALITY FILTER"] + historical_tq.get("reasons", [])))

        entry_valid = bool(entry.get("approved") and direction in {"BULLISH", "BEARISH"})
        final_decision = FinalDecisionEngine.resolve(
            intelligence, final_risk, entry_valid,
            True, execution.get("status") == "PASS",
            {"halted": False, "reason": "HISTORICAL_REPLAY", "timestamp": None},
            trade_quality_valid=bool(historical_tq.get("qualified", False)),
            engine_health_valid=True,
        )
        # Trade Quality is an explicit pre-decision gate. FinalDecision remains
        # the only component that emits BUY/SELL/NO-TRADE.
        if not historical_tq.get("qualified", False) and final_decision.get("final_decision") != "NO-TRADE":
            raise RuntimeError("ARCHITECTURE VIOLATION: trade-quality veto bypassed final decision")

        return {
            "eligible": final_decision.get("final_decision") in {"BUY", "SELL"},
            "final_decision": final_decision,
            "direction": direction,
            "entry": entry,
            "risk": final_risk,
            "trade_quality": historical_tq,
            "data_quality": dq,
            "central_assessment": central,
            "signal_intelligence": intelligence,
            "ensemble": ensemble,
            "no_trade": no_trade,
            "ai": ai,
            "confluence": c,
            "mtf": mtf,
            "analysis": advisory,
            "evidence_packages": evidence_packages,
            "decision_timestamp": str(x.time.iloc[-1]),
        }


def _v14_test_metrics(trades: pd.DataFrame, initial_balance: float, final_equity: float,
                      ambiguous_exits: int = 0, open_at_end: int = 0) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "net_profit": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "profit_factor": 0.0, "expectancy": 0.0, "average_R": 0.0,
            "max_drawdown": 0.0, "final_equity": float(final_equity),
            "return_pct": (float(final_equity) / max(float(initial_balance), 1e-9) - 1.0) * 100.0,
            "consecutive_losses": 0, "consecutive_wins": 0,
            "ambiguous_exits": int(ambiguous_exits), "open_at_test_end": int(open_at_end),
            "status": "NO_CLOSED_TRADES",
        }
    closed = trades[trades["status"].isin(["WIN", "LOSS", "OPEN_AT_TEST_END"])].copy()
    closed = closed[closed["status"].isin(["WIN", "LOSS"])].copy()
    wins = int((closed["status"] == "WIN").sum())
    losses = int((closed["status"] == "LOSS").sum())
    gp = float(closed.loc[closed.pnl > 0, "pnl"].sum()) if not closed.empty else 0.0
    gl = abs(float(closed.loc[closed.pnl < 0, "pnl"].sum())) if not closed.empty else 0.0
    seq_w = seq_l = max_w = max_l = 0
    for outcome in closed.get("status", []):
        if outcome == "WIN":
            seq_w += 1; seq_l = 0; max_w = max(max_w, seq_w)
        else:
            seq_l += 1; seq_w = 0; max_l = max(max_l, seq_l)
    return {
        "trades": int(len(closed)),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(len(closed), 1) * 100.0, 2),
        "net_profit": float(closed.pnl.sum()) if not closed.empty else 0.0,
        "gross_profit": gp,
        "gross_loss": gl,
        "profit_factor": gp / gl if gl else (float("inf") if gp > 0 else 0.0),
        "expectancy": float(closed.pnl.mean()) if not closed.empty else 0.0,
        "average_R": float(closed.return_R.mean()) if not closed.empty else 0.0,
        "max_drawdown": float(trades.drawdown_pct.max()) if "drawdown_pct" in trades else 0.0,
        "final_equity": float(final_equity),
        "return_pct": (float(final_equity) / max(float(initial_balance), 1e-9) - 1.0) * 100.0,
        "consecutive_losses": max_l,
        "consecutive_wins": max_w,
        "ambiguous_exits": int(ambiguous_exits),
        "open_at_test_end": int(open_at_end),
        "status": "OK",
        "statistical_validation": None,
    }


class V14BacktestEngine:
    """Architecture-level chronological historical replay for Forex only."""
    # Robustness gate: 250 candles can execute a smoke test, but cannot support
    # a meaningful accuracy claim. V14 accuracy testing therefore requires 1000.
    MIN_CANDLES = 1000
    WARMUP = 120
    MAX_HOLDING_BARS = 24
    AI_REFRESH_BARS = 5

    @staticmethod
    def run(df: pd.DataFrame, cfg: Config, symbol="EURUSD", timeframe="M5", enforce_minimum=True):
        x = MarketDataEngine.normalize(df)
        minimum = V14BacktestEngine.MIN_CANDLES if enforce_minimum else (V14BacktestEngine.WARMUP + 2)
        if len(x) < minimum:
            return pd.DataFrame(), {
                "error": f"V14 Backtest requires at least {minimum} candles for this run.",
                "rows": int(len(x)), "required": int(minimum),
            }
        initial = float(cfg.initial_balance)
        equity = initial
        peak = equity
        rows = []
        open_trade = None
        ambiguous = 0
        ai_cache = None
        ai_cache_bar = -1
        rejection_counts = {}

        def close_trade(trade, exit_price, exit_time, status, reason, bar_index):
            nonlocal equity, peak
            risk_distance = max(float(trade["risk_distance"]), 1e-12)
            if trade["direction"] == "BUY":
                r_mult = (float(exit_price) - float(trade["entry_fill"])) / risk_distance
            else:
                r_mult = (float(trade["entry_fill"]) - float(exit_price)) / risk_distance
            r_mult = float(r_mult)
            risk_money = equity * cfg.risk_per_trade
            pnl = risk_money * r_mult
            commission = float(getattr(cfg, "commission_per_lot", 0.0) or 0.0) * float(trade.get("lot", 0.0) or 0.0)
            pnl -= commission
            equity += pnl
            peak = max(peak, equity)
            dd = (peak - equity) / max(peak, 1e-9) * 100.0
            trade.update({
                "exit_time": exit_time, "exit_price": float(exit_price),
                "status": status, "exit_reason": reason, "return_R": r_mult,
                "pnl": float(pnl), "equity": float(equity), "drawdown_pct": float(dd),
                "exit_bar": int(bar_index), "commission": commission,
            })
            return dict(trade)

        for i in range(V14BacktestEngine.WARMUP, len(x) - 1):
            # Manage the already-open position first. No new signal can use
            # information from the current bar before its outcome is resolved.
            if open_trade is not None:
                row = x.iloc[i]
                direction = open_trade["direction"]
                sl = float(open_trade["sl"]); tp = float(open_trade["tp"])
                open_bar = int(open_trade["entry_bar"])
                bars_held = i - open_bar
                o = float(row.open); h = float(row.high); l = float(row.low)
                exit_price = None; status = None; reason = None
                if direction == "BUY":
                    if o <= sl:
                        exit_price, status, reason = o, "LOSS", "SL_GAP_AT_OPEN"
                    elif o >= tp:
                        exit_price, status, reason = o, "WIN", "TP_GAP_AT_OPEN"
                    elif l <= sl and h >= tp:
                        exit_price, status, reason = sl, "LOSS", "BOTH_HIT_SAME_CANDLE_CONSERVATIVE_SL"
                        ambiguous += 1
                    elif l <= sl:
                        exit_price, status, reason = sl, "LOSS", "SL"
                    elif h >= tp:
                        exit_price, status, reason = tp, "WIN", "TP"
                else:
                    if o >= sl:
                        exit_price, status, reason = o, "LOSS", "SL_GAP_AT_OPEN"
                    elif o <= tp:
                        exit_price, status, reason = o, "WIN", "TP_GAP_AT_OPEN"
                    elif h >= sl and l <= tp:
                        exit_price, status, reason = sl, "LOSS", "BOTH_HIT_SAME_CANDLE_CONSERVATIVE_SL"
                        ambiguous += 1
                    elif h >= sl:
                        exit_price, status, reason = sl, "LOSS", "SL"
                    elif l <= tp:
                        exit_price, status, reason = tp, "WIN", "TP"
                if exit_price is None and bars_held >= V14BacktestEngine.MAX_HOLDING_BARS:
                    exit_price, status = float(row.close), ("WIN" if direction == "BUY" and row.close > open_trade["entry_fill"] or direction == "SELL" and row.close < open_trade["entry_fill"] else "LOSS")
                    reason = "TIME_EXIT"
                if exit_price is not None:
                    rows.append(close_trade(open_trade, exit_price, row.time, status, reason, i))
                    open_trade = None

            if open_trade is not None:
                continue
            # Decision is made at bar close i; execution occurs at NEXT bar open.
            window = x.iloc[: i + 1]
            if ai_cache is None or i - ai_cache_bar >= V14BacktestEngine.AI_REFRESH_BARS:
                ai_cache = V14WalkForwardProbabilityEngine.predict(window, horizon=3, min_train=300, retrain_window=1500)
                ai_cache_bar = i
            cfg.current_equity = equity
            replay = V14ReplayDecisionEngine.evaluate(window, symbol, timeframe, cfg, equity, ai_override=ai_cache)
            if not replay.get("eligible"):
                reason = str((replay.get("final_decision") or {}).get("decision_explanation", replay.get("reason", "REJECTED")))
                rejection_counts[reason] = int(rejection_counts.get(reason, 0)) + 1
                continue
            next_bar = x.iloc[i + 1]
            direction = "BUY" if replay["final_decision"].get("final_decision") == "BUY" else "SELL"
            entry = replay["entry"]
            sl = float(entry["sl"]); tp = float(entry["tp"])
            raw_open = float(next_bar.open)
            pip = 0.01 if str(symbol).upper().endswith("JPY") else 0.0001
            slip = float(getattr(cfg, "slippage_pips", 0.0) or 0.0) * pip
            entry_fill = raw_open + slip if direction == "BUY" else raw_open - slip
            risk_distance = abs(entry_fill - sl)
            if risk_distance <= 1e-12 or not np.isfinite(risk_distance):
                continue
            open_trade = {
                "time": window.time.iloc[-1], "symbol": canonical_symbol(symbol),
                "timeframe": str(timeframe).upper(), "direction": direction,
                "decision": replay["final_decision"].get("final_decision"),
                "signal_score": replay["trade_quality"].get("score"),
                "ai_confidence": replay["ai"].get("confidence"),
                "confluence_score": replay["confluence"].get("score"),
                "mtf_alignment": replay["mtf"].get("alignment"),
                "entry_signal": float(entry.get("entry", x.close.iloc[i])),
                "entry_fill": float(entry_fill), "sl": sl, "tp": tp,
                "rr": float(entry.get("rr", 0.0)), "lot": float(entry.get("lot", 0.0)),
                "risk_distance": float(risk_distance), "entry_bar": int(i + 1),
                "status": "OPEN",
                "signal_timestamp": window.time.iloc[-1],
                "data_source": "HISTORICAL_REPLAY",
            }

        if open_trade is not None:
            open_trade = dict(open_trade)
            open_trade.update({"status": "OPEN_AT_TEST_END", "pnl": 0.0, "return_R": 0.0, "equity": equity, "drawdown_pct": (peak-equity)/max(peak,1e-9)*100.0})
            rows.append(open_trade)
            open_trade = None
        trades = pd.DataFrame(rows)
        metrics = _v14_test_metrics(trades, initial, equity, ambiguous, int((trades.status == "OPEN_AT_TEST_END").sum()) if not trades.empty else 0)
        metrics.update({
            "engine": "V14BacktestEngine",
            "symbol": canonical_symbol(symbol), "timeframe": str(timeframe).upper(),
            "rows_tested": int(len(x)), "warmup_bars": V14BacktestEngine.WARMUP,
            "no_future_data": True, "execution_model": "NEXT_CANDLE_OPEN_WITH_SLIPPAGE",
            "same_candle_ambiguity": "CONSERVATIVE_SL",
            "probability_engine": "WALK_FORWARD_RF_CALIBRATED",
            "probability_horizon_bars": 3,
            "minimum_accuracy_test_candles": V14BacktestEngine.MIN_CANDLES,
            "rejection_counts": dict(sorted(rejection_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        })
        metrics["statistical_validation"] = V14StatisticalValidationEngine.assess(metrics, cfg, oos=False)
        return trades, metrics


class V14ForwardTestingEngine:
    """Chronological train/OOS evaluation without resetting historical context."""
    @staticmethod
    def run(df: pd.DataFrame, cfg: Config, symbol="EURUSD", timeframe="M5", train_ratio=0.65):
        x = MarketDataEngine.normalize(df)
        if len(x) < V14BacktestEngine.MIN_CANDLES:
            error = f"V14 Forward Test requires at least {V14BacktestEngine.MIN_CANDLES} candles in the COMPLETE dataset."
            return {"train": {"error": error, "rows": int(len(x))}, "out_of_sample": {"error": error, "rows": int(len(x))}}
        cut = max(V14BacktestEngine.WARMUP + 1, int(len(x) * float(train_ratio)))
        cut = min(cut, len(x) - 1)
        # Use the full pre-OOS history as indicator context, but prohibit every
        # trade until the OOS boundary. This avoids cold-start distortion while
        # preserving chronological/no-future-data rules.
        train_df = x.iloc[:cut].copy()
        test_df = x.copy()
        bt_train, train_metrics = V14BacktestEngine.run(train_df, cfg, symbol, timeframe, enforce_minimum=False)
        # Custom OOS replay starts with frozen configuration and prior history.
        initial = float(cfg.initial_balance)
        equity = initial
        peak = equity
        rows = []
        open_trade = None
        ambiguous = 0
        ai_cache = None
        ai_cache_bar = -1
        rejection_counts = {}
        for i in range(cut, len(x) - 1):
            if open_trade is not None:
                row = x.iloc[i]; direction = open_trade["direction"]
                sl = float(open_trade["sl"]); tp = float(open_trade["tp"])
                bars_held = i - int(open_trade["entry_bar"])
                o,h,l = float(row.open),float(row.high),float(row.low)
                exit_price=status=reason=None
                if direction == "BUY":
                    if o <= sl: exit_price,status,reason=o,"LOSS","SL_GAP_AT_OPEN"
                    elif o >= tp: exit_price,status,reason=o,"WIN","TP_GAP_AT_OPEN"
                    elif l <= sl and h >= tp: exit_price,status,reason=sl,"LOSS","BOTH_HIT_SAME_CANDLE_CONSERVATIVE_SL"; ambiguous += 1
                    elif l <= sl: exit_price,status,reason=sl,"LOSS","SL"
                    elif h >= tp: exit_price,status,reason=tp,"WIN","TP"
                else:
                    if o >= sl: exit_price,status,reason=o,"LOSS","SL_GAP_AT_OPEN"
                    elif o <= tp: exit_price,status,reason=o,"WIN","TP_GAP_AT_OPEN"
                    elif h >= sl and l <= tp: exit_price,status,reason=sl,"LOSS","BOTH_HIT_SAME_CANDLE_CONSERVATIVE_SL"; ambiguous += 1
                    elif h >= sl: exit_price,status,reason=sl,"LOSS","SL"
                    elif l <= tp: exit_price,status,reason=tp,"WIN","TP"
                if exit_price is None and bars_held >= V14BacktestEngine.MAX_HOLDING_BARS:
                    exit_price=float(row.close); status="WIN" if (direction=="BUY" and exit_price>open_trade["entry_fill"]) or (direction=="SELL" and exit_price<open_trade["entry_fill"]) else "LOSS"; reason="TIME_EXIT"
                if exit_price is not None:
                    risk_distance=max(float(open_trade["risk_distance"]),1e-12)
                    r_mult=(exit_price-open_trade["entry_fill"])/risk_distance if direction=="BUY" else (open_trade["entry_fill"]-exit_price)/risk_distance
                    pnl=equity*cfg.risk_per_trade*float(r_mult)
                    commission=float(getattr(cfg,"commission_per_lot",0.0) or 0.0)*float(open_trade.get("lot",0.0) or 0.0)
                    pnl-=commission; equity+=pnl; peak=max(peak,equity)
                    open_trade.update({"exit_time":row.time,"exit_price":float(exit_price),"status":status,"exit_reason":reason,"return_R":float(r_mult),"pnl":float(pnl),"equity":float(equity),"drawdown_pct":(peak-equity)/max(peak,1e-9)*100.0})
                    rows.append(dict(open_trade)); open_trade=None
            if open_trade is not None: continue
            window=x.iloc[:i+1]
            if ai_cache is None or i - ai_cache_bar >= V14BacktestEngine.AI_REFRESH_BARS:
                ai_cache = V14WalkForwardProbabilityEngine.predict(window, horizon=3, min_train=300, retrain_window=1500)
                ai_cache_bar = i
            cfg.current_equity = equity
            replay=V14ReplayDecisionEngine.evaluate(window,symbol,timeframe,cfg,equity,ai_override=ai_cache)
            if not replay.get("eligible"):
                reason = str((replay.get("final_decision") or {}).get("decision_explanation", replay.get("reason", "REJECTED")))
                rejection_counts[reason] = int(rejection_counts.get(reason, 0)) + 1
                continue
            next_bar=x.iloc[i+1]; direction="BUY" if replay["final_decision"].get("final_decision")=="BUY" else "SELL"
            ent=replay["entry"]; sl=float(ent["sl"]); tp=float(ent["tp"])
            pip=0.01 if str(symbol).upper().endswith("JPY") else 0.0001; slip=float(getattr(cfg,"slippage_pips",0.0) or 0.0)*pip
            raw_open=float(next_bar.open); entry_fill=raw_open+slip if direction=="BUY" else raw_open-slip
            risk_distance=abs(entry_fill-sl)
            if risk_distance<=1e-12 or not np.isfinite(risk_distance): continue
            open_trade={
                "time":x.time.iloc[i],"symbol":canonical_symbol(symbol),"timeframe":str(timeframe).upper(),"direction":direction,
                "decision":replay["final_decision"].get("final_decision"),"signal_score":replay["trade_quality"].get("score"),
                "ai_confidence":replay["ai"].get("confidence"),"confluence_score":replay["confluence"].get("score"),"mtf_alignment":replay["mtf"].get("alignment"),
                "entry_signal":float(ent.get("entry",x.close.iloc[i])),"entry_fill":float(entry_fill),"sl":sl,"tp":tp,"rr":float(ent.get("rr",0.0)),"lot":float(ent.get("lot",0.0)),
                "risk_distance":float(risk_distance),"entry_bar":int(i+1),"status":"OPEN","signal_timestamp":x.time.iloc[i],"data_source":"HISTORICAL_FORWARD_OOS",
            }
        if open_trade is not None:
            open_trade.update({"status":"OPEN_AT_TEST_END","pnl":0.0,"return_R":0.0,"equity":equity,"drawdown_pct":(peak-equity)/max(peak,1e-9)*100.0})
            rows.append(dict(open_trade))
        oos_trades=pd.DataFrame(rows)
        oos_metrics=_v14_test_metrics(oos_trades,initial,equity,ambiguous,int((oos_trades.status=="OPEN_AT_TEST_END").sum()) if not oos_trades.empty else 0)
        oos_metrics.update({
            "engine":"V14ForwardTestingEngine","symbol":canonical_symbol(symbol),"timeframe":str(timeframe).upper(),
            "train_rows":int(len(train_df)),"oos_rows":int(len(x)-cut),"oos_start":str(x.time.iloc[cut]),"no_future_data":True,
            "parameters_frozen_after_split":True,"full_pre_split_context_used":True,"execution_model":"NEXT_CANDLE_OPEN_WITH_SLIPPAGE",
            "probability_engine":"WALK_FORWARD_RF_CALIBRATED","probability_horizon_bars":3,
            "rejection_counts": dict(sorted(rejection_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        })
        train_metrics["statistical_validation"] = V14StatisticalValidationEngine.assess(train_metrics, cfg, oos=False)
        oos_metrics["statistical_validation"] = V14StatisticalValidationEngine.assess(oos_metrics, cfg, oos=True)
        return {
            "train": train_metrics,
            "out_of_sample": oos_metrics,
            "train_trades": bt_train,
            "test_trades": oos_trades,
            "split_index": int(cut),
            "split_timestamp": str(x.time.iloc[cut]),
        }


class MonteCarloEngine:
    @staticmethod
    def run(trades: pd.DataFrame, simulations=500, seed=11):
        if trades.empty:
            return {"error": "Run a backtest first."}

        rng = np.random.default_rng(seed)
        vals = trades.pnl.to_numpy()
        finals = []
        maxdds = []

        for _ in range(simulations):
            p = rng.permutation(vals)
            eq = 10000 + np.cumsum(p)
            peak = np.maximum.accumulate(eq)
            dd = np.max((peak - eq) / peak * 100)
            finals.append(eq[-1])
            maxdds.append(dd)

        return {
            "simulations": simulations,
            "median_final": float(np.median(finals)),
            "p05_final": float(np.percentile(finals, 5)),
            "p95_final": float(np.percentile(finals, 95)),
            "median_max_dd": float(np.median(maxdds)),
            "p95_max_dd": float(np.percentile(maxdds, 95)),
        }


class V14StatisticalValidationEngine:
    """Determines whether a V14 result is statistically meaningful enough to judge.

    This is an evaluation layer only. It never changes a trade decision, threshold,
    model, or engine output. Small samples are explicitly labelled instead of being
    presented as proof of accuracy.
    """
    @staticmethod
    def assess(metrics: Dict[str, Any], cfg: Config, oos: bool = False) -> Dict[str, Any]:
        m = metrics or {}
        trades = int(m.get("trades", 0) or 0)
        required = int(getattr(cfg, "minimum_statistically_valid_oos_trades" if oos else "minimum_statistically_valid_trades", 20 if oos else 30))
        warnings = []
        if trades < required:
            warnings.append(f"INSUFFICIENT TRADE SAMPLE: {trades} < {required}")
        pf = float(m.get("profit_factor", 0) or 0)
        if trades == 0:
            warnings.append("NO CLOSED TRADES")
        if not np.isfinite(pf):
            if trades and float(m.get("gross_loss", 0) or 0) == 0:
                warnings.append("NO LOSING TRADES IN SAMPLE; PROFIT FACTOR IS NOT STABLE")
        return {
            "status": "STATISTICALLY_VALID" if not warnings else "INSUFFICIENT_EVIDENCE",
            "closed_trades": trades,
            "required_trades": required,
            "warnings": warnings,
            "not_a_profitability_guarantee": True,
        }


class V14RobustnessValidationEngine:
    """Additive parameter-sensitivity test; reports stability without selecting a winner.

    Threshold/slippage perturbations are evaluated independently. Results are diagnostic
    only and must not be used to tune against the final untouched OOS set.
    """
    @staticmethod
    def run(df: pd.DataFrame, cfg: Config, symbol="EURUSD", timeframe="M5") -> pd.DataFrame:
        base = float(getattr(cfg, "min_score", 72.0))
        thresholds = sorted(set([max(50.0, base-5.0), base, min(95.0, base+5.0)]))
        base_slip = float(getattr(cfg, "slippage_pips", 0.3) or 0.3)
        slippages = sorted(set([max(0.0, base_slip), max(0.0, base_slip*1.5), max(0.0, base_slip*2.0)]))
        rows=[]
        for th in thresholds:
            for slip in slippages:
                test_cfg = replace(cfg, min_score=float(th), slippage_pips=float(slip))
                _, m = V14BacktestEngine.run(df, test_cfg, symbol, timeframe)
                rows.append({
                    "threshold": float(th), "slippage_pips": float(slip),
                    "trades": int(m.get("trades",0) or 0),
                    "win_rate": float(m.get("win_rate",0) or 0),
                    "profit_factor": float(m.get("profit_factor",0) or 0),
                    "net_profit": float(m.get("net_profit",0) or 0),
                    "max_drawdown": float(m.get("max_drawdown",0) or 0),
                })
        return pd.DataFrame(rows)


class OptimizerEngine:
    @staticmethod
    def run(df, cfg, thresholds=(65, 70, 75, 80, 85)):
        """Optimize the existing V14 score threshold without restoring the legacy tester."""
        rows = []
        for th in thresholds:
            test_cfg = replace(cfg, min_score=float(th))
            _, m = V14BacktestEngine.run(df, test_cfg)
            if m.get("trades", 0):
                rows.append({
                    "threshold": th,
                    "net_profit": m["net_profit"],
                    "win_rate": m["win_rate"],
                    "profit_factor": m["profit_factor"],
                    "max_drawdown": m["max_drawdown"],
                    "trades": m["trades"],
                })
        return (
            pd.DataFrame(rows).sort_values(
                ["profit_factor", "net_profit"], ascending=False
            )
            if rows else pd.DataFrame()
        )


# ============================================================
# APP STATE / HELPERS
# ============================================================


# ============================================================
# V14 LIVE DATA / AI / QUALITY / TIMING
# V14 architecture services and evidence engines.
# ============================================================

class DataIntegrityEngine:
    """Live-feed quality, freshness and OHLC integrity monitor."""

    @staticmethod
    def assess(df: pd.DataFrame, timeframe="M5", expected_age_seconds=420) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"score": 0.0, "status": "BAD", "reasons": ["NO DATA"], "stale": True}
        x = MarketDataEngine.normalize(df)
        reasons=[]
        score=100.0
        if len(x) < 100:
            score -= 30; reasons.append("INSUFFICIENT HISTORY")
        bad_ohlc = ((x.high < x.low) | (x.high < x.open) | (x.high < x.close) |
                    (x.low > x.open) | (x.low > x.close)).sum()
        if bad_ohlc:
            score -= min(30, bad_ohlc*5); reasons.append("INVALID OHLC")
        dup = int(x.time.duplicated().sum())
        if dup:
            score -= 10; reasons.append("DUPLICATE TIMESTAMPS")
        diffs = x.time.diff().dt.total_seconds().dropna()
        if len(diffs):
            expected = {"M1":60,"M3":180,"M5":300,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}.get(timeframe,300)
            # Daily FX data normally has a Friday-to-Monday weekend gap. Do not
            # misclassify that normal market closure as missing candles.
            gap_multiplier = 4.5 if str(timeframe).upper() == "D1" else 1.8
            missing = int((diffs > expected*gap_multiplier).sum())
            if missing:
                score -= min(20, missing*2); reasons.append(f"MISSING/GAPPED CANDLES: {missing}")
        last = x.time.iloc[-1]
        age = max(0.0, (pd.Timestamp.now(tz="UTC") - last).total_seconds())
        stale = age > expected_age_seconds
        if stale:
            score -= 30; reasons.append(f"STALE DATA ({age:.0f}s)")
        # frozen price: last 8 closes identical
        if len(x) >= 8 and x.close.tail(8).nunique() == 1:
            score -= 20; reasons.append("FROZEN PRICE")
        score=max(0.0,min(100.0,score))
        status="EXCELLENT" if score>=90 else "GOOD" if score>=75 else "DEGRADED" if score>=55 else "BAD"
        return {"score":score,"status":status,"reasons":reasons,"stale":stale,"age_seconds":age,
                "last_timestamp":str(last),"rows":len(x),"signal_allowed":score>=55 and not stale}





class TwelveDataRateLimitError(RuntimeError):
    """Raised when Twelve Data rejects a request because the API quota is exhausted."""

    def __init__(self, message, retry_after=60, credits_left=None):
        super().__init__(message)
        self.retry_after = int(max(1, retry_after or 60))
        self.credits_left = credits_left


class TwelveDataLiveEngine:
    """Twelve Data REST adapter. Read-only; no order endpoints are used.

    Important reliability rules:
      * One canonical provider symbol is used; the provider guard never retries the same request
        as both GBP/USD and GBPUSD.
      * The API key is sent in the Authorization header, not in the URL.
      * 429 responses are surfaced as a controlled data-quality/rate-limit state
        rather than immediately issuing more requests and making the quota worse.
    """
    BASE_URL = "https://api.twelvedata.com/time_series"
    TF_MAP = {"M1":"1min","M3":"3min","M5":"5min","M15":"15min","M30":"30min","H1":"1h","H4":"4h","D1":"1day"}

    @staticmethod
    def fetch(api_key, symbol, timeframe="M5", outputsize=500, timeout=15, end_date=None, start_date=None):
        if not api_key:
            raise ValueError("Twelve Data API key is not configured.")
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("The requests package is required for Twelve Data REST mode.") from e

        # Twelve Data accepts slash notation for forex. Use exactly one form so
        # a failed request never burns a second credit on a symbol alias.
        provider_symbol = display_symbol(canonical_symbol(symbol))
        params = {
            "symbol": provider_symbol,
            "interval": TwelveDataLiveEngine.TF_MAP.get(str(timeframe).upper(), str(timeframe)),
            "outputsize": int(outputsize),
            "format": "JSON",
        }
        # Optional bounded historical pagination. The caller controls the time
        # boundary; no live/future timestamp is fabricated.
        if end_date is not None:
            params["end_date"] = str(end_date)
        if start_date is not None:
            params["start_date"] = str(start_date)
        headers = {"Authorization": f"apikey {api_key}"}

        r = requests.get(
            TwelveDataLiveEngine.BASE_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After") or 60
            try:
                retry_after = int(float(retry_after))
            except Exception:
                retry_after = 60
            credits_left = r.headers.get("api-credits-left")
            raise TwelveDataRateLimitError(
                f"Twelve Data rate limit reached for {provider_symbol}/{str(timeframe).upper()}. "
                f"No duplicate retry was attempted. Wait about {retry_after}s before the next request.",
                retry_after=retry_after,
                credits_left=credits_left,
            )

        r.raise_for_status()
        payload = r.json()
        if str(payload.get("status", "")).lower() == "error":
            code = payload.get("code")
            message = payload.get("message", "Twelve Data returned an error.")
            if str(code) == "429":
                raise TwelveDataRateLimitError(str(message))
            raise RuntimeError(str(message))

        values = payload.get("values", [])
        if not values:
            raise RuntimeError(f"Twelve Data returned no candles for {provider_symbol}/{str(timeframe).upper()}.")

        rows = []
        for z in values:
            rows.append({
                "time": z.get("datetime"),
                "open": z.get("open"),
                "high": z.get("high"),
                "low": z.get("low"),
                "close": z.get("close"),
                "volume": z.get("volume", 0),
            })
        df = MarketDataEngine.normalize(pd.DataFrame(rows)).sort_values("time").reset_index(drop=True)
        return df, {
            "source": "TWELVE DATA",
            "rows": len(df),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "credits_used": r.headers.get("api-credits-used"),
            "credits_left": r.headers.get("api-credits-left"),
            "provider_symbol": provider_symbol,
        }


class LiveConnectionManager:
    """Source selector and safe read-only connection state. No execution endpoints."""
    @staticmethod
    def status(source, df, validation, data_quality, metadata=None):
        metadata=metadata or {}
        return {"source":source,"connected":bool(df is not None and not df.empty),
                "quality":data_quality.get("score",0),"status":data_quality.get("status","BAD"),
                "last_fetch":metadata.get("fetched_at", data_quality.get("last_timestamp","-")),
                "credits_left":metadata.get("credits_left"),"read_only":True,
                "execution_enabled":False,"validation":validation}


class MomentumDirectionEngine:
    """Advanced momentum direction, acceleration and persistence layer."""
    @staticmethod
    def analyze(df):
        x=MarketDataEngine.normalize(df)
        close=x.close
        if len(x)<60:
            return {"direction":"NEUTRAL","score":50.0,"strength":"LOW","acceleration":0.0,"persistence":0.0,"divergence":"UNKNOWN"}
        r=rsi(close,14).iloc[-1]
        macd_line, macd_signal, macd_hist = macd(close)
        hist=float(macd_hist.iloc[-1])
        hist_prev=float(macd_hist.iloc[-2])
        adx_series, plus_series, minus_series = adx(x)
        adxv=float(adx_series.iloc[-1]); plus=float(plus_series.iloc[-1]); minus=float(minus_series.iloc[-1])
        e20=ema(close,20).iloc[-1]; e50=ema(close,50).iloc[-1]
        roc12=float(roc(close,12).iloc[-1])
        accel=float(hist-hist_prev)
        signs=np.sign(close.diff().tail(12).dropna())
        persistence=float((signs>0).mean()*100) if len(signs) else 50
        bull=0; bear=0
        bull += 1 if r>50 else 0; bear += 1 if r<50 else 0
        bull += 1 if hist>0 else 0; bear += 1 if hist<0 else 0
        bull += 1 if hist>=hist_prev else 0; bear += 1 if hist<hist_prev else 0
        bull += 1 if plus>minus else 0; bear += 1 if minus>plus else 0
        bull += 1 if close.iloc[-1]>e20>e50 else 0; bear += 1 if close.iloc[-1]<e20<e50 else 0
        bull += 1 if roc12>0 else 0; bear += 1 if roc12<0 else 0
        if bull-bear>=2: direction="BULLISH"
        elif bear-bull>=2: direction="BEARISH"
        else: direction="NEUTRAL"
        score=50 + (bull-bear)*8 + min(12,max(-12,adxv-20)*0.5) + (persistence-50)*0.12
        score=float(max(0,min(100,score)))
        strength="HIGH" if score>=78 or score<=22 else "MEDIUM" if score>=65 or score<=35 else "LOW"
        return {"direction":direction,"score":score,"strength":strength,"acceleration":accel,
                "persistence":persistence,"rsi":float(r),"adx":adxv,"plus_di":plus,"minus_di":minus,
                "roc":roc12,"macd_hist":float(hist),"divergence":"NOT CONFIRMED"}


class AIProbabilityEngine:
    """Optional ML layer. Uses sklearn when installed; otherwise a transparent statistical fallback."""
    @staticmethod
    def _features(df):
        x=MarketDataEngine.normalize(df).copy()
        x["ret1"]=x.close.pct_change()
        x["ret3"]=x.close.pct_change(3)
        x["ret10"]=x.close.pct_change(10)
        x["ema_gap"]=ema(x.close,20)/ema(x.close,50)-1
        x["rsi"]=rsi(x.close,14)/100
        x["atr_pct"]=atr(x,14)/x.close
        x["vol_z"]=(x.volume-x.volume.rolling(30).mean())/(x.volume.rolling(30).std()+1e-9)
        # The optional spread field is not available on every feed (e.g. many OHLC-only
        # Twelve Data responses). Keep only model features required by this engine so
        # an absent spread does not discard the entire training set.
        feature_cols=["ret1","ret3","ret10","ema_gap","rsi","atr_pct","vol_z"]
        return x.replace([np.inf,-np.inf],np.nan).dropna(subset=feature_cols)

    @staticmethod
    def predict(df, horizon=1):
        x=AIProbabilityEngine._features(df)
        if len(x)<160:
            return {"up_probability":50.0,"down_probability":50.0,"confidence":0.0,"model":"INSUFFICIENT DATA","trade_quality":"C"}
        cols=["ret1","ret3","ret10","ema_gap","rsi","atr_pct","vol_z"]
        y=(x.close.shift(-horizon)>x.close).astype(int)
        train=x.iloc[:-horizon].copy(); target=y.iloc[:-horizon]
        latest=x[cols].iloc[[-1]]
        try:
            from sklearn.ensemble import RandomForestClassifier
            model=RandomForestClassifier(n_estimators=120,max_depth=5,min_samples_leaf=6,random_state=42,class_weight="balanced_subsample")
            model.fit(train[cols],target)
            prob=float(model.predict_proba(latest)[0][1])
            model_name="RANDOM FOREST"
        except Exception:
            # deterministic fallback when sklearn is unavailable
            z=0.0
            z += float(train.ret1.tail(20).mean())/max(float(train.ret1.tail(60).std()),1e-8)*0.8
            z += float(train.ret3.tail(20).mean())/max(float(train.ret3.tail(60).std()),1e-8)*0.6
            z += float(latest.ema_gap.iloc[0])*100*0.8
            z += (float(latest.rsi.iloc[0])-0.5)*1.2
            prob=float(1/(1+math.exp(-max(-8,min(8,z)))))
            model_name="STATISTICAL FALLBACK"
        conf=abs(prob-0.5)*200
        quality="A" if conf>=65 else "B" if conf>=45 else "C"
        return {"up_probability":prob*100,"down_probability":(1-prob)*100,"confidence":conf,
                "model":model_name,"trade_quality":quality,"training_rows":len(train)}


class V14WalkForwardProbabilityEngine:
    """Leakage-safe adaptive probability model used by V14 historical/OOS replay.

    The model is refit only on observations that were already fully observable at
    the prediction timestamp.  A chronological calibration tail is kept inside
    the available history; the current bar is never used as a training label.
    This is predictive evidence only and cannot create a final trade decision.
    """
    FEATURE_COLS = ["ret1", "ret3", "ret10", "ema_gap", "rsi", "atr_pct", "vol_z"]

    @staticmethod
    def _dataset(df: pd.DataFrame, horizon: int = 3):
        x = AIProbabilityEngine._features(df).copy()
        if x.empty:
            return x, pd.Series(dtype=int)
        future = x.close.shift(-int(horizon))
        valid = future.notna()
        y = (future > x.close).astype(int)
        return x.loc[valid].copy(), y.loc[valid].copy()

    @staticmethod
    def predict(df: pd.DataFrame, horizon: int = 3, min_train: int = 300,
                retrain_window: int = 1500) -> Dict[str, Any]:
        raw = MarketDataEngine.normalize(df)
        latest_features = AIProbabilityEngine._features(raw)
        if latest_features.empty:
            return {"up_probability": 50.0, "down_probability": 50.0,
                    "confidence": 0.0, "model": "WALK-FORWARD INSUFFICIENT",
                    "trade_quality": "C", "training_rows": 0,
                    "calibrated": False, "status": "INSUFFICIENT"}
        latest = latest_features.iloc[[-1]][V14WalkForwardProbabilityEngine.FEATURE_COLS]
        train_x, y = V14WalkForwardProbabilityEngine._dataset(raw, horizon)
        # Never allow the current prediction row to become a training example.
        if len(train_x) > retrain_window:
            train_x = train_x.iloc[-retrain_window:]
            y = y.loc[train_x.index]
        if len(train_x) < min_train or y.nunique() < 2:
            return {"up_probability": 50.0, "down_probability": 50.0,
                    "confidence": 0.0, "model": "WALK-FORWARD INSUFFICIENT",
                    "trade_quality": "C", "training_rows": int(len(train_x)),
                    "calibrated": False, "status": "INSUFFICIENT"}
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            split = int(len(train_x) * 0.80)
            split = max(200, min(split, len(train_x) - 50))
            fit_x, cal_x = train_x.iloc[:split], train_x.iloc[split:]
            fit_y, cal_y = y.loc[fit_x.index], y.loc[cal_x.index]
            model = RandomForestClassifier(
                n_estimators=180, max_depth=6, min_samples_leaf=8,
                random_state=42, class_weight="balanced_subsample",
                n_jobs=-1,
            )
            model.fit(fit_x[V14WalkForwardProbabilityEngine.FEATURE_COLS], fit_y)
            raw_cal = model.predict_proba(cal_x[V14WalkForwardProbabilityEngine.FEATURE_COLS])[:, 1]
            # Platt-style calibration trained only on the chronological calibration tail.
            calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            calibrator.fit(raw_cal.reshape(-1, 1), cal_y)
            raw_latest = float(model.predict_proba(latest)[0, 1])
            prob = float(calibrator.predict_proba(np.array([[raw_latest]]))[0, 1])
            model_name = "WALK-FORWARD RF + CALIBRATION"
            calibrated = True
        except Exception as exc:
            # Transparent fallback: deterministic, still chronological and leakage-safe.
            recent = train_x.tail(80)
            z = 0.0
            z += float(recent.ret1.mean()) / max(float(recent.ret1.std()), 1e-8) * 0.25
            z += float(recent.ret3.mean()) / max(float(recent.ret3.std()), 1e-8) * 0.35
            z += float(latest.ema_gap.iloc[0]) * 80.0
            z += (float(latest.rsi.iloc[0]) - 0.5) * 1.0
            prob = float(1.0 / (1.0 + math.exp(-np.clip(z, -8, 8))))
            model_name = "WALK-FORWARD STATISTICAL FALLBACK"
            calibrated = False
            exc = str(exc)
        conf = float(abs(prob - 0.5) * 200.0)
        quality = "A" if conf >= 65 else "B" if conf >= 45 else "C"
        return {
            "up_probability": prob * 100.0,
            "down_probability": (1.0 - prob) * 100.0,
            "confidence": conf,
            "model": model_name,
            "trade_quality": quality,
            "training_rows": int(len(train_x)),
            "calibration_rows": int(max(0, len(train_x) - int(len(train_x) * 0.80))),
            "calibrated": calibrated,
            "horizon_bars": int(horizon),
            "status": "OK",
        }


class VolumeFlowEngine:
    """Volume/flow direction using available OHLCV data; no synthetic volume is invented."""
    @staticmethod
    def analyze(df):
        x = MarketDataEngine.normalize(df)
        if x.empty or len(x) < 30:
            return {"direction": "NO DATA", "score": 0.0, "flow": 0.0, "status": "INSUFFICIENT"}
        vol = pd.to_numeric(x.get("volume", pd.Series(0.0, index=x.index)), errors="coerce").fillna(0.0)
        if float(vol.abs().sum()) <= 0:
            return {"direction": "NEUTRAL", "score": 50.0, "flow": 0.0, "status": "NO VOLUME"}
        signed = np.sign(x.close.diff().fillna(0.0)) * vol
        flow = float(signed.tail(20).sum() / max(vol.tail(20).sum(), 1e-9))
        score = float(np.clip(50.0 + flow * 50.0, 0, 100))
        direction = "BULLISH" if score >= 58 else "BEARISH" if score <= 42 else "NEUTRAL"
        return {"direction": direction, "score": score, "flow": flow, "status": "OK"}


class CandleTimingEngine:
    """Current-candle state: age/progress plus directional body confirmation."""
    @staticmethod
    def analyze(df, timeframe="M5"):
        x = MarketDataEngine.normalize(df)
        if x.empty:
            return {"direction": "NO DATA", "score": 0.0, "progress_pct": 0.0, "status": "NO DATA"}
        tf_seconds = {"M1":60,"M3":180,"M5":300,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}.get(str(timeframe).upper(),300)
        now = pd.Timestamp.now(tz="UTC")
        last_ts = x.time.iloc[-1]
        age = max(0.0, (now - last_ts).total_seconds())
        progress = float(np.clip((age / tf_seconds) * 100.0, 0, 100)) if age <= tf_seconds else 100.0
        row = x.iloc[-1]
        rng = max(float(row.high - row.low), 1e-12)
        body = float(row.close - row.open) / rng
        direction = "BULLISH" if body > 0.10 else "BEARISH" if body < -0.10 else "NEUTRAL"
        # A forming candle is informational; it never becomes a hard veto merely because it is unfinished.
        score = float(np.clip(50 + body * 50, 0, 100))
        return {"direction": direction, "score": score, "progress_pct": progress,
                "age_seconds": age, "timeframe_seconds": tf_seconds,
                "forming": age < tf_seconds, "status": "OK"}


class MarketRegionEngine:
    """Maps current price to recent range regions: lower/middle/upper."""
    @staticmethod
    def analyze(df, lookback=100):
        x = MarketDataEngine.normalize(df).tail(lookback)
        if len(x) < 20:
            return {"region": "UNKNOWN", "direction": "NO DATA", "score": 0.0, "status": "INSUFFICIENT"}
        lo = float(x.low.min()); hi = float(x.high.max()); last = float(x.close.iloc[-1])
        pos = (last - lo) / max(hi - lo, 1e-12)
        if pos <= 0.33:
            region, direction = "LOWER", "BULLISH"
        elif pos >= 0.67:
            region, direction = "UPPER", "BEARISH"
        else:
            region, direction = "MIDDLE", "NEUTRAL"
        # Region is contextual, not a standalone trigger.
        score = float(np.clip(50 + (0.5 - abs(pos - 0.5)) * 40, 0, 100))
        return {"region": region, "direction": direction, "score": score, "position_pct": pos * 100, "status": "OK"}


class DirectProbabilityEngine:
    """Transparent non-ML probability estimate from independent price features."""
    @staticmethod
    def predict(df):
        x = MarketDataEngine.normalize(df)
        if len(x) < 60:
            return {"up_probability":50.0,"down_probability":50.0,"confidence":0.0,"direction":"NEUTRAL","status":"INSUFFICIENT"}
        r = float(rsi(x.close,14).iloc[-1])
        e20 = float(ema(x.close,20).iloc[-1]); e50 = float(ema(x.close,50).iloc[-1])
        ret = float(x.close.pct_change(10).iloc[-1])
        z = ((r - 50) / 20) + ((e20 / max(e50,1e-12) - 1) * 120) + ret * 100
        prob = float(1 / (1 + math.exp(-np.clip(z, -8, 8))))
        conf = float(abs(prob - 0.5) * 200)
        direction = "BULLISH" if prob >= 0.55 else "BEARISH" if prob <= 0.45 else "NEUTRAL"
        return {"up_probability":prob*100,"down_probability":(1-prob)*100,"confidence":conf,"direction":direction,"status":"OK"}


class SignalConfidenceEngine:
    """Final signal-confidence layer built from independent engine agreement.

    AI probability is advisory rather than a single hard gate. A signal still needs
    strong multi-engine agreement, valid data, and risk/news/session safety.
    """
    @staticmethod
    def calculate(a, ai, data_quality, direction):
        if direction not in ("BULLISH", "BEARISH"):
            return {"confidence": 0.0, "agreement": 0.0, "ai_alignment": 50.0, "components": {}}

        def aligned(value, neutral=50.0):
            v = str(value or "").upper()
            return 100.0 if v == direction else 0.0 if v in ("BULLISH", "BEARISH") else neutral

        votes = [
            a.get("trend", {}).get("direction"),
            a.get("momentum", {}).get("direction"),
            a.get("structure", {}).get("direction"),
            a.get("price_action", {}).get("direction"),
            a.get("mtf", {}).get("direction"),
            a.get("advanced_momentum", {}).get("direction"),
        ]
        valid_votes = [v for v in votes if v in ("BULLISH", "BEARISH")]
        matching = sum(v == direction for v in valid_votes)
        agreement = 100.0 * matching / max(len(valid_votes), 1)

        trend_score = float(a.get("trend", {}).get("strength", 50.0))
        momentum_score = float(a.get("momentum", {}).get("score", 50.0))
        if direction == "BEARISH":
            momentum_score = 100.0 - momentum_score
        adv_score = float(a.get("advanced_momentum", {}).get("score", 50.0))
        if direction == "BEARISH":
            adv_score = 100.0 - adv_score

        mtf_dir = a.get("mtf", {}).get("direction", "MIXED")
        mtf_align = float(a.get("mtf", {}).get("alignment", 0.0))
        mtf_score = mtf_align if mtf_dir == direction else (100.0 - mtf_align if mtf_dir in ("BULLISH", "BEARISH") else 50.0)

        ai_dir = "BULLISH" if ai.get("up_probability", 50.0) > 55 else "BEARISH" if ai.get("down_probability", 50.0) > 55 else "NEUTRAL"
        ai_conf = float(ai.get("confidence", 0.0))
        if ai_dir == direction:
            ai_alignment = 50.0 + ai_conf * 0.50
        elif ai_dir in ("BULLISH", "BEARISH"):
            ai_alignment = 50.0 - ai_conf * 0.50
        else:
            ai_alignment = 50.0

        pa_score = aligned(a.get("price_action", {}).get("direction"))
        structure_score = aligned(a.get("structure", {}).get("direction"))
        data_score = float(data_quality.get("score", 0.0))

        confidence = (
            agreement * 0.25
            + trend_score * 0.15
            + momentum_score * 0.12
            + adv_score * 0.12
            + mtf_score * 0.12
            + ai_alignment * 0.10
            + data_score * 0.08
            + ((pa_score + structure_score) / 2.0) * 0.06
        )
        confidence = float(np.clip(confidence, 0.0, 100.0))
        return {
            "confidence": confidence,
            "agreement": float(agreement),
            "ai_alignment": float(ai_alignment),
            "components": {
                "engine_agreement": float(agreement),
                "trend": trend_score,
                "momentum": momentum_score,
                "advanced_momentum": adv_score,
                "mtf": mtf_score,
                "ai_alignment": ai_alignment,
                "data_quality": data_score,
                "price_action_structure": (pa_score + structure_score) / 2.0,
            },
        }


class EnsembleDecisionEngine:
    @staticmethod
    def decide(a, ai, data_quality, cfg):
        score = float(a["confluence"]["score"])
        direction = a["confluence"]["direction"]
        ai_dir = "BULLISH" if ai["up_probability"] > 55 else "BEARISH" if ai["down_probability"] > 55 else "NEUTRAL"
        momentum = a.get("advanced_momentum", {}).get("direction", "NEUTRAL")

        # Determine a directional consensus before confidence is calculated.
        vote_set = [
            a.get("trend", {}).get("direction"),
            a.get("momentum", {}).get("direction"),
            a.get("structure", {}).get("direction"),
            a.get("price_action", {}).get("direction"),
            a.get("mtf", {}).get("direction"),
            momentum,
            ai_dir,
        ]
        bull = sum(v == "BULLISH" for v in vote_set)
        bear = sum(v == "BEARISH" for v in vote_set)
        if direction not in ("BULLISH", "BEARISH"):
            direction = "BULLISH" if bull > bear else "BEARISH" if bear > bull else "WAIT"

        confidence = SignalConfidenceEngine.calculate(a, ai, data_quality, direction)
        final = confidence["confidence"]
        reasons = []

        if not data_quality.get("signal_allowed", False):
            reasons.append("DATA QUALITY / STALE FEED")
        if direction == "WAIT":
            reasons.append("NO CLEAR CONFLUENCE DIRECTION")
        if confidence["agreement"] < cfg.min_engine_agreement:
            reasons.append("ENGINE DIRECTION CONFLICT")
        if score < cfg.min_score:
            reasons.append("CONFLUENCE BELOW THRESHOLD")

        # AI is a soft advisory gate: it can reduce confidence, but does not veto a
        # signal when the independent technical stack strongly agrees.
        if ai["confidence"] < cfg.ai_soft_floor and confidence["agreement"] < 80:
            reasons.append("AI MODEL TOO UNCERTAIN")

        allowed = (
            direction in ("BULLISH", "BEARISH")
            and final >= cfg.min_signal_confidence
            and not reasons
        )
        return {
            "score": final,
            "confidence": final,
            "direction": direction,
            "ai_direction": ai_dir,
            "agreement": confidence["agreement"],
            "ai_alignment": confidence["ai_alignment"],
            "approved": allowed,
            "reasons": list(dict.fromkeys(reasons)),
            "components": confidence["components"],
            "confluence_score": score,
            "grade": "A" if final >= 85 else "B" if final >= 75 else "C" if final >= 65 else "D",
        }


class NoTradeEngine:
    @staticmethod
    def evaluate(a, ensemble, data_quality, cfg, signal_age=0):
        reasons = list(ensemble.get("reasons", []))
        if data_quality.get("score", 0) < 55:
            reasons.append("DATA QUALITY BELOW 55")
        if signal_age > cfg.signal_max_age_seconds:
            reasons.append("SIGNAL STALE")
        if a["volatility"].get("regime") == "EXTREME":
            reasons.append("EXTREME VOLATILITY")
        if a["economic"].get("blocked"):
            reasons.append("HIGH-IMPACT NEWS BLACKOUT")
        if a["session"].get("session_tradeable") is False:
            reasons.append("SESSION NOT TRADEABLE")
        if not a.get("risk", {}).get("approved", False):
            reasons.append("RISK VETO")
        return {
            "trade_allowed": len(reasons) == 0,
            "reasons": list(dict.fromkeys(reasons)),
            "status": "TRADE" if not reasons else "NO TRADE",
        }


class SignalTimingEngine:
    @staticmethod
    def assess(signal_time=None, max_age_seconds=60):
        if signal_time is None: return {"fresh":True,"age_seconds":0.0,"status":"FRESH"}
        t=pd.Timestamp(signal_time)
        if t.tzinfo is None: t=t.tz_localize("UTC")
        age=max(0,(pd.Timestamp.now(tz="UTC")-t).total_seconds())
        return {"fresh":age<=max_age_seconds,"age_seconds":age,"status":"FRESH" if age<=max_age_seconds else "EXPIRED"}


class EngineInformationBus:
    """Central normalized evidence layer used by Trade Quality and diagnostics.

    Every engine publishes the same contract: direction, strength/confidence,
    health, role and whether it may contribute a directional vote. Missing or
    invalid outputs are marked unavailable rather than silently treated as neutral.
    """
    SPECS = [
        ("Trend", "trend", "directional", 1.00),
        ("Momentum", "momentum", "directional", 1.00),
        ("EMA", "ema", "directional", 0.90),
        ("MACD", "macd", "directional", 0.90),
        ("Current Volatility", "volatility", "context", 0.75),
        ("Market Structure", "structure", "directional", 1.00),
        ("Price Action", "price_action", "directional", 1.00),
        ("Support/Resistance", "sr", "context", 0.80),
        ("Breakout", "breakout", "directional", 0.90),
        ("Fair Value Gap", "fvg", "context", 0.80),
        ("Market Microstructure", "microstructure", "context", 0.70),
        ("Order Flow / Volume Flow", "order_flow", "directional", 0.90),
        ("Reversal Detection", "reversal", "context", 0.70),
        ("Liquidity", "liquidity", "context", 0.80),
        ("Regime", "regime", "context", 0.80),
        ("Session", "session", "timing", 0.70),
        ("Currency Strength", "currency_strength", "directional", 0.90),
        ("Correlation", "correlation", "risk", 0.60),
        ("Multi-Timeframe Alignment", "mtf", "directional", 1.20),
        ("News/Economic", "economic", "risk", 0.80),
        ("COT/Positioning", "cot", "context", 0.40),
        ("Advanced Momentum", "advanced_momentum", "directional", 1.00),
        ("Volume Flow", "volume_flow", "directional", 0.90),
        ("Candle Timing", "candle_timing", "timing", 0.60),
        ("Market Region", "market_region", "context", 0.60),
        ("Direct Probability", "direct_probability", "directional", 0.90),
        ("AI/ML Prediction", "ai", "directional", 1.00),
        ("Confluence", "confluence", "summary", 1.00),
        ("Ensemble Decision", "ensemble", "summary", 1.10),
        ("Risk Control", "risk", "risk", 1.00),
        ("Risk Veto", "risk_veto", "risk", 1.20),
        ("Signal Timing", "signal_timing", "timing", 0.70),
        ("Data Integrity", "data_quality", "data", 1.30),
        ("Market Tracker", "market_tracker", "context", 0.50),
    ]

    @staticmethod
    def _direction(obj):
        if not isinstance(obj, dict):
            return "NEUTRAL"
        d = str(obj.get("direction", obj.get("bias", "NEUTRAL"))).upper()
        return d if d in {"BULLISH","BEARISH"} else "NEUTRAL"

    @staticmethod
    def _strength(obj):
        if not isinstance(obj, dict):
            return 0.0
        for key in ("confidence","score","strength","alignment","quality"):
            value = obj.get(key)
            if isinstance(value, (int,float,np.integer,np.floating)):
                return float(np.clip(value,0,100))
        return 0.0

    @classmethod
    def build(cls, a, data_quality):
        out = {}
        for label, key, role, weight in cls.SPECS:
            # Data Integrity is computed outside the advisory bundle and must be
            # injected explicitly.  The previous integration omitted it from
            # `advisory`, leaving bus["Data Integrity"]["raw"] as None and causing
            # the dashboard error: "NoneType object has no attribute get".
            obj = data_quality if key == "data_quality" else a.get(key)
            if key == "fvg":
                li = a.get("liquidity", {}) or {}
                obj = {"direction": a.get("liquidity",{}).get("direction","NEUTRAL"),
                       "score": 70.0 if li.get("FVG") else 30.0,
                       "present": bool(li.get("FVG")), "status":"OK"}
            if key == "risk_veto":
                rv = a.get("risk_veto") or a.get("risk", {}) or {}
                obj = {"direction":"NEUTRAL","score":100.0 if not rv.get("reasons") else 0.0,
                       "status":"PASS" if not rv.get("reasons") else "VETO"}
            if key == "signal_timing":
                obj = a.get("signal_timing") or {"direction":"NEUTRAL","score":100.0,"status":"FRESH"}
            if key == "market_tracker":
                obj = a.get("market_tracker") or {"direction":a.get("confluence",{}).get("direction","NEUTRAL"),"score":a.get("confluence",{}).get("score",0),"status":"CURRENT PAIR"}

            if isinstance(obj, str):
                raw_text = obj.upper()
                obj = {"direction": raw_text if raw_text in {"BULLISH","BEARISH"} else "NEUTRAL", "status": raw_text}
            direction = cls._direction(obj)
            strength = cls._strength(obj)
            if key == "data_quality":
                healthy = bool(obj) and not bool(obj.get("stale",False)) and bool(obj.get("signal_allowed",False)) and float(obj.get("score",0)) >= 70
            elif key == "session":
                healthy = bool(obj) and bool(obj.get("session_tradeable",False))
            elif key == "risk":
                healthy = bool(obj) and bool(obj.get("approved",False))
            elif key == "risk_veto":
                healthy = bool(obj) and not bool(obj.get("reasons"))
            elif key == "economic":
                healthy = bool(obj) and not bool(obj.get("blocked",False))
            elif key == "mtf":
                healthy = bool(obj) and bool(obj.get("d1_current_available",True)) and float(obj.get("alignment",0)) >= 60
            else:
                healthy = bool(obj) and str(obj.get("status","OK")).upper() not in {"FAILED","ERROR","INSUFFICIENT","NO DATA","UNAVAILABLE"}
            out[label] = {
                "key": key, "role": role, "weight": weight,
                "direction": direction, "strength": round(strength,1),
                "healthy": bool(healthy), "available": bool(obj),
                "raw": obj,
            }
        return out


class TradeQualityEngine:
    """Central trade-quality evaluator fed by the complete Engine Information Bus.

    Directional evidence is aggregated only from healthy, available engines.
    Context/risk/timing engines can strengthen or veto a setup. Data quality,
    session, MTF coverage and risk remain hard gates.
    """

    @staticmethod
    def evaluate(a, ensemble, no_trade, data_quality):
        direction = str(ensemble.get("direction", a.get("confluence",{}).get("direction","WAIT"))).upper()
        bus = EngineInformationBus.build(a, data_quality)

        # Directional consensus across every healthy directional engine.
        votes = []
        for label, item in bus.items():
            if item["role"] not in {"directional"} or not item["healthy"]:
                continue
            if item["direction"] in {"BULLISH","BEARISH"}:
                votes.append((label, item["direction"], item["weight"]))
        total_weight = sum(w for _,_,w in votes)
        aligned_weight = sum(w for _,d,w in votes if d == direction) if direction in {"BULLISH","BEARISH"} else 0.0
        agreement = 100.0 * aligned_weight / max(total_weight,1e-9)

        dq = bus["Data Integrity"]
        session = bus["Session"]
        mtf = bus["Multi-Timeframe Alignment"]
        risk = bus["Risk Control"]
        risk_veto = bus["Risk Veto"]
        news = bus["News/Economic"]
        ai = bus["AI/ML Prediction"]
        confluence = bus["Confluence"]
        ensemble_item = bus["Ensemble Decision"]
        mtf_raw = mtf["raw"] or {}
        mtf_alignment = float(mtf_raw.get("alignment",0) or 0)
        mtf_coverage = float(mtf_raw.get("coverage_percent",0) or 0)
        mtf_ok = bool(mtf_raw.get("d1_current_available", True)) and mtf_alignment >= 60 and mtf_coverage >= 50

        dq_ok = dq["healthy"]
        session_ok = session["healthy"]
        risk_ok = risk["healthy"] and risk_veto["healthy"]
        news_ok = news["healthy"]
        ensemble_ok = bool(ensemble.get("approved",False))
        no_trade_ok = bool(no_trade.get("trade_allowed",False))

        # Weighted central quality score. Data/MTF/directional agreement carry
        # more weight than any single indicator so no isolated engine dominates.
        engine_direction_score = agreement
        ai_score = ai["strength"]
        confluence_score = confluence["strength"]
        ensemble_score = ensemble_item["strength"]
        context_scores = [
            bus["Current Volatility"]["strength"], bus["Volume Flow"]["strength"],
            bus["Candle Timing"]["strength"], bus["Market Region"]["strength"],
            bus["Fair Value Gap"]["strength"], bus["Currency Strength"]["strength"],
            bus["Correlation"]["strength"], bus["Breakout"]["strength"],
            bus["Support/Resistance"]["strength"], bus["Regime"]["strength"],
        ]
        context_score = float(np.mean([x for x in context_scores if x > 0])) if any(x > 0 for x in context_scores) else 0.0

        score = (
            engine_direction_score * 0.25
            + confluence_score * 0.15
            + ai_score * 0.10
            + ensemble_score * 0.10
            + mtf_alignment * 0.15
            + context_score * 0.10
            + float(dq["raw"].get("score",0)) * 0.10
            + (100.0 if risk_ok and news_ok and session_ok else 0.0) * 0.05
        )
        score = float(np.clip(score,0,100))

        hard_vetoes=[]
        if not dq_ok: hard_vetoes.append("DATA QUALITY / STALE OR INVALID FEED")
        if not session_ok: hard_vetoes.append("SESSION NOT TRADEABLE")
        if direction not in {"BULLISH","BEARISH"}: hard_vetoes.append("NO VALID DIRECTION")
        if not mtf_ok: hard_vetoes.append("CURRENT-MARKET MTF INSUFFICIENT")
        if agreement < 65: hard_vetoes.append("ENGINE INFORMATION AGREEMENT BELOW 65")
        if not ensemble_ok: hard_vetoes.append("ENSEMBLE NOT APPROVED")
        if not no_trade_ok: hard_vetoes.extend(no_trade.get("reasons",[]))
        if not risk_ok: hard_vetoes.append("RISK CONTROL / RISK VETO")
        if not news_ok: hard_vetoes.append("HIGH-IMPACT NEWS BLACKOUT")
        if float(a.get("volatility",{}).get("regime_score",50) or 50) < 0: hard_vetoes.append("INVALID VOLATILITY")

        hard_vetoes=list(dict.fromkeys(hard_vetoes))
        decision = "TRADE" if score >= 85 and not hard_vetoes else "NO TRADE"
        grade = "A" if decision == "TRADE" and score >= 90 else "B" if decision == "TRADE" else "C" if score >= 70 else "D"

        checks = {
            "DATA QUALITY": dq_ok,
            "CURRENT-MARKET MTF": mtf_ok,
            "ALL-ENGINE INFORMATION BUS": agreement >= 65,
            "ENSEMBLE": ensemble_ok,
            "RISK CONTROL": risk_ok,
            "NEWS FILTER": news_ok,
            "SESSION": session_ok,
            "NO-TRADE GATE": no_trade_ok,
        }
        return {
            "checks": checks,
            "components": {
                "Engine information agreement": round(engine_direction_score*0.25,2),
                "Confluence": round(confluence_score*0.15,2),
                "AI/ML": round(ai_score*0.10,2),
                "Ensemble": round(ensemble_score*0.10,2),
                "MTF": round(mtf_alignment*0.15,2),
                "Technical context": round(context_score*0.10,2),
                "Data integrity": round(float(dq["raw"].get("score",0))*0.10,2),
                "Risk/news/session": round((100.0 if risk_ok and news_ok and session_ok else 0.0)*0.05,2),
            },
            "score": round(score,1), "grade": grade, "decision": decision,
            "direction": direction, "directional_agreement": round(agreement,1),
            "directional_engines": [(x,d,round(w,2)) for x,d,w in votes],
            "engine_information_bus": bus,
            "mtf_alignment": round(mtf_alignment,1),
            "mtf_coverage": round(mtf_coverage,1),
            "hard_vetoes": hard_vetoes,
            "reasons": hard_vetoes,
        }


class TimeSynchronizationEngine:
    """Authoritative UTC/system clock service. It never makes trading decisions."""
    @staticmethod
    def snapshot() -> Dict[str, Any]:
        now = pd.Timestamp.now(tz="UTC")
        return {"utc_timestamp": now.isoformat(), "unix": float(now.timestamp()),
                "timezone": "UTC", "clock_status": "SYNCHRONIZED"}

class APIConnectionManager:
    """V14 provider/connection registry. Credentials stay outside analysis engines."""
    @staticmethod
    def status(cfg: Config) -> Dict[str, Any]:
        source = str(getattr(cfg, "data_source", "DEMO")).upper()
        return {"provider": source, "read_only": True, "execution_enabled": False,
                "configured": source == "DEMO" or bool(getattr(cfg, "twelve_data_api_key", "")) or source == "DERIV REAL FOREX"}

class EconomicCalendarEventEngine:
    """Collects and validates scheduled event data; it does not decide trade direction."""
    @staticmethod
    def get_events() -> pd.DataFrame:
        return load_events()
    @staticmethod
    def validate(events: pd.DataFrame) -> Dict[str, Any]:
        if events is None or events.empty:
            return {"status": "NO_DATA", "rows": 0, "valid": False}
        cols = {str(c).lower().strip().replace(" ", "_") for c in events.columns}
        return {"status": "PASS" if "time" in cols else "FAIL", "rows": int(len(events)), "valid": "time" in cols}

class NewsEventFilterEngine:
    """Converts validated event context into PASS/CAUTION/BLOCK risk state."""
    @staticmethod
    def evaluate(events: pd.DataFrame, symbol: str, cfg: Config) -> Dict[str, Any]:
        e = EconomicEngine.analyze(events, symbol)
        status = "BLOCK" if e.get("blocked") else "CAUTION" if str(e.get("risk", "")).upper() in {"HIGH", "CRITICAL", "UNKNOWN"} else "PASS"
        return {**e, "filter_status": status}

class EntryZoneAssessmentEngine:
    """Assesses a candidate entry zone after direction is supplied; it never chooses direction."""
    @staticmethod
    def assess(df: pd.DataFrame, direction: str, confluence: Dict[str, Any], cfg: Config, symbol: str) -> Dict[str, Any]:
        candidate = ForexEntryEngine.calculate(df, direction, confluence, cfg, symbol=symbol)
        return {"entry": candidate.get("entry"), "zone_low": candidate.get("zone_low"),
                "zone_high": candidate.get("zone_high"), "rr": candidate.get("rr"),
                "quality": candidate.get("quality", 0.0), "approved": bool(candidate.get("approved")),
                "direction_input": direction, "status": "ASSESSED"}

class StrikeEntryValidationEngine:
    """Validates entry price/zone/market conditions; it never selects direction."""
    @staticmethod
    def validate(candidate: Dict[str, Any], current_price: Optional[float] = None) -> Dict[str, Any]:
        entry = candidate.get("entry")
        lo, hi = candidate.get("zone_low"), candidate.get("zone_high")
        reasons=[]
        valid = entry is not None and lo is not None and hi is not None and float(lo) <= float(entry) <= float(hi)
        if not valid: reasons.append("ENTRY_OUTSIDE_VALID_ZONE")
        if current_price is not None and not np.isfinite(float(current_price)): reasons.append("INVALID_CURRENT_PRICE")
        return {"status": "PASS" if valid and not reasons else "FAIL", "valid": bool(valid and not reasons), "reasons": reasons}

class TradeQualityFilterLayer:
    """Pre-decision quality gate; final BUY/SELL authority remains FinalDecisionEngine."""
    @staticmethod
    def evaluate(a, ensemble, no_trade, data_quality):
        return TradeQualityEngine.evaluate(a, ensemble, no_trade, data_quality)

class EngineHealthVerificationEngine:
    """Checks engine output availability/integrity without making trading decisions."""
    @staticmethod
    def verify(evidence_packages: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        total=len(evidence_packages or {}); healthy=0; failed=[]
        for name,pkg in (evidence_packages or {}).items():
            q=pkg.get("quality",{}) if isinstance(pkg,dict) else {}
            ok=str(q.get("status","OK")).upper() not in {"FAIL","FAILED","INVALID"}
            if ok: healthy+=1
            else: failed.append(name)
        pct=healthy/max(total,1)*100.0
        return {"status":"HEALTHY" if pct>=90 else "DEGRADED" if pct>=70 else "FAILED",
                "healthy_engines":healthy,"total_engines":total,"health_percent":pct,"failed_engines":failed}

class MTFConfirmationEngine:
    """MTF confirmation facade. It consumes existing timeframe calculations and does not duplicate indicators."""
    @staticmethod
    def analyze(df: pd.DataFrame, symbol: Optional[str]=None, cfg: Optional[Config]=None) -> Dict[str, Any]:
        return MultiTimeframeEngine.analyze(df, symbol=symbol, cfg=cfg) if symbol and cfg else MultiTimeframeEngine.analyze(df)

class AIMLPredictionEngine:
    """Predictive evidence facade; probabilities never become final signals by themselves."""
    @staticmethod
    def predict(df: pd.DataFrame) -> Dict[str, Any]:
        return AIProbabilityEngine.predict(df)

class RiskControlEngine:
    """Risk-control facade. Direction and final decision remain separate."""
    @staticmethod
    def evaluate(*args, **kwargs):
        return RiskEngine.evaluate(*args, **kwargs)

class BreakEvenEngine:
    """Post-entry break-even management; it cannot create a new trade signal."""
    @staticmethod
    def evaluate(trade: Dict[str, Any], current_price: float, trigger_R: float = 1.0) -> Dict[str, Any]:
        if not trade or str(trade.get("market","FOREX")).upper() != "FOREX":
            return {"eligible": False, "status": "NO_OPEN_FOREX_TRADE"}
        entry=float(trade.get("entry", current_price)); sl=trade.get("sl"); direction=str(trade.get("direction","")).upper()
        if sl is None or direction not in {"BUY","SELL"}: return {"eligible": False, "status":"INSUFFICIENT_TRADE_DATA"}
        risk=abs(entry-float(sl)); reward=(float(current_price)-entry if direction=="BUY" else entry-float(current_price))
        r_multiple=reward/max(risk,1e-12)
        return {"eligible": bool(r_multiple>=trigger_R), "r_multiple": r_multiple,
                "break_even_price": entry, "status":"TRIGGERED" if r_multiple>=trigger_R else "WAITING"}

class RealForexTradeEngine:
    """Forex paper-execution facade; live order placement is disabled in this build."""
    @staticmethod
    def paper_execute(symbol: str, direction: str, entry: float, sl: float, tp: float, lot: float) -> Dict[str, Any]:
        return ExecutionEngine.paper_order("FOREX", symbol, direction, entry, sl, tp, lot=lot)

class AutoRefreshEngine:
    """Dashboard orchestration only; never creates a signal."""
    @staticmethod
    def configuration(enabled: bool, interval_seconds: int) -> Dict[str, Any]:
        return {"enabled": bool(enabled), "interval_seconds": int(interval_seconds), "role": "ORCHESTRATION_ONLY"}

class LeakageFreeTrainingEngine:
    """Creates chronological train/OOS datasets without future-data leakage."""
    @staticmethod
    def split(df: pd.DataFrame, train_ratio: float = 0.70) -> Dict[str, Any]:
        x=MarketDataEngine.normalize(df)
        cut=max(1,min(len(x)-1,int(len(x)*float(train_ratio)))) if len(x)>1 else len(x)
        return {"train":x.iloc[:cut].copy(),"out_of_sample":x.iloc[cut:].copy(),
                "chronological":True,"future_data_used_in_train":False,"status":"READY" if len(x)>1 else "INSUFFICIENT"}

class LiveTradeEligibilityEngine:
    """Final fail-closed gate separating analysis quality from live-data eligibility.

    Important separation:
      * analysis_quality = how strongly the analysis engines agree
      * ai_confidence = model confidence
      * data_integrity_score = structural quality of the loaded dataset
      * data_freshness = whether the latest candle is recent enough
      * trade_eligible = final permission to consider a live/paper entry

    High confluence or AI confidence can never override a stale/invalid feed.
    """

    @staticmethod
    def evaluate(
        a,
        ensemble,
        no_trade,
        trade_quality,
        data_quality,
        timing,
        emergency=False,
    ):
        dq = data_quality or {}
        reasons = list(no_trade.get("reasons", []))

        if dq.get("stale", False) or not dq.get("signal_allowed", False):
            if "DATA QUALITY / STALE FEED" not in reasons:
                reasons.append("DATA QUALITY / STALE FEED")

        if timing and not timing.get("fresh", True):
            reasons.append("SIGNAL TIMING EXPIRED")

        if emergency:
            reasons.append("EMERGENCY STOP")

        if not ensemble.get("approved", False):
            reasons.extend(ensemble.get("reasons", []))

        if trade_quality.get("decision") != "TRADE":
            reasons.append("TRADE QUALITY GATE")

        reasons = list(dict.fromkeys(reasons))

        analysis_score = float(
            np.clip(
                ensemble.get("confluence_score", a.get("confluence", {}).get("score", 0.0)),
                0,
                100,
            )
        )
        ai_confidence = float(np.clip(a.get("ai", {}).get("confidence", 0.0), 0, 100))
        integrity_score = float(np.clip(dq.get("score", 0.0), 0, 100))
        freshness_ok = bool(not dq.get("stale", True))
        quality_score = float(np.clip(trade_quality.get("score", 0.0), 0, 100))

        eligible = (
            not reasons
            and freshness_ok
            and bool(dq.get("signal_allowed", False))
            and bool(ensemble.get("approved", False))
            and bool(no_trade.get("trade_allowed", False))
            and trade_quality.get("decision") == "TRADE"
            and not emergency
        )

        return {
            "eligible": bool(eligible),
            "status": "APPROVED" if eligible else "BLOCKED",
            "reasons": reasons,
            "analysis_score": analysis_score,
            "ai_confidence": ai_confidence,
            "data_integrity_score": integrity_score,
            "data_freshness_seconds": float(dq.get("age_seconds", 0.0)),
            "data_freshness_ok": freshness_ok,
            "trade_quality_score": quality_score,
            "trade_quality_grade": trade_quality.get("grade", "D"),
        }


class PerformanceIntelligenceEngine:
    @staticmethod
    def summarize(journal):
        if not journal: return {"signals":0,"wins":0,"losses":0,"win_rate":0.0,"by_key":pd.DataFrame()}
        j=pd.DataFrame(journal)
        result={"signals":len(j),"wins":int((j.get("result",pd.Series(dtype=str))=="WIN").sum()),"losses":int((j.get("result",pd.Series(dtype=str))=="LOSS").sum())}
        result["win_rate"]=result["wins"]/max(1,result["wins"]+result["losses"])*100
        keys=[c for c in ["symbol","market","session","regime","expiry"] if c in j.columns]
        result["by_key"]=j.groupby(keys).size().reset_index(name="trades") if keys else pd.DataFrame()
        return result

def init_state():
    """Initialize V14 Streamlit session state only.

    This contains only V14 session state; no daily signal-lock mechanism exists.
    Empty market data means no verified data; it is never synthetic data.
    """
    defaults = {
        "data": _empty_market_data(),
        "data_meta": {},
        "data_symbol": None,
        "data_timeframe": None,
        "data_source_loaded": None,
        "events": pd.DataFrame(),
        "cot": pd.DataFrame(),
        "journal": [],
        "paper_balance": 10000.0,
        "bot_enabled": False,
        "emergency": False,
        "live_status": {"connected": False, "healthy": False},
        "td_key": "",
        "td_budget": {
            "request_log": [],
            "daily_log": [],
            "cooldown_until": 0.0,
            "key_cooldowns": {},
            "last_request": None,
            "last_error": None,
            "last_error_at": None,
        },
        "live_pair_cache": {},
        "v14_backtest": pd.DataFrame(),
        "v14_backtest_metrics": {},
        "v14_forward": {},
        "mc": {},
        "optimizer": pd.DataFrame(),
        "historical_builder_status": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, pd.DataFrame):
                st.session_state[key] = value.copy()
            elif isinstance(value, dict):
                st.session_state[key] = dict(value)
            elif isinstance(value, list):
                st.session_state[key] = list(value)
            else:
                st.session_state[key] = value


def canonical_symbol(symbol: str) -> str:
    """Normalize a dashboard/feed symbol to one canonical key."""
    s = str(symbol or "").upper().strip().replace("/", "").replace("_", "").replace("-", "")
    return s


def display_symbol(symbol: str) -> str:
    """Human-readable dashboard symbol, e.g. EURUSD -> EUR/USD."""
    s = canonical_symbol(symbol)
    return f"{s[:3]}/{s[3:]}" if len(s) == 6 else s


def _empty_market_data() -> pd.DataFrame:
    """Return a typed empty OHLCV frame. Empty means NO DATA, never synthetic data."""
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


def _get_twelve_data_key(cfg: Config) -> str:
    """Resolve the Twelve Data key from the UI, environment, or Streamlit secrets."""
    key = str(getattr(cfg, "twelve_data_api_key", "") or "").strip()
    if key:
        return key
    try:
        session_key = str(st.session_state.get("td_key", "") or "").strip()
        if session_key:
            return session_key
    except Exception:
        pass
    for env_name in ("TWELVE_DATA_API_KEY", "TWELVEDATA_API_KEY"):
        value = str(os.getenv(env_name, "") or "").strip()
        if value:
            return value
    try:
        value = str(st.secrets.get("TWELVE_DATA_API_KEY", "") or "").strip()
        if value:
            return value
    except Exception:
        pass
    return ""


def data_matches_selection(symbol: str, timeframe: str, df=None, source: str | None = None) -> bool:
    """Strict identity check: pair + timeframe + source must match the dashboard."""
    if df is None or df.empty:
        return False
    meta = st.session_state.get("data_meta", {}) or {}
    identity_ok = (
        canonical_symbol(meta.get("symbol", "")) == canonical_symbol(symbol)
        and str(meta.get("timeframe", "")).upper() == str(timeframe).upper()
    )
    if not identity_ok:
        return False
    if source is None:
        return True
    return str(meta.get("source", "")).upper() == str(source).upper()


def store_market_data(symbol: str, timeframe: str, df: pd.DataFrame, meta=None, source=None):
    """Atomically replace candles and their full identity. Never mixes pair/source data."""
    symbol = canonical_symbol(symbol)
    x = MarketDataEngine.normalize(df)
    m = dict(meta or {})
    actual_source = str(source or m.get("source", "UNKNOWN")).upper()
    m.update({
        "symbol": symbol,
        "timeframe": str(timeframe).upper(),
        "rows": len(x),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "source": actual_source,
    })
    # Validate before replacing the currently loaded dataset.
    if x.empty:
        raise RuntimeError(f"No market data returned for {symbol}/{timeframe} from {actual_source}.")
    st.session_state.data = x
    st.session_state.data_meta = m
    st.session_state.data_symbol = symbol
    st.session_state.data_timeframe = str(timeframe).upper()
    st.session_state.data_source_loaded = actual_source
    return x, m


def clear_market_data_identity(clear_candles: bool = True):
    """Hard-reset loaded-data identity so stale candles cannot survive a failed pair switch."""
    st.session_state.data_symbol = None
    st.session_state.data_timeframe = None
    st.session_state.data_source_loaded = None
    st.session_state.data_meta = {}
    if clear_candles:
        st.session_state.data = _empty_market_data()


class TwelveDataRateGuard:
    """Central gate for every Twelve Data request made by V14.

    The manager is intentionally conservative. It prevents Streamlit reruns,
    scanner rotation, MTF/D1 analysis and manual refreshes from competing for
    the same API allowance. A 429 starts a cooldown instead of triggering a
    retry storm. Cached data may be served only when it remains inside the
    configured maximum-staleness window; otherwise the system fails closed.
    """

    @staticmethod
    def state() -> dict:
        return st.session_state.setdefault("td_budget", {
            "request_log": [],
            "daily_log": [],
            "cooldown_until": 0.0,
            "key_cooldowns": {},
            "last_request": None,
            "last_error": None,
            "last_error_at": None,
        })

    @staticmethod
    def _prune(now: float, minute_window: int = 60):
        state = TwelveDataRateGuard.state()
        state["request_log"] = [t for t in state.get("request_log", []) if now - float(t) < minute_window]
        today = datetime.now(timezone.utc).date().isoformat()
        state["daily_log"] = [
            item for item in state.get("daily_log", [])
            if str(item.get("day", "")) == today
        ]
        state["key_cooldowns"] = {
            k: float(v) for k, v in state.get("key_cooldowns", {}).items() if float(v) > now
        }

    @staticmethod
    def allow(symbol: str, timeframe: str, cfg: Config) -> tuple[bool, int, str]:
        now = datetime.now(timezone.utc).timestamp()
        TwelveDataRateGuard._prune(now, int(getattr(cfg, "twelve_data_rate_guard_seconds", 60)))
        state = TwelveDataRateGuard.state()
        key = f"{canonical_symbol(symbol)}|{str(timeframe).upper()}"

        global_until = float(state.get("cooldown_until", 0.0) or 0.0)
        key_until = float(state.get("key_cooldowns", {}).get(key, 0.0) or 0.0)
        until = max(global_until, key_until)
        if until > now:
            return False, max(1, int(math.ceil(until - now))), "API_COOLDOWN"

        minute_budget = max(1, int(getattr(cfg, "twelve_data_request_budget_per_minute", 7)))
        if len(state["request_log"]) >= minute_budget:
            oldest = min(state["request_log"]) if state["request_log"] else now
            wait = max(1, int(math.ceil(60 - (now - oldest))))
            return False, wait, "MINUTE_BUDGET"

        daily_budget = max(minute_budget, int(getattr(cfg, "twelve_data_daily_budget", 760)))
        if len(state["daily_log"]) >= daily_budget:
            return False, 86400, "DAILY_BUDGET"

        return True, 0, "OK"

    @staticmethod
    def reserve(symbol: str, timeframe: str, cfg: Config):
        allowed, wait, reason = TwelveDataRateGuard.allow(symbol, timeframe, cfg)
        if not allowed:
            raise TwelveDataRateLimitError(
                f"Twelve Data rate protection blocked {display_symbol(symbol)}/{str(timeframe).upper()} "
                f"({reason}). Wait about {wait}s; no duplicate request was sent.",
                retry_after=wait,
            )
        now = datetime.now(timezone.utc).timestamp()
        state = TwelveDataRateGuard.state()
        state["request_log"].append(now)
        state["daily_log"].append({
            "time": now,
            "day": datetime.now(timezone.utc).date().isoformat(),
            "symbol": canonical_symbol(symbol),
            "timeframe": str(timeframe).upper(),
        })
        state["last_request"] = now
        return True

    @staticmethod
    def record_429(symbol: str, timeframe: str, retry_after: int):
        now = datetime.now(timezone.utc).timestamp()
        until = now + max(1, int(retry_after or 60))
        state = TwelveDataRateGuard.state()
        state["cooldown_until"] = max(float(state.get("cooldown_until", 0.0) or 0.0), until)
        state["key_cooldowns"][f"{canonical_symbol(symbol)}|{str(timeframe).upper()}"] = until
        state["last_error"] = "429 RATE LIMIT"
        state["last_error_at"] = now

    @staticmethod
    def snapshot(cfg: Config) -> dict:
        now = datetime.now(timezone.utc).timestamp()
        TwelveDataRateGuard._prune(now, int(getattr(cfg, "twelve_data_rate_guard_seconds", 60)))
        state = TwelveDataRateGuard.state()
        cooldown = max(0, int(math.ceil(float(state.get("cooldown_until", 0.0) or 0.0) - now)))
        return {
            "minute_used": len(state["request_log"]),
            "minute_budget": int(getattr(cfg, "twelve_data_request_budget_per_minute", 7)),
            "daily_used": len(state["daily_log"]),
            "daily_budget": int(getattr(cfg, "twelve_data_daily_budget", 760)),
            "cooldown_seconds": cooldown,
            "last_error": state.get("last_error"),
        }


def _td_cache_ttl(timeframe: str) -> int:
    """How long a successfully fetched Twelve Data dataset may be reused."""
    return {
        "M1": 30,
        "M3": 45,
        "M5": 75,
        "M15": 150,
        "M30": 300,
        "H1": 600,
        "H4": 1800,
        "D1": 3600,
    }.get(str(timeframe).upper(), 300)


def _td_cache_age_seconds(meta: dict) -> float:
    fetched = meta.get("fetched_at") or meta.get("loaded_at")
    if not fetched:
        return float("inf")
    try:
        ts = pd.Timestamp(fetched)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return max(0.0, (pd.Timestamp.now(tz="UTC") - ts.tz_convert("UTC")).total_seconds())
    except Exception:
        return float("inf")


def _td_request_allowed(cfg: Config) -> tuple[bool, int]:
    """Backward-compatible view of the central Twelve Data rate guard."""
    allowed, wait, _ = TwelveDataRateGuard.allow("GLOBAL", "GLOBAL", cfg)
    return allowed, wait


def _td_record_request():
    # Kept only for compatibility with legacy provider code paths. New provider calls
    # reserve their credit atomically through TwelveDataRateGuard.reserve().
    return None


def _td_cached_pair(symbol: str, timeframe: str, cfg: Config, allow_stale_seconds: int | None = None):
    """Return cached data without spending an API credit.

    Fresh cache is preferred. When the provider is rate-limited, a bounded stale
    cache may be returned only if its fetch age remains inside the data-integrity
    maximum. This is explicitly marked so downstream quality/health layers know
    that a provider refresh was unavailable.
    """
    cache = st.session_state.setdefault("live_pair_cache", {})
    source = str(getattr(cfg, "data_source", "TWELVE DATA")).upper()
    key = (source, canonical_symbol(symbol), str(timeframe).upper(), int(getattr(cfg, "twelve_data_outputsize", 500)))
    cached = cache.get(key)
    if not cached:
        return None
    cdf = cached.get("df")
    meta = dict(cached.get("meta", {}) or {})
    if cdf is None or cdf.empty:
        return None
    ttl = _td_cache_ttl(timeframe)
    age = _td_cache_age_seconds(meta)
    stale_limit = int(allow_stale_seconds if allow_stale_seconds is not None else ttl)
    if age > stale_limit:
        return None
    meta["cache_age_seconds"] = round(age, 1)
    meta["cache_fresh"] = bool(age <= ttl)
    meta["cache_stale"] = bool(age > ttl)
    return cdf.copy(), meta



def _fetch_twelve_data(symbol: str, timeframe: str, cfg: Config, end_date=None, start_date=None, outputsize=None):
    key = _get_twelve_data_key(cfg)
    if not key:
        raise RuntimeError(
            "Twelve Data is selected but no API key is configured. "
            "Enter the key in the sidebar or set TWELVE_DATA_API_KEY in Streamlit secrets."
        )

    # Reserve exactly one provider credit before the network call. This is the
    # single gate used by the selected pair, D1 scanner and multi-timeframe analysis.
    TwelveDataRateGuard.reserve(symbol, timeframe, cfg)
    try:
        df, meta = TwelveDataLiveEngine.fetch(
            key, symbol, timeframe,
            int(outputsize if outputsize is not None else cfg.twelve_data_outputsize),
            end_date=end_date, start_date=start_date
        )
    except TwelveDataRateLimitError as ex:
        # The provider has authoritative quota information. Start a cooldown
        # immediately and never retry inside this call.
        TwelveDataRateGuard.record_429(symbol, timeframe, ex.retry_after)
        raise
    meta = dict(meta or {})
    meta.update({
        "symbol": canonical_symbol(symbol),
        "requested_symbol": canonical_symbol(symbol),
        "timeframe": str(timeframe).upper(),
        "source": "TWELVE DATA",
        "read_only": True,
        "execution_enabled": False,
    })
    return df, meta



class HistoricalDatasetBuilder:
    """Build a larger real historical dataset from the selected real provider.

    This is an ingestion/orchestration service, not an analysis or signal engine.
    It only concatenates provider-returned candles, walks backward from the oldest
    received candle, removes duplicates, validates OHLC/timestamps, and fails closed
    if the provider cannot supply the requested history. It never creates candles.
    """
    @staticmethod
    def _step_seconds(timeframe: str) -> int:
        return {"M1":60,"M3":180,"M5":300,"M15":900,"M30":1800,
                "H1":3600,"H4":14400,"D1":86400}.get(str(timeframe).upper(),300)

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return _empty_market_data()
        x = MarketDataEngine.normalize(df)
        x = x.dropna(subset=["time","open","high","low","close"])
        x = x.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
        return x

    @classmethod
    def build_twelve_data(cls, symbol: str, timeframe: str, cfg: Config,
                          target_rows: int = None, max_requests: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        target = max(1000, int(target_rows or getattr(cfg, "historical_target_rows", 10000)))
        max_req = max(1, int(max_requests or getattr(cfg, "historical_max_requests", 4)))
        chunk = min(5000, max(1000, int(getattr(cfg, "twelve_data_outputsize", 5000))))
        key = _get_twelve_data_key(cfg)
        if not key:
            raise RuntimeError("Twelve Data historical builder requires a configured API key.")

        pieces = []
        end_date = None
        request_count = 0
        seen_oldest = None
        diagnostics = []
        while sum(len(p) for p in pieces) < target and request_count < max_req:
            # Each request is independently quota-gated. No duplicate retry is made.
            df, meta = _fetch_twelve_data(
                symbol, timeframe, cfg,
                end_date=end_date,
                outputsize=chunk,
            )
            x = cls._clean(df)
            if x.empty:
                break
            oldest = x.time.min()
            newest = x.time.max()
            diagnostics.append({"request": request_count + 1, "rows": int(len(x)),
                                "oldest": str(oldest), "newest": str(newest)})
            pieces.append(x)
            request_count += 1
            if seen_oldest is not None and pd.Timestamp(oldest) >= pd.Timestamp(seen_oldest):
                break
            seen_oldest = oldest
            # Move the next boundary strictly before the oldest returned candle.
            step = cls._step_seconds(timeframe)
            boundary = pd.Timestamp(oldest) - pd.Timedelta(seconds=step)
            end_date = boundary.strftime("%Y-%m-%d %H:%M:%S")
            if len(x) < max(10, int(chunk * 0.20)):
                # Provider returned a short page; another backward page may be
                # legitimate, but avoid burning quota on a clearly exhausted range.
                break

        if not pieces:
            raise RuntimeError("Historical builder received no candles from Twelve Data.")
        combined = cls._clean(pd.concat(pieces, ignore_index=True))
        validation = MarketDataEngine.validate(combined)
        if not validation.get("data_ok", False):
            raise RuntimeError(f"Historical dataset validation failed: {validation}")
        meta = {
            "source": "TWELVE DATA",
            "symbol": canonical_symbol(symbol),
            "timeframe": str(timeframe).upper(),
            "historical_dataset_builder": True,
            "target_rows": target,
            "rows": int(len(combined)),
            "requests_used": request_count,
            "request_diagnostics": diagnostics,
            "complete_target_reached": bool(len(combined) >= target),
            "no_synthetic_data": True,
            "validation": validation,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        if len(combined) < target:
            meta["warning"] = f"Provider supplied {len(combined)} valid candles; requested target was {target}."
        return combined, meta


def ensure_historical_test_dataset(df: pd.DataFrame, cfg: Config, symbol: str, timeframe: str):
    """Return a test-ready real dataset, expanding it only when necessary.

    DEMO/synthetic data is never silently upgraded into a statistical proof set.
    """
    x = HistoricalDatasetBuilder._clean(df)
    source = str(getattr(cfg, "data_source", "")).upper()
    target = max(1000, int(getattr(cfg, "historical_target_rows", 12000) or 12000))
    if source == "DERIV REAL FOREX":
        # Deriv is the sole historical source in this mode. Always request enough
        # real Deriv history to satisfy the statistical target; never substitute
        # Twelve Data, DEMO, synthetic candles, or fabricated values.
        if len(x) >= target:
            return x, {"status": "READY", "rows": len(x), "expanded": False,
                       "target_rows": target, "source": "DERIV REAL FOREX"}
        try:
            cfg.deriv_stream_outputsize = max(int(getattr(cfg, "deriv_stream_outputsize", 12000) or 12000), target)
            built, meta = get_deriv_real_forex_data(symbol, timeframe, cfg, force=False)
            built = HistoricalDatasetBuilder._clean(built)
            return built, {
                "status": "READY" if len(built) >= target else "INSUFFICIENT_REAL_HISTORY",
                "rows": len(built), "expanded": True, "target_rows": target,
                "message": (f"Deriv supplied {len(built):,} real-Forex candles; {target:,} are required for the V14 statistical test."
                            if len(built) < target else f"Deriv real-Forex historical dataset ready: {len(built):,} candles."),
                **meta,
            }
        except Exception as exc:
            return x, {
                "status": "DERIV_HISTORICAL_ERROR",
                "rows": len(x), "expanded": False, "target_rows": target,
                "message": f"Deriv real-Forex historical data unavailable: {exc}",
            }
    if len(x) >= int(getattr(cfg, "minimum_history_candles", 1000) or 1000):
        return x, {"status": "READY", "rows": len(x), "expanded": False}
    if source == "TWELVE DATA":
        target = max(1000, int(getattr(cfg, "historical_target_rows", 10000)))
        built, meta = HistoricalDatasetBuilder.build_twelve_data(symbol, timeframe, cfg, target_rows=target)
        return built, {"status": "READY" if len(built) >= 1000 else "INSUFFICIENT_REAL_HISTORY",
                       "rows": len(built), "expanded": True, **meta}
    return x, {
        "status": "INSUFFICIENT_REAL_HISTORY",
        "rows": len(x),
        "expanded": False,
        "message": "Select DERIV REAL FOREX or TWELVE DATA; DEMO/synthetic data cannot be used as statistical proof."
    }

def get_live_pair_data(symbol: str, timeframe: str, cfg: Config, force=False):
    """Resolve exactly one pair/timeframe while respecting the central data budget.

    Normal calls are cache-first. ``force=True`` is an explicit operator refresh;
    even then, if the provider refuses the request, the system may use a bounded stale
    cache rather than inventing data. Live modes never fall back to synthetic data.
    """
    symbol = canonical_symbol(symbol)
    timeframe = str(timeframe).upper()
    requested_source = str(cfg.data_source).upper()
    cache = st.session_state.setdefault("live_pair_cache", {})
    key = (requested_source, symbol, timeframe, int(getattr(cfg, "twelve_data_outputsize", 500)))

    # Cache-first is the default and is the critical protection against Streamlit
    # reruns repeatedly spending provider credits.
    if requested_source == "TWELVE DATA":
        fresh = _td_cached_pair(symbol, timeframe, cfg)
        if fresh is not None and not force:
            return fresh
    elif not force and key in cache:
        cached = cache[key]
        cached_df = cached.get("df")
        if cached_df is not None and not cached_df.empty:
            return cached_df.copy(), dict(cached.get("meta", {}))

    try:
        if requested_source == "TWELVE DATA":
            df, meta = _fetch_twelve_data(symbol, timeframe, cfg)
            actual_source = "TWELVE DATA"

        elif requested_source == "DERIV REAL FOREX":
            df, meta = get_deriv_real_forex_data(symbol, timeframe, cfg, force=force)
            actual_source = "DERIV REAL FOREX"
        elif requested_source == "DEMO":
            base_rows = int(cfg.demo_rows)
            if timeframe == "D1":
                base_rows = max(base_rows, 120 * 288)
            elif timeframe == "H4":
                base_rows = max(base_rows, 120 * 48)
            base = MarketDataEngine.synthetic(symbol, base_rows, seed=7 + sum(map(ord, symbol)))
            df = MarketDataEngine.resample(base, timeframe) if timeframe != "M5" else base
            meta = {"symbol": symbol, "timeframe": timeframe, "source": "DEMO", "synthetic": True,
                    "read_only": True, "execution_enabled": False,
                    "fetched_at": datetime.now(timezone.utc).isoformat()}
            actual_source = "DEMO"
        else:
            raise RuntimeError(f"Unsupported data source: {requested_source}")
    except TwelveDataRateLimitError as ex:
        # Provider quota is authoritative. Use stale cache only inside the
        # configured integrity window; otherwise fail closed.
        if requested_source == "TWELVE DATA" or str(getattr(ex, "provider", "")).upper() == "TWELVE DATA":
            stale = _td_cached_pair(symbol, timeframe, cfg, allow_stale_seconds=int(getattr(cfg, "twelve_data_cache_stale_seconds", cfg.data_max_age_seconds)))
            if stale is not None:
                stale_df, stale_meta = stale
                stale_meta.update({
                    "source": "TWELVE DATA",
                    "rate_limited": True,
                    "rate_limit_fallback": True,
                    "rate_limit_retry_after": ex.retry_after,
                    "read_only": True,
                    "execution_enabled": False,
                })
                return stale_df, stale_meta
        raise

    df = MarketDataEngine.normalize(df)
    if df.empty:
        raise RuntimeError(f"No market data returned for {symbol}/{timeframe} from {actual_source}.")
    meta = dict(meta or {})
    meta.update({"symbol": symbol, "timeframe": timeframe, "source": actual_source,
                 "requested_source": requested_source})
    if not meta.get("fetched_at"):
        meta["fetched_at"] = datetime.now(timezone.utc).isoformat()
    cache_key = (actual_source, symbol, timeframe, int(getattr(cfg, "twelve_data_outputsize", 500)))
    cache[cache_key] = {"df": df.copy(), "meta": dict(meta)}
    cache[key] = {"df": df.copy(), "meta": dict(meta)}
    return df, meta


def get_selected_market_data(symbol: str, timeframe: str, cfg: Config):
    """Resolve the dashboard selection with strict pair/timeframe/source isolation."""
    symbol = canonical_symbol(symbol)
    timeframe = str(timeframe).upper()
    requested_source = str(cfg.data_source).upper()

    # IMPORTANT: source is part of identity. A source change must trigger a new fetch.
    if data_matches_selection(symbol, timeframe, st.session_state.get("data"), requested_source):
        return st.session_state.data, st.session_state.get("data_meta", {})

    # A pair or source change invalidates the old dataset BEFORE fetching.
    clear_market_data_identity(clear_candles=True)
    df, meta = get_live_pair_data(symbol, timeframe, cfg, force=False)
    return store_market_data(symbol, timeframe, df, meta, meta.get("source", requested_source))


def load_events():
    return st.session_state.get("events", pd.DataFrame())


def load_cot():
    return st.session_state.get("cot", pd.DataFrame())


def build_no_data_analysis(symbol, cfg, data_quality=None, reason="NO MARKET DATA"):
    """Return a complete, dashboard-safe analysis schema when candles are unavailable.

    Every key used by the dashboard is present so a failed feed cannot turn into a
    secondary KeyError such as a["cs"]["matrix"]. No synthetic candles are created here.
    """
    dq = data_quality or {
        "score": 0.0, "status": "BAD", "reasons": [reason], "stale": True,
        "age_seconds": 0.0, "last_timestamp": "-", "rows": 0,
        "signal_allowed": False,
    }
    c = {"score": 0.0, "direction": "WAIT", "components": {}, "quality": 0.0}
    trend = {"label": "NO DATA", "direction": "SIDEWAYS"}
    momentum = {"direction": "NO DATA", "score": 0.0, "strength": "LOW"}
    volatility = {"regime": "NO DATA"}
    structure = {"direction": "NO DATA", "bias": "NO DATA"}
    price_action = {"direction": "NO DATA", "bias": "NO DATA"}
    currency_strength = {
        "matrix": {}, "base": str(symbol)[:3], "quote": str(symbol)[3:6],
        "spread": 0.0, "strongest": "-", "weakest": "-",
    }
    mtf = {"alignment": 0.0, "strength": "WEAK", "states": {}}
    economic = {"bias": "NEUTRAL", "score": 0.0, "risk": "UNKNOWN", "blocked": False, "next_event": "-"}
    cot = {"bias": "NEUTRAL", "net": 0.0, "weekly_change": 0.0, "percentile": 50.0}
    correlation = {"average_abs_corr": 0.0, "risk": "LOW", "pairs": []}
    risk = {"approved": False, "status": "VETO", "reasons": [reason], "risk_pct": cfg.risk_per_trade * 100}
    advanced_momentum = {"direction": "NO DATA", "score": 0.0, "strength": "LOW"}
    ai = {"up_probability": 50.0, "down_probability": 50.0, "confidence": 0.0}
    ensemble = {"score": 0.0, "direction": "WAIT", "approved": False, "reasons": [reason]}
    no_trade = {"trade_allowed": False, "reasons": [reason], "status": "NO TRADE"}
    trade_quality = {
        "grade": "D", "score": 0.0, "decision": "NO TRADE",
        "checks": {"DATA QUALITY": False, "NO MARKET DATA": False},
    }
    session = SessionEngine.analyze(live=True)
    breakout = {"direction": "NO DATA", "state": "NO DATA"}
    liquidity = {"direction": "NO DATA", "FVG": False}
    ema_evidence = {"periods": list(EMAEngine.PERIODS), "values": {}, "direction": "NO DATA", "alignment": 0.0, "status": "NO DATA"}
    macd_evidence = {"direction": "NO DATA", "status": "NO DATA"}
    fvg = {"detected": False, "direction": "NO DATA", "status": "NO DATA"}
    microstructure = {"order_book_available": False, "depth_data": None, "status": "NO DATA"}
    order_flow = {"direction": "NO DATA", "score": 0.0, "status": "NO DATA", "order_book_used": False}
    reversal = {"detected": False, "evidence": [], "status": "NO DATA"}
    sr = {}
    regime = "NO-TRADE"
    volume_flow = {"direction": "NO DATA", "score": 0.0, "status": "NO DATA"}
    candle_timing = {"direction": "NO DATA", "score": 0.0, "status": "NO DATA"}
    market_region = {"direction": "NO DATA", "score": 0.0, "status": "NO DATA"}
    direct_probability = {"up_probability": 50.0, "down_probability": 50.0, "confidence": 0.0, "direction": "NEUTRAL", "status": "NO DATA"}
    signal_timing = {"fresh": False, "age_seconds": float("inf"), "status": "NO DATA"}

    return {
        "trend": trend, "momentum": momentum, "volatility": volatility,
        "structure": structure, "price_action": price_action, "sr": sr,
        "breakout": breakout, "liquidity": liquidity,
        "ema": ema_evidence, "macd": macd_evidence, "fvg": fvg,
        "microstructure": microstructure, "order_flow": order_flow, "reversal": reversal,
        "regime": regime,
        "session": session, "currency_strength": currency_strength, "mtf": mtf,
        "economic": economic, "cot": cot, "correlation": correlation,
        "volume_flow": volume_flow, "candle_timing": candle_timing,
        "market_region": market_region, "direct_probability": direct_probability,
        "market_tracker": {"direction":"WAIT","score":0.0,"status":"NO DATA"},
        "signal_timing": signal_timing,
        "confluence": c, "risk": risk, "advanced_momentum": advanced_momentum,
        "ai": ai, "ensemble": ensemble, "no_trade": no_trade,
        "trade_quality": trade_quality, "data_quality": dq,
        # V14 compatibility aliases.
        "t": trend, "m": momentum, "v": volatility, "s": structure,
        "pa": price_action, "bo": breakout, "li": liquidity, "re": regime,
        "se": session, "cs": currency_strength, "eco": economic,
        "corr": correlation, "c": c, "vf": volume_flow, "ct": candle_timing,
        "mr": market_region, "dp": direct_probability,
        "_error": reason,
    }



def analyze_market(df, symbol, cfg, timeframe=None, *, deep_mtf=True):
    validation = DataValidationEngine.validate(df)
    t = TrendEngine.analyze(df)
    ema_evidence = EMAEngine.analyze(df)
    macd_evidence = MACDEngine.analyze(df)
    m = MomentumEngine.analyze(df)
    v = VolatilityEngine.analyze(df)
    s = StructureEngine.analyze(df)
    pa = PriceActionEngine.analyze(df)
    sr = SupportResistanceEngine.analyze(df)
    bo = BreakoutEngine.analyze(df)
    li = LiquidityEngine.analyze(df)
    fvg = FVGEngine.analyze(df)
    microstructure = MarketMicrostructureEngine.analyze(df)
    volume_order_flow = OrderFlowVolumeFlowEngine.analyze(df)
    reversal = ReversalDetectionEngine.analyze(df)
    re = derive_market_context(t, v, s, bo)
    # LIVE session gate: use the current UTC clock, never the timestamp of
    # the last candle. This prevents stale/historical candles from showing
    # London during Tokyo/Sydney hours or during the weekend.
    se = SessionEngine.analyze(live=True)
    cs = CurrencyStrengthEngine.analyze(df, symbol)
    # The selected pair gets the full current-market MTF pass. Historical replay
    # can explicitly use the historical/resampled path so provider requests are
    # not multiplied for every pair.
    mtf = (
        MultiTimeframeEngine.analyze(df, symbol=symbol, cfg=cfg)
        if deep_mtf
        else MultiTimeframeEngine.analyze(df)
    )
    volume_flow = VolumeFlowEngine.analyze(df)
    candle_timing = CandleTimingEngine.analyze(df, timeframe or st.session_state.get("data_timeframe", "M5") or "M5")
    market_region = MarketRegionEngine.analyze(df)
    direct_probability = DirectProbabilityEngine.predict(df)
    economic_events = EconomicCalendarEventEngine.get_events()
    eco = EconomicEngine.analyze(economic_events, symbol)
    news_filter = NewsEventFilterEngine.evaluate(economic_events, symbol, cfg)
    cot = COTEngine.analyze(load_cot(), symbol)

    # Single-symbol correlation is LOW by definition unless peer histories are supplied.
    history = {symbol: df}
    corr = CorrelationEngine.analyze(history, symbol)

    c = ConfluenceEngine.score(
        t, m, v, s, pa, sr, bo, li, re, se, mtf, eco, cot
    )
    dq = DataIntegrityEngine.assess(df, timeframe or st.session_state.get("data_timeframe", "M5") or "M5", cfg.data_max_age_seconds)
    adv_m = MomentumDirectionEngine.analyze(df)
    ai = AIProbabilityEngine.predict(df)

    risk = RiskEngine.evaluate(
        {
            "daily_loss_pct": 0,
            "drawdown_pct": 0,
            "open_positions": len(
                [x for x in st.session_state.journal if x.get("status") == "OPEN"]
            ),
        },
        cfg,
        c,
        v,
        eco,
        corr,
    )
    # Advanced layers are advisory gates; they do not replace V14 specialist engines.
    advisory = {"trend":t,"momentum":m,"volatility":v,"structure":s,"price_action":pa,"sr":sr,
                "breakout":bo,"liquidity":li,"regime":re,"session":se,"currency_strength":cs,"mtf":mtf,"economic":eco,
                "cot":cot,"correlation":corr,"confluence":c,"risk":risk,"data_quality":dq}
    advisory["ema"] = ema_evidence
    advisory["macd"] = macd_evidence
    advisory["fvg"] = fvg
    advisory["microstructure"] = microstructure
    advisory["order_flow"] = volume_order_flow
    advisory["reversal"] = reversal
    advisory["advanced_momentum"] = adv_m
    advisory["ai"] = ai
    advisory["volume_flow"] = volume_flow
    advisory["candle_timing"] = candle_timing
    advisory["market_region"] = market_region
    advisory["direct_probability"] = direct_probability
    advisory["market_tracker"] = {"direction": c.get("direction","WAIT"), "score": c.get("score",0), "status":"CURRENT PAIR"}
    advisory["signal_timing"] = SignalTimingEngine.assess(pd.Timestamp.now(tz="UTC"), cfg.signal_max_age_seconds)
    ensemble = EnsembleDecisionEngine.decide(advisory, ai, dq, cfg)
    no_trade = NoTradeEngine.evaluate(advisory, ensemble, dq, cfg)
    advisory["ensemble"] = ensemble
    advisory["no_trade"] = no_trade
    quality = TradeQualityFilterLayer.evaluate(advisory, ensemble, no_trade, dq)

    result = {
        "trend": t,
        "momentum": m,
        "volatility": v,
        "structure": s,
        "price_action": pa,
        "sr": sr,
        "breakout": bo,
        "liquidity": li,
        "ema": ema_evidence,
        "macd": macd_evidence,
        "fvg": fvg,
        "microstructure": microstructure,
        "order_flow": volume_order_flow,
        "reversal": reversal,
        "regime": re,
        "session": se,
        "currency_strength": cs,
        "volume_flow": volume_flow,
        "candle_timing": candle_timing,
        "market_region": market_region,
        "direct_probability": direct_probability,
        "market_tracker": advisory["market_tracker"],
        "signal_timing": advisory["signal_timing"],
        "mtf": mtf,
        "economic": eco,
        "news_filter": news_filter,
        "cot": cot,
        "correlation": corr,
        "confluence": c,
        "risk": risk,
        "advanced_momentum": adv_m,
        "ai": ai,
        "ensemble": ensemble,
        "no_trade": no_trade,
        "trade_quality": quality,
        "data_quality": dq,
        # Compatibility aliases used by earlier code.
        "t": t, "m": m, "v": v, "s": s, "pa": pa, "bo": bo, "li": li,
        "re": re, "se": se, "cs": cs, "eco": eco, "corr": corr, "c": c,
        "vf": volume_flow, "ct": candle_timing, "mr": market_region, "dp": direct_probability,
    }
    # V14 architecture routing: every specialist result is packaged as evidence
    # and published to Central Assessment. V14 result keys are kept stable for dashboard rendering.
    tf = timeframe or st.session_state.get("data_timeframe", "M5") or "M5"
    evidence_inputs = {
        "TrendEngine": t,
        "MomentumEngine": m,
        "VolatilityEngine": v,
        "StructureEngine": s,
        "PriceActionEngine": pa,
        "SupportResistanceEngine": sr,
        "BreakoutEngine": bo,
        "LiquidityEngine": li,
        "EMAEngine": ema_evidence,
        "MACDEngine": macd_evidence,
        "FVGEngine": fvg,
        "MarketMicrostructureEngine": microstructure,
        "OrderFlowVolumeFlowEngine": volume_order_flow,
        "ReversalDetectionEngine": reversal,
        "CurrencyStrengthEngine": cs,
        "CorrelationEngine": corr,
        "VolumeFlowEngine": volume_flow,
        "MTFConfirmation": mtf,
        "AIProbability": ai,
        "ReversalContext": {"direction": re if re in {"BULLISH","BEARISH"} else "NEUTRAL", "context": re},
    }
    evidence_packages = {}
    data_ts = df["time"].iloc[-1] if isinstance(df, pd.DataFrame) and "time" in df.columns and not df.empty else None
    for engine_name, evidence in evidence_inputs.items():
        evidence_packages[engine_name] = package_engine_evidence(
            engine_id=engine_name.upper(),
            engine_name=engine_name,
            symbol=symbol,
            timeframe=tf,
            evidence=evidence,
            data_timestamp=data_ts,
            quality={
                "score": _evidence_quality_score(evidence, dq),
                "status": _evidence_quality_status(evidence, dq),
                "weight": _evidence_weight(engine_name),
                "input_sufficient": _evidence_input_sufficient(evidence),
            },
            provenance={"source": dq.get("source", st.session_state.get("data_source_loaded", "UNKNOWN"))},
        )

    central_assessment = CentralAssessmentDataQualityEngine.assess(
        symbol, tf, dq, evidence_packages
    )
    signal_intelligence = SignalIntelligenceLayer.assess(
        central_assessment, evidence_packages
    )
    result["engine_evidence_packages"] = evidence_packages
    result["central_assessment"] = central_assessment
    result["signal_intelligence"] = signal_intelligence
    # V14 decision authority: specialized engines and advisory/legacy layers may
    # provide evidence, but ONLY FinalDecisionEngine can publish BUY/SELL/NO-TRADE.
    intelligence_direction = str(
        signal_intelligence.get("market_context", "NEUTRAL")
    ).upper()
    entry_zone = EntryZoneAssessmentEngine.assess(
        df, intelligence_direction, c, cfg, symbol
    )
    entry_candidate = ForexEntryEngine.calculate(
        df, intelligence_direction, c, cfg, symbol=symbol
    )
    last_price = float(df.close.iloc[-1]) if not df.empty else None
    strike_validation = StrikeEntryValidationEngine.validate(
        entry_candidate, current_price=last_price
    )
    engine_health = EngineHealthVerificationEngine.verify(evidence_packages)
    timing_package = advisory["signal_timing"]
    exec_package = ExecutionQualityEngine.assess(
        current_price=last_price,
        bid=(st.session_state.get("data_meta", {}).get("tick") or {}).get("bid"),
        ask=(st.session_state.get("data_meta", {}).get("tick") or {}).get("ask"),
        spread_pips=(st.session_state.get("data_meta", {}).get("tick") or {}).get("spread"),
    )
    entry_valid = bool(
        entry_candidate.get("approved")
        and strike_validation.get("valid", False)
        and intelligence_direction in {"BULLISH", "BEARISH"}
    )
    timing_valid = bool(timing_package.get("fresh"))
    execution_valid = bool(exec_package.get("status") == "PASS")
    final_decision = FinalDecisionEngine.resolve(
        signal_intelligence,
        risk,
        entry_valid,
        timing_valid,
        execution_valid,
        SYSTEM_CIRCUIT_BREAKER.status(),
        trade_quality_valid=bool(quality.get("decision") == "TRADE"),
        engine_health_valid=bool(engine_health.get("status") == "HEALTHY"),
    )
    result["forex_entry"] = entry_candidate
    result["entry_zone_assessment"] = entry_zone
    result["strike_entry_validation"] = strike_validation
    result["entry_validation"] = entry_candidate
    result["engine_health"] = engine_health
    result["api_connection"] = APIConnectionManager.status(cfg)
    result["time_synchronization"] = TimeSynchronizationEngine.snapshot()
    result["execution_quality"] = exec_package
    result["final_decision"] = final_decision
    result["configuration_governance"] = ConfigurationRulesGovernanceLayer.snapshot(cfg)
    result["system_circuit_breaker"] = SYSTEM_CIRCUIT_BREAKER.status()
    result["decision_replay"] = TradeJournalDecisionReplayEngine.record(
        final_decision, central_assessment, signal_intelligence
    )
    return result




def fmt(v, n=2):
    try:
        return f"{float(v):,.{n}f}"
    except Exception:
        return str(v)


def warm_deriv_multi_pair_streams(symbols, cfg):
    """Ensure every scanner pair has its own authenticated Deriv stream.

    Each pair gets exactly one authenticated read-only WebSocket. That WebSocket
    supplies all supported timeframes (M1 through D1). Existing healthy streams are
    reused, so Streamlit reruns do not repeatedly authenticate the same pair.
    """
    if str(getattr(cfg, "data_source", "")).upper() != "DERIV REAL FOREX":
        return {}

    results = {}
    unique_symbols = [canonical_symbol(s) for s in symbols]
    unique_symbols = list(dict.fromkeys(unique_symbols))

    def start_one(sym):
        app_id, token, configured_account = DerivRealForexStream.resolve_credentials(
            getattr(cfg, "deriv_options_account_id", "")
        )
        stream = DerivRealForexStream.get(
            sym,
            cfg.deriv_api_base,
            app_id,
            token,
            configured_account,
            cfg.deriv_stream_outputsize,
            cfg.deriv_stream_reconnect_seconds,
        )
        snap = stream.snapshot()
        if not (snap.get("healthy") and stream.frames):
            snap = stream.start(wait_seconds=12)
        return sym, stream, snap

    # Keep concurrency deliberately modest so the Options-account OTP endpoint
    # is not hammered while still bringing the complete watchlist online quickly.
    with ThreadPoolExecutor(max_workers=min(2, max(len(unique_symbols), 1))) as pool:
        futures = {pool.submit(start_one, sym): sym for sym in unique_symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                pair, stream, snap = future.result()
                results[pair] = {
                    "ok": True,
                    "stream": stream,
                    "status": snap,
                }
            except Exception as exc:
                results[sym] = {
                    "ok": False,
                    "stream": None,
                    "status": {"error": str(exc), "healthy": False},
                }
    return results


def build_full_pair_signal_snapshot(sym, x, cfg, timeframe):
    """Run the V14 evidence → assessment → intelligence pipeline for one pair."""
    aa = analyze_market(x, sym, cfg, timeframe, deep_mtf=True)
    timing = SignalTimingEngine.assess(
        pd.Timestamp.now(tz="UTC"),
        cfg.signal_max_age_seconds,
    )
    live_gate = LiveTradeEligibilityEngine.evaluate(
        a=aa,
        ensemble=aa.get("ensemble", {}),
        no_trade=aa.get("no_trade", {}),
        trade_quality=aa.get("trade_quality", {}),
        data_quality=aa.get("data_quality", {}),
        timing=timing,
        emergency=False,
    )
    final_direction = str(
        aa.get("final_decision", {}).get("final_decision", "NO-TRADE")
    ).upper()
    signal = final_direction if live_gate.get("eligible") and final_direction in {"BUY", "SELL"} else "NO-TRADE"
    mtf = aa.get("mtf", {}) or {}
    return {
        "analysis": aa,
        "live_gate": live_gate,
        "signal": signal,
        "raw_direction": final_direction,
        "mtf_alignment": float(mtf.get("alignment", 0.0) or 0.0),
        "mtf_coverage": float(mtf.get("coverage_percent", 0.0) or 0.0),
        "data_quality": aa.get("data_quality", {}),
        "trade_quality": aa.get("trade_quality", {}),
    }


# ============================================================
# DASHBOARD
# ============================================================



# ============================================================
# V14 ADDITION: DERIV AUTHENTICATED REAL-FOREX LIVE STREAM (READ-ONLY)
# ============================================================
class DerivRealForexStream:
    """Authenticated, read-only Deriv real-Forex market-data stream.

    Proven authentication flow:
      1. Read DERIV_APP_ID + DERIV_API_TOKEN from Streamlit secrets.
      2. GET /trading/v1/options/accounts to discover the user's Options demo account.
      3. POST /trading/v1/options/accounts/{accountId}/otp to obtain a short-lived
         authenticated WebSocket URL.
      4. Connect directly to the returned URL (the OTP is already embedded).
      5. Subscribe to frx* ticks and bootstrap OHLC candles with ticks_history.

    The adapter is strictly read-only: it does not call buy, sell, proposal-buy,
    or any order/execution endpoint. The authenticated WebSocket is used for the
    proven Deriv Options account-scoped market-data channel.
    """
    TF_SECONDS = {"M1":60, "M3":180, "M5":300, "M15":900, "M30":1800, "H1":3600, "H4":14400, "D1":86400}
    _registry = {}
    _registry_lock = threading.RLock()

    @classmethod
    def symbol_map(cls):
        return {
            "EURUSD":"frxEURUSD", "GBPUSD":"frxGBPUSD", "USDJPY":"frxUSDJPY",
            "USDCHF":"frxUSDCHF", "AUDUSD":"frxAUDUSD", "USDCAD":"frxUSDCAD",
            "NZDUSD":"frxNZDUSD", "EURGBP":"frxEURGBP", "EURJPY":"frxEURJPY",
            "GBPJPY":"frxGBPJPY", "AUDCAD":"frxAUDCAD",
        }

    @staticmethod
    def _secret(name: str) -> str:
        try:
            return str(st.secrets.get(name, "")).strip()
        except Exception:
            return ""

    @classmethod
    def resolve_credentials(cls, configured_account_id: str = ""):
        app_id = cls._secret("DERIV_APP_ID")
        token = cls._secret("DERIV_API_TOKEN")
        account_id = str(configured_account_id or cls._secret("DERIV_OPTIONS_ACCOUNT_ID") or "").strip()
        if not app_id:
            raise RuntimeError("DERIV_APP_ID is not configured in Streamlit Secrets.")
        if not token:
            raise RuntimeError("DERIV_API_TOKEN is not configured in Streamlit Secrets.")
        return app_id, token, account_id

    @staticmethod
    def _response_json(response):
        try:
            return response.json()
        except Exception:
            return {"raw_response": response.text[:2000]}

    def _validate_active_symbols(self, rows):
        """Compatibility validator retained from the earlier provider adapter.

        The authenticated integration uses a fixed verified Forex symbol map and
        does not depend on an unauthenticated active_symbols request, but the
        method remains available so no prior adapter capability is silently removed.
        """
        for item in rows or []:
            if not isinstance(item, dict):
                continue
            sym = item.get("underlying_symbol", item.get("symbol"))
            market = str(item.get("market", "")).lower()
            typ = str(item.get("underlying_symbol_type", item.get("symbol_type", ""))).lower()
            if sym == self.deriv_symbol and market == "forex" and typ == "forex":
                return True
        return False

    @classmethod
    def discover_demo_account(cls, api_base: str, app_id: str, token: str, preferred_account_id: str = ""):
        headers = {
            "Deriv-App-ID": app_id,
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        try:
            response = requests.get(
                f"{api_base}/trading/v1/options/accounts",
                headers=headers,
                timeout=15,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Deriv Options account discovery request failed: {exc}") from exc

        body = cls._response_json(response)
        if not response.ok:
            # Never include credentials in the diagnostic error.
            raise RuntimeError(f"Deriv Options account discovery failed HTTP {response.status_code}: {body}")

        raw = body.get("data", [])
        accounts = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
        usable = [
            a for a in accounts
            if isinstance(a, dict)
            and str(a.get("account_type", "")).lower() == "demo"
            and str(a.get("status", "")).lower() == "active"
            and a.get("account_id")
        ]

        if preferred_account_id:
            for a in usable:
                if str(a.get("account_id")) == preferred_account_id:
                    return dict(a)
            raise RuntimeError(f"Configured DERIV_OPTIONS_ACCOUNT_ID was not found as an active demo Options account: {preferred_account_id}")

        if not usable:
            raise RuntimeError("Deriv authentication succeeded, but no active demo Options account was returned.")
        return dict(usable[0])

    @classmethod
    def request_authenticated_ws_url(cls, api_base: str, app_id: str, token: str, account_id: str):
        headers = {
            "Deriv-App-ID": app_id,
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        try:
            response = requests.post(
                f"{api_base}/trading/v1/options/accounts/{account_id}/otp",
                headers=headers,
                timeout=15,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Deriv OTP request failed: {exc}") from exc

        body = cls._response_json(response)
        if not response.ok:
            raise RuntimeError(f"Deriv OTP request failed HTTP {response.status_code}: {body}")
        data = body.get("data", {})
        ws_url = data.get("url") if isinstance(data, dict) else None
        if not isinstance(ws_url, str) or not ws_url.startswith("wss://"):
            raise RuntimeError("Deriv OTP response did not contain a usable authenticated WebSocket URL.")
        return ws_url

    def __init__(self, symbol, api_base, app_id, token, account_id="", outputsize=500, reconnect_seconds=5):
        if websocket is None:
            raise RuntimeError("websocket-client is not installed. Add websocket-client to requirements.txt.")
        self.symbol = canonical_symbol(symbol)
        self.deriv_symbol = self.symbol_map().get(self.symbol)
        if not self.deriv_symbol:
            raise RuntimeError(f"Deriv real-Forex mapping is unavailable for {self.symbol}.")
        self.api_base = str(api_base).rstrip("/")
        self.app_id = str(app_id)
        self.token = str(token)
        self.account_id = str(account_id or "")
        self.outputsize = int(outputsize)
        self.reconnect_seconds = int(reconnect_seconds)
        self.ws = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.frames = {}
        self.raw_ticks = []
        self.raw_tick_limit = max(5000, self.outputsize * 20)
        self.history_page_size = min(5000, max(1000, self.outputsize))
        self.history_max_pages = max(1, int(math.ceil(self.outputsize / float(self.history_page_size))) + 1)
        self.history_pages = {tf: 0 for tf in self.TF_SECONDS}
        self.history_oldest = {tf: None for tf in self.TF_SECONDS}
        self.status = {
            "connected":False,
            "authenticated":False,
            "validated":True,
            "source":"DERIV REAL FOREX",
            "symbol":self.symbol,
            "deriv_symbol":self.deriv_symbol,
            "account_type":"demo",
            "account_id":self.account_id,
            "last_tick":None,
            "last_quote":None,
            "error":None,
            "streaming":False,
            "synthetic":False,
            "read_only":True,
            "execution_enabled":False,
        }

    @classmethod
    def get(cls, symbol, api_base, app_id, token, account_id="", outputsize=500, reconnect_seconds=5):
        key = (canonical_symbol(symbol), str(account_id or ""))
        with cls._registry_lock:
            obj = cls._registry.get(key)
            if obj is None:
                obj = cls(key[0], api_base, app_id, token, account_id, outputsize, reconnect_seconds)
                cls._registry[key] = obj
            else:
                # Refresh non-secret configuration without replacing the live object.
                obj.api_base = str(api_base).rstrip("/")
                obj.app_id = str(app_id)
                obj.token = str(token)
                obj.outputsize = int(outputsize)
                obj.reconnect_seconds = int(reconnect_seconds)
            return obj

    def _send(self, payload):
        if self.ws is not None:
            self.ws.send(json.dumps(payload))

    def _on_open(self, ws):
        with self.lock:
            self.status.update({"connected":True,"authenticated":True,"streaming":False,"error":None})
        # The authenticated Options websocket accepts the same market-data
        # requests used by the proven diagnostic: live ticks + candle history.
        # Bootstrap real Deriv candle history in bounded backward pages.
        # The first page is newest; each subsequent page ends strictly before
        # the oldest candle already received. This keeps the entire historical
        # dataset on the same Deriv WebSocket market-data stream.
        with self.lock:
            for tf in self.TF_SECONDS:
                self.history_pages[tf] = 0
                self.history_oldest[tf] = None
        for idx, (tf, seconds) in enumerate(self.TF_SECONDS.items()):
            self._request_history_page(tf, seconds, end="latest")
        self._send({"ticks":self.deriv_symbol,"subscribe":1,"req_id":3001})

    def _request_history_page(self, timeframe: str, seconds: int, end="latest"):
        tf = str(timeframe).upper()
        with self.lock:
            page = int(self.history_pages.get(tf, 0))
            if page >= self.history_max_pages:
                return
            self.history_pages[tf] = page + 1
        # req_id encodes timeframe index + page for diagnostics/backward compatibility.
        tf_index = list(self.TF_SECONDS).index(tf)
        req_id = 2000 + tf_index * 100 + page
        self._send({
            "ticks_history": self.deriv_symbol,
            "count": self.history_page_size,
            "end": end,
            "style": "candles",
            "granularity": seconds,
            "req_id": req_id,
        })

    def _on_message(self, ws, raw):
        try:
            data = json.loads(raw)
        except Exception:
            return
        if data.get("error"):
            with self.lock:
                err = data.get("error") or {}
                self.status["error"] = err.get("message", "Deriv WebSocket error")
            return
        typ = data.get("msg_type")
        if typ == "candles":
            req = data.get("echo_req", {}) or {}
            req_id = int(req.get("req_id", 0) or 0)
            tfs = list(self.TF_SECONDS)
            idx = (req_id - 2000) // 100
            page = (req_id - 2000) % 100
            if 0 <= idx < len(tfs):
                tf = tfs[idx]
                rows = data.get("candles", []) or []
                frame = pd.DataFrame(rows)
                if not frame.empty:
                    frame = frame.rename(columns={"epoch":"time"})
                    frame["time"] = pd.to_datetime(pd.to_numeric(frame["time"], errors="coerce"), unit="s", utc=True)
                    for c in ["open","high","low","close"]:
                        frame[c] = pd.to_numeric(frame[c], errors="coerce")
                    frame["volume"] = 0.0
                    frame = frame[["time","open","high","low","close","volume"]].dropna()
                    # Route provider candles through the V14 OHLCV Builder.
                    # No values are synthesized; Deriv remains the sole source.
                    built = OHLCVBuilderEngine.build(frame, tf)
                    with self.lock:
                        existing = self.frames.get(tf)
                        combined = built if existing is None or existing.empty else pd.concat([existing, built], ignore_index=True)
                        combined = MarketDataEngine.normalize(combined).sort_values("time").drop_duplicates("time", keep="last")
                        self.frames[tf] = combined.tail(self.outputsize).reset_index(drop=True)
                        oldest = self.frames[tf]["time"].min() if not self.frames[tf].empty else None
                        self.history_oldest[tf] = oldest
                        have = len(self.frames[tf])
                    # Continue backward pagination until the requested target is reached.
                    if len(frame) > 0 and have < self.outputsize and page + 1 < self.history_max_pages:
                        oldest_page = frame["time"].min()
                        end_epoch = int(pd.Timestamp(oldest_page).timestamp()) - 1
                        self._request_history_page(tf, self.TF_SECONDS[tf], end=end_epoch)
            return
        if typ == "tick" and data.get("tick"):
            tick = data["tick"]
            try:
                quote = float(tick["quote"])
                epoch = int(tick["epoch"])
                if tick.get("symbol") != self.deriv_symbol or not math.isfinite(quote):
                    return
                ts = pd.Timestamp(epoch, unit="s", tz="UTC")
            except Exception:
                return
            with self.lock:
                self.status.update({"last_tick":ts.isoformat(),"last_quote":quote,"streaming":True})
                self.raw_ticks.append({"symbol": self.symbol, "price": quote, "timestamp": ts.isoformat(),
                                       "source": "DERIV REAL FOREX", "sequence": tick.get("id")})
                if len(self.raw_ticks) > self.raw_tick_limit:
                    self.raw_ticks = self.raw_ticks[-self.raw_tick_limit:]
                for tf, seconds in self.TF_SECONDS.items():
                    start = pd.Timestamp((epoch // seconds) * seconds, unit="s", tz="UTC")
                    frame = self.frames.get(tf)
                    if frame is None or frame.empty:
                        continue
                    if frame.iloc[-1]["time"] < start:
                        new = pd.DataFrame([{"time":start,"open":quote,"high":quote,"low":quote,"close":quote,"volume":0.0}])
                        frame = pd.concat([frame,new], ignore_index=True)
                    elif frame.iloc[-1]["time"] == start:
                        i = frame.index[-1]
                        frame.at[i,"high"] = max(float(frame.at[i,"high"]), quote)
                        frame.at[i,"low"] = min(float(frame.at[i,"low"]), quote)
                        frame.at[i,"close"] = quote
                    self.frames[tf] = frame.tail(self.outputsize).reset_index(drop=True)

    def _on_error(self, ws, error):
        with self.lock:
            self.status["error"] = str(error)
            self.status["connected"] = False
            self.status["authenticated"] = False

    def _on_close(self, ws, code, msg):
        with self.lock:
            self.status["connected"] = False
            self.status["authenticated"] = False
            self.status["streaming"] = False

    def _run(self):
        while not self.stop_event.is_set():
            try:
                # OTPs are short-lived and single-use. Request a fresh URL for
                # every reconnect instead of trying to reuse an expired OTP.
                if not self.account_id:
                    account = self.discover_demo_account(self.api_base, self.app_id, self.token)
                    self.account_id = str(account["account_id"])
                    with self.lock:
                        self.status["account_id"] = self.account_id
                ws_url = self.request_authenticated_ws_url(self.api_base, self.app_id, self.token, self.account_id)
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as ex:
                with self.lock:
                    self.status["error"] = str(ex)
                    self.status["connected"] = False
                    self.status["authenticated"] = False
                    self.status["streaming"] = False
                # If an account ID has become invalid, clear it so the next
                # cycle rediscoveries the active demo Options account.
                if "AccountNotFound" in str(ex) or "account not found" in str(ex).lower():
                    self.account_id = ""
            if not self.stop_event.is_set():
                _time.sleep(self.reconnect_seconds)

    def start(self, wait_seconds=20):
        if self.thread and self.thread.is_alive():
            snap = self.snapshot()
            if snap.get("healthy") or self.frames:
                return snap
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name=f"deriv-auth-forex-{self.symbol}", daemon=True)
        self.thread.start()
        deadline = _time.time() + wait_seconds
        while _time.time() < deadline:
            snap = self.snapshot()
            if snap.get("authenticated") and snap.get("streaming") and self.frames:
                return snap
            err = snap.get("error")
            if err:
                # Give the reconnect loop a brief chance to recover transient errors.
                if _time.time() + 0.5 >= deadline:
                    raise RuntimeError(err)
            _time.sleep(0.1)
        snap = self.snapshot()
        if snap.get("authenticated") and self.frames:
            return snap
        raise RuntimeError(f"Deriv authenticated stream did not produce usable candles within the startup window: {snap}")

    def frame(self, timeframe):
        with self.lock:
            df = self.frames.get(str(timeframe).upper())
            return None if df is None else df.copy()

    def ticks(self):
        with self.lock:
            return list(self.raw_ticks)

    def snapshot(self):
        with self.lock:
            s = dict(self.status)
        last = s.get("last_tick")
        if last:
            try:
                age = max(0.0, (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last)).total_seconds())
            except Exception:
                age = float("inf")
        else:
            age = float("inf")
        s["tick_age_seconds"] = round(age, 2) if math.isfinite(age) else None
        s["healthy"] = bool(s.get("connected") and s.get("authenticated") and s.get("validated") and s.get("streaming"))
        return s

    def stop(self):
        self.stop_event.set()
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
        with self.lock:
            self.status["connected"] = False
            self.status["authenticated"] = False
            self.status["streaming"] = False


def get_deriv_real_forex_data(symbol: str, timeframe: str, cfg: Config, force=False):
    """Return validated live Deriv real-Forex candles for the V14 data layer."""
    if websocket is None:
        raise RuntimeError("Deriv streaming requires websocket-client. Add websocket-client to requirements.txt.")
    symbol = canonical_symbol(symbol)
    timeframe = str(timeframe).upper()
    if symbol not in DerivRealForexStream.symbol_map():
        raise RuntimeError(f"No verified Deriv real-Forex mapping configured for {symbol}.")

    app_id, token, configured_account = DerivRealForexStream.resolve_credentials(
        getattr(cfg, "deriv_options_account_id", "")
    )
    stream = DerivRealForexStream.get(
        symbol,
        cfg.deriv_api_base,
        app_id,
        token,
        configured_account,
        cfg.deriv_stream_outputsize,
        cfg.deriv_stream_reconnect_seconds,
    )
    status = stream.start()
    df = stream.frame(timeframe)
    if df is None or df.empty:
        raise RuntimeError(f"Deriv authenticated stream did not produce {symbol}/{timeframe} candles. Diagnostic: {status}")

    # The Deriv historical bootstrap is retained, while the live tail is rebuilt
    # from the same authenticated Deriv ticks through the V14 OHLCV Builder.
    live_ticks = stream.ticks()
    tick_bars = OHLCVBuilderEngine.build_from_ticks(
        live_ticks, timeframe, symbol=symbol, source="DERIV REAL FOREX"
    )
    if not tick_bars.empty:
        df = OHLCVBuilderEngine._merge_provider_and_tick_bars(df, tick_bars)

    validation = MarketDataEngine.validate(df)
    if not validation.get("data_ok"):
        raise RuntimeError(f"Deriv {symbol}/{timeframe} input validation failed: {validation}")

    if not status.get("healthy") or (status.get("tick_age_seconds") is not None and status["tick_age_seconds"] > cfg.deriv_stream_stale_seconds):
        raise RuntimeError(f"Deriv authenticated real-Forex stream is stale/unhealthy for {symbol}: {status}")

    meta = {
        "symbol":symbol,
        "timeframe":timeframe,
        "source":"DERIV REAL FOREX",
        "provider_symbol":stream.deriv_symbol,
        "synthetic":False,
        "market":"FOREX",
        "underlying_symbol_type":"forex",
        "read_only":True,
        "execution_enabled":False,
        "streaming":True,
        "authenticated":True,
        "account_type":"demo",
        "account_id":stream.account_id,
        "stream_health":status,
        "fetched_at":datetime.now(timezone.utc).isoformat(),
    }
    return MarketDataEngine.normalize(df), meta

# ============================================================
# END DERIV AUTHENTICATED REAL-FOREX LIVE STREAM
# ============================================================



def _v14_json(value):
    """Safe dashboard renderer for nested engine packages."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {"value": value}


def _v14_metric_value(blob, *keys, default="—"):
    if not isinstance(blob, dict):
        return default
    for key in keys:
        if key in blob and blob[key] is not None:
            return blob[key]
    return default


def _v14_direction(blob):
    if not isinstance(blob, dict):
        return "NEUTRAL"
    for key in ("final_decision", "direction", "market_context", "bias"):
        value = str(blob.get(key, "")).upper()
        if value in {"BUY", "SELL", "BULLISH", "BEARISH", "NEUTRAL", "NO-TRADE", "WAIT"}:
            return value
    return "NEUTRAL"


def _v14_status(blob, default="UNKNOWN"):
    if not isinstance(blob, dict):
        return default
    for key in ("status", "state", "assessment_status", "validation_status", "health_status"):
        value = blob.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _v14_render_engine_ledger(analysis):
    """Render evidence packages safely whether they arrive as dicts or dataclasses."""
    packages = (
        analysis.get("engine_evidence_packages", {})
        if isinstance(analysis, dict)
        else {}
    )
    if not isinstance(packages, dict):
        packages = {}

    rows = []
    for name, raw_pkg in packages.items():
        # The architecture normally stores finalized dictionaries, but this
        # renderer also accepts an EngineEvidencePackage object so a partially
        # initialized/rerun Streamlit state can never crash the dashboard.
        if isinstance(raw_pkg, dict):
            pkg = raw_pkg
        else:
            try:
                pkg = asdict(raw_pkg)
            except Exception:
                pkg = {}

        ts = pkg.get("timestamps", {})
        if not isinstance(ts, dict):
            try:
                ts = asdict(ts)
            except Exception:
                ts = {}

        q = pkg.get("quality", {})
        if not isinstance(q, dict):
            q = {}

        ev = pkg.get("evidence", {})
        if not isinstance(ev, dict):
            try:
                ev = asdict(ev)
            except Exception:
                ev = {"value": str(ev)}

        output_ts = (
            pkg.get("output_timestamp")
            or ts.get("output_timestamp")
            or "—"
        )
        freshness = (
            ts.get("freshness_status")
            or ts.get("timestamp_validity")
            or pkg.get("freshness_status")
            or "—"
        )

        rows.append({
            "Engine": name,
            "Status": _v14_status(pkg),
            "Direction/Evidence": _v14_direction(ev),
            "Quality": q.get("score", "—"),
            "Freshness": freshness,
            "Data Age (s)": ts.get("data_age_seconds", "—"),
            "Output Timestamp": output_ts,
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No engine evidence packages are available.")


def _v14_render_architecture_testing(df, cfg, symbol, timeframe):
    """Render the corrected V14 architecture-level testing tools."""
    st.markdown("### 🧪 V14 Architecture Backtest / Forward Test")
    st.caption("If the loaded real dataset is below the statistical minimum, V14 can expand it from Twelve Data using bounded historical pagination. No synthetic candles are created.")
    st.caption(
        "Tests the actual V14 decision path chronologically. These V14 Backtest and V14 Forward Test tools are the authoritative testing path. Results with small trade counts are evidence-insufficient, not proof of accuracy."
    )
    hb = st.session_state.get("historical_builder_status", {}) or {}
    if hb:
        st.info(f"Historical dataset: {hb.get('rows', 0):,} candles · status={hb.get('status', 'UNKNOWN')} · requests={hb.get('requests_used', 0)}")
    t0, t1, t2, t3 = st.columns(4)
    with t0:
        if st.button("📚 Build Historical Dataset", use_container_width=True):
            try:
                with st.spinner("Building validated historical dataset from the selected real provider…"):
                    built, build_meta = ensure_historical_test_dataset(df, cfg, symbol, timeframe)
                if build_meta.get("status") == "READY":
                    built, stored_meta = store_market_data(symbol, timeframe, built, build_meta, build_meta.get("source", cfg.data_source))
                    st.success(f"Historical dataset ready: {len(built):,} valid candles.")
                    st.session_state.historical_builder_status = build_meta
                else:
                    st.session_state.historical_builder_status = build_meta
                    st.warning(build_meta.get("message", "Historical dataset is insufficient."))
            except Exception as ex:
                st.session_state.historical_builder_status = {"status":"ERROR", "message":str(ex)}
                st.error(f"Historical dataset builder error: {ex}")
    with t1:
        if st.button("▶ Run V14 Backtest", use_container_width=True, disabled=df.empty):
            with st.spinner("Preparing validated historical dataset…"):
                test_df, prep = ensure_historical_test_dataset(df, cfg, symbol, timeframe)
            if prep.get("status") != "READY":
                st.session_state.v14_backtest = pd.DataFrame()
                st.session_state.v14_backtest_metrics = {"status":"INSUFFICIENT_REAL_HISTORY", **prep}
                st.warning(prep.get("message", "Insufficient real historical data for statistical testing."))
            else:
                with st.spinner("Running V14 chronological backtest…"):
                    trades, metrics = V14BacktestEngine.run(test_df, cfg, symbol, timeframe)
                st.session_state.v14_backtest = trades
                st.session_state.v14_backtest_metrics = metrics
    with t2:
        if st.button("↔ Run V14 Forward Test", use_container_width=True, disabled=df.empty):
            with st.spinner("Preparing validated historical dataset…"):
                test_df, prep = ensure_historical_test_dataset(df, cfg, symbol, timeframe)
            if prep.get("status") != "READY":
                st.session_state.v14_forward = {"out_of_sample":{"status":"INSUFFICIENT_REAL_HISTORY", **prep}}
                st.warning(prep.get("message", "Insufficient real historical data for statistical testing."))
            else:
                with st.spinner("Running V14 chronological forward test…"):
                    result = V14ForwardTestingEngine.run(test_df, cfg, symbol, timeframe)
                st.session_state.v14_forward = result
    with t3:
        if st.button("🧹 Clear V14 Test Results", use_container_width=True):
            st.session_state.v14_backtest = pd.DataFrame()
            st.session_state.v14_backtest_metrics = {}
            st.session_state.v14_forward = {}

    bm = st.session_state.get("v14_backtest_metrics", {}) or {}
    if bm:
        st.markdown("#### V14 Backtest")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trades", bm.get("trades", 0))
        m2.metric("Win Rate", f'{bm.get("win_rate", 0):.2f}%')
        m3.metric("Profit Factor", f'{bm.get("profit_factor", 0):.2f}' if np.isfinite(float(bm.get("profit_factor", 0) or 0)) else "∞")
        m4.metric("Net Profit", f'${float(bm.get("net_profit", 0) or 0):,.2f}')
        m5.metric("Max Drawdown", f'{float(bm.get("max_drawdown", 0) or 0):.2f}%')
        st.json(bm)
        sv = bm.get("statistical_validation") or {}
        if sv:
            if sv.get("status") == "STATISTICALLY_VALID":
                st.success("Statistical sample threshold passed; this still does not guarantee future profitability.")
            else:
                st.warning("Backtest evidence is statistically too small to claim high accuracy: " + "; ".join(sv.get("warnings", [])))
        if bm.get("rejection_counts"):
            st.markdown("**Backtest rejection diagnostics**")
            st.dataframe(
                pd.DataFrame(
                    [{"Reason": k, "Rejected Bars": v} for k, v in bm["rejection_counts"].items()]
                ),
                use_container_width=True,
                hide_index=True,
            )
        bt = st.session_state.get("v14_backtest", pd.DataFrame())
        if isinstance(bt, pd.DataFrame) and not bt.empty:
            chart = bt.copy()
            if "exit_time" in chart.columns:
                chart["time"] = chart["exit_time"].fillna(chart["time"])
            if "equity" in chart.columns and "time" in chart.columns:
                st.line_chart(chart.set_index("time")[["equity"]])
            st.dataframe(bt.tail(100), use_container_width=True, hide_index=True)

    fw = st.session_state.get("v14_forward", {}) or {}
    if fw:
        st.markdown("#### V14 Forward Test — Out of Sample")
        train_m = fw.get("train", {}) or {}
        oos_m = fw.get("out_of_sample", {}) or {}
        if "error" in oos_m:
            st.error(oos_m["error"])
        else:
            f1, f2, f3, f4, f5 = st.columns(5)
            f1.metric("OOS Trades", oos_m.get("trades", 0))
            f2.metric("OOS Win Rate", f'{float(oos_m.get("win_rate", 0) or 0):.2f}%')
            pf = float(oos_m.get("profit_factor", 0) or 0)
            f3.metric("OOS Profit Factor", f'{pf:.2f}' if np.isfinite(pf) else "∞")
            f4.metric("OOS Net Profit", f'${float(oos_m.get("net_profit", 0) or 0):,.2f}')
            f5.metric("OOS Max DD", f'{float(oos_m.get("max_drawdown", 0) or 0):.2f}%')
            st.caption(
                f"Train rows: {train_m.get('rows_tested', train_m.get('trades', '—'))} · "
                f"OOS rows: {oos_m.get('oos_rows', '—')} · "
                f"Split: {fw.get('split_timestamp', '—')} · "
                "Pre-split context is retained; no OOS trade is evaluated before the split."
            )
            st.json({"train": train_m, "out_of_sample": oos_m})
            sv = oos_m.get("statistical_validation") or {}
            if sv:
                if sv.get("status") == "STATISTICALLY_VALID":
                    st.success("OOS sample threshold passed; this is evidence, not a guarantee of future performance.")
                else:
                    st.warning("OOS evidence is statistically too small to claim high accuracy: " + "; ".join(sv.get("warnings", [])))
            if oos_m.get("rejection_counts"):
                st.markdown("**Forward-test OOS rejection diagnostics**")
                st.dataframe(
                    pd.DataFrame(
                        [{"Reason": k, "Rejected OOS Bars": v} for k, v in oos_m["rejection_counts"].items()]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            ot = fw.get("test_trades", pd.DataFrame())
            if isinstance(ot, pd.DataFrame) and not ot.empty:
                st.dataframe(ot.tail(100), use_container_width=True, hide_index=True)

    # Monte Carlo and optimizer consume the authoritative V14 testing results.
    st.markdown("#### V14 Stress / Optimization")
    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button(
            "🎲 Run Monte Carlo",
            use_container_width=True,
            disabled=not isinstance(st.session_state.get("v14_backtest"), pd.DataFrame)
            or st.session_state.get("v14_backtest", pd.DataFrame()).empty,
        ):
            st.session_state.mc = MonteCarloEngine.run(st.session_state.v14_backtest)
        if st.session_state.get("mc"):
            st.json(st.session_state.mc)
    with r2:
        if st.button("⚙ Run V14 Threshold Optimizer", use_container_width=True, disabled=df.empty):
            with st.spinner("Running V14 threshold optimization…"):
                st.session_state.optimizer = OptimizerEngine.run(df, cfg)
        if not st.session_state.get("optimizer", pd.DataFrame()).empty:
            st.dataframe(st.session_state.optimizer, use_container_width=True, hide_index=True)
    with r3:
        if st.button("🧪 Run Robustness Stress Test", use_container_width=True, disabled=df.empty):
            with st.spinner("Running threshold/slippage robustness test…"):
                st.session_state.v14_robustness = V14RobustnessValidationEngine.run(df, cfg, symbol, timeframe)
        robust = st.session_state.get("v14_robustness", pd.DataFrame())
        if isinstance(robust, pd.DataFrame) and not robust.empty:
            st.dataframe(robust, use_container_width=True, hide_index=True)
            st.caption("Diagnostic sensitivity only. Do not tune the untouched OOS set with these results.")


def dashboard():
    """
    V14 AI Trading Dashboard.

    This is the primary V14 dashboard using the newly designed 18-panel layout.
    V14 Backtest and V14 Forward Test are the authoritative testing path.
    """
    st.set_page_config(
        page_title="V14 AI Forex Trading Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    cfg = Config()

    # ---------------------------
    # V14 control plane
    # ---------------------------
    with st.sidebar:
        st.header("⚙️ V14 Control Center")
        pair_options = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD", "AUD/CAD",
            "EUR/GBP", "EUR/JPY", "GBP/JPY", "XAU/USD",
        ]
        selected_pair_label = st.selectbox("Primary currency pair", pair_options, index=0)
        symbol = canonical_symbol(selected_pair_label)

        timeframe = st.selectbox(
            "Primary timeframe",
            ["M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1"],
            index=2,
        )

        source_options = ["DEMO", "TWELVE DATA", "DERIV REAL FOREX"]
        configured_td_key = _get_twelve_data_key(cfg)
        default_source_index = 1 if configured_td_key else 0
        data_source = st.selectbox("Primary data source", source_options, index=default_source_index)
        cfg.data_source = data_source

        cfg.initial_balance = st.number_input(
            "Paper balance", 100.0, 10000000.0, 10000.0, 100.0
        )
        st.session_state.paper_balance = cfg.initial_balance
        cfg.current_equity = float(st.session_state.paper_balance)
        cfg.risk_per_trade = st.slider("Risk / trade %", 0.1, 2.0, 0.5, 0.1) / 100
        cfg.min_score = st.slider("Minimum confluence score", 50, 95, 72)
        cfg.max_spread_pips = st.slider("Max spread (pips)", 0.2, 10.0, 2.0, 0.1)
        cfg.allow_live_source_failover = st.checkbox(
            "Allow live source failover",
            value=True,
            help="Only genuine live feeds may fail over. Synthetic data is never used as a live fallback.",
        )

        if data_source == "TWELVE DATA":
            cfg.twelve_data_api_key = st.text_input(
                "Twelve Data API key",
                type="password",
                value=st.session_state.get("td_key", configured_td_key),
            )
            st.session_state.td_key = cfg.twelve_data_api_key
            cfg.twelve_data_outputsize = st.number_input(
                "Initial historical candles", 200, 5000, 5000, 100
            )
            if st.button("🔄 Refresh Twelve Data", use_container_width=True):
                try:
                    live_df, meta = _fetch_twelve_data(symbol, timeframe, cfg)
                    live_df, meta = store_market_data(
                        symbol, timeframe, live_df, meta, "TWELVE DATA"
                    )
                    st.session_state.live_status = LiveConnectionManager.status(
                        "TWELVE DATA",
                        live_df,
                        MarketDataEngine.validate(live_df),
                        DataIntegrityEngine.assess(
                            live_df, timeframe, cfg.data_max_age_seconds
                        ),
                        meta,
                    )
                except Exception as ex:
                    st.session_state.live_status = {
                        "source": "TWELVE DATA",
                        "connected": False,
                        "read_only": True,
                        "execution_enabled": False,
                        "error": str(ex),
                    }
                    st.error(f"Twelve Data error: {ex}")


        elif data_source == "DERIV REAL FOREX":
            cfg.deriv_stream_enabled = True
            cfg.deriv_stream_outputsize = st.number_input(
                "Deriv historical candles", 1000, 20000, 12000, 100
            )
            cfg.deriv_stream_stale_seconds = st.number_input(
                "Live tick stale limit (seconds)", 5, 120, 15, 1
            )
            st.caption(
                "Authenticated Deriv real-Forex stream · read-only · "
                "PAT credentials are read from Streamlit Secrets."
            )
            if st.button("🟢 Connect Deriv Real-Forex Stream", use_container_width=True):
                try:
                    live_df, meta = get_deriv_real_forex_data(
                        symbol, timeframe, cfg, force=True
                    )
                    live_df, meta = store_market_data(
                        symbol, timeframe, live_df, meta, "DERIV REAL FOREX"
                    )
                    dq0 = DataIntegrityEngine.assess(
                        live_df, timeframe, cfg.data_max_age_seconds
                    )
                    validation0 = MarketDataEngine.validate(live_df)
                    st.session_state.live_status = LiveConnectionManager.status(
                        "DERIV REAL FOREX", live_df, validation0, dq0, meta
                    )
                    st.session_state.live_status.update(meta.get("stream_health", {}))
                except Exception as ex:
                    st.session_state.live_status = {
                        "source": "DERIV REAL FOREX",
                        "connected": False,
                        "read_only": True,
                        "execution_enabled": False,
                        "error": str(ex),
                        "synthetic": False,
                    }
                    st.error(f"Deriv stream error: {ex}")

        st.divider()
        st.write("**Safety / execution**")
        st.session_state.bot_enabled = st.toggle(
            "Enable paper bot", st.session_state.bot_enabled
        )
        if st.button("⏸ Pause New Trades", use_container_width=True):
            st.session_state.bot_enabled = False
        if st.button("🚨 EMERGENCY STOP", use_container_width=True):
            st.session_state.emergency = True
            st.session_state.bot_enabled = False
            SYSTEM_CIRCUIT_BREAKER.halt("MANUAL_EMERGENCY_STOP")
        if st.button("Reset Emergency", use_container_width=True):
            st.session_state.emergency = False
            SYSTEM_CIRCUIT_BREAKER.release()

        auto_refresh = st.checkbox(
            "Auto refresh",
            value=False,
            help="Refreshes the dashboard only; it never creates a trading decision by itself.",
        )
        refresh_seconds = st.number_input("Refresh interval (seconds)", 15, 3600, 60, 15)

    st.title("V14 AI Forex Trading Dashboard")
    st.caption(
        "V14 18-panel dashboard · specialist evidence → Central Assessment → "
        "Signal Intelligence → Final Decision → display"
    )

    # ---------------------------
    # Exact pair/timeframe data lock
    # ---------------------------
    upload = st.file_uploader("Upload OHLCV CSV", type=["csv"])
    if upload is not None:
        try:
            csv_df = MarketDataEngine.normalize(pd.read_csv(upload))
            store_market_data(
                symbol,
                timeframe,
                csv_df,
                {"source": "CSV", "symbol": symbol, "timeframe": timeframe},
                "CSV",
            )
        except Exception as ex:
            st.error(f"CSV error: {ex}")

    try:
        df, resolved_meta = get_selected_market_data(symbol, timeframe, cfg)
    except Exception as ex:
        clear_market_data_identity(clear_candles=True)
        st.session_state.live_status = {
            "source": cfg.data_source,
            "connected": False,
            "read_only": True,
            "execution_enabled": False,
            "error": str(ex),
        }
        df, resolved_meta = _empty_market_data(), {}

    validation = (
        MarketDataEngine.validate(df)
        if not df.empty
        else {
            "rows": 0,
            "data_ok": False,
            "duplicates_removed": 0,
            "missing_ohlc": 0,
            "large_gaps": 0,
            "timezone": "UTC",
        }
    )
    data_quality = DataIntegrityEngine.assess(
        df, timeframe, cfg.data_max_age_seconds
    )

    if not df.empty:
        try:
            analysis = analyze_market(df, symbol, cfg, timeframe, deep_mtf=True)
        except Exception as ex:
            analysis = build_no_data_analysis(
                symbol, cfg, data_quality, f"ANALYSIS ERROR: {ex}"
            )
            st.error(f"Analysis error: {ex}")
    else:
        analysis = build_no_data_analysis(
            symbol, cfg, data_quality, "NO VERIFIED MARKET DATA"
        )

    resolved_source = str(
        st.session_state.get("data_meta", {}).get(
            "source", cfg.data_source
        )
    ).upper()
    loaded_symbol = canonical_symbol(st.session_state.get("data_symbol", ""))
    loaded_tf = str(
        st.session_state.get("data_timeframe", "") or ""
    ).upper()
    sync_ok = bool(
        not df.empty
        and loaded_symbol == canonical_symbol(symbol)
        and loaded_tf == str(timeframe).upper()
        and resolved_source
    )

    if sync_ok:
        st.success(
            f"🔗 DATA SYNC LOCKED · {display_symbol(symbol)}/{timeframe} · {resolved_source}"
        )
    else:
        st.error(
            f"⛔ DATA SYNC BLOCKED · selected {display_symbol(symbol)}/{timeframe} · "
            f"loaded {display_symbol(loaded_symbol or 'NONE')}/{loaded_tf or 'NONE'}"
        )


    # ---------------------------
    # 18-panel V14 dashboard
    # ---------------------------
    panels = st.tabs(
        [
            "01 Market Overview",
            "02 Live Price",
            "03 Chart",
            "04 Trend & Structure",
            "05 Volatility & Liquidity",
            "06 MTF Analysis",
            "07 Entry Zone",
            "08 Signal Intelligence",
            "09 AI Probability",
            "10 Risk",
            "11 News/Event",
            "12 Signal",
            "13 System Health",
            "14 Performance",
            "15 Bot Control",
            "16 Data Source",
            "17 Alerts",
            "18 Audit / Activity",
        ]
    )

    central = analysis.get("central_assessment", {})
    intelligence = analysis.get("signal_intelligence", {})
    final_decision = analysis.get("final_decision", {})
    entry = analysis.get("entry_validation", analysis.get("forex_entry", {}))
    timing = analysis.get("signal_timing", {})
    execution = analysis.get("execution_quality", {})
    risk = analysis.get("risk", {})
    ai = analysis.get("ai", {})
    mtf = analysis.get("mtf", {})
    volatility = analysis.get("volatility", {})
    liquidity = analysis.get("liquidity", {})
    trend = analysis.get("trend", {})
    structure = analysis.get("structure", {})
    sr = analysis.get("sr", {})
    economic = analysis.get("economic", {})
    tracker = analysis.get("market_tracker", {})
    circuit = analysis.get("system_circuit_breaker", {})
    governance = analysis.get("configuration_governance", {})
    architecture_audit = V14ArchitectureAuditEngine.run(Path(__file__).read_text(errors="ignore"))

    last_price = float(df.close.iloc[-1]) if not df.empty else None
    bid = ask = spread = None
    tick = st.session_state.get("data_meta", {}).get("tick", {})
    if isinstance(tick, dict):
        bid, ask, spread = tick.get("bid"), tick.get("ask"), tick.get("spread")

    with panels[0]:
        st.subheader("Market Overview")
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Pair", display_symbol(symbol))
        a2.metric("Timeframe", timeframe)
        a3.metric("Direction", _v14_direction(intelligence))
        a4.metric("Central Readiness", "READY" if central.get("assessment_readiness") else "BLOCKED")
        a5.metric("Final Decision", final_decision.get("final_decision", "NO-TRADE"))
        st.caption(
            f"Rows: {len(df):,} · Source: {resolved_source or 'NONE'} · "
            f"Data quality: {data_quality.get('status', 'UNKNOWN')} "
            f"({data_quality.get('score', 0):.0f}/100)"
        )

    with panels[1]:
        st.subheader("Live Price")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Last", fmt(last_price) if last_price is not None else "—")
        p2.metric("Bid", fmt(bid) if bid is not None else "—")
        p3.metric("Ask", fmt(ask) if ask is not None else "—")
        p4.metric("Spread", fmt(spread) if spread is not None else "—")
        if not df.empty:
            st.caption(f"Latest candle: {df.time.iloc[-1]} UTC")

    with panels[2]:
        st.subheader("Chart")
        if not df.empty:
            chart_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
            st.line_chart(df.set_index("time")[chart_cols[-1:]])
            with st.expander("Latest OHLCV"):
                st.dataframe(df.tail(100), use_container_width=True, hide_index=True)
        else:
            st.warning("No verified candles available.")

    with panels[3]:
        st.subheader("Trend & Structure")
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Trend", _v14_direction(trend))
        x2.metric("Trend Strength", fmt(_v14_metric_value(trend, "strength", default=0), 1))
        x3.metric("Structure", _v14_direction(structure))
        x4.metric("Structure State", _v14_status(structure))
        st.json({"trend": trend, "EMA": analysis.get("ema", {}), "MACD": analysis.get("macd", {}),
                 "structure": structure, "price_action": analysis.get("price_action", {}),
                 "reversal": analysis.get("reversal", {})})

    with panels[4]:
        st.subheader("Volatility & Liquidity")
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Volatility Regime", _v14_metric_value(volatility, "regime"))
        x2.metric("ATR", fmt(_v14_metric_value(volatility, "atr")))
        x3.metric("Liquidity High", fmt(_v14_metric_value(liquidity, "liquidity_high")))
        x4.metric("Liquidity Low", fmt(_v14_metric_value(liquidity, "liquidity_low")))
        st.json({"volatility": volatility, "liquidity": liquidity, "support_resistance": sr,
                 "FVG": analysis.get("fvg", {}), "microstructure": analysis.get("microstructure", {}),
                 "order_flow": analysis.get("order_flow", {})})

    with panels[5]:
        st.subheader("MTF Analysis")
        st.json(_v14_json(mtf))
        st.caption("MTF consumes existing specialized-engine outputs; it does not replace their calculations.")

    with panels[6]:
        st.subheader("Entry Zone")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Preferred Entry", fmt(_v14_metric_value(entry, "entry", "preferred_entry")))
        e2.metric("Zone Low", fmt(_v14_metric_value(entry, "zone_low", "low")))
        e3.metric("Zone High", fmt(_v14_metric_value(entry, "zone_high", "high")))
        e4.metric("R:R", fmt(_v14_metric_value(entry, "rr"), 2))
        st.json(_v14_json(entry))

    with panels[7]:
        st.subheader("Signal Intelligence")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Status", intelligence.get("status", "BLOCKED"))
        s2.metric("Context", intelligence.get("market_context", "NEUTRAL"))
        s3.metric("Score", fmt(intelligence.get("overall_intelligence_score", 0), 1))
        s4.metric("Readiness", "YES" if intelligence.get("decision_readiness") else "NO")
        st.json(intelligence)

    with panels[8]:
        st.subheader("AI Probability")
        probs = {
            "AI": ai,
            "Direct Probability": analysis.get("direct_probability", {}),
            "Ensemble": analysis.get("ensemble", {}),
        }
        st.json(probs)

    with panels[9]:
        st.subheader("Risk")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Risk Status", "APPROVED" if risk.get("approved") else "BLOCKED")
        r2.metric("Risk Score", fmt(risk.get("risk_score", 0), 1))
        r3.metric("Risk Level", risk.get("risk_level", "UNKNOWN"))
        r4.metric("Emergency", "STOPPED" if st.session_state.emergency else "NORMAL")
        st.json(risk)

    with panels[10]:
        st.subheader("News / Event")
        n1, n2, n3 = st.columns(3)
        n1.metric("Event Status", _v14_status(economic))
        n2.metric("Blocked", "YES" if economic.get("blocked") else "NO")
        n3.metric("Session", analysis.get("session", {}).get("session", "UNKNOWN"))
        st.json({"economic_calendar": economic, "news_event_filter": analysis.get("news_filter", {})})
        st.caption("Economic Calendar/Event data and News/Event filtering remain separate responsibilities.")

    with panels[11]:
        st.subheader("Final Signal")
        d = final_decision.get("final_decision", "NO-TRADE")
        st.metric("FINAL DECISION", d)
        st.caption(
            f"Direction: {final_decision.get('direction', 'NONE')} · "
            f"Reason: {final_decision.get('decision_explanation', '—')} · "
            f"Decision ID: {final_decision.get('decision_id', '—')}"
        )
        st.json(final_decision)
        if d == "NO-TRADE":
            st.warning("NO-TRADE is fail-closed unless every required final gate passes.")

    with panels[12]:
        st.subheader("System Health")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Central Assessment", "READY" if central.get("assessment_readiness") else "BLOCKED")
        h2.metric("Circuit Breaker", "HALTED" if circuit.get("halted") else "ARMED")
        h3.metric("Data Validation", validation.get("data_ok", False))
        h4.metric("Timestamp", central.get("output_timestamp", "—"))
        st.json({"Engine Health": analysis.get("engine_health", {}),
                 "API Connection": analysis.get("api_connection", {}),
                 "Time Sync": analysis.get("time_synchronization", {})})
        _v14_render_engine_ledger(analysis)
        st.markdown("### V14 Architecture Audit")
        st.json(architecture_audit)

    with panels[13]:
        st.subheader("Performance")
        open_trades = [
            x for x in st.session_state.journal
            if x.get("status") == "OPEN"
        ]
        closed_trades = [
            x for x in st.session_state.journal
            if x.get("status") != "OPEN"
        ]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Balance", f"${st.session_state.paper_balance:,.2f}")
        p2.metric("Open Trades", len(open_trades))
        p3.metric("Closed Trades", len(closed_trades))
        p4.metric("V14 Backtest Trades", len(st.session_state.get("v14_backtest", pd.DataFrame())))
        if st.session_state.get("v14_backtest_metrics"):
            st.json(st.session_state.get("v14_backtest_metrics"))

        # V14 testing is a first-class dashboard capability; these are the
        # authoritative V14 tools and are not the removed legacy test engines.
        _v14_render_architecture_testing(df, cfg, symbol, timeframe)

    with panels[14]:
        st.subheader("Bot Control")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Paper Bot", "ON" if st.session_state.bot_enabled else "PAUSED")
        c2.metric("Execution", "DISABLED")
        c3.metric("Read Only", "YES")
        c4.metric("Circuit", "HALTED" if circuit.get("halted") else "ARMED")
        st.info(
            "V14 remains fail-closed for live execution. The existing paper-trading "
            "and execution adapters are retained, but the new dashboard does not bypass safety controls."
        )

    with panels[15]:
        st.subheader("Data Source")
        st.json({
            "requested_source": cfg.data_source,
            "resolved_source": resolved_source,
            "symbol": symbol,
            "timeframe": timeframe,
            "loaded_symbol": loaded_symbol,
            "loaded_timeframe": loaded_tf,
            "validation": validation,
            "data_quality": data_quality,
            "live_status": st.session_state.live_status,
            "configuration_governance": governance,
        })

    with panels[16]:
        st.subheader("Alerts")
        alerts = []
        if not sync_ok:
            alerts.append("DATA SYNC BLOCKED")
        if not data_quality.get("signal_allowed", False):
            alerts.append("DATA QUALITY GATE BLOCKED")
        if not central.get("assessment_readiness", False):
            alerts.append("CENTRAL ASSESSMENT NOT READY")
        if not intelligence.get("decision_readiness", False):
            alerts.append("SIGNAL INTELLIGENCE NOT READY")
        if not timing.get("fresh", False):
            alerts.append("SIGNAL TIMING STALE")
        if circuit.get("halted"):
            alerts.append("SYSTEM CIRCUIT BREAKER HALTED")
        if not alerts:
            st.success("No active dashboard-level safety alerts.")
        else:
            for alert in alerts:
                st.warning(alert)

    with panels[17]:
        st.subheader("Audit / Activity")
        audit = {
            "architecture_version": getattr(cfg, "architecture_version", "V14-FROZEN-ARCH"),
            "symbol": symbol,
            "timeframe": timeframe,
            "data_source": resolved_source,
            "data_timestamp": central.get("data_timestamp"),
            "calculation_timestamp": central.get("calculation_timestamp"),
            "output_timestamp": central.get("output_timestamp"),
            "final_decision": final_decision,
            "central_assessment": central,
            "signal_intelligence": intelligence,
            "execution_quality": execution,
            "signal_timing": timing,
            "market_tracker": tracker,
            "auto_refresh": AutoRefreshEngine.configuration(auto_refresh, refresh_seconds),
        }
        st.json(audit)
        if st.session_state.journal:
            st.dataframe(
                pd.DataFrame(st.session_state.journal).tail(100),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("📚 Complete Engine Output Ledger", expanded=False):
        _v14_render_engine_ledger(analysis)

    if auto_refresh:
        # Streamlit's rerun is orchestration only; it does not manufacture data
        # or create a new trading decision independently.
        st.caption(f"Auto refresh enabled · interval {refresh_seconds}s")
        try:
            st.autorefresh(interval=int(refresh_seconds * 1000), key="v14_auto_refresh")
        except Exception:
            st.caption("Automatic refresh is unavailable in this Streamlit version; use the browser refresh control.")


if __name__ == "__main__":
    dashboard()
