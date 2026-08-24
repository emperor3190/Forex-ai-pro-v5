import json
import time
from datetime import datetime, timezone
import streamlit as st
import websocket

ENDPOINT = "wss://ws.binaryws.com/websockets/v3"
TARGET_SYMBOL = "frxEURUSD"
TEST_SECONDS = 30

st.set_page_config(page_title="Deriv Real Forex Test", page_icon="📡")
st.title("📡 Deriv REAL Forex — Read-Only Live Test")
st.warning("Diagnostic only. No trading/account execution. Synthetic/derived instruments are rejected.")

def utc_time():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def normalize(item):
    return {
        "symbol": item.get("underlying_symbol") or item.get("symbol"),
        "name": item.get("underlying_symbol_name") or item.get("display_name"),
        "market": item.get("market"),
        "symbol_type": item.get("underlying_symbol_type") or item.get("symbol_type"),
        "submarket": item.get("submarket"),
    }

def show_error(title, detail=None):
    st.error(title)
    if detail is not None:
        st.code(json.dumps(detail, indent=2, default=str), language="json")

def run_test():
    ws = None
    try:
        st.subheader("1. Connection")
        ws = websocket.create_connection(ENDPOINT, timeout=10, enable_multithread=True)
        st.success("✅ Connected to Deriv public market-data WebSocket")

        st.subheader("2. Real-Forex Symbol Validation")
        ws.send(json.dumps({
            "active_symbols": "brief",
            "product_type": "basic",
            "req_id": 1001
        }))

        response = None
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                data = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            if data.get("error"):
                show_error("❌ Deriv rejected active_symbols.", data["error"])
                return
            if data.get("msg_type") == "active_symbols":
                response = data
                break

        if response is None:
            show_error("❌ No active_symbols response received within 10 seconds.")
            return

        target = None
        for raw in response.get("active_symbols", []):
            item = normalize(raw)
            if item["symbol"] == TARGET_SYMBOL:
                target = item
                break

        if target is None:
            show_error("❌ frxEURUSD was not returned by Deriv.")
            return

        st.json(target)

        market_ok = target["market"] == "forex"
        type_ok = target["symbol_type"] == "forex"

        if market_ok and type_ok:
            st.success("✅ REAL FOREX VALIDATION PASSED")
        else:
            show_error("❌ REAL FOREX GATE FAILED — symbol rejected.", target)
            return

        st.subheader("3. Live EUR/USD Tick Subscription")
        request = {"ticks": TARGET_SYMBOL, "subscribe": 1, "req_id": 1002}
        st.code(json.dumps(request, indent=2), language="json")
        ws.send(json.dumps(request))

        status = st.empty()
        quote_box = st.empty()
        table_box = st.empty()
        ticks = []
        start = time.time()
        status.info(f"Waiting up to {TEST_SECONDS} seconds for live EUR/USD ticks...")

        while time.time() - start < TEST_SECONDS:
            try:
                data = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue

            if data.get("error"):
                show_error("❌ Deriv rejected the EUR/USD tick subscription.",
                           {"error": data["error"], "request": request})
                return

            if data.get("msg_type") != "tick":
                continue

            tick = data.get("tick") or {}
            if tick.get("symbol") != TARGET_SYMBOL:
                continue

            if tick.get("quote") is None or tick.get("epoch") is None:
                continue

            ticks.append({
                "symbol": tick["symbol"],
                "quote": tick["quote"],
                "epoch": tick["epoch"],
                "received_utc": utc_time()
            })

            if len(ticks) == 1:
                status.success("🎯 FIRST LIVE EUR/USD TICK RECEIVED")

            quote_box.metric("LIVE EUR/USD QUOTE", str(tick["quote"]))
            table_box.dataframe(ticks[-10:], use_container_width=True, hide_index=True)

            if len(ticks) >= 3:
                break

        if not ticks:
            status.error(f"❌ No live EUR/USD tick received within {TEST_SECONDS} seconds.")
            st.code(json.dumps({
                "endpoint": ENDPOINT,
                "symbol": TARGET_SYMBOL,
                "subscription": request,
                "real_forex_validation": True,
                "ticks_received": 0
            }, indent=2), language="json")
            return

        st.subheader("4. Final Diagnostic")
        if len(ticks) >= 3:
            st.success("🟢 SUCCESS — Deriv REAL Forex EUR/USD live tick stream is working.")
        else:
            st.warning("🟡 EUR/USD live data was received, but fewer than 3 ticks arrived.")

        st.write("Ticks received:", len(ticks))
        st.write("First live quote:", ticks[0]["quote"])
        st.write("Latest quote:", ticks[-1]["quote"])
        st.write("Deriv symbol:", TARGET_SYMBOL)
        st.write("Market:", target["market"])
        st.write("Type:", target["symbol_type"])
        st.write("Read-only:", "YES")
        st.write("Synthetic/derived:", "REJECTED")
        st.write("Execution:", "DISABLED")

    except websocket.WebSocketBadStatusException as exc:
        show_error("❌ WebSocket handshake failed.", str(exc))
    except websocket.WebSocketException as exc:
        show_error("❌ WebSocket error.", str(exc))
    except Exception as exc:
        show_error("❌ Unexpected diagnostic error.", repr(exc))
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

st.markdown("**Target:** EUR/USD (`frxEURUSD`) — Forex classification required.")
if st.button("🔌 TEST DERIV REAL EUR/USD LIVE STREAM", type="primary", use_container_width=True):
    run_test()
else:
    st.info("Press the button above to start the read-only Deriv real-Forex connectivity test.")
