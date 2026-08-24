import json, time
from datetime import datetime, timezone
import requests, streamlit as st, websocket

API_BASE="https://api.derivws.com"
SYMBOL="frxEURUSD"
TEST_SECONDS=30

st.set_page_config(page_title="Deriv Authenticated Real Forex Test", page_icon="📡")
st.title("📡 Deriv Authenticated REAL Forex — Read-Only Test")
st.info("Isolated diagnostic only. No trade execution and no changes to your V13 app.py.")

def secret(name):
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return ""

APP_ID=secret("DERIV_APP_ID")
TOKEN=secret("DERIV_API_TOKEN")
ACCOUNT=secret("DERIV_ACCOUNT_ID")

st.write("Target: EUR/USD (`frxEURUSD`)")
st.write("App ID configured:", "Yes" if APP_ID else "No")
st.write("API token configured:", "Yes" if TOKEN else "No")
st.write("Account ID configured:", "Yes" if ACCOUNT else "No")

if not (APP_ID and TOKEN and ACCOUNT):
    st.error("Missing one or more secrets: DERIV_APP_ID, DERIV_API_TOKEN, DERIV_ACCOUNT_ID")
    st.stop()

def show(title, obj):
    st.write(title)
    st.code(json.dumps(obj, indent=2, default=str), language="json")

def run():
    ws=None
    try:
        # 1) REST OTP
        st.subheader("1. Deriv REST Authentication / OTP")
        url=f"{API_BASE}/trading/v1/options/accounts/{ACCOUNT}/otp"
        headers={
            "Deriv-App-ID": APP_ID,
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
        }
        try:
            r=requests.post(url, headers=headers, timeout=15)
        except requests.RequestException as e:
            st.error("❌ REST connection failed.")
            show("REST diagnostic", {"exception_type":type(e).__name__,"exception":str(e)})
            return

        if not r.ok:
            st.error(f"❌ Deriv OTP request failed: HTTP {r.status_code}")
            try: body=r.json()
            except Exception: body=r.text[:2000]
            show("Deriv REST diagnostic", {
                "http_status":r.status_code,
                "response":body,
                "credentials_exposed":False,
            })
            st.warning("HTTP 401 usually means the token is invalid/missing. HTTP 403 may indicate insufficient token scope. HTTP 400 may indicate an invalid account ID.")
            return

        try: data=r.json()
        except Exception:
            st.error("❌ Deriv returned a non-JSON OTP response.")
            return

        ws_url=((data.get("data") or {}).get("url"))
        if not isinstance(ws_url,str) or not ws_url.startswith("wss://"):
            st.error("❌ OTP succeeded but no usable WebSocket URL was returned.")
            show("Safe OTP diagnostic", {"http_status":r.status_code,"temporary_websocket_url_received":False,"credentials_exposed":False})
            return

        st.success("✅ Temporary authenticated WebSocket URL obtained.")
        show("Safe OTP diagnostic", {"http_status":r.status_code,"temporary_websocket_url_received":True,"otp_or_url_exposed":False})

        # 2) Authenticated demo WebSocket
        st.subheader("2. Authenticated Demo WebSocket")
        try:
            ws=websocket.create_connection(
                ws_url, timeout=10,
                header=[f"Deriv-App-ID: {APP_ID}"],
                enable_multithread=True
            )
            st.success("✅ Authenticated Deriv WebSocket connected.")
        except Exception as e:
            st.error("❌ Authenticated WebSocket connection failed.")
            show("WebSocket diagnostic", {
                "exception_type":type(e).__name__,
                "exception":str(e),
                "websocket_url_exposed":False,
                "credentials_exposed":False
            })
            return

        # 3) Ping
        st.subheader("3. WebSocket Ping")
        ws.send(json.dumps({"ping":1,"req_id":3000}))
        ping=False
        deadline=time.time()+8
        while time.time()<deadline:
            try: d=json.loads(ws.recv())
            except websocket.WebSocketTimeoutException: continue
            if d.get("error"):
                show("❌ Ping error",d); return
            if d.get("msg_type")=="ping" or "ping" in d:
                ping=True; break
        if ping: st.success("✅ WebSocket ping response received.")
        else: st.warning("⚠️ Connected but no ping response was observed.")

        # 4) EUR/USD ticks
        st.subheader("4. EUR/USD Live Tick Subscription")
        req={"ticks":SYMBOL,"subscribe":1,"req_id":3001}
        show("Subscription request",req)
        ws.send(json.dumps(req))
        status=st.empty(); quote=st.empty(); table=st.empty()
        status.info("Waiting up to 30 seconds for a real EUR/USD tick...")
        ticks=[]; start=time.time()

        while time.time()-start<TEST_SECONDS:
            try: raw=ws.recv()
            except websocket.WebSocketTimeoutException: continue
            except Exception as e:
                status.error("❌ WebSocket receive failed.")
                show("Receive diagnostic",{"exception_type":type(e).__name__,"exception":str(e)})
                break
            try: d=json.loads(raw)
            except Exception: continue
            if d.get("error"):
                status.error("❌ Deriv returned a subscription/API error.")
                show("Deriv subscription diagnostic",{"error":d.get("error"),"msg_type":d.get("msg_type"),"symbol_requested":SYMBOL})
                break
            if d.get("msg_type")!="tick": continue
            t=d.get("tick") or {}
            if t.get("symbol")!=SYMBOL or t.get("quote") is None: continue
            rec={
                "symbol":t.get("symbol"),
                "quote":t.get("quote"),
                "epoch":t.get("epoch"),
                "received_utc":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            }
            ticks.append(rec)
            if len(ticks)==1: status.success("🎯 FIRST LIVE EUR/USD TICK RECEIVED")
            quote.metric("LIVE EUR/USD QUOTE",str(rec["quote"]))
            table.dataframe(ticks[-10:],use_container_width=True,hide_index=True)
            if len(ticks)>=3: break

        # 5) Result
        st.subheader("5. Final Diagnostic")
        if not ticks:
            status.error("❌ No EUR/USD tick was received within 30 seconds.")
            show("Final diagnostic",{
                "source":"DERIV AUTHENTICATED REAL FOREX",
                "symbol":SYMBOL,"ticks_received":0,
                "read_only":True,"synthetic":False,"execution_enabled":False
            })
        else:
            st.success("🟢 SUCCESS — authenticated Deriv EUR/USD live stream is working.")
            show("Verified result",{
                "source":"DERIV AUTHENTICATED REAL FOREX",
                "symbol":SYMBOL,"ticks_received":len(ticks),
                "first_quote":ticks[0]["quote"],"latest_quote":ticks[-1]["quote"],
                "read_only":True,"synthetic":False,"execution_enabled":False
            })
    except Exception as e:
        st.error("❌ Unexpected diagnostic error.")
        show("Exception",{"exception_type":type(e).__name__,"exception":str(e),"credentials_exposed":False})
    finally:
        if ws:
            try: ws.close()
            except Exception: pass

if st.button("🔌 TEST AUTHENTICATED DERIV EUR/USD LIVE STREAM", type="primary", use_container_width=True):
    run()
else:
    st.info("Press the button to test the authenticated read-only Deriv EUR/USD live stream.")
