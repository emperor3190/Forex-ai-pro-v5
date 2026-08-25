
import json
import time
from datetime import datetime, timezone
import requests
import streamlit as st
import websocket

BASE = "https://api.derivws.com"
SYMBOL = "frxEURUSD"

st.set_page_config(page_title="Deriv Options Account Test", page_icon="📡")
st.title("📡 Deriv Options Account + EUR/USD Read-Only Test")
st.caption("Diagnostic only — no trade execution.")

def secret(name):
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return ""

APP_ID = secret("DERIV_APP_ID")
TOKEN = secret("DERIV_API_TOKEN")

st.write("PAT App ID configured:", "Yes" if APP_ID else "No")
st.write("PAT token configured:", "Yes" if TOKEN else "No")

if not APP_ID or not TOKEN:
    st.error("Missing DERIV_APP_ID or DERIV_API_TOKEN in Streamlit Secrets.")
    st.stop()

HEADERS = {
    "Deriv-App-ID": APP_ID,
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
}

def body_json(response):
    try:
        return response.json()
    except Exception:
        return {"raw_response": response.text[:2000]}

def run_test():
    st.subheader("1. Discover Options trading accounts")

    try:
        response = requests.get(
            f"{BASE}/trading/v1/options/accounts",
            headers=HEADERS,
            timeout=15,
        )
    except requests.RequestException as exc:
        st.error("REST request failed.")
        st.code(str(exc))
        return

    body = body_json(response)

    if not response.ok:
        st.error(f"Options account discovery failed — HTTP {response.status_code}")
        st.json({
            "http_status": response.status_code,
            "response": body,
            "credentials_exposed": False,
        })
        return

    raw = body.get("data", [])
    accounts = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []

    rows = []
    for account in accounts:
        if isinstance(account, dict):
            rows.append({
                "account_id": account.get("account_id"),
                "account_type": account.get("account_type"),
                "currency": account.get("currency"),
                "status": account.get("status"),
                "group": account.get("group"),
            })

    if not rows:
        st.warning("Authentication succeeded, but Deriv returned no Options accounts.")
        return

    st.success(f"Found {len(rows)} Options account(s).")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    demos = [
        account for account in accounts
        if isinstance(account, dict)
        and str(account.get("account_type", "")).lower() == "demo"
        and account.get("account_id")
    ]

    if not demos:
        st.warning("No DEMO Options account was returned.")
        return

    account = demos[0]
    account_id = account["account_id"]

    st.success("Demo Options account found.")
    st.code(f"Selected demo Options account: {account_id}")

    st.subheader("2. Request authenticated WebSocket URL")

    try:
        otp_response = requests.post(
            f"{BASE}/trading/v1/options/accounts/{account_id}/otp",
            headers=HEADERS,
            timeout=15,
        )
    except requests.RequestException as exc:
        st.error("OTP request failed.")
        st.code(str(exc))
        return

    otp_body = body_json(otp_response)

    if not otp_response.ok:
        st.error(f"OTP request failed — HTTP {otp_response.status_code}")
        st.json({
            "http_status": otp_response.status_code,
            "response": otp_body,
            "credentials_exposed": False,
        })
        return

    data = otp_body.get("data", {})
    ws_url = data.get("url") if isinstance(data, dict) else None

    if not isinstance(ws_url, str) or not ws_url.startswith("wss://"):
        st.error("OTP succeeded but no usable WebSocket URL was returned.")
        return

    st.success("OTP succeeded — authenticated WebSocket URL received.")
    st.info("The returned URL/OTP is intentionally hidden.")

    st.subheader("3. Connect directly to Deriv's returned WebSocket")

    try:
        ws = websocket.create_connection(ws_url, timeout=10)
        st.success("Authenticated Options WebSocket connected.")
    except Exception as exc:
        st.error("Authenticated WebSocket connection failed.")
        st.json({
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "websocket_url_exposed": False,
            "credentials_exposed": False,
        })
        return

    st.subheader("4. EUR/USD live tick test")

    ws.send(json.dumps({
        "ticks": SYMBOL,
        "subscribe": 1,
        "req_id": 4001,
    }))

    status = st.empty()
    quote = st.empty()
    status.info("Waiting up to 30 seconds for the first EUR/USD tick...")

    ticks = []
    deadline = time.time() + 30

    while time.time() < deadline:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as exc:
            status.error("WebSocket receive error.")
            st.json({
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            })
            break

        try:
            message = json.loads(raw)
        except Exception:
            continue

        if message.get("error"):
            status.error("Deriv returned a WebSocket error.")
            st.json({
                "error": message.get("error"),
                "msg_type": message.get("msg_type"),
            })
            break

        if message.get("msg_type") != "tick":
            continue

        tick = message.get("tick") or {}
        if tick.get("symbol") != SYMBOL or tick.get("quote") is None:
            continue

        record = {
            "symbol": tick.get("symbol"),
            "quote": tick.get("quote"),
            "epoch": tick.get("epoch"),
            "received_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        ticks.append(record)

        if len(ticks) == 1:
            status.success("🎯 FIRST LIVE EUR/USD TICK RECEIVED")

        quote.metric("EUR/USD", str(record["quote"]))

        if len(ticks) >= 3:
            break

    if ticks:
        st.success("🟢 SUCCESS — authenticated Deriv EUR/USD live tick stream is working.")
        st.dataframe(ticks, use_container_width=True, hide_index=True)
        st.json({
            "source": "DERIV AUTHENTICATED OPTIONS WEBSOCKET",
            "symbol": SYMBOL,
            "account_type": account.get("account_type"),
            "ticks_received": len(ticks),
            "read_only_test": True,
            "execution_enabled": False,
            "credentials_exposed": False,
        })
    else:
        status.error("❌ No EUR/USD tick received during the test window.")

    try:
        ws.close()
    except Exception:
        pass

if st.button(
    "🔌 DISCOVER OPTIONS ACCOUNT + TEST EUR/USD LIVE STREAM",
    type="primary",
    use_container_width=True,
):
    run_test()
else:
    st.info("Press the button to discover the correct Options demo account and test EUR/USD.")
