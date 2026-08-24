import json
import sys
import time
from datetime import datetime, timezone
import websocket

ENDPOINT = "wss://ws.binaryws.com/websockets/v3"
TARGET_SYMBOL = "frxEURUSD"
TEST_SECONDS = 30

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def fail(message, detail=None):
    print("\n" + "=" * 70)
    print("DERIV REAL-FOREX TEST FAILED")
    print(message)
    if detail:
        print("DETAIL:", detail)
    print("=" * 70)
    return 1

def normalize(item):
    return {
        "symbol": item.get("underlying_symbol") or item.get("symbol"),
        "type": item.get("underlying_symbol_type") or item.get("symbol_type"),
        "name": item.get("underlying_symbol_name") or item.get("display_name"),
        "market": item.get("market"),
        "submarket": item.get("submarket"),
    }

def main():
    print("=" * 70)
    print("DERIV REAL FOREX — READ-ONLY EUR/USD LIVE STREAM TEST")
    print("=" * 70)
    print("Time:", utc_now())
    print("Endpoint:", ENDPOINT)
    print("Symbol:", TARGET_SYMBOL)
    print("Synthetic/derived instruments: REJECTED")
    print("Trading/execution: DISABLED")

    ws = None
    try:
        print("\n[1] Connecting...")
        ws = websocket.create_connection(ENDPOINT, timeout=10)
        print("CONNECTED")

        print("\n[2] Requesting active symbols...")
        ws.send(json.dumps({
            "active_symbols": "brief",
            "product_type": "basic",
            "req_id": 1
        }))

        response = None
        deadline = time.time() + 10
        while time.time() < deadline:
            data = json.loads(ws.recv())
            if data.get("error"):
                return fail("active_symbols was rejected.", data["error"])
            if data.get("msg_type") == "active_symbols":
                response = data
                break

        if response is None:
            return fail("No active_symbols response received.")

        found = None
        for raw in response.get("active_symbols", []):
            item = normalize(raw)
            if item["symbol"] == TARGET_SYMBOL:
                found = item
                break

        if found is None:
            return fail("frxEURUSD was not returned by active_symbols.")

        print("Symbol:", found["symbol"])
        print("Name:", found["name"])
        print("Market:", found["market"])
        print("Type:", found["type"])

        if found["market"] != "forex" or found["type"] != "forex":
            return fail("EUR/USD failed the REAL FOREX gate.", found)

        print("REAL FOREX VALIDATION: PASS")

        print("\n[3] Sending minimal tick subscription...")
        request = {
            "ticks": TARGET_SYMBOL,
            "subscribe": 1,
            "req_id": 2
        }
        print(json.dumps(request))
        ws.send(json.dumps(request))

        print("\n[4] Waiting for live ticks...")
        first = None
        count = 0
        deadline = time.time() + TEST_SECONDS

        while time.time() < deadline:
            try:
                data = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue

            if data.get("error"):
                return fail("Tick subscription was rejected.", {
                    "error": data["error"],
                    "request": request
                })

            if data.get("msg_type") != "tick":
                continue

            tick = data.get("tick") or {}
            if tick.get("symbol") != TARGET_SYMBOL:
                continue

            if tick.get("quote") is None or tick.get("epoch") is None:
                return fail("Tick lacked quote or epoch.", tick)

            count += 1
            if first is None:
                first = tick
                print("\nFIRST LIVE EUR/USD TICK RECEIVED")
            print(
                "Tick #{} | symbol={} | quote={} | epoch={}".format(
                    count, tick["symbol"], tick["quote"], tick["epoch"]
                )
            )

        if first is None:
            return fail(
                "No live EUR/USD tick received within {} seconds.".format(TEST_SECONDS),
                {"endpoint": ENDPOINT, "symbol": TARGET_SYMBOL, "request": request}
            )

        print("\n" + "=" * 70)
        print("SUCCESS — DERIV REAL FOREX EUR/USD STREAM VERIFIED")
        print("Ticks received:", count)
        print("Last quote:", first["quote"] if count == 1 else tick["quote"])
        print("Read-only: YES")
        print("Synthetic/derived: REJECTED")
        print("=" * 70)
        print("\nDo not integrate into V13 until this feed-level test passes.")
        return 0

    except Exception as exc:
        return fail("Runtime/WebSocket error.", repr(exc))
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

if __name__ == "__main__":
    sys.exit(main())
