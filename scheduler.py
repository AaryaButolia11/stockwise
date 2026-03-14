"""
scheduler.py — Background thread that:
  1. Every 5 min: checks open positions against stop-loss / take-profit
  2. Auto-sells and sends SMS notification if triggered
"""
import threading
import time
from msg import fetch_current_price, send_alert_sms
import db

CHECK_INTERVAL = int(__import__("os").getenv("AUTO_SELL_INTERVAL", "300"))  # seconds


def _check_auto_sell():
    """Check all open positions for stop-loss / take-profit triggers."""
    positions = db.get_open_positions()
    for pos in positions:
        symbol = pos["stock_symbol"]
        pid    = pos["id"]

        if pos["stop_loss"] is None and pos["take_profit"] is None:
            continue

        price, _ = fetch_current_price(symbol)
        if price is None:
            continue

        db.update_current_price(pid, price)

        triggered = False
        reason    = ""

        stop  = float(pos["stop_loss"])   if pos["stop_loss"]   else None
        tp    = float(pos["take_profit"]) if pos["take_profit"] else None
        buy   = float(pos["buy_price"])

        if stop and price <= stop:
            triggered = True
            reason    = f"Stop-loss hit at ${price:.2f} (limit ${stop:.2f})"
        elif tp and price >= tp:
            triggered = True
            reason    = f"Take-profit hit at ${price:.2f} (target ${tp:.2f})"

        if triggered:
            ok  = db.sell_stock(pid, price, action="auto_sell")
            pnl = round((price - buy) * float(pos["quantity"]), 2)
            print(f"[AutoSell] {symbol} pos#{pid} — {reason} | PnL ${pnl}")

            if ok and pos.get("phone_number"):
                sign = "+" if pnl >= 0 else ""
                msg  = (
                    f"📊 StockWise Auto-Sell\n"
                    f"{symbol} ({pos['company_name']})\n"
                    f"{reason}\n"
                    f"Qty: {pos['quantity']} @ ${price:.2f}\n"
                    f"P&L: {sign}${pnl}\n"
                    f"Position closed automatically."
                )
                send_alert_sms(pos["phone_number"], msg)


def _run():
    print("[Scheduler] Auto-sell monitor started.")
    while True:
        try:
            _check_auto_sell()
        except Exception as e:
            print(f"[Scheduler] Error: {e}")
        time.sleep(CHECK_INTERVAL)


def start():
    """Call once at app startup to launch the background monitor."""
    t = threading.Thread(target=_run, daemon=True)
    t.start()
