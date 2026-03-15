"""
scheduler.py — Background scheduler
  • Every 5 min: auto-sell check (stop-loss / take-profit)
  • Every day 9:15 AM IST: generate AI recommendations
  • Every day 3:30 PM IST: track closing prices
"""
import os
import threading
import time
from datetime import datetime
import pytz
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

from msg import fetch_current_price, send_alert_sms
import db

CHECK_INTERVAL = int(__import__("os").getenv("AUTO_SELL_INTERVAL", "300"))
IST = pytz.timezone("Asia/Kolkata")


# ── Auto-sell check ──────────────────────────────────────────────────────────

def _check_auto_sell():
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
        stop = float(pos["stop_loss"])   if pos["stop_loss"]   else None
        tp   = float(pos["take_profit"]) if pos["take_profit"] else None
        buy  = float(pos["buy_price"])
        triggered = False
        reason    = ""
        if stop and price <= stop:
            triggered = True
            reason    = f"Stop-loss hit at ₹{price:.2f} (limit ₹{stop:.2f})"
        elif tp and price >= tp:
            triggered = True
            reason    = f"Take-profit hit at ₹{price:.2f} (target ₹{tp:.2f})"
        if triggered:
            ok  = db.sell_stock(pid, price, action="auto_sell")
            pnl = round((price - buy) * float(pos["quantity"]), 2)
            sign = "+" if pnl >= 0 else ""
            print(f"[AutoSell] {symbol} pos#{pid} — {reason} | PnL ₹{pnl}")
            if ok and pos.get("phone_number"):
                msg = (f"📊 StockWise Auto-Sell\n"
                       f"{symbol} — {reason}\n"
                       f"Qty: {pos['quantity']} @ ₹{price:.2f}\n"
                       f"P&L: {sign}₹{pnl}")
                send_alert_sms(pos["phone_number"], msg)


# ── Market schedule ──────────────────────────────────────────────────────────

_last_recommendation_date = None
_last_close_track_date    = None

def _run_market_jobs():
    global _last_recommendation_date, _last_close_track_date

    now       = datetime.now(IST)
    today     = now.date()
    weekday   = now.weekday()  # 0=Mon, 4=Fri

    # Only run on weekdays (Mon-Fri)
    if weekday >= 5:
        return

    hour   = now.hour
    minute = now.minute

    # 9:15 AM IST — generate AI recommendations
    if hour == 9 and 15 <= minute <= 20:
        if _last_recommendation_date != today:
            _last_recommendation_date = today
            print(f"[Scheduler] Running morning AI recommendations for {today}...")
            try:
                from recommender import generate_recommendations, save_recommendations, track_daily_prices
                track_daily_prices()          # track opening prices
                recs = generate_recommendations()
                if recs:
                    save_recommendations(recs)
                    # Send SMS to users with alerts set
                    _notify_recommendations(recs)
            except Exception as e:
                print(f"[Scheduler] Recommendation error: {e}")

    # 3:30 PM IST — track closing prices
    if hour == 15 and 30 <= minute <= 35:
        if _last_close_track_date != today:
            _last_close_track_date = today
            print(f"[Scheduler] Tracking closing prices for {today}...")
            try:
                from recommender import track_daily_prices
                track_daily_prices()
            except Exception as e:
                print(f"[Scheduler] Close price tracking error: {e}")


def _notify_recommendations(recs: list):
    """Send top 5 recommendations to all users who have alerts set."""
    try:
        alerts = db.get_all_alerts()
        phones = list(set(a["phone_number"] for a in alerts if a.get("phone_number")))

        if not phones:
            return

        msg = "📈 StockWise Morning Picks\nTop 5 stocks to watch today:\n\n"
        for r in recs:
            sym    = r["symbol"].replace(".NS", "")
            sign   = "+" if r["predicted_gain"] >= 0 else ""
            msg   += f"{r['rank']}. {sym} — {sign}{r['predicted_gain']:.1f}% potential\n"
            msg   += f"   ₹{r['current_price']:.2f} → ₹{r['target_price']:.2f}\n"

        msg += "\n⚠️ For informational purposes only. Not financial advice."

        for phone in phones[:50]:  # limit to 50 users
            send_alert_sms(phone, msg)
    except Exception as e:
        print(f"[Scheduler] Notify error: {e}")


# ── Main loop ────────────────────────────────────────────────────────────────

def _run():
    print("[Scheduler] Started — auto-sell + AI recommendations active.")
    while True:
        try:
            _check_auto_sell()
        except Exception as e:
            print(f"[Scheduler] Auto-sell error: {e}")
        try:
            _run_market_jobs()
        except Exception as e:
            print(f"[Scheduler] Market job error: {e}")
        time.sleep(CHECK_INTERVAL)


def start():
    t = threading.Thread(target=_run, daemon=True)
    t.start()