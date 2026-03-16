# msg.py — price fetching + Twilio alerts
# Price fetching delegated to data_fetch.py (Twelve Data → NSE → Stooq)

import os
import requests
from twilio.rest import Client
from data_fetch import fetch_price as _df_fetch_price

# All credentials read lazily inside functions — so updating env vars on
# Render/Railway takes effect on next call without a full redeploy.

def _twilio_client():
    sid   = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not sid or not token:
        return None
    try:
        return Client(sid, token)
    except Exception as e:
        print(f"[Twilio] Client init error: {e}")
        return None

# ── Symbol helpers ──────────────────────────────────────────────────────────

def fetch_current_price(symbol: str) -> tuple:
    """
    Fetch latest price using data_fetch (Twelve Data → NSE → Stooq).
    Returns (price, symbol) or (None, None) on failure.
    """
    return _df_fetch_price(symbol)


# ── SMS ──────────────────────────────────────────────────────────────────────

def send_alert_sms(to_phone_number: str, message: str) -> bool:
    sid    = os.getenv("TWILIO_ACCOUNT_SID", "")
    sms_from = os.getenv("TWILIO_SMS_NUMBER", "")
    if not sid or not sms_from:
        print("[Twilio] SMS credentials not set.")
        return False
    client = _twilio_client()
    if not client:
        return False
    try:
        resp = client.messages.create(
            body=message, from_=sms_from, to=to_phone_number
        )
        print(f"[SMS] Sent to {to_phone_number}: {resp.sid}")
        return True
    except Exception as e:
        print(f"[SMS] Error to {to_phone_number}: {e}")
        return False


# ── WhatsApp ─────────────────────────────────────────────────────────────────

def send_alert_whatsapp(to_number: str, message: str) -> bool:
    wa_from = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    client  = _twilio_client()
    if not client:
        print("[Twilio] WhatsApp credentials not set.")
        return False
    if not to_number.startswith("whatsapp:"):
        to_number = "whatsapp:" + to_number
    try:
        resp = client.messages.create(
            body=message, from_=wa_from, to=to_number
        )
        print(f"[WhatsApp] Sent to {to_number}: {resp.sid}")
        return True
    except Exception as e:
        print(f"[WhatsApp] Error to {to_number}: {e}")
        return False


# ── News + price change alert ─────────────────────────────────────────────────

def send_stock_news_alert(stock_symbol: str, company_name: str,
                          phone_number: str, threshold_percent: int = 1) -> bool:
    news_api_key = os.getenv("NEWS_API_KEY", "")   # lazy read
    try:
        price, _ = fetch_current_price(stock_symbol)
        if not price:
            return False

        # Get previous close via Twelve Data history
        prev_price = None
        key = os.getenv("TWELVE_DATA_KEY", "")
        if key:
            try:
                td_sym = _to_td_symbol(stock_symbol)
                url    = (
                    f"https://api.twelvedata.com/time_series"
                    f"?symbol={td_sym}&interval=1day&outputsize=2&apikey={key}&format=JSON"
                )
                resp   = requests.get(url, timeout=8).json()
                vals   = resp.get("values", [])
                if len(vals) >= 2:
                    prev_price = float(vals[1]["close"])
            except Exception:
                pass

        if prev_price is None:
            # fallback: use yfinance history for prev close
            try:
                hist = yf.Ticker(stock_symbol).history(period="5d")
                if len(hist) >= 2:
                    prev_price = float(hist["Close"].iloc[-2])
                    price      = float(hist["Close"].iloc[-1])
            except Exception:
                pass

        if not prev_price:
            return False

        diff     = price - prev_price
        diff_pct = round((diff / prev_price) * 100, 2)
        up_down  = "🔺" if diff > 0 else "🔻"
        display  = stock_symbol.replace(".NS", "").replace(".BO", "")

        if abs(diff_pct) < threshold_percent:
            return False

        articles = []
        if news_api_key:
            try:
                news_resp = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={"apiKey": news_api_key, "qInTitle": company_name, "pageSize": 3},
                    timeout=10
                )
                articles = news_resp.json().get("articles", [])
            except Exception as e:
                print(f"[NewsAPI] Error: {e}")

        if not articles:
            msg = (f"📊 StockWise Alert\n{display} ({company_name})\n"
                   f"{up_down} {diff_pct}% price change\nCurrent: ₹{price:.2f}")
            send_alert_sms(phone_number, msg)
            send_alert_whatsapp(phone_number, msg)
            return True

        sent = False
        for article in articles:
            msg = (f"📊 {display}: {up_down}{diff_pct}%\n"
                   f"📰 {article.get('title','')}\n"
                   f"💬 {article.get('description','')[:100]}")
            if send_alert_sms(phone_number, msg):
                sent = True
            send_alert_whatsapp(phone_number, msg)
        return sent

    except Exception as e:
        print(f"[send_stock_news_alert] Error for {stock_symbol}: {e}")
        return False
