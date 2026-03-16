"""
msg.py - Twilio SMS/WhatsApp alerts + price fetching
"""
import os
import requests
from twilio.rest import Client
from data_fetch import fetch_price


def _twilio():
    sid   = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not sid or not token:
        return None
    try:
        return Client(sid, token)
    except Exception as e:
        print(f"[Twilio] init error: {e}")
        return None


def fetch_current_price(symbol):
    return fetch_price(symbol)


def send_alert_sms(to_number, message):
    client   = _twilio()
    sms_from = os.getenv("TWILIO_SMS_NUMBER", "")
    if not client or not sms_from:
        print("[SMS] Twilio not configured")
        return False
    try:
        msg = client.messages.create(body=message, from_=sms_from, to=to_number)
        print(f"[SMS] sent {msg.sid}")
        return True
    except Exception as e:
        print(f"[SMS] error: {e}")
        return False


def send_alert_whatsapp(to_number, message):
    client  = _twilio()
    wa_from = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    if not client:
        print("[WA] Twilio not configured")
        return False
    if not to_number.startswith("whatsapp:"):
        to_number = "whatsapp:" + to_number
    try:
        msg = client.messages.create(body=message, from_=wa_from, to=to_number)
        print(f"[WA] sent {msg.sid}")
        return True
    except Exception as e:
        print(f"[WA] error: {e}")
        return False


def send_stock_news_alert(stock_symbol, company_name, phone_number, threshold_percent=1):
    news_key = os.getenv("NEWS_API_KEY", "")
    try:
        price, _ = fetch_current_price(stock_symbol)
        if not price:
            return False

        # get previous close via Stooq
        from data_fetch import _hist_stooq
        hist = _hist_stooq(stock_symbol, days=5)
        if hist is None or len(hist) < 2:
            return False

        prev      = float(hist["Close"].iloc[-2])
        curr      = float(hist["Close"].iloc[-1])
        diff_pct  = round(((curr - prev) / prev) * 100, 2)
        arrow     = "🔺" if diff_pct > 0 else "🔻"
        sym       = stock_symbol.replace(".NS", "").replace(".BO", "")

        if abs(diff_pct) < threshold_percent:
            return False

        articles = []
        if news_key:
            try:
                resp     = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={"apiKey": news_key, "qInTitle": company_name, "pageSize": 3},
                    timeout=10
                )
                articles = resp.json().get("articles", [])
            except Exception:
                pass

        if not articles:
            msg = (f"📊 StockWise Alert\n{sym} ({company_name})\n"
                   f"{arrow} {diff_pct}% price change\nCurrent: ₹{curr:.2f}")
            send_alert_sms(phone_number, msg)
            send_alert_whatsapp(phone_number, msg)
            return True

        sent = False
        for a in articles:
            msg = (f"📊 {sym}: {arrow}{diff_pct}%\n"
                   f"📰 {a.get('title','')}\n"
                   f"💬 {a.get('description','')[:100]}")
            if send_alert_sms(phone_number, msg):
                sent = True
            send_alert_whatsapp(phone_number, msg)
        return sent

    except Exception as e:
        print(f"[NewsAlert] {stock_symbol}: {e}")
        return False
