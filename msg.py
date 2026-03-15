# msg.py — yfinance for prices (no API limit), Twilio for alerts
import os
import requests
import yfinance as yf
from twilio.rest import Client

# ── Twilio credentials ──────────────────────────────────────────────────────
account_sid           = os.getenv("TWILIO_ACCOUNT_SID")
auth_token            = os.getenv("TWILIO_AUTH_TOKEN")
twilio_sms_number     = os.getenv("TWILIO_SMS_NUMBER")
twilio_whatsapp_number= os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
NEWS_API_KEY          = os.getenv("NEWS_API_KEY", "451415d68a1b4f3e8a055047d2509f38")

client = Client(account_sid, auth_token)


# ── Price fetching via yfinance (unlimited, no API key needed) ──────────────

def fetch_current_price(symbol: str):
    """
    Fetch latest closing price using yfinance.
    Works for Indian stocks (RELIANCE.NS) and US stocks (AAPL).
    Returns (price, symbol) or (None, None) on failure.
    Includes retry with custom headers to avoid Render/cloud IP blocks.
    """
    import time, requests

    # Custom session with browser-like headers to avoid Yahoo rate limiting on cloud IPs
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
    })

    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol, session=session)
            # Try fast_info first (single API call)
            try:
                price = ticker.fast_info.last_price
                if price and price > 0:
                    return float(price), symbol
            except Exception:
                pass
            # Fallback: history
            hist = ticker.history(period="2d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1]), symbol
        except Exception as e:
            print(f"[yfinance] Attempt {attempt+1} failed for {symbol}: {e}")
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))   # 1.5s, 3s backoff

    print(f"[yfinance] All attempts failed for {symbol}")
    return None, None


# ── SMS ─────────────────────────────────────────────────────────────────────

def send_alert_sms(to_phone_number: str, message: str) -> bool:
    if not all([account_sid, auth_token, twilio_sms_number]):
        print("Twilio SMS credentials not set.")
        return False
    try:
        resp = client.messages.create(
            body=message, from_=twilio_sms_number, to=to_phone_number
        )
        print(f"SMS sent: {resp.sid}")
        return True
    except Exception as e:
        print(f"SMS error to {to_phone_number}: {e}")
        return False


# ── WhatsApp ─────────────────────────────────────────────────────────────────

def send_alert_whatsapp(to_number: str, message: str) -> bool:
    if not all([account_sid, auth_token, twilio_whatsapp_number]):
        print("Twilio WhatsApp credentials not set.")
        return False
    if not to_number.startswith("whatsapp:"):
        to_number = "whatsapp:" + to_number
    try:
        resp = client.messages.create(
            body=message, from_=twilio_whatsapp_number, to=to_number
        )
        print(f"WhatsApp sent: {resp.sid}")
        return True
    except Exception as e:
        print(f"WhatsApp error to {to_number}: {e}")
        return False


# ── News + price change alert ─────────────────────────────────────────────────

def send_stock_news_alert(stock_symbol: str, company_name: str,
                          phone_number: str, threshold_percent: int = 1) -> bool:
    """
    Checks 2-day price change using yfinance.
    Sends SMS + WhatsApp news alert if change >= threshold.
    Works for both Indian (.NS) and US stocks.
    """
    try:
        ticker = yf.Ticker(stock_symbol)
        hist   = ticker.history(period="5d")

        if len(hist) < 2:
            print(f"Not enough data for {stock_symbol}")
            return False

        yesterday  = float(hist["Close"].iloc[-1])
        day_before = float(hist["Close"].iloc[-2])
        diff       = yesterday - day_before
        diff_pct   = round((diff / day_before) * 100, 2)
        up_down    = "🔺" if diff > 0 else "🔻"

        # Clean symbol for display (remove .NS .BO suffix)
        display_sym = stock_symbol.replace(".NS", "").replace(".BO", "")

        print(f"[Alert] {display_sym}: {up_down}{diff_pct}% change")

        if abs(diff_pct) < threshold_percent:
            print(f"Below threshold ({threshold_percent}%). No alert sent.")
            return False

        # Fetch news
        news_resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"apiKey": NEWS_API_KEY, "qInTitle": company_name, "pageSize": 3},
            timeout=10
        )
        articles = news_resp.json().get("articles", [])

        if not articles:
            # Send price alert even without news
            msg = (f"📊 StockWise Alert\n"
                   f"{display_sym} ({company_name})\n"
                   f"{up_down} {diff_pct}% price change\n"
                   f"Current: ₹{yesterday:.2f}")
            send_alert_sms(phone_number, msg)
            send_alert_whatsapp(phone_number, msg)
            return True

        sent = False
        for article in articles:
            msg = (f"📊 {display_sym}: {up_down}{diff_pct}%\n"
                   f"📰 {article.get('title', '')}\n"
                   f"💬 {article.get('description', '')[:100]}")
            if send_alert_sms(phone_number, msg):
                sent = True
            send_alert_whatsapp(phone_number, msg)

        return sent

    except Exception as e:
        print(f"[send_stock_news_alert] Error for {stock_symbol}: {e}")
        return False