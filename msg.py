# msg.py — price fetching + Twilio alerts
# Price source priority:
#   1. Twelve Data API  (reliable on cloud, needs TWELVE_DATA_KEY)
#   2. NSE India public API  (free, no key, works on cloud, India stocks only)
#   3. yfinance  (often rate-limited on cloud IPs — last resort)

import os
import requests
import yfinance as yf
from twilio.rest import Client
from data_fetch import fetch_price as _df_fetch_price

# ── Twilio — read at import time is fine, these don't change ────────────────
account_sid            = os.getenv("TWILIO_ACCOUNT_SID")
auth_token             = os.getenv("TWILIO_AUTH_TOKEN")
twilio_sms_number      = os.getenv("TWILIO_SMS_NUMBER")
twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

# NOTE: API keys are read inside functions (lazy) so Render env vars added
# after deploy are always picked up without a restart.

client = Client(account_sid, auth_token)

# ── Symbol helpers ──────────────────────────────────────────────────────────

def _to_td_symbol(symbol: str) -> str:
    """Convert yfinance symbol to Twelve Data format."""
    if symbol.endswith(".NS"): return symbol.replace(".NS", "") + ":NSE"
    if symbol.endswith(".BO"): return symbol.replace(".BO", "") + ":BSE"
    return symbol

def _to_nse_symbol(symbol: str) -> str:
    """
    Convert yfinance symbol to NSE ticker.
    e.g. RELIANCE.NS -> RELIANCE, M&M.NS -> M%26M (URL-encoded)
    """
    sym = symbol.replace(".NS", "").replace(".BO", "")
    return sym  # requests handles URL encoding automatically


# ── Price source 1: Twelve Data ─────────────────────────────────────────────

def _fetch_twelvedata(symbol: str) -> float | None:
    key = os.getenv("TWELVE_DATA_KEY", "")   # lazy read every call
    if not key:
        return None
    try:
        td_sym = _to_td_symbol(symbol)
        url    = f"https://api.twelvedata.com/price?symbol={td_sym}&apikey={key}"
        resp   = requests.get(url, timeout=8)
        data   = resp.json()
        if "price" in data:
            price = float(data["price"])
            if price > 0:
                print(f"[TwelveData] {symbol} = {price}")
                return price
        print(f"[TwelveData] No price for {symbol}: {data.get('message','unknown error')}")
    except Exception as e:
        print(f"[TwelveData] Error for {symbol}: {e}")
    return None


# ── Price source 2: NSE India public API (no key, India stocks only) ────────

def _fetch_nse(symbol: str) -> float | None:
    """
    Uses NSE India's public quote API.
    Only works for .NS stocks. Returns None for US/other symbols.
    """
    if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
        return None
    try:
        nse_sym = _to_nse_symbol(symbol)
        session = requests.Session()
        # NSE requires a cookie from the main page first
        session.get(
            "https://www.nseindia.com",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        url  = f"https://www.nseindia.com/api/quote-equity?symbol={nse_sym}"
        resp = session.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/",
            },
            timeout=8,
        )
        data  = resp.json()
        price = (
            data.get("priceInfo", {}).get("lastPrice") or
            data.get("priceInfo", {}).get("close")
        )
        if price and float(price) > 0:
            print(f"[NSE] {symbol} = {price}")
            return float(price)
    except Exception as e:
        print(f"[NSE] Error for {symbol}: {e}")
    return None


# ── Price source 3: yfinance (last resort, often blocked on cloud) ───────────

def _fetch_yfinance(symbol: str) -> float | None:
    import time
    # yfinance often blocks .NS symbols on cloud — try with a browser UA
    yf.utils.requests = requests.Session()
    yf.utils.requests.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    for attempt in range(2):
        try:
            ticker = yf.Ticker(symbol)
            try:
                p = ticker.fast_info.last_price
                if p and float(p) > 0:
                    return float(p)
            except Exception:
                pass
            hist = ticker.history(period="2d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            print(f"[yfinance] Attempt {attempt+1} for {symbol}: {e}")
            if attempt < 1:
                time.sleep(1)
    return None


# ── Public interface ─────────────────────────────────────────────────────────

def fetch_current_price(symbol: str) -> tuple:
    """
    Fetch latest price using data_fetch (Twelve Data → NSE → Stooq).
    Returns (price, symbol) or (None, None) on failure.
    """
    return _df_fetch_price(symbol)


# ── SMS ──────────────────────────────────────────────────────────────────────

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
