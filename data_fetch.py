"""
data_fetch.py - Cloud-safe market data. No yfinance (blocked on cloud IPs).
Sources: Twelve Data -> NSE India -> Stooq
"""
import os
import time
import requests
import pandas as pd
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta


def _td_sym(symbol):
    if symbol.endswith(".NS"):
        return symbol.replace(".NS", "") + ":NSE"
    if symbol.endswith(".BO"):
        return symbol.replace(".BO", "") + ":BSE"
    return symbol


def _nse_sym(symbol):
    return symbol.replace(".NS", "").replace(".BO", "")


def _stooq_sym(symbol):
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol.lower()
    return symbol.lower() + ".us"


# ---------- current price ----------

def _price_td(symbol):
    key = os.getenv("TWELVE_DATA_KEY", "")
    if not key:
        return None
    try:
        r = requests.get(
            f"https://api.twelvedata.com/price?symbol={_td_sym(symbol)}&apikey={key}",
            timeout=8
        ).json()
        p = r.get("price")
        if p and float(p) > 0:
            print(f"[TD price] {symbol}={p}")
            return float(p)
    except Exception as e:
        print(f"[TD price] {symbol}: {e}")
    return None


def _price_nse(symbol):
    if not symbol.endswith(".NS"):
        return None
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com",
              headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        r = s.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={_nse_sym(symbol)}",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                     "Referer": "https://www.nseindia.com/"},
            timeout=8
        ).json()
        p = r.get("priceInfo", {}).get("lastPrice") or r.get("priceInfo", {}).get("close")
        if p and float(p) > 0:
            print(f"[NSE price] {symbol}={p}")
            return float(p)
    except Exception as e:
        print(f"[NSE price] {symbol}: {e}")
    return None


def _price_stooq(symbol):
    try:
        r = requests.get(
            f"https://stooq.com/q/d/l/?s={_stooq_sym(symbol)}&i=d",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"}
        )
        df = pd.read_csv(StringIO(r.text.strip()))
        if not df.empty and "Close" in df.columns:
            p = float(df["Close"].iloc[-1])
            if p > 0:
                print(f"[Stooq price] {symbol}={p}")
                return p
    except Exception as e:
        print(f"[Stooq price] {symbol}: {e}")
    return None


def fetch_price(symbol):
    for fn in (_price_td, _price_nse, _price_stooq):
        try:
            p = fn(symbol)
            if p and p > 0:
                return p, symbol
        except Exception:
            pass
    print(f"[Price] All sources failed for {symbol}")
    return None, None


# ---------- history ----------

def _hist_td(symbol, days=365):
    key = os.getenv("TWELVE_DATA_KEY", "")
    if not key:
        return None
    try:
        r = requests.get(
            f"https://api.twelvedata.com/time_series?symbol={_td_sym(symbol)}"
            f"&interval=1day&outputsize={min(days,5000)}&apikey={key}&format=JSON",
            timeout=15
        ).json()
        if r.get("status") == "error":
            print(f"[TD hist] {symbol}: {r.get('message')}")
            return None
        rows = r.get("values", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["Date"]   = pd.to_datetime(df["datetime"])
        df["Open"]   = pd.to_numeric(df["open"],   errors="coerce")
        df["High"]   = pd.to_numeric(df["high"],   errors="coerce")
        df["Low"]    = pd.to_numeric(df["low"],    errors="coerce")
        df["Close"]  = pd.to_numeric(df["close"],  errors="coerce")
        df["Volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df = df.set_index("Date").sort_index()
        df["ds"] = df.index
        df["y"]  = df["Close"]
        print(f"[TD hist] {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[TD hist] {symbol}: {e}")
    return None


def _hist_stooq(symbol, days=365):
    try:
        start = (date.today() - timedelta(days=int(days * 1.6) + 90)).strftime("%Y%m%d")
        end   = date.today().strftime("%Y%m%d")
        r = requests.get(
            f"https://stooq.com/q/d/l/?s={_stooq_sym(symbol)}&d1={start}&d2={end}&i=d",
            timeout=20, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code != 200:
            return None
        text = r.text.strip()
        if not text or "No data" in text or len(text) < 30:
            return None
        df = pd.read_csv(StringIO(text))
        if df.empty:
            return None
        df.columns = [c.strip().title() for c in df.columns]
        if "Close" not in df.columns or "Date" not in df.columns:
            return None
        df["Date"]   = pd.to_datetime(df["Date"], errors="coerce")
        df           = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        df["Close"]  = pd.to_numeric(df["Close"],  errors="coerce")
        df["Open"]   = pd.to_numeric(df["Open"],   errors="coerce") if "Open"   in df.columns else df["Close"]
        df["High"]   = pd.to_numeric(df["High"],   errors="coerce") if "High"   in df.columns else df["Close"]
        df["Low"]    = pd.to_numeric(df["Low"],    errors="coerce") if "Low"    in df.columns else df["Close"]
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0) if "Volume" in df.columns else 0
        df = df.dropna(subset=["Close"])
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        df["ds"] = df.index
        df["y"]  = df["Close"]
        print(f"[Stooq hist] {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[Stooq hist] {symbol}: {e}")
    return None


def fetch_history(symbol, days=365):
    for fn in (_hist_td, _hist_stooq):
        try:
            df = fn(symbol, days)
            if df is not None and len(df) >= 10:
                return df
        except Exception:
            pass
    print(f"[History] All sources failed for {symbol}")
    return None


def fetch_history_batch(symbols, days=60):
    if not symbols:
        return {}
    result = {}
    key = os.getenv("TWELVE_DATA_KEY", "")

    if key:
        for i in range(0, len(symbols), 8):
            batch = symbols[i:i+8]
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = {ex.submit(_hist_td, s, days): s for s in batch}
                for f in as_completed(futs):
                    s = futs[f]
                    try:
                        df = f.result()
                        if df is not None and len(df) >= 10:
                            result[s] = df
                    except Exception:
                        pass
            if i + 8 < len(symbols):
                time.sleep(61)
        print(f"[Batch TD] {len(result)}/{len(symbols)}")

    missing = [s for s in symbols if s not in result]
    if missing:
        print(f"[Batch Stooq] fetching {len(missing)} symbols...")
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_hist_stooq, s, days): s for s in missing}
            for f in as_completed(futs):
                s = futs[f]
                try:
                    df = f.result()
                    if df is not None and len(df) >= 10:
                        result[s] = df
                except Exception:
                    pass
        print(f"[Batch total] {len(result)}/{len(symbols)}")
    return result
