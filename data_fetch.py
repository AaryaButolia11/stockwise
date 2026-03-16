"""
data_fetch.py — Market data fetcher. Zero yfinance dependency.
Yahoo Finance blocks all cloud IPs. Use these instead:
  1. Twelve Data  — needs TWELVE_DATA_KEY env var
  2. NSE India    — free, no key, .NS stocks only
  3. Stooq        — free, no key, global coverage
"""

import os
import time
import requests
import pandas as pd
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed


def to_td_symbol(symbol):
    if symbol.endswith(".NS"):
        return symbol.replace(".NS", "") + ":NSE"
    if symbol.endswith(".BO"):
        return symbol.replace(".BO", "") + ":BSE"
    return symbol


def to_stooq_symbol(symbol):
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol.lower()
    return symbol.lower() + ".us"


def to_nse_symbol(symbol):
    return symbol.replace(".NS", "").replace(".BO", "")


# ── Current price ─────────────────────────────────────────────────────────────

def _price_twelvedata(symbol):
    key = os.getenv("TWELVE_DATA_KEY", "")
    if not key:
        return None
    try:
        url  = f"https://api.twelvedata.com/price?symbol={to_td_symbol(symbol)}&apikey={key}"
        data = requests.get(url, timeout=8).json()
        p    = data.get("price")
        if p and float(p) > 0:
            print(f"[TD price] {symbol} = {p}")
            return float(p)
    except Exception as e:
        print(f"[TD price] {symbol}: {e}")
    return None


def _price_nse(symbol):
    if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
        return None
    try:
        session = requests.Session()
        session.get(
            "https://www.nseindia.com",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        resp = session.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={to_nse_symbol(symbol)}",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/",
            },
            timeout=8,
        )
        data  = resp.json()
        price = (data.get("priceInfo", {}).get("lastPrice") or
                 data.get("priceInfo", {}).get("close"))
        if price and float(price) > 0:
            print(f"[NSE price] {symbol} = {price}")
            return float(price)
    except Exception as e:
        print(f"[NSE price] {symbol}: {e}")
    return None


def _price_stooq(symbol):
    try:
        url  = f"https://stooq.com/q/d/l/?s={to_stooq_symbol(symbol)}&i=d"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        df = pd.read_csv(StringIO(resp.text.strip()))
        if df.empty or "Close" not in df.columns:
            return None
        p = float(df["Close"].iloc[-1])
        if p > 0:
            print(f"[Stooq price] {symbol} = {p}")
            return p
    except Exception as e:
        print(f"[Stooq price] {symbol}: {e}")
    return None


def fetch_price(symbol):
    """Returns (price, symbol) or (None, None)."""
    for fn in (_price_twelvedata, _price_nse, _price_stooq):
        try:
            p = fn(symbol)
            if p and p > 0:
                return p, symbol
        except Exception:
            pass
    print(f"[Price] All sources failed for {symbol}")
    return None, None


# ── Historical OHLCV ──────────────────────────────────────────────────────────

def _history_twelvedata(symbol, days=365):
    key = os.getenv("TWELVE_DATA_KEY", "")
    if not key:
        return None
    try:
        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol={to_td_symbol(symbol)}&interval=1day"
            f"&outputsize={min(days, 5000)}&apikey={key}&format=JSON"
        )
        data = requests.get(url, timeout=15).json()
        if data.get("status") == "error":
            print(f"[TD hist] {symbol}: {data.get('message')}")
            return None
        rows = data.get("values", [])
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


def _history_stooq(symbol, days=365):
    try:
        from datetime import date, timedelta
        start = (date.today() - timedelta(days=int(days * 1.5) + 90)).strftime("%Y%m%d")
        end   = date.today().strftime("%Y%m%d")
        url   = f"https://stooq.com/q/d/l/?s={to_stooq_symbol(symbol)}&d1={start}&d2={end}&i=d"
        resp  = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        text = resp.text.strip()
        if not text or "No data" in text or len(text) < 30:
            return None
        df = pd.read_csv(StringIO(text))
        if df.empty:
            return None
        df.columns = [c.strip().title() for c in df.columns]
        if "Close" not in df.columns or "Date" not in df.columns:
            print(f"[Stooq hist] {symbol}: bad columns {list(df.columns)}")
            return None
        df["Date"]   = pd.to_datetime(df["Date"], errors="coerce")
        df           = df.dropna(subset=["Date"])
        df           = df.set_index("Date").sort_index()
        df["Close"]  = pd.to_numeric(df["Close"],  errors="coerce")
        df["Open"]   = pd.to_numeric(df.get("Open",  df["Close"]), errors="coerce")
        df["High"]   = pd.to_numeric(df.get("High",  df["Close"]), errors="coerce")
        df["Low"]    = pd.to_numeric(df.get("Low",   df["Close"]), errors="coerce")
        df["Volume"] = pd.to_numeric(df.get("Volume", 0),          errors="coerce").fillna(0)
        df = df.dropna(subset=["Close"])
        df.index = df.index.tz_localize(None) if df.index.tzinfo is not None else df.index
        df["ds"] = df.index
        df["y"]  = df["Close"]
        print(f"[Stooq hist] {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[Stooq hist] {symbol}: {e}")
    return None


def fetch_history(symbol, days=365):
    """Returns DataFrame with OHLCV + ds + y columns, or None."""
    for fn in (_history_twelvedata, _history_stooq):
        try:
            df = fn(symbol, days)
            if df is not None and len(df) >= 10:
                return df
        except Exception:
            pass
    print(f"[History] All sources failed for {symbol}")
    return None


# ── Batch history fetch ───────────────────────────────────────────────────────

def fetch_history_batch(symbols, days=60):
    """Returns {symbol: DataFrame} for all successful fetches."""
    if not symbols:
        return {}

    result = {}
    key    = os.getenv("TWELVE_DATA_KEY", "")

    if key:
        batch_size = 8
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(_history_twelvedata, s, days): s for s in batch}
                for fut in as_completed(futures):
                    sym = futures[fut]
                    try:
                        df = fut.result()
                        if df is not None and len(df) >= 10:
                            result[sym] = df
                    except Exception:
                        pass
            if i + batch_size < len(symbols):
                time.sleep(61)
        print(f"[Batch TD] Got {len(result)}/{len(symbols)}")

    missing = [s for s in symbols if s not in result]
    if missing:
        print(f"[Batch Stooq] Fetching {len(missing)} symbols...")
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(_history_stooq, s, days): s for s in missing}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    df = fut.result()
                    if df is not None and len(df) >= 10:
                        result[sym] = df
                except Exception:
                    pass
        print(f"[Batch total] {len(result)}/{len(symbols)}")

    return result
