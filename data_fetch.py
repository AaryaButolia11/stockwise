"""
data_fetch.py — Centralised market data fetcher. Zero yfinance dependency.
Yahoo Finance blocks all cloud datacenter IPs. This module uses sources that work.

Price sources (tried in order):
  1. Twelve Data  — reliable on cloud, needs TWELVE_DATA_KEY env var
  2. NSE India API — free, no key, works on cloud for .NS stocks
  3. Stooq         — free CSV feed, works on cloud, global coverage

History sources (tried in order):
  1. Twelve Data  — up to 5 years daily OHLCV, needs TWELVE_DATA_KEY
  2. Stooq        — free daily OHLCV, works on cloud, good coverage
"""

import os
import time
import requests
import pandas as pd
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Symbol converters ────────────────────────────────────────────────────────

def to_td_symbol(symbol: str) -> str:
    """yfinance .NS/.BO → Twelve Data :NSE/:BSE"""
    if symbol.endswith(".NS"): return symbol.replace(".NS", "") + ":NSE"
    if symbol.endswith(".BO"): return symbol.replace(".BO", "") + ":BSE"
    return symbol

def to_stooq_symbol(symbol: str) -> str:
    """
    yfinance symbol → Stooq symbol.
    RELIANCE.NS → RELIANCE.NS  (Stooq uses same format for NSE)
    AAPL → AAPL.US
    """
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol.lower()
    return symbol.lower() + ".us"

def to_nse_symbol(symbol: str) -> str:
    """RELIANCE.NS → RELIANCE"""
    return symbol.replace(".NS", "").replace(".BO", "")


# ── Current price ─────────────────────────────────────────────────────────────

def _price_twelvedata(symbol: str) -> float | None:
    key = os.getenv("TWELVE_DATA_KEY", "")
    if not key:
        return None
    try:
        td_sym = to_td_symbol(symbol)
        url    = f"https://api.twelvedata.com/price?symbol={td_sym}&apikey={key}"
        resp   = requests.get(url, timeout=8)
        data   = resp.json()
        p = data.get("price")
        if p and float(p) > 0:
            print(f"[TD price] {symbol} = {p}")
            return float(p)
        print(f"[TD price] {symbol}: {data.get('message','no price')}")
    except Exception as e:
        print(f"[TD price] {symbol}: {e}")
    return None


def _price_nse(symbol: str) -> float | None:
    """NSE public API — Indian stocks only, no key needed."""
    if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
        return None
    try:
        nse_sym = to_nse_symbol(symbol)
        session = requests.Session()
        session.get(
            "https://www.nseindia.com",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=5,
        )
        resp = session.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={nse_sym}",
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


def _price_stooq(symbol: str) -> float | None:
    """Stooq last price via 5d CSV."""
    try:
        stooq_sym = to_stooq_symbol(symbol)
        url  = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200 or not resp.text.strip():
            return None
        df = pd.read_csv(StringIO(resp.text))
        if df.empty or "Close" not in df.columns:
            return None
        price = float(df["Close"].iloc[-1])
        if price > 0:
            print(f"[Stooq price] {symbol} = {price}")
            return price
    except Exception as e:
        print(f"[Stooq price] {symbol}: {e}")
    return None


def fetch_price(symbol: str) -> tuple:
    """
    Returns (price, symbol) or (None, None).
    Tries: Twelve Data → NSE → Stooq
    """
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

def _history_twelvedata(symbol: str, days: int = 365) -> pd.DataFrame | None:
    key = os.getenv("TWELVE_DATA_KEY", "")
    if not key:
        return None
    try:
        td_sym     = to_td_symbol(symbol)
        outputsize = min(days, 5000)
        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol={td_sym}&interval=1day&outputsize={outputsize}"
            f"&apikey={key}&format=JSON"
        )
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if data.get("status") == "error":
            print(f"[TD hist] {symbol}: {data.get('message')}")
            return None
        rows = data.get("values", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["Date"]   = pd.to_datetime(df["datetime"])
        df["Open"]   = df["open"].astype(float)
        df["High"]   = df["high"].astype(float)
        df["Low"]    = df["low"].astype(float)
        df["Close"]  = df["close"].astype(float)
        df["Volume"] = df["volume"].astype(float)
        df = df.set_index("Date").sort_index()
        # Also expose ds/y columns for LSTM compatibility
        df["ds"] = df.index
        df["y"]  = df["Close"]
        print(f"[TD hist] {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[TD hist] {symbol}: {e}")
    return None


def _history_stooq(symbol: str, days: int = 365) -> pd.DataFrame | None:
    """
    Stooq free daily OHLCV. Works on cloud, no API key.
    Coverage: all NSE stocks, US stocks, indices.
    """
    try:
        from datetime import date, timedelta
        stooq_sym  = to_stooq_symbol(symbol)
        # Request extra days to account for weekends/holidays
        start_date = (date.today() - timedelta(days=int(days * 1.5) + 90)).strftime("%Y%m%d")
        end_date   = date.today().strftime("%Y%m%d")
        url = (f"https://stooq.com/q/d/l/?s={stooq_sym}"
               f"&d1={start_date}&d2={end_date}&i=d")
        resp = requests.get(url, timeout=20,
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        if resp.status_code != 200:
            print(f"[Stooq hist] {symbol}: HTTP {resp.status_code}")
            return None
        text = resp.text.strip()
        if not text or "No data" in text or len(text) < 30:
            print(f"[Stooq hist] {symbol}: empty response")
            return None

        df = pd.read_csv(StringIO(text))
        if df.empty:
            return None

        # Normalize column names to Title Case (Stooq returns mixed case)
        df.columns = [c.strip().title() for c in df.columns]

        if "Close" not in df.columns or "Date" not in df.columns:
            print(f"[Stooq hist] {symbol}: missing columns: {list(df.columns)}")
            return None

        # Parse date and set as index
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date").sort_index()

        # Ensure all standard OHLCV columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = df["Close"]  # fallback to close

        df["Close"]  = pd.to_numeric(df["Close"],  errors="coerce")
        df["Open"]   = pd.to_numeric(df["Open"],   errors="coerce")
        df["High"]   = pd.to_numeric(df["High"],   errors="coerce")
        df["Low"]    = pd.to_numeric(df["Low"],    errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        df = df.dropna(subset=["Close"])

        # ds/y columns for LSTM compatibility — strip timezone safely
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert(None)
        else:
            idx = idx.tz_localize(None) if idx.tzinfo is not None else idx  # type: ignore
        df.index = idx
        df["ds"] = df.index
        df["y"]  = df["Close"]

        print(f"[Stooq hist] {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[Stooq hist] {symbol}: {e}")
    return None


def fetch_history(symbol: str, days: int = 365) -> pd.DataFrame | None:
    """
    Returns DataFrame with columns: Date(index), Open, High, Low, Close, Volume, ds, y
    Tries: Twelve Data → Stooq
    """
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

def fetch_history_batch(symbols: list, days: int = 60) -> dict:
    """
    Fetch history for multiple symbols in parallel.
    Returns {symbol: DataFrame} for successful fetches.
    """
    if not symbols:
        return {}

    result = {}

    # Try Twelve Data batch first (8 req/min free tier)
    key = os.getenv("TWELVE_DATA_KEY", "")
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
                time.sleep(61)   # Twelve Data free: 8 req/min

        print(f"[Batch TD] Got {len(result)}/{len(symbols)} symbols")
        if len(result) >= len(symbols) * 0.7:   # 70%+ success rate → use TD
            return result

    # Stooq fallback for missing symbols
    missing = [s for s in symbols if s not in result]
    if missing:
        print(f"[Batch] Fetching {len(missing)} symbols from Stooq...")
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
        print(f"[Batch Stooq] Total now: {len(result)}/{len(symbols)}")

    return result
