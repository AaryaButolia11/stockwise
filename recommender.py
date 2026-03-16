"""
recommender.py — Fast AI Stock Recommender for Nifty 50
Runs every morning at 9:15 AM IST.

SPEED PHILOSOPHY:
  • Recommendations use ONLY fast technical indicators (no LSTM).
    LSTM is expensive (~3-5 min per stock) and is reserved for the
    on-demand /get_forecast endpoint when a user picks a specific stock.
  • All 50 stocks are fetched in ONE yfinance batch call.
  • Results cached in DB — repeated calls within same day are instant.

Scoring factors (pure technical, runs in < 15 seconds for all 50):
  1. Momentum score  (40%) — 5d / 10d / 20d price trend
  2. Volatility score(30%) — lower vol = safer, higher score
  3. Volume surge    (20%) — unusual buying activity vs 20d avg
  4. Gap score       (10%) — today's open vs yesterday's close
"""

import os, csv, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import pytz

IST = pytz.timezone("Asia/Kolkata")

def _ist_now():
    return datetime.now(IST)

def _is_market_day():
    return _ist_now().weekday() < 5


# ── Load all Nifty 50 symbols ─────────────────────────────────────────────────

def load_nifty50():
    path = os.path.join(os.path.dirname(__file__), "companies_india.csv")
    out  = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "Symbol" in row and "Company" in row:
                    out.append((row["Symbol"].strip(), row["Company"].strip()))
    except Exception as e:
        print(f"[Recommender] Error loading companies: {e}")
    return out


# ── Fast batch data fetch ─────────────────────────────────────────────────────

TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "")

def _to_td_symbol(symbol: str) -> str:
    if symbol.endswith(".NS"): return symbol.replace(".NS", "") + ":NSE"
    if symbol.endswith(".BO"): return symbol.replace(".BO", "") + ":BSE"
    return symbol

def _batch_fetch_twelvedata(symbols: list, period: str = "30d") -> dict:
    """
    Fetch all symbols from Twelve Data in parallel threads.
    Free plan: 800 req/day, 8 req/min — throttled accordingly.
    """
    import requests as _req, time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not TWELVE_DATA_KEY:
        return {}

    outputsize = 30 if period == "30d" else 5

    def fetch_one(sym):
        try:
            td_sym = _to_td_symbol(sym)
            url    = (
                f"https://api.twelvedata.com/time_series"
                f"?symbol={td_sym}&interval=1day&outputsize={outputsize}"
                f"&apikey={TWELVE_DATA_KEY}&format=JSON"
            )
            resp = _req.get(url, timeout=10)
            data = resp.json()
            if data.get("status") == "error":
                return sym, None
            rows = data.get("values", [])
            if not rows:
                return sym, None
            df = pd.DataFrame(rows)
            df["Date"]   = pd.to_datetime(df["datetime"])
            df["Open"]   = df["open"].astype(float)
            df["High"]   = df["high"].astype(float)
            df["Low"]    = df["low"].astype(float)
            df["Close"]  = df["close"].astype(float)
            df["Volume"] = df["volume"].astype(float)
            df = df.set_index("Date").sort_index()
            return sym, df
        except Exception as e:
            print(f"[TwelveData] {sym}: {e}")
            return sym, None

    result = {}
    batch_size = 8
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(fetch_one, s): s for s in batch}
            for fut in as_completed(futures):
                sym, df = fut.result()
                if df is not None and not df.empty:
                    result[sym] = df
        if i + batch_size < len(symbols):
            time.sleep(60)

    print(f"[TwelveData] Fetched {len(result)}/{len(symbols)} symbols")
    return result


def _batch_fetch(symbols: list, period: str = "30d") -> dict:
    """
    Fetch all symbols — Twelve Data first, yfinance as fallback.
    """
    if not symbols:
        return {}

    result = _batch_fetch_twelvedata(symbols, period)
    if result:
        return result

    print("[Recommender] Twelve Data empty, falling back to yfinance...")
    import time
    for attempt in range(3):
        try:
            raw = yf.download(
                tickers=symbols, period=period, interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )
            fb = {}
            if len(symbols) == 1:
                if not raw.empty: fb[symbols[0]] = raw
            else:
                for sym in symbols:
                    try:
                        df = raw[sym].dropna(how="all")
                        if not df.empty: fb[sym] = df
                    except Exception:
                        pass
            if fb:
                return fb
            time.sleep(2)
        except Exception as e:
            print(f"[yfinance] Batch attempt {attempt+1}: {e}")
            if attempt < 2: time.sleep(2)
    return {}


# ── Scoring functions ─────────────────────────────────────────────────────────

def _momentum_score(hist: pd.DataFrame) -> float:
    if len(hist) < 21:
        return 50.0
    close = hist["Close"].values
    try:
        m5  = (close[-1] - close[-6])  / close[-6]  * 100
        m10 = (close[-1] - close[-11]) / close[-11] * 100
        m20 = (close[-1] - close[-21]) / close[-21] * 100
        score = (m5 * 0.5) + (m10 * 0.3) + (m20 * 0.2)
        return float(min(100, max(0, 50 + score * 5)))
    except Exception:
        return 50.0


def _volatility_score(hist: pd.DataFrame) -> float:
    if len(hist) < 10:
        return 50.0
    try:
        returns = hist["Close"].pct_change().dropna()
        vol     = returns.std() * 100
        score   = max(10, 100 - (vol * 20))
        return float(min(100, score))
    except Exception:
        return 50.0


def _volume_score(hist: pd.DataFrame) -> float:
    if len(hist) < 5:
        return 50.0
    try:
        avg_vol  = hist["Volume"].iloc[:-1].mean()
        last_vol = hist["Volume"].iloc[-1]
        if avg_vol == 0:
            return 50.0
        ratio = last_vol / avg_vol
        return float(min(100, ratio * 50))
    except Exception:
        return 50.0


def _gap_score(hist: pd.DataFrame) -> float:
    if len(hist) < 2:
        return 50.0
    try:
        prev_close = float(hist["Close"].iloc[-2])
        today_open = float(hist["Open"].iloc[-1])
        gap_pct    = ((today_open - prev_close) / prev_close) * 100
        return float(min(100, max(0, 50 + gap_pct * 25)))
    except Exception:
        return 50.0


def _estimate_gain(hist: pd.DataFrame) -> float:
    if len(hist) < 10:
        return 0.0
    try:
        closes    = hist["Close"].values[-10:]
        x         = np.arange(len(closes))
        slope, intercept = np.polyfit(x, closes, 1)
        current   = closes[-1]
        predicted = intercept + slope * (len(closes) + 4)
        gain      = ((predicted - current) / current) * 100
        return float(round(gain, 2))
    except Exception:
        return 0.0


# ── Score a single stock from pre-fetched data ────────────────────────────────

def _score_from_hist(symbol: str, company: str, hist: pd.DataFrame):
    try:
        if hist.empty or len(hist) < 5:
            return None

        current_price = float(hist["Close"].iloc[-1])
        open_price    = float(hist["Open"].iloc[-1])

        mom_score = _momentum_score(hist)
        vol_score = _volatility_score(hist)
        vum_score = _volume_score(hist)
        gap_score = _gap_score(hist)
        est_gain  = _estimate_gain(hist)

        total_score = (
            mom_score * 0.40 +
            vol_score * 0.30 +
            vum_score * 0.20 +
            gap_score * 0.10
        )

        reasons = []
        if mom_score > 65: reasons.append("strong upward momentum")
        if vol_score > 70: reasons.append("low volatility")
        if vum_score > 70: reasons.append("high buying volume")
        if gap_score > 65: reasons.append("gap-up open")
        if est_gain  > 1:  reasons.append(f"trend projects +{est_gain:.1f}%")

        reason       = ("Based on " + ", ".join(reasons)) if reasons else "Balanced risk-reward profile"
        target_price = round(current_price * (1 + est_gain / 100), 2)

        return {
            "symbol":         symbol,
            "company":        company,
            "score":          round(total_score, 2),
            "predicted_gain": est_gain,
            "current_price":  current_price,
            "open_price":     open_price,
            "target_price":   target_price,
            "reason":         reason,
            "momentum":       round(mom_score, 1),
            "volatility":     round(vol_score, 1),
            "volume":         round(vum_score, 1),
        }
    except Exception as e:
        print(f"[Recommender] Score error for {symbol}: {e}")
        return None


# ── Main fast scoring ─────────────────────────────────────────────────────────

def generate_recommendations() -> list:
    """
    Score all Nifty 50 in < 15 seconds using batch fetch + parallel scoring.
    """
    print(f"[Recommender] Fast recommendations for {date.today()}...")
    stocks = load_nifty50()
    if not stocks:
        print("[Recommender] No stocks loaded.")
        return []

    symbols     = [s for s, _ in stocks]
    company_map = {s: c for s, c in stocks}

    print(f"[Recommender] Batch fetching {len(symbols)} symbols...")
    hist_map = _batch_fetch(symbols, period="30d")
    print(f"[Recommender] Got data for {len(hist_map)}/{len(symbols)} symbols.")

    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {
            ex.submit(_score_from_hist, sym, company_map[sym], hist_map[sym]): sym
            for sym in hist_map
        }
        for fut in as_completed(futures):
            scored = fut.result()
            if scored:
                results.append(scored)

    if not results:
        print("[Recommender] No results generated.")
        return []

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    for i, r in enumerate(top5):
        r["rank"] = i + 1

    print(f"[Recommender] Done. Top 5: {[r['symbol'] for r in top5]}")
    return top5


# ── Legacy single-stock scorer ────────────────────────────────────────────────

def score_stock(symbol: str, company: str):
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="30d")
        return _score_from_hist(symbol, company, hist)
    except Exception as e:
        print(f"[Recommender] score_stock error for {symbol}: {e}")
        return None


# ── DB persistence ────────────────────────────────────────────────────────────

def save_recommendations(recommendations: list):
    import db
    conn = cur = None
    try:
        conn  = db.get_conn()
        cur   = conn.cursor()
        today = date.today()
        cur.execute("DELETE FROM ai_recommendations WHERE date=%s", (today,))
        for r in recommendations:
            cur.execute("""
                INSERT INTO ai_recommendations
                  (date, stock_symbol, company_name, score, predicted_gain,
                   current_price, target_price, reason, rank)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                today, r["symbol"], r["company"], r["score"],
                r["predicted_gain"], r["current_price"],
                r["target_price"], r["reason"], r["rank"]
            ))
        conn.commit()
        print(f"[Recommender] Saved {len(recommendations)} recommendations.")
        return True
    except Exception as e:
        if conn: conn.rollback()
        print(f"[Recommender] DB save error: {e}")
        traceback.print_exc()
        return False
    finally:
        if cur:  cur.close()
        if conn: db.release_conn(conn)


def get_todays_recommendations() -> list:
    """Fetch today's cached recommendations — instant DB read."""
    import db
    import psycopg2.extras
    conn = cur = None
    try:
        conn = db.get_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # PostgreSQL uses CURRENT_DATE (not MySQL's CURDATE())
        cur.execute("""
            SELECT * FROM ai_recommendations
            WHERE date = CURRENT_DATE
            ORDER BY rank ASC
        """)
        rows = cur.fetchall()
        result = []
        for r in rows:
            row = dict(r)
            if row.get("date"):       row["date"]       = str(row["date"])
            if row.get("created_at"): row["created_at"] = str(row["created_at"])
            result.append(row)
        return result
    except Exception as e:
        print(f"[Recommender] DB fetch error: {e}")
        return []
    finally:
        if cur:  cur.close()
        if conn: db.release_conn(conn)


def track_daily_prices():
    """Batch-fetch open/close prices for all Nifty 50."""
    import db
    stocks  = load_nifty50()
    symbols = [s for s, _ in stocks]
    today   = date.today()

    hist_map = _batch_fetch(symbols, period="2d")

    for symbol, _ in stocks:
        hist = hist_map.get(symbol)
        if hist is None or hist.empty:
            continue
        conn = cur = None
        try:
            row  = hist.iloc[-1]
            conn = db.get_conn()
            cur  = conn.cursor()
            # PostgreSQL upsert — replaces MySQL's ON DUPLICATE KEY UPDATE
            cur.execute("""
                INSERT INTO daily_prices
                  (date, stock_symbol, open_price, close_price, high_price, low_price, volume, pct_change)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date, stock_symbol) DO UPDATE SET
                  close_price = EXCLUDED.close_price,
                  high_price  = EXCLUDED.high_price,
                  low_price   = EXCLUDED.low_price,
                  volume      = EXCLUDED.volume,
                  pct_change  = EXCLUDED.pct_change
            """, (
                today, symbol,
                float(row["Open"]),  float(row["Close"]),
                float(row["High"]),  float(row["Low"]),
                int(row["Volume"]),
                round(((float(row["Close"]) - float(row["Open"])) / float(row["Open"])) * 100, 2)
            ))
            conn.commit()
        except Exception as e:
            if conn: conn.rollback()
            print(f"[Prices] Error tracking {symbol}: {e}")
        finally:
            if cur:  cur.close()
            if conn: db.release_conn(conn)
