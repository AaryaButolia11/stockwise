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

def _yf_session():
    """Browser-like session to avoid Yahoo rate-limiting on cloud/Render IPs."""
    import requests as _req
    s = _req.Session()
    s.headers.update({
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
    return s


def _batch_fetch(symbols: list, period: str = "30d") -> dict:
    """
    Download all symbols in ONE yfinance call — far faster than one-by-one.
    Uses custom session to avoid Yahoo rate-limiting on cloud/Render IPs.
    Returns dict: symbol -> DataFrame
    """
    if not symbols:
        return {}
    import time
    for attempt in range(3):
        try:
            raw = yf.download(
                tickers=symbols,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
                session=_yf_session(),
            )
            result = {}
            if len(symbols) == 1:
                sym = symbols[0]
                if not raw.empty:
                    result[sym] = raw
            else:
                for sym in symbols:
                    try:
                        df = raw[sym].dropna(how="all")
                        if not df.empty:
                            result[sym] = df
                    except Exception:
                        pass
            if result:
                return result
            print(f"[Recommender] Empty result attempt {attempt+1}, retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"[Recommender] Batch fetch error attempt {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(2)
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
    """Gap-up open = buying interest."""
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
    """
    Fast trend-based gain estimate using linear regression on last 10 days.
    No ML — runs in microseconds. Replaces slow LSTM for recommendations.
    """
    if len(hist) < 10:
        return 0.0
    try:
        closes    = hist["Close"].values[-10:]
        x         = np.arange(len(closes))
        slope, intercept = np.polyfit(x, closes, 1)
        current   = closes[-1]
        predicted = intercept + slope * (len(closes) + 4)   # 5 days ahead
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
    No LSTM training — that happens only on-demand in /get_forecast.
    """
    print(f"[Recommender] Fast recommendations for {date.today()}...")
    stocks = load_nifty50()
    if not stocks:
        print("[Recommender] No stocks loaded.")
        return []

    symbols     = [s for s, _ in stocks]
    company_map = {s: c for s, c in stocks}

    # One HTTP round-trip for all 50 symbols
    print(f"[Recommender] Batch fetching {len(symbols)} symbols...")
    hist_map = _batch_fetch(symbols, period="30d")
    print(f"[Recommender] Got data for {len(hist_map)}/{len(symbols)} symbols.")

    # Score in parallel
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

    print(f"[Recommender] Done in fast mode. Top 5: {[r['symbol'] for r in top5]}")
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
        print(f"[Recommender] DB save error: {e}")
        traceback.print_exc()
        return False
    finally:
        try: cur.close(); conn.close()
        except: pass


def get_todays_recommendations() -> list:
    """Fetch today's cached recommendations — instant DB read."""
    import db
    try:
        conn = db.get_conn()
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT * FROM ai_recommendations
            WHERE date = CURDATE()
            ORDER BY rank ASC
        """)
        rows = cur.fetchall()
        for r in rows:
            if r.get("date"):       r["date"]       = str(r["date"])
            if r.get("created_at"): r["created_at"] = str(r["created_at"])
        return rows
    except Exception as e:
        print(f"[Recommender] DB fetch error: {e}")
        return []
    finally:
        try: cur.close(); conn.close()
        except: pass


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
        try:
            row  = hist.iloc[-1]
            conn = db.get_conn()
            cur  = conn.cursor()
            cur.execute("""
                INSERT INTO daily_prices
                  (date, stock_symbol, open_price, close_price, high_price, low_price, volume, pct_change)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                  close_price=VALUES(close_price),
                  high_price=VALUES(high_price),
                  low_price=VALUES(low_price),
                  volume=VALUES(volume),
                  pct_change=VALUES(pct_change)
            """, (
                today, symbol,
                float(row["Open"]),   float(row["Close"]),
                float(row["High"]),   float(row["Low"]),
                int(row["Volume"]),
                round(((float(row["Close"]) - float(row["Open"])) / float(row["Open"])) * 100, 2)
            ))
            conn.commit()
            cur.close(); conn.close()
        except Exception as e:
            print(f"[Prices] Error tracking {symbol}: {e}")