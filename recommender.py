"""
recommender.py — Fast AI Stock Recommender for Nifty 50
Schema-aligned to existing Supabase tables:
  - ai_recommendations.rank  (not rec_rank)
  - No UNIQUE constraint on ai_recommendations → use DELETE+INSERT
  - No UNIQUE constraint on daily_prices       → use DELETE+INSERT
"""

import os, csv, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
import numpy as np
import pandas as pd
import yfinance as yf
import pytz

IST = pytz.timezone("Asia/Kolkata")

def _ist_now():      return datetime.now(IST)
def _is_market_day(): return _ist_now().weekday() < 5


# ── Load Nifty 50 ─────────────────────────────────────────────────────────────

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


# ── Batch yfinance fetch ──────────────────────────────────────────────────────

def _batch_fetch(symbols: list, period: str = "30d") -> dict:
    if not symbols:
        return {}
    try:
        raw = yf.download(
            tickers=symbols, period=period, interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
        result = {}
        if len(symbols) == 1:
            if not raw.empty:
                result[symbols[0]] = raw
        else:
            for sym in symbols:
                try:
                    df = raw[sym].dropna(how="all")
                    if not df.empty:
                        result[sym] = df
                except Exception:
                    pass
        return result
    except Exception as e:
        print(f"[Recommender] Batch fetch error: {e}")
        return {}


# ── Technical scoring ─────────────────────────────────────────────────────────

def _momentum_score(hist) -> float:
    if len(hist) < 21: return 50.0
    c = hist["Close"].values
    try:
        score = ((c[-1]-c[-6])/c[-6]*.5 + (c[-1]-c[-11])/c[-11]*.3 + (c[-1]-c[-21])/c[-21]*.2) * 100 * 5
        return float(min(100, max(0, 50 + score)))
    except: return 50.0

def _volatility_score(hist) -> float:
    if len(hist) < 10: return 50.0
    try:
        vol = hist["Close"].pct_change().dropna().std() * 100
        return float(min(100, max(10, 100 - vol * 20)))
    except: return 50.0

def _volume_score(hist) -> float:
    if len(hist) < 5: return 50.0
    try:
        avg = hist["Volume"].iloc[:-1].mean()
        return float(min(100, (hist["Volume"].iloc[-1] / avg * 50) if avg else 50.0))
    except: return 50.0

def _gap_score(hist) -> float:
    if len(hist) < 2: return 50.0
    try:
        gap = ((float(hist["Open"].iloc[-1]) - float(hist["Close"].iloc[-2]))
               / float(hist["Close"].iloc[-2]) * 100)
        return float(min(100, max(0, 50 + gap * 25)))
    except: return 50.0

def _estimate_gain(hist) -> float:
    if len(hist) < 10: return 0.0
    try:
        closes = hist["Close"].values[-10:]
        x = np.arange(len(closes))
        slope, intercept = np.polyfit(x, closes, 1)
        predicted = intercept + slope * (len(closes) + 4)
        return float(round(((predicted - closes[-1]) / closes[-1]) * 100, 2))
    except: return 0.0


def _score_from_hist(symbol: str, company: str, hist):
    try:
        if hist is None or hist.empty or len(hist) < 5:
            return None
        current_price = float(hist["Close"].iloc[-1])
        mom  = _momentum_score(hist)
        vol  = _volatility_score(hist)
        vum  = _volume_score(hist)
        gap  = _gap_score(hist)
        gain = _estimate_gain(hist)
        total = mom*.40 + vol*.30 + vum*.20 + gap*.10
        reasons = []
        if mom  > 65: reasons.append("strong upward momentum")
        if vol  > 70: reasons.append("low volatility")
        if vum  > 70: reasons.append("high buying volume")
        if gap  > 65: reasons.append("gap-up open")
        if gain > 1:  reasons.append(f"trend projects +{gain:.1f}%")
        return {
            "symbol":         symbol,
            "company":        company,
            "score":          round(total, 2),
            "predicted_gain": gain,
            "current_price":  current_price,
            "open_price":     float(hist["Open"].iloc[-1]),
            "target_price":   round(current_price * (1 + gain / 100), 2),
            "reason":         ("Based on " + ", ".join(reasons)) if reasons
                              else "Balanced risk-reward profile",
            "momentum":       round(mom, 1),
            "volatility":     round(vol, 1),
            "volume":         round(vum, 1),
        }
    except Exception as e:
        print(f"[Recommender] Score error {symbol}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_recommendations() -> list:
    print(f"[Recommender] Fast recommendations for {date.today()}...")
    stocks = load_nifty50()
    if not stocks:
        return []
    symbols     = [s for s, _ in stocks]
    company_map = {s: c for s, c in stocks}
    print(f"[Recommender] Batch fetching {len(symbols)} symbols...")
    hist_map = _batch_fetch(symbols, period="30d")
    print(f"[Recommender] Got data for {len(hist_map)}/{len(symbols)} symbols.")
    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {
            ex.submit(_score_from_hist, s, company_map[s], hist_map[s]): s
            for s in hist_map
        }
        for fut in as_completed(futures):
            r = fut.result()
            if r: results.append(r)
    if not results:
        return []
    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    for i, r in enumerate(top5):
        r["rank"] = i + 1
    print(f"[Recommender] Top 5: {[r['symbol'] for r in top5]}")
    return top5


def score_stock(symbol: str, company: str):
    try:
        return _score_from_hist(symbol, company, yf.Ticker(symbol).history(period="30d"))
    except Exception as e:
        print(f"[Recommender] score_stock error {symbol}: {e}")
        return None


# ── DB persistence — matches exact Supabase schema ───────────────────────────

def save_recommendations(recommendations: list):
    """
    Supabase ai_recommendations has no UNIQUE constraint,
    so we DELETE today's rows first, then INSERT fresh ones.
    Column name is 'rank' (as in the actual table).
    """
    import db
    conn = cur = None
    try:
        conn  = db.get_conn()
        cur   = conn.cursor()
        today = date.today()
        # Clear today's existing rows
        cur.execute("DELETE FROM ai_recommendations WHERE date=%s", (today,))
        for r in recommendations:
            cur.execute("""
                INSERT INTO ai_recommendations
                  (date, stock_symbol, company_name, score, predicted_gain,
                   current_price, target_price, reason, rank)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                today,
                r["symbol"],
                r["company"],
                r["score"],
                r["predicted_gain"],
                r["current_price"],
                r["target_price"],
                r["reason"],
                r["rank"],
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
    """
    Fetch today's cached recs. Uses CURRENT_DATE (standard SQL).
    Column is 'rank' matching the Supabase table.
    """
    import db
    conn = cur = None
    try:
        conn = db.get_conn()
        cur  = db._dict_cursor(conn)
        cur.execute("""
            SELECT * FROM ai_recommendations
            WHERE date = CURRENT_DATE
            ORDER BY rank ASC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            if r.get("date"):       r["date"]       = str(r["date"])
            if r.get("created_at"): r["created_at"] = str(r["created_at"])
        return rows
    except Exception as e:
        print(f"[Recommender] DB fetch error: {e}")
        return []
    finally:
        if cur:  cur.close()
        if conn: db.release_conn(conn)


def track_daily_prices():
    """
    Batch-fetch open/close for all Nifty 50.
    daily_prices has no UNIQUE constraint → DELETE today's rows first.
    """
    import db
    stocks   = load_nifty50()
    symbols  = [s for s, _ in stocks]
    today    = date.today()
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
            # No UNIQUE constraint — delete existing row for today first
            cur.execute(
                "DELETE FROM daily_prices WHERE date=%s AND stock_symbol=%s",
                (today, symbol)
            )
            cur.execute("""
                INSERT INTO daily_prices
                  (date, stock_symbol, open_price, close_price,
                   high_price, low_price, volume, pct_change)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                today, symbol,
                float(row["Open"]),  float(row["Close"]),
                float(row["High"]),  float(row["Low"]),
                int(row["Volume"]),
                round(((float(row["Close"]) - float(row["Open"]))
                       / float(row["Open"])) * 100, 2),
            ))
            conn.commit()
        except Exception as e:
            if conn: conn.rollback()
            print(f"[Prices] Error {symbol}: {e}")
        finally:
            if cur:  cur.close()
            if conn: db.release_conn(conn)