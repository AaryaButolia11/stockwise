"""
recommender.py — Fast AI Stock Recommender for Nifty 50
Data: Twelve Data → Stooq (zero yfinance — blocked on cloud IPs)

Scoring factors:
  1. Momentum score  (40%) — 5d / 10d / 20d price trend
  2. Volatility score(30%) — lower vol = safer, higher score
  3. Volume surge    (20%) — unusual buying vs 20d avg
  4. Gap score       (10%) — today's open vs yesterday's close
"""

import os, csv, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
import numpy as np
import pandas as pd
import pytz
import psycopg2.extras

from data_fetch import fetch_history_batch, fetch_history

IST = pytz.timezone("Asia/Kolkata")


# ── Load Nifty 50 symbols ─────────────────────────────────────────────────────

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
        return float(min(100, max(10, 100 - (vol * 20))))
    except Exception:
        return 50.0


def _volume_score(hist: pd.DataFrame) -> float:
    if len(hist) < 5 or "Volume" not in hist.columns:
        return 50.0
    try:
        avg_vol  = hist["Volume"].iloc[:-1].mean()
        last_vol = hist["Volume"].iloc[-1]
        if avg_vol == 0:
            return 50.0
        return float(min(100, (last_vol / avg_vol) * 50))
    except Exception:
        return 50.0


def _gap_score(hist: pd.DataFrame) -> float:
    if len(hist) < 2 or "Open" not in hist.columns:
        return 50.0
    try:
        prev_close = float(hist["Close"].iloc[-2])
        today_open = float(hist["Open"].iloc[-1])
        gap_pct    = ((today_open - prev_close) / prev_close) * 100
        return float(min(100, max(0, 50 + gap_pct * 25)))
    except Exception:
        return 50.0


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


def _estimate_gain(hist: pd.DataFrame) -> float:
    if len(hist) < 20:
        return 0.0
    try:
        closes  = hist["Close"].values
        current = closes[-1]

        slope5,  _ = np.polyfit(np.arange(5),  closes[-5:],  1)
        slope20, _ = np.polyfit(np.arange(20), closes[-20:], 1)
        trend5     = (slope5  / current) * 100
        trend20    = (slope20 / current) * 100

        ma20       = closes[-20:].mean()
        ma_gap     = ((current - ma20) / ma20) * 100

        rsi        = _rsi(closes)
        rsi_signal = (rsi - 50) / 50 * 3

        recovery   = ((closes[-1] - closes[-4]) / closes[-4]) * 100 if len(closes) >= 5 else 0

        gain = (trend5 * 2.0 + trend20 * 1.0 + ma_gap * 0.3 +
                rsi_signal * 1.0 + recovery * 0.5)
        return float(round(max(-10.0, min(10.0, gain)), 2))
    except Exception:
        return 0.0


def _score_from_hist(symbol: str, company: str, hist: pd.DataFrame):
    try:
        if hist is None or hist.empty or len(hist) < 10:
            return None

        current_price = float(hist["Close"].iloc[-1])
        open_price    = float(hist["Open"].iloc[-1]) if "Open" in hist.columns else current_price
        closes        = hist["Close"].values

        mom_score = _momentum_score(hist)
        vol_score = _volatility_score(hist)
        vum_score = _volume_score(hist)
        gap_score = _gap_score(hist)
        est_gain  = _estimate_gain(hist)
        rsi_val   = _rsi(closes)

        total_score = (
            mom_score * 0.40 +
            vol_score * 0.30 +
            vum_score * 0.20 +
            gap_score * 0.10
        )

        reasons = []
        if mom_score > 65:  reasons.append("strong upward momentum")
        if vol_score > 70:  reasons.append("low volatility")
        if vum_score > 70:  reasons.append("high buying volume")
        if gap_score > 65:  reasons.append("gap-up open")
        if rsi_val > 55:    reasons.append(f"RSI bullish ({rsi_val:.0f})")
        if rsi_val < 35:    reasons.append(f"oversold RSI ({rsi_val:.0f}) — bounce potential")
        if est_gain > 0.5:  reasons.append(f"trend projects +{est_gain:.1f}%")

        reason       = ("Based on " + ", ".join(reasons)) if reasons else "Neutral technical signals"
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
            "rsi":            round(rsi_val, 1),
        }
    except Exception as e:
        print(f"[Recommender] Score error for {symbol}: {e}")
        return None


# ── Main scoring ──────────────────────────────────────────────────────────────

def generate_recommendations() -> list:
    print(f"[Recommender] Generating recommendations for {date.today()}...")
    stocks = load_nifty50()
    if not stocks:
        print("[Recommender] No stocks loaded.")
        return []

    symbols     = [s for s, _ in stocks]
    company_map = {s: c for s, c in stocks}

    print(f"[Recommender] Fetching 60-day history for {len(symbols)} symbols...")
    hist_map = fetch_history_batch(symbols, days=60)
    print(f"[Recommender] Got data for {len(hist_map)}/{len(symbols)} symbols.")

    if not hist_map:
        print("[Recommender] No market data returned.")
        return []

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
        print("[Recommender] Scoring returned no results.")
        return []

    bullish = [r for r in results if r["predicted_gain"] > 0]
    print(f"[Recommender] {len(bullish)} bullish / {len(results) - len(bullish)} bearish")

    pool = bullish if len(bullish) >= 5 else results

    for r in pool:
        r["_rank"] = r["score"] * 0.6 + min(r["predicted_gain"] * 10, 40)
    pool.sort(key=lambda x: x["_rank"], reverse=True)

    top5 = pool[:5]
    for i, r in enumerate(top5):
        r["rank"] = i + 1
        r.pop("_rank", None)

    print(f"[Recommender] Top 5: {[(r['symbol'], r['predicted_gain']) for r in top5]}")
    return top5


def score_stock(symbol: str, company: str):
    try:
        hist = fetch_history(symbol, days=60)
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
    import db
    conn = cur = None
    try:
        conn = db.get_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT * FROM ai_recommendations
            WHERE date = CURRENT_DATE
            ORDER BY rank ASC
        """)
        rows   = cur.fetchall()
        result = []
        for r in rows:
            row = dict(r)
            if row.get("date"):       row["date"]       = str(row["date"])
            if row.get("created_at"): row["created_at"] = str(row["created_at"])
            for field in ("score", "predicted_gain", "current_price", "target_price"):
                if row.get(field) is not None:
                    row[field] = float(row[field])
            result.append(row)
        print(f"[Recommender] Fetched {len(result)} recommendations from DB.")
        return result
    except Exception as e:
        print(f"[Recommender] DB fetch error: {e}")
        traceback.print_exc()
        return []
    finally:
        if cur:  cur.close()
        if conn: db.release_conn(conn)


def track_daily_prices():
    import db
    stocks   = load_nifty50()
    symbols  = [s for s, _ in stocks]
    today    = date.today()
    hist_map = fetch_history_batch(symbols, days=2)

    for symbol, _ in stocks:
        hist = hist_map.get(symbol)
        if hist is None or hist.empty:
            continue
        conn = cur = None
        try:
            row  = hist.iloc[-1]
            conn = db.get_conn()
            cur  = conn.cursor()
            cur.execute("""
                INSERT INTO daily_prices
                  (date, stock_symbol, open_price, close_price, high_price,
                   low_price, volume, pct_change)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date, stock_symbol) DO UPDATE SET
                  close_price = EXCLUDED.close_price,
                  high_price  = EXCLUDED.high_price,
                  low_price   = EXCLUDED.low_price,
                  volume      = EXCLUDED.volume,
                  pct_change  = EXCLUDED.pct_change
            """, (
                today, symbol,
                float(row.get("Open",  row["Close"])),
                float(row["Close"]),
                float(row.get("High",  row["Close"])),
                float(row.get("Low",   row["Close"])),
                int(row.get("Volume",  0)),
                round(((float(row["Close"]) - float(row.get("Open", row["Close"]))) /
                        float(row.get("Open", row["Close"]))) * 100, 2)
                if row.get("Open") else 0.0
            ))
            conn.commit()
        except Exception as e:
            if conn: conn.rollback()
            print(f"[Prices] Error tracking {symbol}: {e}")
        finally:
            if cur:  cur.close()
            if conn: db.release_conn(conn)
