"""
recommender.py — AI Stock Recommender for Nifty 50
Runs every morning at 9:15 AM IST.
Scores all 50 stocks and picks top 5 to buy.

Scoring factors:
  1. LSTM predicted gain (40%) — our trained model's forecast
  2. Momentum score (30%)      — recent price trend (5d, 10d, 20d)
  3. Volatility score (20%)    — lower volatility = safer bet
  4. Volume surge (10%)        — unusual buying activity
"""

import os, csv, traceback
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import pytz

# ── IST timezone ─────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

def _ist_now():
    return datetime.now(IST)

def _is_market_day():
    """Monday–Friday only (simplified — doesn't check NSE holidays)."""
    return _ist_now().weekday() < 5


# ── Load all Nifty 50 symbols ────────────────────────────────────────────────

def load_nifty50():
    path = os.path.join(os.path.dirname(__file__), "companies_india.csv")
    out  = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "Symbol" in row and "Company" in row:
                    out.append((row["Symbol"], row["Company"]))
    except Exception as e:
        print(f"[Recommender] Error loading companies: {e}")
    return out


# ── Scoring functions ─────────────────────────────────────────────────────────

def _momentum_score(hist: pd.DataFrame) -> float:
    """
    Returns score 0-100.
    Combines 5-day, 10-day, 20-day price momentum.
    """
    if len(hist) < 21:
        return 50.0
    close = hist["Close"].values
    try:
        m5  = (close[-1] - close[-6])  / close[-6]  * 100
        m10 = (close[-1] - close[-11]) / close[-11] * 100
        m20 = (close[-1] - close[-21]) / close[-21] * 100
        # Weighted average — short term matters more
        score = (m5 * 0.5) + (m10 * 0.3) + (m20 * 0.2)
        # Normalize to 0-100
        return min(100, max(0, 50 + score * 5))
    except Exception:
        return 50.0


def _volatility_score(hist: pd.DataFrame) -> float:
    """
    Returns score 0-100.
    Lower volatility = higher score (safer stocks preferred).
    """
    if len(hist) < 10:
        return 50.0
    try:
        returns = hist["Close"].pct_change().dropna()
        vol     = returns.std() * 100  # daily std in %
        # vol of 1% = score 80, vol of 3% = score 40, vol of 5%+ = score 10
        score = max(10, 100 - (vol * 20))
        return min(100, score)
    except Exception:
        return 50.0


def _volume_score(hist: pd.DataFrame) -> float:
    """
    Returns score 0-100.
    Checks if today's volume is higher than 20-day average.
    Volume surge = strong buying interest.
    """
    if len(hist) < 5:
        return 50.0
    try:
        avg_vol  = hist["Volume"].iloc[:-1].mean()
        last_vol = hist["Volume"].iloc[-1]
        if avg_vol == 0:
            return 50.0
        ratio = last_vol / avg_vol
        # 2x volume = score 100, 1x = score 50, 0.5x = score 25
        return min(100, ratio * 50)
    except Exception:
        return 50.0


def _lstm_predicted_gain(symbol: str, current_price: float) -> float:
    """
    Uses our trained LSTM model to predict next-day gain %.
    Returns predicted % change (positive = gain, negative = loss).
    """
    try:
        from ml_model import get_aggregated_forecast, fetch_stock_data, get_or_train_model, _forecast_days, _ci, LOOKBACK
        df = fetch_stock_data(symbol)
        if df is None or df.empty or len(df) < LOOKBACK + 10:
            return 0.0

        model, scaler, last_win = get_or_train_model(symbol, df)
        # Predict next 5 days and take average
        preds = _forecast_days(model, last_win, 5, scaler)
        avg_5d_price    = float(np.mean(preds))
        predicted_gain  = ((avg_5d_price - current_price) / current_price) * 100
        return round(predicted_gain, 2)
    except Exception as e:
        print(f"[LSTM] Prediction error for {symbol}: {e}")
        return 0.0


# ── Main scoring ──────────────────────────────────────────────────────────────

def score_stock(symbol: str, company: str) -> dict | None:
    """
    Score a single stock. Returns dict with all metrics or None on failure.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="30d")

        if hist.empty or len(hist) < 5:
            return None

        current_price = float(hist["Close"].iloc[-1])
        open_price    = float(hist["Open"].iloc[-1])

        mom_score = _momentum_score(hist)
        vol_score = _volatility_score(hist)
        vum_score = _volume_score(hist)
        lstm_gain = _lstm_predicted_gain(symbol, current_price)

        # LSTM gain converted to 0-100 score
        # +5% gain = 100, 0% = 50, -5% = 0
        lstm_score = min(100, max(0, 50 + lstm_gain * 10))

        # Weighted total score
        total_score = (
            lstm_score  * 0.40 +
            mom_score   * 0.30 +
            vol_score   * 0.20 +
            vum_score   * 0.10
        )

        # Build human-readable reason
        reasons = []
        if mom_score > 65:
            reasons.append("strong upward momentum")
        if vol_score > 70:
            reasons.append("low volatility")
        if vum_score > 70:
            reasons.append("high buying volume")
        if lstm_gain > 1:
            reasons.append(f"LSTM predicts +{lstm_gain:.1f}% gain")
        elif lstm_gain > 0:
            reasons.append(f"positive LSTM outlook")

        reason = "Based on " + ", ".join(reasons) if reasons else "Balanced risk-reward profile"

        # Target price based on LSTM prediction
        target_price = round(current_price * (1 + lstm_gain / 100), 2)

        return {
            "symbol":         symbol,
            "company":        company,
            "score":          round(total_score, 2),
            "predicted_gain": lstm_gain,
            "current_price":  current_price,
            "open_price":     open_price,
            "target_price":   target_price,
            "reason":         reason,
            "momentum":       round(mom_score, 1),
            "volatility":     round(vol_score, 1),
            "volume":         round(vum_score, 1),
        }
    except Exception as e:
        print(f"[Recommender] Error scoring {symbol}: {e}")
        return None


def generate_recommendations() -> list:
    """
    Score all Nifty 50 stocks and return top 5.
    This is called every morning at 9:15 AM IST.
    """
    print(f"[Recommender] Generating recommendations for {date.today()}...")
    stocks  = load_nifty50()
    results = []

    for symbol, company in stocks:
        print(f"[Recommender] Scoring {symbol}...")
        scored = score_stock(symbol, company)
        if scored:
            results.append(scored)

    if not results:
        print("[Recommender] No results generated.")
        return []

    # Sort by total score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]

    # Add rank
    for i, r in enumerate(top5):
        r["rank"] = i + 1

    print(f"[Recommender] Top 5: {[r['symbol'] for r in top5]}")
    return top5


# ── DB persistence ────────────────────────────────────────────────────────────

def save_recommendations(recommendations: list):
    """Save today's top 5 to database."""
    import db
    try:
        conn = db.get_conn()
        cur  = conn.cursor()
        today = date.today()

        # Clear today's old recommendations first
        cur.execute("DELETE FROM ai_recommendations WHERE date=%s", (today,))

        for r in recommendations:
            cur.execute("""
                INSERT INTO ai_recommendations
                  (date, stock_symbol, company_name, score, predicted_gain,
                   current_price, target_price, reason, rank)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date, stock_symbol) DO UPDATE SET
                  score=EXCLUDED.score,
                  predicted_gain=EXCLUDED.predicted_gain,
                  current_price=EXCLUDED.current_price,
                  target_price=EXCLUDED.target_price,
                  reason=EXCLUDED.reason,
                  rank=EXCLUDED.rank
            """, (
                today, r["symbol"], r["company"], r["score"],
                r["predicted_gain"], r["current_price"],
                r["target_price"], r["reason"], r["rank"]
            ))
        conn.commit()
        print(f"[Recommender] Saved {len(recommendations)} recommendations to DB.")
        return True
    except Exception as e:
        print(f"[Recommender] DB save error: {e}")
        traceback.print_exc()
        return False
    finally:
        try: cur.close(); db.release_conn(conn)
        except: pass


def get_todays_recommendations() -> list:
    """Fetch today's recommendations from DB."""
    import db
    try:
        conn = db.get_conn()
        cur  = conn.cursor(cursor_factory=__import__('psycopg2.extras', fromlist=['RealDictCursor']).RealDictCursor)
        cur.execute("""
            SELECT * FROM ai_recommendations
            WHERE date = CURRENT_DATE
            ORDER BY rank ASC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            if r.get("date"): r["date"] = str(r["date"])
            if r.get("created_at"): r["created_at"] = str(r["created_at"])
        return rows
    except Exception as e:
        print(f"[Recommender] DB fetch error: {e}")
        return []
    finally:
        try: cur.close(); db.release_conn(conn)
        except: pass


def track_daily_prices():
    """
    Track open + close prices for all Nifty 50.
    Called at market open (9:15 AM) and close (3:30 PM).
    """
    import db
    stocks = load_nifty50()
    today  = date.today()

    for symbol, _ in stocks:
        try:
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period="2d")
            if hist.empty:
                continue
            row = hist.iloc[-1]
            conn = db.get_conn()
            cur  = conn.cursor()
            cur.execute("""
                INSERT INTO daily_prices
                  (date, stock_symbol, open_price, close_price, high_price, low_price, volume, pct_change)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date, stock_symbol) DO UPDATE SET
                  close_price=EXCLUDED.close_price,
                  high_price=EXCLUDED.high_price,
                  low_price=EXCLUDED.low_price,
                  volume=EXCLUDED.volume,
                  pct_change=EXCLUDED.pct_change
            """, (
                today, symbol,
                float(row["Open"]),
                float(row["Close"]),
                float(row["High"]),
                float(row["Low"]),
                int(row["Volume"]),
                round(((float(row["Close"]) - float(row["Open"])) / float(row["Open"])) * 100, 2)
            ))
            conn.commit()
            db.release_conn(conn)
        except Exception as e:
            print(f"[Prices] Error tracking {symbol}: {e}")
