"""
recommender.py - AI stock recommender for Nifty 50
Data: Twelve Data -> Stooq (no yfinance)
"""
import os, csv, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import numpy as np
import pandas as pd
import pytz
import psycopg2.extras

from data_fetch import fetch_history_batch, fetch_history

IST = pytz.timezone("Asia/Kolkata")


def load_nifty50():
    path = os.path.join(os.path.dirname(__file__), "companies_india.csv")
    out  = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "Symbol" in row and "Company" in row:
                    out.append((row["Symbol"].strip(), row["Company"].strip()))
    except Exception as e:
        print(f"[Recommender] load error: {e}")
    return out


# ── scoring ────────────────────────────────────────────────────────────────────

def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag, al = gains[-period:].mean(), losses[-period:].mean()
    if al == 0:
        return 100.0
    return float(100 - 100 / (1 + ag / al))


def _momentum(hist):
    if len(hist) < 21:
        return 50.0
    c = hist["Close"].values
    try:
        s = ((c[-1]-c[-6])/c[-6]*0.5 + (c[-1]-c[-11])/c[-11]*0.3 + (c[-1]-c[-21])/c[-21]*0.2) * 100
        return float(min(100, max(0, 50 + s * 5)))
    except Exception:
        return 50.0


def _volatility(hist):
    if len(hist) < 10:
        return 50.0
    try:
        vol = hist["Close"].pct_change().dropna().std() * 100
        return float(min(100, max(10, 100 - vol * 20)))
    except Exception:
        return 50.0


def _volume(hist):
    if len(hist) < 5 or "Volume" not in hist.columns:
        return 50.0
    try:
        avg = hist["Volume"].iloc[:-1].mean()
        if avg == 0:
            return 50.0
        return float(min(100, (hist["Volume"].iloc[-1] / avg) * 50))
    except Exception:
        return 50.0


def _gap(hist):
    if len(hist) < 2 or "Open" not in hist.columns:
        return 50.0
    try:
        g = ((float(hist["Open"].iloc[-1]) - float(hist["Close"].iloc[-2])) /
             float(hist["Close"].iloc[-2])) * 100
        return float(min(100, max(0, 50 + g * 25)))
    except Exception:
        return 50.0


def _gain(hist):
    if len(hist) < 20:
        return 0.0
    try:
        c       = hist["Close"].values
        cur     = c[-1]
        s5,  _  = np.polyfit(np.arange(5),  c[-5:],  1)
        s20, _  = np.polyfit(np.arange(20), c[-20:], 1)
        ma20    = c[-20:].mean()
        rsi_sig = (_rsi(c) - 50) / 50 * 3
        rec     = ((c[-1] - c[-4]) / c[-4]) * 100 if len(c) >= 5 else 0
        g       = (s5/cur*100*2 + s20/cur*100 + (cur-ma20)/ma20*100*0.3 +
                   rsi_sig + rec*0.5)
        return float(round(max(-10.0, min(10.0, g)), 2))
    except Exception:
        return 0.0


def _score(symbol, company, hist):
    try:
        if hist is None or hist.empty or len(hist) < 10:
            return None
        cur   = float(hist["Close"].iloc[-1])
        opn   = float(hist["Open"].iloc[-1]) if "Open" in hist.columns else cur
        mom   = _momentum(hist)
        vol   = _volatility(hist)
        vum   = _volume(hist)
        gap   = _gap(hist)
        gain  = _gain(hist)
        rsi   = _rsi(hist["Close"].values)
        total = mom*0.40 + vol*0.30 + vum*0.20 + gap*0.10
        reasons = []
        if mom > 65:    reasons.append("strong momentum")
        if vol > 70:    reasons.append("low volatility")
        if vum > 70:    reasons.append("high volume")
        if gap > 65:    reasons.append("gap-up open")
        if rsi > 55:    reasons.append(f"RSI bullish ({rsi:.0f})")
        if rsi < 35:    reasons.append(f"oversold RSI ({rsi:.0f})")
        if gain > 0.5:  reasons.append(f"trend +{gain:.1f}%")
        return {
            "symbol": symbol, "company": company,
            "score": round(total, 2), "predicted_gain": gain,
            "current_price": cur, "open_price": opn,
            "target_price": round(cur * (1 + gain/100), 2),
            "reason": ("Based on " + ", ".join(reasons)) if reasons else "Neutral signals",
            "momentum": round(mom, 1), "volatility": round(vol, 1),
            "volume": round(vum, 1), "rsi": round(rsi, 1),
        }
    except Exception as e:
        print(f"[Score] {symbol}: {e}")
        return None


# ── main ────────────────────────────────────────────────────────────────────────

def generate_recommendations():
    print(f"[Recommender] {date.today()}")
    stocks = load_nifty50()
    if not stocks:
        return []
    symbols     = [s for s, _ in stocks]
    company_map = {s: c for s, c in stocks}
    hist_map    = fetch_history_batch(symbols, days=60)
    print(f"[Recommender] got {len(hist_map)}/{len(symbols)} symbols")
    if not hist_map:
        return []
    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_score, sym, company_map[sym], hist_map[sym]): sym
                for sym in hist_map}
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)
    if not results:
        return []
    bullish = [r for r in results if r["predicted_gain"] > 0]
    pool    = bullish if len(bullish) >= 5 else results
    for r in pool:
        r["_rank"] = r["score"] * 0.6 + min(r["predicted_gain"] * 10, 40)
    pool.sort(key=lambda x: x["_rank"], reverse=True)
    top5 = pool[:5]
    for i, r in enumerate(top5):
        r["rank"] = i + 1
        r.pop("_rank", None)
    print(f"[Recommender] top5: {[(r['symbol'], r['predicted_gain']) for r in top5]}")
    return top5


def score_stock(symbol, company):
    try:
        return _score(symbol, company, fetch_history(symbol, days=60))
    except Exception as e:
        print(f"[score_stock] {symbol}: {e}")
        return None


# ── DB ──────────────────────────────────────────────────────────────────────────

def save_recommendations(recs):
    import db
    conn = cur = None
    try:
        conn = db.get_conn()
        cur  = conn.cursor()
        today = date.today()
        cur.execute("DELETE FROM ai_recommendations WHERE date=%s", (today,))
        for r in recs:
            cur.execute("""
                INSERT INTO ai_recommendations
                  (date,stock_symbol,company_name,score,predicted_gain,
                   current_price,target_price,reason,rank)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (today, r["symbol"], r["company"], r["score"],
                  r["predicted_gain"], r["current_price"],
                  r["target_price"], r["reason"], r["rank"]))
        conn.commit()
        print(f"[Recommender] saved {len(recs)}")
        return True
    except Exception as e:
        if conn: conn.rollback()
        traceback.print_exc()
        return False
    finally:
        if cur:  cur.close()
        if conn: db.release_conn(conn)


def get_todays_recommendations():
    import db
    conn = cur = None
    try:
        conn = db.get_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT * FROM ai_recommendations
            WHERE date = CURRENT_DATE ORDER BY rank ASC
        """)
        result = []
        for row in cur.fetchall():
            r = dict(row)
            if r.get("date"):       r["date"]       = str(r["date"])
            if r.get("created_at"): r["created_at"] = str(r["created_at"])
            for f in ("score","predicted_gain","current_price","target_price"):
                if r.get(f) is not None: r[f] = float(r[f])
            result.append(r)
        print(f"[Recommender] fetched {len(result)} from DB")
        return result
    except Exception as e:
        print(f"[Recommender] DB fetch: {e}")
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
                  (date,stock_symbol,open_price,close_price,high_price,low_price,volume,pct_change)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date,stock_symbol) DO UPDATE SET
                  close_price=EXCLUDED.close_price, high_price=EXCLUDED.high_price,
                  low_price=EXCLUDED.low_price, volume=EXCLUDED.volume,
                  pct_change=EXCLUDED.pct_change
            """, (today, symbol,
                  float(row.get("Open",  row["Close"])),
                  float(row["Close"]),
                  float(row.get("High",  row["Close"])),
                  float(row.get("Low",   row["Close"])),
                  int(row.get("Volume",  0)),
                  round(((float(row["Close"]) - float(row.get("Open", row["Close"]))) /
                         float(row.get("Open", row["Close"]))) * 100, 2)
                  if row.get("Open") else 0.0))
            conn.commit()
        except Exception as e:
            if conn: conn.rollback()
            print(f"[Prices] {symbol}: {e}")
        finally:
            if cur:  cur.close()
            if conn: db.release_conn(conn)
