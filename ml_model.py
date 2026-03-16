"""
ml_model.py - LSTM price forecast + fast statistical fallback
Data: Twelve Data -> Stooq (no yfinance)
"""
import os, io, base64, warnings, threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_fetch import fetch_history

warnings.filterwarnings("ignore")

MODEL_CACHE_DIR   = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")
MODEL_EXPIRY_DAYS = int(os.getenv("MODEL_EXPIRY_DAYS", "7"))
LOOKBACK          = 60

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

_training_lock      = threading.Lock()
_currently_training = set()


# ── cache ──────────────────────────────────────────────────────────────────────

def _paths(symbol):
    safe = symbol.replace(".", "_").replace("&", "n").upper()
    base = os.path.join(MODEL_CACHE_DIR, safe)
    return {"model": base + "_model.keras",
            "scaler": base + "_scaler.pkl",
            "meta":   base + "_meta.txt"}


def _is_fresh(symbol):
    p = _paths(symbol)
    if not all(os.path.exists(v) for v in p.values()):
        return False
    try:
        with open(p["meta"]) as f:
            age = (datetime.now() - datetime.fromisoformat(f.read().strip())).days
        return age < MODEL_EXPIRY_DAYS
    except Exception:
        return False


def _save_model(symbol, model, scaler):
    p = _paths(symbol)
    model.save(p["model"])
    joblib.dump(scaler, p["scaler"])
    with open(p["meta"], "w") as f:
        f.write(datetime.now().isoformat())
    print(f"[Cache] saved {symbol}")


def _load_model(symbol):
    p = _paths(symbol)
    return load_model(p["model"]), joblib.load(p["scaler"])


# ── data ───────────────────────────────────────────────────────────────────────

def fetch_stock_data(symbol):
    df = fetch_history(symbol, days=1825)
    if df is None or df.empty:
        return None
    if "ds" not in df.columns:
        df = df.reset_index()
        df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    if df["ds"].dt.tz is not None:
        df["ds"] = df["ds"].dt.tz_convert(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df = df[df["y"] > 0]
    if len(df) < LOOKBACK + 10:
        print(f"[LSTM] {symbol}: only {len(df)} rows, need {LOOKBACK+10}")
        return None
    return df[["ds", "y"]].sort_values("ds").reset_index(drop=True)


# ── LSTM ────────────────────────────────────────────────────────────────────────

def _sequences(scaled):
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i, 0])
        y.append(scaled[i, 0])
    return np.array(X).reshape(-1, LOOKBACK, 1), np.array(y)


def _build_model():
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


def _train(symbol, df):
    prices = df["y"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    X, y   = _sequences(scaled)
    split  = int(len(X) * 0.8)
    model  = _build_model()
    es     = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X[:split], y[:split],
              validation_data=(X[split:], y[split:]),
              epochs=80, batch_size=32, callbacks=[es], verbose=0)
    _save_model(symbol, model, scaler)
    with _training_lock:
        _currently_training.discard(symbol)


def _spawn_train(symbol, df):
    with _training_lock:
        if symbol in _currently_training:
            return
        _currently_training.add(symbol)
    threading.Thread(target=_train, args=(symbol, df), daemon=True).start()
    print(f"[BG] training {symbol}")


def _get_model(symbol, df):
    prices = df["y"].values.reshape(-1, 1)
    if _is_fresh(symbol):
        model, scaler = _load_model(symbol)
        return model, scaler, scaler.transform(prices)[-LOOKBACK:, 0]
    p = _paths(symbol)
    if all(os.path.exists(v) for v in p.values()):
        model, scaler = _load_model(symbol)
        _spawn_train(symbol, df)
        return model, scaler, scaler.transform(prices)[-LOOKBACK:, 0]
    print(f"[Train] first time for {symbol}")
    _train(symbol, df)
    model, scaler = _load_model(symbol)
    return model, scaler, scaler.transform(prices)[-LOOKBACK:, 0]


def _predict(model, window, n, scaler):
    win   = window.copy().reshape(1, LOOKBACK, 1)
    preds = []
    for _ in range(n):
        p = model.predict(win, verbose=0)[0, 0]
        preds.append(p)
        win = np.roll(win, -1, axis=1)
        win[0, -1, 0] = p
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


def _ci(preds):
    s = preds.std() * 0.05
    d = 1.65 * s * np.sqrt(np.arange(1, len(preds)+1))
    return preds - d, preds + d


# ── fast fallback (no LSTM needed) ─────────────────────────────────────────────

def _fast_forecast(df, forecast_type):
    closes     = df["y"].values
    last_date  = df["ds"].max()
    last_price = closes[-1]
    lb         = min(60, len(closes))
    slope, intercept = np.polyfit(np.arange(lb), closes[-lb:], 1)
    vol        = float(np.std(np.diff(closes[-lb:]) / closes[-lb:-1])) * last_price

    rows = []
    if forecast_type == "6m":
        for m in range(1, 7):
            d_ahead = m * 21
            pred    = last_price + slope * d_ahead
            sigma   = vol * float(np.sqrt(d_ahead)) * 1.65
            rows.append({"ds": last_date + timedelta(days=m*30),
                         "yhat": round(pred, 2),
                         "yhat_lower": round(pred - sigma, 2),
                         "yhat_upper": round(pred + sigma, 2)})
    else:
        for yr in range(1, 6):
            d_ahead = yr * 252
            pred    = last_price + slope * d_ahead
            sigma   = vol * float(np.sqrt(d_ahead)) * 1.65
            rows.append({"ds": pd.Timestamp(f"{last_date.year + yr}-01-01"),
                         "yhat": round(pred, 2),
                         "yhat_lower": round(pred - sigma, 2),
                         "yhat_upper": round(pred + sigma, 2)})
    return pd.DataFrame(rows)


# ── public API ──────────────────────────────────────────────────────────────────

def get_aggregated_forecast(symbol, forecast_type="6m"):
    print(f"[Forecast] {symbol} {forecast_type}")
    df = fetch_stock_data(symbol)
    if df is None or df.empty:
        print(f"[Forecast] no data for {symbol}")
        return None
    print(f"[Forecast] {len(df)} rows for {symbol}")

    last_date = df["ds"].max()

    # try LSTM
    try:
        model, scaler, win = _get_model(symbol, df)
        if forecast_type == "6m":
            n      = 180
            preds  = _predict(model, win, n, scaler)
            dates  = pd.bdate_range(start=last_date + timedelta(days=1), periods=n)[:n]
            lo, hi = _ci(preds)
            tmp    = pd.DataFrame({"ds": dates, "yhat": preds,
                                   "yhat_lower": lo, "yhat_upper": hi})
            tmp["period"] = tmp["ds"].dt.to_period("M")
            agg    = tmp.groupby("period")[["yhat","yhat_lower","yhat_upper"]].mean().reset_index()
            agg["ds"] = agg["period"].dt.to_timestamp()
            print(f"[Forecast] LSTM done {symbol}")
            return agg[["ds","yhat","yhat_lower","yhat_upper"]].head(6)
        else:
            n      = 365*5+30
            preds  = _predict(model, win, n, scaler)
            dates  = pd.bdate_range(start=last_date + timedelta(days=1), periods=n)[:n]
            lo, hi = _ci(preds)
            tmp    = pd.DataFrame({"ds": dates, "yhat": preds,
                                   "yhat_lower": lo, "yhat_upper": hi})
            tmp["year"] = tmp["ds"].dt.year
            agg    = tmp.groupby("year")[["yhat","yhat_lower","yhat_upper"]].mean().reset_index()
            agg["ds"] = pd.to_datetime(agg["year"].astype(str) + "-01-01")
            print(f"[Forecast] LSTM done {symbol}")
            return agg[["ds","yhat","yhat_lower","yhat_upper"]].head(5)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Forecast] LSTM failed for {symbol}: {e} — using fast fallback")

    # fast fallback — always works
    try:
        result = _fast_forecast(df, forecast_type)
        print(f"[Forecast] fast fallback done for {symbol}")
        return result
    except Exception as e2:
        print(f"[Forecast] fast fallback failed: {e2}")
        return None


def generate_stock_plot(symbol, forecast_type="6m"):
    fdf = get_aggregated_forecast(symbol, forecast_type)
    if fdf is None or fdf.empty:
        return None
    cur     = "₹" if symbol.endswith(".NS") or symbol.endswith(".BO") else "$"
    display = symbol.replace(".NS","").replace(".BO","")
    title   = f"{display} — {'6-Month' if forecast_type=='6m' else '5-Year'} Forecast"
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=110)
    ax.plot(fdf["ds"], fdf["yhat"], marker="o", color="#6c5ce7", linewidth=2, label="Predicted")
    ax.fill_between(fdf["ds"], fdf["yhat_lower"], fdf["yhat_upper"],
                    alpha=0.25, color="#a29bfe", label="90% CI")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Price ({cur})")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=35)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
