"""
ml_model.py — LSTM price forecast
Data: Twelve Data → Stooq (zero yfinance — blocked on cloud IPs)
Caching: models saved to MODEL_CACHE_DIR, retrained weekly in background
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

from data_fetch import fetch_history, fetch_price

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_CACHE_DIR   = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")
MODEL_EXPIRY_DAYS = int(os.getenv("MODEL_EXPIRY_DAYS", "7"))
LOOKBACK          = 60

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

_training_lock      = threading.Lock()
_currently_training: set = set()


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _paths(symbol: str) -> dict:
    safe = symbol.replace(".", "_").replace("&", "n").upper()
    base = os.path.join(MODEL_CACHE_DIR, safe)
    return {
        "model":  base + "_model.keras",
        "scaler": base + "_scaler.pkl",
        "meta":   base + "_meta.txt",
    }

def _is_fresh(symbol: str) -> bool:
    p = _paths(symbol)
    if not all(os.path.exists(v) for v in p.values()):
        return False
    try:
        with open(p["meta"]) as f:
            age = (datetime.now() - datetime.fromisoformat(f.read().strip())).days
        return age < MODEL_EXPIRY_DAYS
    except Exception:
        return False

def _save(symbol: str, model, scaler):
    p = _paths(symbol)
    model.save(p["model"])
    joblib.dump(scaler, p["scaler"])
    with open(p["meta"], "w") as f:
        f.write(datetime.now().isoformat())
    print(f"[Cache] Saved model for {symbol}")

def _load(symbol: str):
    p = _paths(symbol)
    model  = load_model(p["model"])
    scaler = joblib.load(p["scaler"])
    print(f"[Cache] Loaded model for {symbol}")
    return model, scaler


# ── Data fetch (uses data_fetch.py — no yfinance) ────────────────────────────

def fetch_stock_data(symbol: str) -> pd.DataFrame | None:
    """
    Fetch 5 years of daily closing prices.
    Returns DataFrame with columns ds (datetime) and y (close price).
    """
    df = fetch_history(symbol, days=1825)   # ~5 years
    if df is None or df.empty:
        return None

    # Ensure ds/y columns exist (data_fetch already sets them, but be safe)
    if "ds" not in df.columns:
        df = df.reset_index()
        df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

    # Safely strip timezone — handle both tz-aware and tz-naive
    def _strip_tz(s: pd.Series) -> pd.Series:
        s = pd.to_datetime(s, errors="coerce")
        if s.dt.tz is not None:
            return s.dt.tz_convert(None)
        return s

    df["ds"] = _strip_tz(df["ds"])
    df["y"]  = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df = df[df["y"] > 0]   # drop zero/negative prices

    if len(df) < LOOKBACK + 10:
        print(f"[LSTM] {symbol}: not enough data ({len(df)} rows, need {LOOKBACK + 10})")
        return None

    return df[["ds", "y"]].sort_values("ds").reset_index(drop=True)


# ── LSTM helpers ──────────────────────────────────────────────────────────────

def _build_sequences(scaled: np.ndarray):
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i - LOOKBACK: i, 0])
        y.append(scaled[i, 0])
    return np.array(X).reshape(-1, LOOKBACK, 1), np.array(y)

def _build_model() -> Sequential:
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


# ── Training ───────────────────────────────────────────────────────────────────

def _train_and_save(symbol: str, df: pd.DataFrame):
    prices = df["y"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    X, y   = _build_sequences(scaled)
    split  = int(len(X) * 0.8)
    model  = _build_model()
    es     = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(
        X[:split], y[:split],
        validation_data=(X[split:], y[split:]),
        epochs=80, batch_size=32,
        callbacks=[es], verbose=0,
    )
    _save(symbol, model, scaler)
    with _training_lock:
        _currently_training.discard(symbol)

def _spawn_training(symbol: str, df: pd.DataFrame):
    with _training_lock:
        if symbol in _currently_training:
            return
        _currently_training.add(symbol)
    t = threading.Thread(target=_train_and_save, args=(symbol, df), daemon=True)
    t.start()
    print(f"[BG] Training started for {symbol}")


# ── Get or train ───────────────────────────────────────────────────────────────

def get_or_train_model(symbol: str, df: pd.DataFrame):
    prices = df["y"].values.reshape(-1, 1)
    if _is_fresh(symbol):
        model, scaler = _load(symbol)
        scaled        = scaler.transform(prices)
        return model, scaler, scaled[-LOOKBACK:, 0]
    p = _paths(symbol)
    if all(os.path.exists(v) for v in p.values()):
        model, scaler = _load(symbol)
        scaled        = scaler.transform(prices)
        _spawn_training(symbol, df)
        return model, scaler, scaled[-LOOKBACK:, 0]
    print(f"[Train] First-time training for {symbol}…")
    _train_and_save(symbol, df)
    model, scaler = _load(symbol)
    scaled        = scaler.transform(prices)
    return model, scaler, scaled[-LOOKBACK:, 0]


# ── Forecast helpers ───────────────────────────────────────────────────────────

def _forecast_days(model, last_window: np.ndarray, n_days: int, scaler) -> np.ndarray:
    win   = last_window.copy().reshape(1, LOOKBACK, 1)
    preds = []
    for _ in range(n_days):
        p = model.predict(win, verbose=0)[0, 0]
        preds.append(p)
        win = np.roll(win, -1, axis=1)
        win[0, -1, 0] = p
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def _ci(preds: np.ndarray, z: float = 1.65):
    sigma = preds.std() * 0.05
    delta = z * sigma * np.sqrt(np.arange(1, len(preds) + 1))
    return preds - delta, preds + delta

def _currency(symbol: str) -> str:
    return "₹" if (symbol.endswith(".NS") or symbol.endswith(".BO")) else "$"


# ── Public API ─────────────────────────────────────────────────────────────────

def _fast_forecast(df: pd.DataFrame, forecast_type: str) -> pd.DataFrame:
    """
    Fast statistical fallback forecast using linear trend + volatility bands.
    Runs in milliseconds. Used when LSTM model is not ready yet.
    """
    closes    = df["y"].values
    last_date = df["ds"].max()
    last_price = closes[-1]

    # Fit linear trend on last 60 days
    lookback = min(60, len(closes))
    x = np.arange(lookback)
    y = closes[-lookback:]
    slope, intercept = np.polyfit(x, y, 1)
    vol = np.std(np.diff(y) / y[:-1]) * last_price  # daily vol in price units

    if forecast_type == "6m":
        months = 6
        periods = []
        for m in range(1, months + 1):
            days_ahead = m * 21  # ~21 trading days per month
            pred = last_price + slope * days_ahead
            sigma = vol * np.sqrt(days_ahead) * 1.65
            dt = last_date + timedelta(days=m * 30)
            periods.append({"ds": dt, "yhat": round(pred, 2),
                           "yhat_lower": round(pred - sigma, 2),
                           "yhat_upper": round(pred + sigma, 2)})
        return pd.DataFrame(periods)
    else:
        years = 5
        periods = []
        for yr in range(1, years + 1):
            days_ahead = yr * 252
            pred = last_price + slope * days_ahead
            sigma = vol * np.sqrt(days_ahead) * 1.65
            dt = pd.Timestamp(str(last_date.year + yr) + "-01-01")
            periods.append({"ds": dt, "yhat": round(pred, 2),
                           "yhat_lower": round(pred - sigma, 2),
                           "yhat_upper": round(pred + sigma, 2)})
        return pd.DataFrame(periods)


def get_aggregated_forecast(symbol: str, forecast_type: str = "6m") -> pd.DataFrame | None:
    print(f"[Forecast] Starting {forecast_type} forecast for {symbol}")

    df = fetch_stock_data(symbol)
    if df is None or df.empty:
        print(f"[Forecast] No data returned for {symbol}")
        return None
    print(f"[Forecast] Got {len(df)} rows of history for {symbol}")

    last_date = df["ds"].max()

    # Try LSTM first
    try:
        model, scaler, last_win = get_or_train_model(symbol, df)
        print(f"[Forecast] LSTM model ready for {symbol}, running inference...")

        if forecast_type == "6m":
            n      = 180
            preds  = _forecast_days(model, last_win, n, scaler)
            dates  = pd.bdate_range(start=last_date + timedelta(days=1), periods=n)[:n]
            lo, hi = _ci(preds)
            tmp    = pd.DataFrame({"ds": dates, "yhat": preds, "yhat_lower": lo, "yhat_upper": hi})
            tmp["period"] = tmp["ds"].dt.to_period("M")
            agg    = tmp.groupby("period")[["yhat","yhat_lower","yhat_upper"]].mean().reset_index()
            agg["ds"] = agg["period"].dt.to_timestamp()
            result = agg[["ds","yhat","yhat_lower","yhat_upper"]].head(6)

        elif forecast_type == "5y":
            n      = 365 * 5 + 30
            preds  = _forecast_days(model, last_win, n, scaler)
            dates  = pd.bdate_range(start=last_date + timedelta(days=1), periods=n)[:n]
            lo, hi = _ci(preds)
            tmp    = pd.DataFrame({"ds": dates, "yhat": preds, "yhat_lower": lo, "yhat_upper": hi})
            tmp["year"] = tmp["ds"].dt.year
            agg    = tmp.groupby("year")[["yhat","yhat_lower","yhat_upper"]].mean().reset_index()
            agg["ds"] = pd.to_datetime(agg["year"].astype(str) + "-01-01")
            result = agg[["ds","yhat","yhat_lower","yhat_upper"]].head(5)
        else:
            return None

        print(f"[Forecast] LSTM done for {symbol}")
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Forecast] LSTM failed for {symbol}: {e} — using fast fallback")

    # Fast statistical fallback — always works
    try:
        result = _fast_forecast(df, forecast_type)
        print(f"[Forecast] Fast fallback done for {symbol} ({len(result)} points)")
        return result
    except Exception as e2:
        print(f"[Forecast] Fast fallback also failed for {symbol}: {e2}")
        return None


def generate_stock_plot(symbol: str, forecast_type: str = "6m") -> str | None:
    fdf = get_aggregated_forecast(symbol, forecast_type)
    if fdf is None or fdf.empty:
        return None

    cur     = _currency(symbol)
    display = symbol.replace(".NS", "").replace(".BO", "")
    label   = "Monthly Avg" if forecast_type == "6m" else "Yearly Avg"
    title   = (f"{display} — {'6-Month' if forecast_type=='6m' else '5-Year'} "
               f"LSTM Forecast ({label})")

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
    
