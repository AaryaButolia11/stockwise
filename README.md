# StockWise — AI-Powered Stock Intelligence for Nifty 50

> **LSTM price forecasting · Live NSE data · Paper trading portfolio · Auto stop-loss/take-profit · SMS + WhatsApp alerts**

StockWise is a full-stack intelligent trading assistant built for Indian markets. It combines a deep learning LSTM model for price forecasting, real-time NSE stock data via yfinance, a paper trading portfolio with automated risk management, and Twilio-powered SMS/WhatsApp alerts — all packaged in a Flask web app backed by PostgreSQL on Supabase.

Whether you're a student learning about markets, a developer exploring ML in finance, or just someone who wants to simulate trading without risking real money, StockWise gives you a production-grade platform to do it.

---

## Features

| Feature | Description |
|---|---|
| **LSTM Forecast** | 6-month and 5-year price forecasts with 90% confidence intervals |
| **Model Caching** | Trained models saved to disk, retrained weekly in background — instant responses after first run |
| **Price Alerts** | SMS + WhatsApp via Twilio when your stocks move |
| **Paper Portfolio** | Buy/sell Nifty 50 stocks (simulated — no real money) |
| **Auto Stop-Loss** | Background scheduler auto-sells when price hits your stop-loss or take-profit |
| **AI Picks** | Daily top-5 stock recommendations scored on momentum, RSI, volatility, and volume |
| **PostgreSQL / Supabase** | All positions, alerts, transactions, and recommendations persisted in cloud DB |

---

## Screenshots

### Sign Up
<img width="1918" height="1028" alt="image" src="https://github.com/user-attachments/assets/9e43b2e9-a77f-4014-a9fc-58433f0d42e3" />

### Login
<img width="1918" height="1031" alt="image" src="https://github.com/user-attachments/assets/968ac026-06ab-4454-a333-30fbf018640b" />

### Dashboard
<img width="1918" height="1033" alt="image" src="https://github.com/user-attachments/assets/3c53dd4a-7157-4ef9-ae5a-76d602a4f6e5" />

### Price Prediction
<img width="1917" height="1031" alt="image" src="https://github.com/user-attachments/assets/1c59739b-76cd-4a35-8fc8-dfcd81d0b945" />

### Buying Stocks
<img width="1917" height="1028" alt="image" src="https://github.com/user-attachments/assets/5ee6dc4b-5243-4698-a155-6cb69e684d5f" />

### SMS Confirmation
<img width="1917" height="1018" alt="image" src="https://github.com/user-attachments/assets/5e1547f6-fd18-4109-aa6d-41fabf977aff" />

<img width="1897" height="1030" alt="image" src="https://github.com/user-attachments/assets/f8c104a4-a95c-478b-9f11-e6fce791cb9e" />

### AI Picks
<img width="1917" height="1028" alt="image" src="https://github.com/user-attachments/assets/03cf86d1-98b5-4fa4-8e0b-2da753b8f447" />

---

## Why LSTM for Stock Forecasting?

<img width="674" height="345" alt="image" src="https://github.com/user-attachments/assets/8db83131-cffd-42c8-94b4-4d3bf3b267c1" />

Stock prices are **sequential time-series data** — the price today is influenced by what happened yesterday, last week, and last month. This is exactly the kind of problem where LSTMs (Long Short-Term Memory networks) outperform traditional approaches.

Standard models like linear regression or ARIMA treat each data point independently and struggle with long-range dependencies. An LSTM is a special kind of recurrent neural network (RNN) that solves this by maintaining a **cell state** — an internal memory that can carry information across hundreds of time steps, deciding what to remember and what to forget via learned gates.

**Why not a Transformer or Prophet?**
- Prophet is fast but produces unrealistically smooth forecasts with no learned market patterns
- Transformers need much larger datasets and are expensive to run on free-tier cloud
- LSTM hits the sweet spot: captures non-linear price patterns, trains in minutes on 2–3 years of daily data, and runs inference in milliseconds from cache

**How LSTM works in StockWise, step by step:**

1. **Input:** 3 years (~750 trading days) of daily closing prices for a Nifty 50 stock, fetched via yfinance
2. **Preprocessing:** Prices are normalized to [0, 1] using `MinMaxScaler` to stabilize training
3. **Sequence creation:** A sliding window of 60 days is created — each sample is "given the last 60 days, predict day 61"
4. **Architecture:** Two stacked LSTM layers (64 → 32 units) with Dropout(0.2) to prevent overfitting, followed by Dense layers down to a single price output
5. **Training:** 80/20 train-val split, Adam optimizer, MSE loss, EarlyStopping with patience=8 to avoid overtraining
6. **Inference:** The last 60 days of real prices seed the model; it then predicts one day ahead, feeds that prediction back as input, and repeats for 180 days (6m) or 5 years (5y)
7. **Confidence intervals:** Built from historical daily return volatility (`std(daily_returns)`), expanding with `√t` — uncertainty grows the further into the future you predict
8. **Aggregation:** Daily predictions are averaged into monthly buckets (6m) or yearly buckets (5y) for a clean chart

If there isn't enough data for LSTM (e.g. recently listed stocks), a fast **linear trend + volatility fallback** runs instead — so forecasts always load.

---

## How LSTM Caching Works

Training an LSTM from scratch takes 3–5 minutes. To make the app fast for all subsequent users, models are cached to disk and served in ~2 seconds.

```
First request for RELIANCE.NS:
  → Fetch 3 years of data via yfinance
  → Train LSTM model (~3-5 min)
  → Save model + scaler + timestamp to /data/model_cache/

All subsequent requests (within 7 days):
  → Load model from disk (~2 sec)
  → Run inference on last 60 days
  → Return forecast instantly

After 7 days (model is stale):
  → Serve stale model instantly (user gets fast response)
  → Trigger background retraining thread
  → Next request gets the freshly trained model
```

This cache-then-retrain pattern means **no user ever waits for training** after the first run.

---

## Program Flow

Here is the complete end-to-end flow of how StockWise works:

```
User opens browser
        │
        ▼
[Flask app.py] ── auth check (session) ──► Login/Register page
        │                                         │
        │ (logged in)                    [db.py] register_user()
        ▼                                         │ bcrypt hash + Supabase INSERT
[Dashboard - index.html]
        │
        ├──► GET /get_current_stock_info?symbol=INFY.NS
        │         │
        │    [msg.py] fetch_current_price()
        │         │
        │    [data_fetch.py]
        │         ├── _price_yf()      ← yfinance (primary, free, no IP blocks)
        │         ├── _price_td()      ← TwelveData (fallback 1, needs API key)
        │         ├── _price_nse()     ← NSE India API (fallback 2)
        │         └── _price_stooq()   ← Stooq (fallback 3)
        │         └── Return price to UI
        │
        ├──► GET /get_forecast?symbol=INFY.NS&forecast_type=6m
        │         │
        │    [ml_model.py] get_aggregated_forecast()
        │         │
        │         ├── fetch_stock_data()
        │         │       └── [data_fetch.py] fetch_history() via yfinance
        │         │               → 3 years of OHLCV data
        │         │
        │         ├── Has cached model? Is it fresh (< 7 days)?
        │         │       ├── YES → load model from disk (2 sec)
        │         │       ├── STALE → load stale model + spawn background retraining
        │         │       └── NO → train LSTM now (~3-5 min first time)
        │         │
        │         ├── _predict() → auto-regressive forecast for 180 days
        │         ├── _ci() → confidence bands from historical volatility
        │         ├── Aggregate into 6 monthly data points
        │         ├── generate_stock_plot() → matplotlib chart → base64 PNG
        │         └── Return JSON {dates, yhat, yhat_lower, yhat_upper, plot_img}
        │
        ├──► POST /portfolio/buy
        │         │
        │    Fetch live price via yfinance
        │    [db.py] buy_stock() → INSERT into portfolio + transactions tables
        │    [msg.py] send_alert_sms() → Twilio SMS confirmation
        │
        ├──► GET /recommendations
        │         │
        │    [recommender.py] get_todays_recommendations()
        │         ├── Check DB for today's cached recommendations
        │         ├── If none → spawn background generation thread
        │         │       └── fetch_history_batch() → all 50 Nifty symbols via yfinance
        │         │       └── Score each stock: RSI, momentum, volatility, volume, gap
        │         │       └── Rank top 5 → save to ai_recommendations table
        │         └── Return top 5 to UI
        │
        └──► Background: [scheduler.py] runs every 5 minutes
                  ├── get_open_positions() from DB
                  ├── fetch live price for each position
                  ├── If price ≤ stop_loss OR price ≥ take_profit:
                  │       └── sell_stock() → UPDATE portfolio, INSERT transaction
                  │       └── send_alert_sms() → "Auto-sell triggered" SMS
                  └── At 9:15 AM IST (weekdays):
                          └── generate_recommendations() + save to DB
```

---

## Database Schema Design

The schema is designed around **five core tables** in PostgreSQL (hosted on Supabase), normalized to avoid redundancy while keeping query patterns fast.

```
┌─────────────┐         ┌──────────────────┐
│    users    │         │   user_alerts    │
├─────────────┤  1:many ├──────────────────┤
│ id (PK)     │────────►│ id (PK)          │
│ username    │         │ user_id (FK)     │
│ email       │         │ stock_symbol     │
│ password_   │         │ phone_number     │
│   hash      │         │ is_active        │
│ phone_number│         └──────────────────┘
│ created_at  │
└──────┬──────┘
       │ 1:many
       ▼
┌─────────────────────┐         ┌─────────────────────┐
│      portfolio      │  1:many │    transactions     │
├─────────────────────┤────────►├─────────────────────┤
│ id (PK)             │         │ id (PK)             │
│ user_id (FK)        │         │ portfolio_id (FK)   │
│ stock_symbol        │         │ action (buy/sell/   │
│ company_name        │         │         auto_sell)  │
│ quantity            │         │ stock_symbol        │
│ buy_price           │         │ quantity            │
│ current_price       │         │ price               │
│ stop_loss           │         │ total_value         │
│ take_profit         │         │ note                │
│ status (open/closed/│         │ created_at          │
│         auto_sell)  │         └─────────────────────┘
│ bought_at           │
│ sold_at             │
│ sell_price          │
│ pnl                 │
└─────────────────────┘

┌──────────────────────────┐     ┌──────────────────────┐
│    ai_recommendations    │     │    daily_prices      │
├──────────────────────────┤     ├──────────────────────┤
│ id (PK)                  │     │ id (PK)              │
│ date (UNIQUE per symbol) │     │ date                 │
│ stock_symbol             │     │ stock_symbol         │
│ company_name             │     │ open_price           │
│ score                    │     │ close_price          │
│ predicted_gain           │     │ high_price           │
│ current_price            │     │ low_price            │
│ target_price             │     │ volume               │
│ reason                   │     │ pct_change           │
│ rank                     │     │ UNIQUE(date, symbol) │
└──────────────────────────┘     └──────────────────────┘
```

**Design decisions:**

- `portfolio` stores **both open and closed** positions — `status` field differentiates them. This lets you see full trade history without a separate archive table.
- `transactions` is an **immutable audit log** — every buy, sell, and auto-sell is appended, never updated. PnL is computed at sell time and stored in `portfolio.pnl`.
- `ai_recommendations` has a `UNIQUE(date, stock_symbol)` constraint — the scheduler deletes and re-inserts for today's date, preventing duplicate recommendations.
- `daily_prices` uses `ON CONFLICT DO UPDATE` (upsert) — the scheduler can safely run at both 9:15 AM and 3:30 PM without duplicates.
- Passwords are stored as **bcrypt hashes** via `werkzeug.security`, never in plain text.
- All foreign keys point to `users.id`, so every piece of data is user-scoped — one user cannot see another's portfolio.

---

## Project Structure

```
stockwise/
├── app.py              # Flask routes + auth + all API endpoints
├── ml_model.py         # LSTM model training, caching, inference, plotting
├── data_fetch.py       # Market data: yfinance (primary) → TwelveData → Stooq
├── db.py               # PostgreSQL connection pool + all DB operations
├── recommender.py      # Nifty 50 scoring engine + AI picks + DB persistence
├── scheduler.py        # Background: auto-sell monitor + daily recommendations
├── msg.py              # Twilio SMS/WhatsApp alerts
├── schema.sql          # PostgreSQL schema (run once in Supabase SQL editor)
├── companies_india.csv # Nifty 50 symbols and company names
├── templates/
│   ├── index.html      # Main dashboard UI
│   └── auth.html       # Login / Register page
├── requirements.txt
├── Dockerfile
├── fly.toml            # Fly.io deployment config (Mumbai region)
├── render.yaml         # Render deployment config
└── .env.example        # Environment variable template
```

---

## Local Setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/AaryaButolia11/stockwise
cd stockwise
pip install -r requirements.txt

# 2. Create PostgreSQL database
# Option A: Supabase (recommended — free tier)
#   → Create project at supabase.com
#   → Run schema.sql in the SQL Editor
#   → Copy the connection string to DATABASE_URL

# Option B: Local PostgreSQL
psql -U postgres -c "CREATE DATABASE stockwise_db;"
psql -U postgres -d stockwise_db < schema.sql

# 3. Configure environment
cp .env.example .env
# Fill in: DATABASE_URL, FLASK_SECRET_KEY, TWELVE_DATA_KEY (optional),
#          TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_SMS_NUMBER

# 4. Run
python app.py
# Visit http://localhost:8080
```

---

## Deploy to Fly.io

```bash
# 1. Install flyctl and login
curl -L https://fly.io/install.sh | sh
fly auth login

# 2. Create app + persistent volume for model cache
fly launch --name stockwise-app --region bom   # bom = Mumbai
fly volumes create stockwise_data --size 3 --region bom

# 3. Set secrets
fly secrets set FLASK_SECRET_KEY="your_long_random_key"
fly secrets set DATABASE_URL="postgresql://..."
fly secrets set TWILIO_ACCOUNT_SID="ACxxx"
fly secrets set TWILIO_AUTH_TOKEN="xxx"
fly secrets set TWILIO_SMS_NUMBER="+1xxx"
fly secrets set TWELVE_DATA_KEY="xxx"    # optional
fly secrets set NEWS_API_KEY="xxx"       # optional

# 4. Deploy
fly deploy

# 5. Monitor logs
fly logs
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Main dashboard (login required) |
| GET/POST | `/login` `/register` | Auth pages |
| POST | `/api/login` | Login → returns session |
| POST | `/api/register` | Register new user |
| POST | `/api/logout` | Clear session |
| GET | `/get_current_stock_info?symbol=INFY.NS` | Live NSE price |
| GET | `/get_forecast?symbol=INFY.NS&forecast_type=6m` | LSTM forecast + chart |
| POST | `/set_alert` | Set price alert for a stock |
| POST | `/portfolio/buy` | Open a paper trade position |
| POST | `/portfolio/sell` | Close a position at live price |
| GET | `/portfolio` | All positions + P&L summary |
| GET | `/recommendations` | Today's AI top-5 stock picks |
| POST | `/recommendations/refresh` | Trigger fresh recommendation generation |
| GET | `/health` | Health check for Fly.io / Render |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `FLASK_SECRET_KEY` | ✅ | Random secret for session signing |
| `DATABASE_URL` | ✅ | Full PostgreSQL connection string (Supabase) |
| `TWILIO_ACCOUNT_SID` | For SMS | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | For SMS | Twilio auth token |
| `TWILIO_SMS_NUMBER` | For SMS | Your Twilio phone number |
| `TWILIO_WHATSAPP_NUMBER` | For WA | WhatsApp sender (default: Twilio sandbox) |
| `TWELVE_DATA_KEY` | Optional | TwelveData API key (yfinance used if absent) |
| `NEWS_API_KEY` | Optional | NewsAPI key for alert news context |
| `MODEL_CACHE_DIR` | Optional | Path for model cache (default: `/tmp/model_cache`) |
| `MODEL_EXPIRY_DAYS` | Optional | Days before model retraining (default: `7`) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask 3.0 |
| ML | TensorFlow/Keras (LSTM), scikit-learn, NumPy |
| Data | yfinance, TwelveData API, Stooq |
| Database | PostgreSQL via Supabase, psycopg2 |
| Alerts | Twilio SMS + WhatsApp |
| Charting | Matplotlib (server-side PNG → base64) |
| Deployment | Docker, Fly.io (Mumbai), Render |

---

## Disclaimer

StockWise is built for **educational and research purposes only**. The LSTM model's predictions are based on historical price patterns and do not account for fundamental analysis, news events, or macroeconomic factors. Nothing in this project constitutes financial advice. Always consult a qualified financial advisor before making real investment decisions.
