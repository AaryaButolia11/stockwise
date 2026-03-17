# StockWise — AI-Powered Stock Intelligence

> LSTM forecasting · Live price alerts · Paper trading portfolio · Auto stop-loss/take-profit

---

## Features

| Feature | Description |
|---|---|
| **LSTM Forecast** | 6-month and 5-year forecasts with confidence intervals |
| **Model Caching** | Models saved to disk, retrained weekly in background |
| **Price Alerts** | SMS + WhatsApp via Twilio when stocks move |
| **Paper Portfolio** | Buy/sell stocks (demo, no real money) |
| **Auto Stop-Loss** | Background scheduler sells automatically when price hits limits |
| **MySQL Persistence** | All positions, alerts, and transactions stored in DB |


---
# Screenshots
## Sign Up
<img width="1918" height="1028" alt="image" src="https://github.com/user-attachments/assets/9e43b2e9-a77f-4014-a9fc-58433f0d42e3" />

## Login
<img width="1918" height="1031" alt="image" src="https://github.com/user-attachments/assets/968ac026-06ab-4454-a333-30fbf018640b" />

## Dashboard
<img width="1918" height="1033" alt="image" src="https://github.com/user-attachments/assets/3c53dd4a-7157-4ef9-ae5a-76d602a4f6e5" />
<img width="1917" height="1031" alt="image" src="https://github.com/user-attachments/assets/1c59739b-76cd-4a35-8fc8-dfcd81d0b945" />

<img width="1897" height="1030" alt="image" src="https://github.com/user-attachments/assets/f8c104a4-a95c-478b-9f11-e6fce791cb9e" />
<img width="1917" height="1018" alt="image" src="https://github.com/user-attachments/assets/5e1547f6-fd18-4109-aa6d-41fabf977aff" />

<img width="1917" height="1028" alt="image" src="https://github.com/user-attachments/assets/5ee6dc4b-5243-4698-a155-6cb69e684d5f" />
<img width="1917" height="1028" alt="image" src="https://github.com/user-attachments/assets/03cf86d1-98b5-4fa4-8e0b-2da753b8f447" />





## Project Structure

```
stockwise/
├── app.py              # Flask routes
├── ml_model.py         # LSTM model with caching
├── db.py               # MySQL connection pool + all queries
├── scheduler.py        # Background auto-sell monitor
├── msg.py              # Twilio SMS/WhatsApp + Alpha Vantage
├── schema.sql          # Database schema (run once)
├── companies.csv       # S&P 500 company list
├── templates/
│   └── index.html      # Full UI
├── requirements.txt
├── Dockerfile
├── fly.toml
└── .env.example
```

---

## Local Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set up MySQL
mysql -u root -p < schema.sql

# 3. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 4. Run
python app.py
# Visit http://localhost:8080
```

---

## Deploy to Fly.io

```bash
# 1. Install flyctl
curl -L https://fly.io/install.sh | sh

# 2. Login and create app
fly auth login
fly launch --name stockwise-app --region bom

# 3. Create persistent volume (for model cache)
fly volumes create stockwise_data --size 3 --region bom

# 4. Set secrets (one command per secret)
fly secrets set FLASK_SECRET_KEY="your_long_random_key"
fly secrets set DB_HOST="your_db_host"
fly secrets set DB_USER="your_db_user"
fly secrets set DB_PASSWORD="your_db_password"
fly secrets set DB_NAME="stockwise_db"
fly secrets set TWILIO_ACCOUNT_SID="ACxxx"
fly secrets set TWILIO_AUTH_TOKEN="xxx"
fly secrets set TWILIO_SMS_NUMBER="+1xxx"
fly secrets set POLYGON_API_KEY="xxx"
fly secrets set ALPHA_VANTAGE_API_KEY="xxx"
fly secrets set NEWS_API_KEY="xxx"

# 5. Deploy
fly deploy

# 6. Check logs
fly logs
```

### Recommended free MySQL for Fly.io
- **PlanetScale** (planetscale.com) — free tier, MySQL-compatible
- Or **Fly MySQL** addon: `fly postgres create`

---

## How the LSTM caching works

```
First request for AAPL:
  → Train model (~3-5 min) → save to /data/model_cache/AAPL_model.keras
  
All subsequent requests:
  → Load from disk (~2-3 sec) → run inference

After 7 days:
  → Serve stale model instantly → retrain in background thread
  → Next request gets fresh model
```

---

## Auto Stop-Loss / Take-Profit

The background scheduler (`scheduler.py`) runs every 5 minutes:
1. Fetches all open positions with stop-loss or take-profit set
2. Gets live price from Alpha Vantage
3. If triggered → sells automatically, updates DB, sends SMS

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Main UI |
| GET | `/get_current_stock_info?symbol=AAPL` | Live price |
| GET | `/get_forecast?symbol=AAPL&forecast_type=6m` | LSTM forecast |
| POST | `/set_alert` | Set price alert |
| POST | `/portfolio/buy` | Open a position |
| POST | `/portfolio/sell` | Close a position |
| GET | `/portfolio` | All positions + summary |
| GET | `/health` | Health check (Fly.io) |
