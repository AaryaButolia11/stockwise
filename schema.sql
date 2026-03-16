-- StockWise Database Schema — PostgreSQL (Supabase)
-- Run once in your Supabase SQL Editor to initialize tables

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(50)  NOT NULL UNIQUE,
    email         VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    phone_number  VARCHAR(20),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_users_email    ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

-- Price alerts (linked to user)
CREATE TABLE IF NOT EXISTS user_alerts (
    id            SERIAL PRIMARY KEY,
    user_id       INT          NOT NULL REFERENCES users(id),
    stock_symbol  VARCHAR(20)  NOT NULL,
    phone_number  VARCHAR(20)  NOT NULL,
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    is_active     BOOLEAN      DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_alerts_user   ON user_alerts (user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON user_alerts (stock_symbol);

-- Portfolio (linked to user)
CREATE TABLE IF NOT EXISTS portfolio (
    id             SERIAL PRIMARY KEY,
    user_id        INT           NOT NULL REFERENCES users(id),
    stock_symbol   VARCHAR(20)   NOT NULL,
    company_name   VARCHAR(255)  NOT NULL,
    quantity       NUMERIC(15,4) NOT NULL DEFAULT 1,
    buy_price      NUMERIC(15,4) NOT NULL,
    current_price  NUMERIC(15,4),
    stop_loss      NUMERIC(15,4),
    take_profit    NUMERIC(15,4),
    status         VARCHAR(20)   DEFAULT 'open' CHECK (status IN ('open','closed','auto_sell')),
    bought_at      TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    sold_at        TIMESTAMP     NULL,
    sell_price     NUMERIC(15,4) NULL,
    pnl            NUMERIC(15,4) NULL,
    phone_number   VARCHAR(20)
);
CREATE INDEX IF NOT EXISTS idx_portfolio_user   ON portfolio (user_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio (stock_symbol);
CREATE INDEX IF NOT EXISTS idx_portfolio_status ON portfolio (status);

-- Transaction log
CREATE TABLE IF NOT EXISTS transactions (
    id            SERIAL PRIMARY KEY,
    portfolio_id  INT NOT NULL,
    action        VARCHAR(20) NOT NULL CHECK (action IN ('buy','sell','auto_sell')),
    stock_symbol  VARCHAR(20)   NOT NULL,
    quantity      NUMERIC(15,4) NOT NULL,
    price         NUMERIC(15,4) NOT NULL,
    total_value   NUMERIC(15,4) NOT NULL,
    note          VARCHAR(255),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily AI stock recommendations
CREATE TABLE IF NOT EXISTS ai_recommendations (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    stock_symbol    VARCHAR(20) NOT NULL,
    company_name    VARCHAR(255),
    score           NUMERIC(8,4),
    predicted_gain  NUMERIC(8,4),
    current_price   NUMERIC(15,4),
    target_price    NUMERIC(15,4),
    reason          TEXT,
    rank            INT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stock_symbol)
);

-- Daily price tracking (open + close)
CREATE TABLE IF NOT EXISTS daily_prices (
    id           SERIAL PRIMARY KEY,
    date         DATE NOT NULL,
    stock_symbol VARCHAR(20) NOT NULL,
    open_price   NUMERIC(15,4),
    close_price  NUMERIC(15,4),
    high_price   NUMERIC(15,4),
    low_price    NUMERIC(15,4),
    volume       BIGINT,
    pct_change   NUMERIC(8,4),
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stock_symbol)
);
