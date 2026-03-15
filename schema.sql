-- StockWise Database Schema
-- Run this once to initialize your database

CREATE DATABASE IF NOT EXISTS stockwise_db;
USE stockwise_db;

-- Users table (login/logout)
CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    username      VARCHAR(50)  NOT NULL UNIQUE,
    email         VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    phone_number  VARCHAR(20),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email    (email),
    INDEX idx_username (username)
);

-- Price alerts (linked to user)
CREATE TABLE IF NOT EXISTS user_alerts (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    user_id       INT          NOT NULL,
    stock_symbol  VARCHAR(20)  NOT NULL,
    phone_number  VARCHAR(20)  NOT NULL,
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    is_active     BOOLEAN      DEFAULT TRUE,
    INDEX idx_user   (user_id),
    INDEX idx_symbol (stock_symbol),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Portfolio (linked to user)
CREATE TABLE IF NOT EXISTS portfolio (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    user_id        INT           NOT NULL,
    stock_symbol   VARCHAR(20)   NOT NULL,
    company_name   VARCHAR(255)  NOT NULL,
    quantity       DECIMAL(15,4) NOT NULL DEFAULT 1,
    buy_price      DECIMAL(15,4) NOT NULL,
    current_price  DECIMAL(15,4),
    stop_loss      DECIMAL(15,4),
    take_profit    DECIMAL(15,4),
    status         ENUM('open','closed','auto_sold') DEFAULT 'open',
    bought_at      TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    sold_at        TIMESTAMP     NULL,
    sell_price     DECIMAL(15,4) NULL,
    pnl            DECIMAL(15,4) NULL,
    phone_number   VARCHAR(20),
    INDEX idx_user   (user_id),
    INDEX idx_symbol (stock_symbol),
    INDEX idx_status (status),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Transaction log
CREATE TABLE IF NOT EXISTS transactions (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id  INT NOT NULL,
    action        ENUM('buy','sell','auto_sell') NOT NULL,
    stock_symbol  VARCHAR(20)   NOT NULL,
    quantity      DECIMAL(15,4) NOT NULL,
    price         DECIMAL(15,4) NOT NULL,
    total_value   DECIMAL(15,4) NOT NULL,
    note          VARCHAR(255),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily AI stock recommendations
CREATE TABLE IF NOT EXISTS ai_recommendations (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    stock_symbol    VARCHAR(20) NOT NULL,
    company_name    VARCHAR(255),
    score           DECIMAL(8,4),
    predicted_gain  DECIMAL(8,4),   -- % predicted gain for the day
    current_price   DECIMAL(15,4),
    target_price    DECIMAL(15,4),
    reason          TEXT,
    rank            INT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, stock_symbol)
);

-- Daily price tracking (open + close)
CREATE TABLE IF NOT EXISTS daily_prices (
    id           SERIAL PRIMARY KEY,
    date         DATE NOT NULL,
    stock_symbol VARCHAR(20) NOT NULL,
    open_price   DECIMAL(15,4),
    close_price  DECIMAL(15,4),
    high_price   DECIMAL(15,4),
    low_price    DECIMAL(15,4),
    volume       BIGINT,
    pct_change   DECIMAL(8,4),
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, stock_symbol)
);
