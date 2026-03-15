"""
db.py — PostgreSQL connection pool + all DB operations for StockWise
Includes user auth (register/login) + user-scoped portfolio/alerts
"""
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from psycopg2 import pool, Error
from psycopg2.extras import RealDictCursor

# ── Connection config ────────────────────────────────────────────────────────
# On Render, set DATABASE_URL env var (Render PostgreSQL provides this automatically).
# Falls back to individual vars for local dev.

DATABASE_URL = os.getenv("DATABASE_URL")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "dbname":   os.getenv("DB_NAME", "stockwise_db"),
    "port":     int(os.getenv("DB_PORT", "5432")),
}

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        if DATABASE_URL:
            _pool = pool.SimpleConnectionPool(1, 5, dsn=DATABASE_URL, sslmode="require")
        else:
            _pool = pool.SimpleConnectionPool(1, 5, **DB_CONFIG)
    return _pool

def get_conn():
    return get_pool().getconn()

def release_conn(conn):
    try:
        get_pool().putconn(conn)
    except Exception:
        pass

def _dict_cursor(conn):
    return conn.cursor(cursor_factory=RealDictCursor)


# ── Auth ─────────────────────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str, phone: str = None):
    """Returns (True, user_dict) or (False, error_message)."""
    conn = None
    try:
        conn = get_conn()
        cur  = _dict_cursor(conn)
        cur.execute("SELECT id FROM users WHERE email=%s OR username=%s", (email, username))
        if cur.fetchone():
            return False, "Email or username already exists."
        hashed = generate_password_hash(password)
        cur.execute(
            """INSERT INTO users (username, email, password_hash, phone_number)
               VALUES (%s,%s,%s,%s) RETURNING id""",
            (username, email, hashed, phone)
        )
        uid = cur.fetchone()["id"]
        conn.commit()
        return True, {"id": uid, "username": username, "email": email, "phone_number": phone}
    except Error as e:
        print(f"[DB] register_user error: {e}")
        if conn: conn.rollback()
        return False, "Database error during registration."
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)


def login_user(email: str, password: str):
    """Returns (True, user_dict) or (False, error_message)."""
    conn = None
    try:
        conn = get_conn()
        cur  = _dict_cursor(conn)
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        if not user:
            return False, "No account found with that email."
        user = dict(user)
        if not check_password_hash(user["password_hash"], password):
            return False, "Incorrect password."
        user.pop("password_hash", None)
        if user.get("created_at"):
            user["created_at"] = str(user["created_at"])
        return True, user
    except Error as e:
        print(f"[DB] login_user error: {e}")
        return False, "Database error during login."
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)


def get_user_by_id(user_id: int):
    conn = None
    try:
        conn = get_conn()
        cur  = _dict_cursor(conn)
        cur.execute(
            "SELECT id, username, email, phone_number, created_at FROM users WHERE id=%s",
            (user_id,)
        )
        u = cur.fetchone()
        if u:
            u = dict(u)
            if u.get("created_at"):
                u["created_at"] = str(u["created_at"])
        return u
    except Error as e:
        print(f"[DB] get_user_by_id error: {e}")
        return None
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)


# ── Alerts ───────────────────────────────────────────────────────────────────

def save_alert(stock_symbol: str, phone_number: str, user_id: int) -> bool:
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO user_alerts (user_id, stock_symbol, phone_number) VALUES (%s,%s,%s)",
            (user_id, stock_symbol, phone_number)
        )
        conn.commit()
        return True
    except Error as e:
        print(f"[DB] save_alert error: {e}")
        if conn: conn.rollback()
        return False
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)


def get_all_alerts():
    conn = None
    try:
        conn = get_conn()
        cur  = _dict_cursor(conn)
        cur.execute("SELECT * FROM user_alerts WHERE is_active=TRUE")
        return [dict(r) for r in cur.fetchall()]
    except Error as e:
        print(f"[DB] get_all_alerts error: {e}")
        return []
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)


# ── Portfolio ─────────────────────────────────────────────────────────────────

def buy_stock(symbol, company, quantity, price, stop_loss, take_profit, phone, user_id):
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO portfolio
              (user_id, stock_symbol, company_name, quantity, buy_price,
               current_price, stop_loss, take_profit, status, phone_number)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'open',%s)
            RETURNING id
        """, (user_id, symbol, company, quantity, price, price, stop_loss, take_profit, phone))
        pid = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO transactions
              (portfolio_id, action, stock_symbol, quantity, price, total_value, note)
            VALUES (%s,'buy',%s,%s,%s,%s,'Manual buy')
        """, (pid, symbol, quantity, price, round(quantity * price, 4)))
        conn.commit()
        return pid
    except Error as e:
        print(f"[DB] buy_stock error: {e}")
        if conn: conn.rollback()
        return None
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)


def sell_stock(portfolio_id: int, sell_price: float, action: str = "sell", user_id: int = None) -> bool:
    conn = None
    try:
        conn = get_conn()
        cur  = _dict_cursor(conn)
        query  = "SELECT * FROM portfolio WHERE id=%s AND status='open'"
        params = [portfolio_id]
        if user_id:
            query += " AND user_id=%s"
            params.append(user_id)
        cur.execute(query, params)
        pos = cur.fetchone()
        if not pos:
            return False
        pos = dict(pos)
        pnl = round((sell_price - float(pos["buy_price"])) * float(pos["quantity"]), 4)
        cur.execute("""
            UPDATE portfolio
            SET status=%s, sold_at=NOW(), sell_price=%s, pnl=%s, current_price=%s
            WHERE id=%s
        """, (action, sell_price, pnl, sell_price, portfolio_id))
        cur.execute("""
            INSERT INTO transactions
              (portfolio_id, action, stock_symbol, quantity, price, total_value, note)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (portfolio_id, action, pos["stock_symbol"], pos["quantity"],
              sell_price, round(float(pos["quantity"]) * sell_price, 4), f"PnL: ₹{pnl}"))
        conn.commit()
        return True
    except Error as e:
        print(f"[DB] sell_stock error: {e}")
        if conn: conn.rollback()
        return False
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)


def get_open_positions(user_id: int = None):
    conn = None
    try:
        conn = get_conn()
        cur  = _dict_cursor(conn)
        if user_id:
            cur.execute(
                "SELECT * FROM portfolio WHERE status='open' AND user_id=%s ORDER BY bought_at DESC",
                (user_id,)
            )
        else:
            cur.execute("SELECT * FROM portfolio WHERE status='open' ORDER BY bought_at DESC")
        return [dict(r) for r in cur.fetchall()]
    except Error as e:
        print(f"[DB] get_open_positions error: {e}")
        return []
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)


def get_all_positions(user_id: int = None):
    conn = None
    try:
        conn = get_conn()
        cur  = _dict_cursor(conn)
        if user_id:
            cur.execute(
                "SELECT * FROM portfolio WHERE user_id=%s ORDER BY bought_at DESC",
                (user_id,)
            )
        else:
            cur.execute("SELECT * FROM portfolio ORDER BY bought_at DESC")
        return [dict(r) for r in cur.fetchall()]
    except Error as e:
        print(f"[DB] get_all_positions error: {e}")
        return []
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)


def update_current_price(portfolio_id: int, price: float):
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("UPDATE portfolio SET current_price=%s WHERE id=%s", (price, portfolio_id))
        conn.commit()
    except Error as e:
        print(f"[DB] update_current_price error: {e}")
        if conn: conn.rollback()
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)


def get_portfolio_summary(user_id: int = None):
    conn = None
    try:
        conn = get_conn()
        cur  = _dict_cursor(conn)
        if user_id:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE status='open')  AS open_count,
                    COUNT(*) FILTER (WHERE status!='open') AS closed_count,
                    COALESCE(SUM(CASE WHEN status='open' THEN quantity*buy_price END), 0) AS invested,
                    COALESCE(SUM(pnl), 0) AS total_pnl
                FROM portfolio WHERE user_id=%s
            """, (user_id,))
        else:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE status='open')  AS open_count,
                    COUNT(*) FILTER (WHERE status!='open') AS closed_count,
                    COALESCE(SUM(CASE WHEN status='open' THEN quantity*buy_price END), 0) AS invested,
                    COALESCE(SUM(pnl), 0) AS total_pnl
                FROM portfolio
            """)
        row = cur.fetchone()
        return dict(row) if row else {}
    except Error as e:
        print(f"[DB] get_portfolio_summary error: {e}")
        return {}
    finally:
        try: cur.close()
        except: pass
        if conn: release_conn(conn)