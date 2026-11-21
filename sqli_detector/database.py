"""SQLite security-log storage."""
import sqlite3
import pandas as pd
import config as C

COLUMNS = ["request_id", "timestamp", "endpoint", "method", "payload", "algorithm", "predicted_label",
           "probability", "action", "attack_category", "patterns", "exec_ms", "true_label",
           "true_category", "source"]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT, timestamp TEXT, endpoint TEXT, method TEXT, payload TEXT,
    algorithm TEXT, predicted_label TEXT, probability REAL, action TEXT,
    attack_category TEXT, patterns TEXT, exec_ms REAL, true_label TEXT,
    true_category TEXT, source TEXT
);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_req ON logs(request_id);
"""


def _conn():
    C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(C.DB_PATH, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def insert_logs(rows):
    if not rows:
        return
    con = _conn()
    with con:
        con.executemany(
            f"INSERT INTO logs ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
            [tuple(r.get(c) for c in COLUMNS) for r in rows])
    con.close()


def read_logs():
    con = _conn()
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", con)
    con.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df = pd.DataFrame(columns=["id"] + COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clear_logs():
    con = _conn()
    with con:
        con.execute("DELETE FROM logs")
    con.execute("VACUUM")
    con.close()
