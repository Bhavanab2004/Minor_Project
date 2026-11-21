"""Optional data augmentation (enabled by default in train.py, disable with --no-augment).

The provided dataset is machine-generated: benign rows are template SQL with unusual syntax and attacks follow fixed
templates. Models trained on it alone can score ~100% on a held-out split yet misjudge ordinary inputs such as an
e-mail address or "SELECT * FROM users WHERE id = 5", or classic payloads like  ' OR '1'='1 .
This module adds a modest number (~10%) of clean everyday benign inputs and textbook attack payloads so the
detector also behaves sensibly in the Live API Testing panel.
"""
import random
import pandas as pd

FIRST = ["alice", "bob", "carol", "david", "emma", "farid", "gita", "hiro", "ivan", "julia", "karthik", "lena",
         "mohan", "nina", "omar", "priya", "quinn", "rahul", "sara", "tomas", "uma", "vikram", "wei", "zoe",
         "O'Brien", "D'Angelo", "McDonald's", "N'Diaye"]
LAST = ["smith", "kumar", "garcia", "chen", "patel", "nguyen", "kim", "lopez", "brown", "singh", "ali", "rossi"]
WORDS = ["red", "running", "shoes", "laptop", "coffee", "table", "blue", "shirt", "phone", "case", "garden", "book",
         "wireless", "mouse", "organic", "tea", "winter", "jacket", "kids", "toys", "size", "large", "cheap", "best"]
TABLES = ["users", "orders", "products", "customers", "payments", "sessions", "employees", "invoices", "accounts"]
COLS = ["id", "name", "email", "price", "status", "created_at", "city", "quantity", "username", "role"]
DOMAINS = ["example.com", "mail.com", "company.org", "test.io", "gmail.com"]
STATUS = ["active", "pending", "shipped", "cancelled", "paid"]


def _val(r):
    k = r.randrange(11)
    f, l = r.choice(FIRST), r.choice(LAST)
    if k == 0: return f"{f.lower().replace(chr(39), '')}.{l}@{r.choice(DOMAINS)}"
    if k == 1: return f"{f} {l}".title()
    if k == 2: return str(r.randint(1, 99999))
    if k == 3: return f"{r.uniform(1, 999):.2f}"
    if k == 4: return f"{r.randint(2019, 2026)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
    if k == 5: return " ".join(r.choices(WORDS, k=r.randint(1, 4)))
    if k == 6: return f"+{r.randint(1, 91)}{r.randint(10**8, 10**9 - 1)}"
    if k == 7: return r.choice(STATUS)
    if k == 8: return f"{r.randint(1, 999)} {r.choice(LAST).title()} Street"
    if k == 9: return f"{f}_{r.randint(1, 999)}"
    return f"https://{r.choice(DOMAINS)}/{r.choice(WORDS)}/{r.randint(1, 500)}"


def _q(v):
    return "'" + str(v).replace("'", "''") + "'"


def _benign_sql(r):
    t, c1, c2 = r.choice(TABLES), r.choice(COLS), r.choice(COLS)
    v = _val(r)
    k = r.randrange(9)
    lit = _q(v) if not v.replace(".", "").isdigit() else v
    if k == 0: return f"SELECT * FROM {t} WHERE {c1} = {lit}"
    if k == 1: return f"SELECT {c1}, {c2} FROM {t} WHERE {c1} = {lit} AND status = {_q(r.choice(STATUS))}"
    if k == 2: return f"SELECT {c1} FROM {t} ORDER BY {c2} DESC LIMIT {r.randint(1, 100)}"
    if k == 3: return f"SELECT COUNT(*) FROM {t} WHERE {c1} > {r.randint(1, 1000)}"
    if k == 4: return f"INSERT INTO {t} ({c1}, {c2}) VALUES ({lit}, {_q(_val(r))})"
    if k == 5: return f"UPDATE {t} SET {c1} = {lit} WHERE id = {r.randint(1, 9999)}"
    if k == 6: return f"DELETE FROM {t} WHERE id = {r.randint(1, 9999)}"
    if k == 7: return f"SELECT a.{c1}, b.{c2} FROM {t} a JOIN {r.choice(TABLES)} b ON a.id = b.id WHERE a.id = {r.randint(1, 999)}"
    return f"SELECT * FROM {t} WHERE {c1} LIKE {_q('%' + r.choice(WORDS) + '%')}"


def _benign(r):
    k = r.randrange(10)
    if k < 5: return _val(r)
    if k < 8: return _benign_sql(r)
    if k == 8: return '{"name": "%s", "email": "%s", "age": %d}' % (_val(r), _val(r), r.randint(18, 80))
    return f"search={_val(r)}&page={r.randint(1, 20)}&sort={r.choice(COLS)}"


def _pre(r):
    return r.choice([_val(r), str(r.randint(1, 999)), r.choice(FIRST), "admin", ""])


def _wrap(r, inj):
    t, c = r.choice(TABLES), r.choice(["id", "username", "name", "email"])
    k = r.randrange(4)
    if k == 0: return inj
    if k == 1: return f"SELECT * FROM {t} WHERE {c} = '{inj}"
    if k == 2: return f"SELECT * FROM {t} WHERE {c} = {inj}"
    return f"SELECT {r.choice(COLS)} FROM {t} WHERE {c} = '{inj}'"


def _mal(r, cat):
    p, n, m = _pre(r), r.randint(1, 9), r.randint(2, 9)
    cm = r.choice(["--", "-- ", "#", "/*", "-- -"])
    if cat == "Boolean-based":
        c = r.choice([f"{p}' OR '1'='1", f"{p}' OR 1=1{cm}", f"{p}'{cm}", f"{p}\" OR \"\"=\"", f"{p}' OR 'a'='a",
                      f"{p} OR {n}={n}", f"{p}' AND '{n}'='{n}", f"{p}') OR ('1'='1", f"{p}' OR {n}>{n - 1}{cm}",
                      f"{p}' AND ASCII(SUBSTRING((SELECT database()),{n},1))>{r.randint(40, 120)}{cm}"])
    elif cat == "Union-based":
        cols = ",".join(r.choice(["NULL", "1", "username", "password", "@@version", "database()"]) for _ in range(m))
        c = r.choice([f"{p}' UNION SELECT {cols}{cm}", f"{p} UNION ALL SELECT {cols}{cm}",
                      f"{p}' UNION SELECT username, password FROM users{cm}", f"{p}' ORDER BY {r.randint(1, 30)}{cm}",
                      f"{p}' UNION SELECT NULL,NULL,NULL{cm}"])
    elif cat == "Error-based":
        c = r.choice([f"{p}' AND EXTRACTVALUE(1,CONCAT(0x7e,(SELECT @@version))){cm}",
                      f"{p}' AND UPDATEXML(1,CONCAT(0x7e,(SELECT user())),1){cm}",
                      f"{p}' AND (SELECT 1 FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a){cm}",
                      f"{p}' HAVING 1=1{cm}", f"{p}' AND 1=CONVERT(int,(SELECT @@version)){cm}", f"{p}' GROUP BY id HAVING 1=1{cm}"])
    elif cat == "Time-based":
        c = r.choice([f"{p}' OR SLEEP({n}){cm}", f"{p}' AND SLEEP({n}){cm}", f"{p}; WAITFOR DELAY '0:0:{n}'{cm}",
                      f"{p}' AND pg_sleep({n}){cm}", f"{p}' AND BENCHMARK({n}000000,MD5(1)){cm}",
                      f"{p}' AND (SELECT * FROM (SELECT(SLEEP({n})))a){cm}"])
    else:
        t = r.choice(TABLES)
        c = r.choice([f"{p}'; DROP TABLE {t};{cm}", f"{p}; DROP TABLE {t}{cm}", f"{p}'; INSERT INTO users(username,password) VALUES('hacker','x'){cm}",
                      f"{p}'; EXEC xp_cmdshell('dir'){cm}", f"{p}' UNION SELECT table_name FROM information_schema.tables{cm}",
                      f"{p}' AND (SELECT name FROM sqlite_master LIMIT 1) IS NOT NULL{cm}", f"{p}'; UPDATE users SET role='admin'{cm}",
                      f"{p}' AND 1=1 INTO OUTFILE '/tmp/x'{cm}"])
    return _wrap(r, c)


def generate(n_benign, n_per_category, seed=42):
    r = random.Random(seed)
    rows = [(_benign(r), 0, "Benign") for _ in range(n_benign)]
    for cat in ["Boolean-based", "Union-based", "Error-based", "Time-based", "Other"]:
        rows += [(_mal(r, cat), 1, cat) for _ in range(n_per_category)]
    return pd.DataFrame(rows, columns=["payload", "label", "attack_category"]).drop_duplicates("payload")
