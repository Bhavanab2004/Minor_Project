"""Preprocessing, signature matching and feature extraction (TF-IDF + engineered features)."""
import json
import math
import re
import urllib.parse
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

# --------------------------------------------------------------------------- signatures
# (human-readable name, attack category, regex). Used for EXPLANATIONS, attack-category
# labelling and engineered features. The ML models make the actual decision.
_SIG_DEFS = [
    ("UNION SELECT injection", "Union-based", r"\bunion\b(?:\s+all)?\s+select\b"),
    ("ORDER BY column enumeration", "Union-based", r"\border\s+by\s+\d+"),
    ("NULL-column probing", "Union-based", r"\bselect\s+(?:null\s*,\s*)+null"),
    ("Tautology (OR/AND 1=1)", "Boolean-based", r"\b(?:or|and)\b\s+(['\"]?)(\w+)\1\s*=\s*\1\2\1"),
    ("Quote-break with comment (auth bypass)", "Boolean-based", r"['\"]\s*(?:--|#|/\*)"),
    ("Blind extraction (ASCII/SUBSTRING)", "Boolean-based",
     r"\b(?:ascii|ord)\s*\(\s*(?:substring|substr|mid)|\b(?:substring|substr|mid)\s*\(\s*\(?\s*select"),
    ("Error-forcing function", "Error-based",
     r"\b(?:extractvalue|updatexml|xmltype|utl_inaddr|dbms_xmlgen|floor\s*\(\s*rand|exp\s*\(\s*~|convert\s*\(\s*int)"),
    ("HAVING / GROUP BY probing", "Error-based", r"\bhaving\s+\d+\s*=\s*\d+|\bgroup\s+by\b[^;]*\bhaving\b"),
    ("Time-delay function (SLEEP/BENCHMARK)", "Time-based",
     r"\b(?:sleep|pg_sleep|benchmark|randomblob|dbms_pipe\.receive_message)\s*\("),
    ("WAITFOR DELAY", "Time-based", r"\bwaitfor\s+delay\b"),
    ("Stacked query", "Other", r";\s*(?:select|insert|update|delete|drop|exec|execute|create|alter|shutdown|declare)\b"),
    ("Metadata / schema enumeration", "Other",
     r"\b(?:information_schema|sqlite_master|pg_catalog|sysobjects|syscolumns|mysql\.(?:db|user)|sys\.(?:tables|objects|columns)|all_tables)\b"),
    ("Database fingerprinting", "Other", r"@@version|\b(?:version|database|current_user|schema)\s*\(\s*\)"),
    ("Destructive statement (DROP/TRUNCATE)", "Other", r"\b(?:drop|truncate)\s+(?:table|database)\b"),
    ("File / command execution", "Other", r"\b(?:load_file|into\s+(?:out|dump)file|xp_cmdshell)\b"),
    ("SQL comment terminator", "Other", r"(?:--|/\*|\s#)"),
]
SIGNATURES = [(n, c, re.compile(r, re.I)) for n, c, r in _SIG_DEFS]
SIG_TO_CATEGORY = {n: c for n, c, _ in SIGNATURES}
_PRIORITY = ["Time-based", "Error-based", "Union-based", "Boolean-based"]


def match_signatures(text):
    return [n for n, _, rx in SIGNATURES if rx.search(text)]


def categorize(patterns):
    cats = {SIG_TO_CATEGORY[p] for p in patterns}
    for c in _PRIORITY:
        if c in cats:
            return c
    return "Other"


# --------------------------------------------------------------------------- preprocessing
_PCT = re.compile(r"%[0-9a-fA-F]{2}")


def normalize(text):
    """Clean an input: URL-decode (only if it looks encoded), collapse whitespace."""
    t = "" if text is None else str(text)
    for _ in range(2):
        if " " not in t and _PCT.search(t):
            t = urllib.parse.unquote(t)
        else:
            break
    return re.sub(r"\s+", " ", t).strip()


_HTTP_RE = re.compile(r"^\s*(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(\S+)(?:\s+HTTP/[\d.]+)?\s*$", re.I)
_INSPECT_HEADERS = {"user-agent", "referer", "x-forwarded-for", "cookie", "x-api-key", "origin"}


def _flatten_json(obj):
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _flatten_json(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in _flatten_json(x)]
    return [] if obj is None else [str(obj)]


def _body_values(body):
    body = body.strip()
    if not body:
        return []
    try:
        return _flatten_json(json.loads(body))
    except Exception:
        pass
    if "=" in body and "\n" not in body and " " not in body.split("=", 1)[0]:
        return [v for _, v in urllib.parse.parse_qsl(body, keep_blank_values=True)] or [body]
    return [body]


def parse_request(raw):
    """Return (method|None, endpoint|None, [values to inspect]).

    Accepts a raw SQL string, a JSON body, or a full HTTP request
    ("GET /path?x=1 HTTP/1.1 ... headers ... blank line ... body").
    """
    raw = "" if raw is None else str(raw)
    lines = raw.strip().splitlines()
    if lines and _HTTP_RE.match(lines[0]):
        m = _HTTP_RE.match(lines[0])
        method, target = m.group(1).upper(), m.group(2)
        parts = urllib.parse.urlsplit(target)
        values = [v for _, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)]
        body_start = None
        for i, line in enumerate(lines[1:], 1):
            if not line.strip():
                body_start = i + 1
                break
            if ":" in line:
                k, v = line.split(":", 1)
                if k.strip().lower() in _INSPECT_HEADERS:
                    values.append(v.strip())
        if body_start:
            values += _body_values("\n".join(lines[body_start:]))
        values = [normalize(v) for v in values if str(v).strip()]
        return method, parts.path or "/", values or [normalize(urllib.parse.unquote(target))]
    s = raw.strip()
    if s[:1] in "{[":
        try:
            vals = [normalize(v) for v in _flatten_json(json.loads(s)) if str(v).strip()]
            if vals:
                return "POST", None, vals
        except Exception:
            pass
    return None, None, [normalize(raw)]


# --------------------------------------------------------------------------- engineered features
_KEYWORDS = ["select", "union", "insert", "update", "delete", "drop", "from", "where", "or", "and",
             "not", "like", "order", "group", "having", "sleep", "waitfor", "exec", "null", "limit",
             "case", "when", "between", "in"]
_TOKEN = re.compile(r"[a-z_]+")


def handcrafted(text):
    t = text
    tl = t.lower()
    n = len(t) or 1
    tokens = Counter(_TOKEN.findall(tl))
    chars = Counter(t)
    entropy = -sum((c / n) * math.log2(c / n) for c in chars.values()) if t else 0.0
    f = [
        len(t), len(tl.split()), t.count("'"), t.count('"'), t.count(";"), t.count("--"), t.count("#"),
        t.count("/*"), t.count("("), t.count(")"), t.count("="), t.count(","),
        sum(c.isdigit() for c in t) / n,
        sum(c.isupper() for c in t) / n,
        sum((not c.isalnum()) and (not c.isspace()) for c in t) / n,
        entropy,
    ]
    f += [tokens.get(k, 0) for k in _KEYWORDS]
    f += [1.0 if rx.search(t) else 0.0 for _, _, rx in SIGNATURES]
    return f


class FeatureBuilder:
    """Char n-gram TF-IDF + SQL-token TF-IDF + engineered numeric features (shared by all 3 models)."""

    def __init__(self, char_max=30000, word_max=10000):
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=char_max,
                                    sublinear_tf=True, lowercase=True, min_df=3, dtype=np.float32)
        self.word = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]",
                                    ngram_range=(1, 2), max_features=word_max, sublinear_tf=True,
                                    lowercase=True, min_df=3, dtype=np.float32)
        self.scaler = MaxAbsScaler()

    @staticmethod
    def _clean(texts):
        return [normalize(t) for t in texts]

    @staticmethod
    def _hand(texts):
        return np.log1p(np.array([handcrafted(t) for t in texts], dtype=np.float32))

    def fit_transform(self, texts):
        texts = self._clean(texts)
        c = self.char.fit_transform(texts)
        w = self.word.fit_transform(texts)
        h = self.scaler.fit_transform(self._hand(texts))
        return sparse.hstack([c, w, sparse.csr_matrix(h.astype(np.float32))]).tocsr()

    def transform(self, texts):
        texts = self._clean(texts)
        c = self.char.transform(texts)
        w = self.word.transform(texts)
        h = self.scaler.transform(self._hand(texts))
        return sparse.hstack([c, w, sparse.csr_matrix(h.astype(np.float32))]).tocsr()
