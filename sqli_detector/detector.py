"""Detection engine: loads the 3 trained models and turns raw API input into a security decision."""
import time
import uuid
from datetime import datetime

import joblib
import numpy as np

import config as C
import database
from features import categorize, match_signatures, parse_request

_DETECTOR = None


class Detector:
    def __init__(self):
        self.fb = joblib.load(C.MODEL_DIR / "features.joblib")
        self.models = {}
        for key in C.ALGOS:
            p = C.MODEL_DIR / f"{key}.joblib"
            if p.exists():
                self.models[key] = joblib.load(p)
        if not self.models:
            raise FileNotFoundError("No trained models found. Run `python train.py` first.")


def models_ready():
    return (C.MODEL_DIR / "features.joblib").exists() and all((C.MODEL_DIR / f"{k}.joblib").exists() for k in C.ALGOS)


def get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = Detector()
    return _DETECTOR


def decide_action(p):
    if p >= C.BLOCK_T:
        return "Block"
    if p >= C.FLAG_T:
        return "Flag"
    return "Allow"


def run_detection(items, log=True):
    """items: list of dicts {payload, algorithm ('svm'|'xgboost'|'mlp'|'all'), endpoint?, method?,
    source?, true_label?, true_category?, timestamp?, request_id?}.  Returns one result per (item, algorithm)."""
    det = get_detector()
    parsed, flat_texts, spans = [], [], []
    for it in items:
        m, e, vals = parse_request(it.get("payload", ""))
        parsed.append((m, e, vals))
        spans.append((len(flat_texts), len(flat_texts) + len(vals)))
        flat_texts += vals

    needed = set()
    for it in items:
        a = (it.get("algorithm") or "xgboost").lower()
        needed.update(det.models if a == "all" else [a])
    bad = needed - set(det.models)
    if bad:
        raise ValueError(f"Unknown/unavailable algorithm(s): {sorted(bad)}")

    t0 = time.perf_counter()
    X = det.fb.transform(flat_texts)
    feat_ms = (time.perf_counter() - t0) * 1000
    probs, model_ms = {}, {}
    for a in needed:
        t0 = time.perf_counter()
        probs[a] = det.models[a].predict_proba(X)[:, 1]
        model_ms[a] = (time.perf_counter() - t0) * 1000
    total_vals = max(1, len(flat_texts))
    sig_cache = {}

    results = []
    for it, (m, e, vals), (s, t) in zip(items, parsed, spans):
        algos = list(det.models) if (it.get("algorithm") or "xgboost").lower() == "all" else [it["algorithm"].lower()]
        rid = it.get("request_id") or uuid.uuid4().hex[:12]
        ts = it.get("timestamp") or datetime.now().isoformat(timespec="seconds")
        patterns = []
        for v in vals:
            if v not in sig_cache:
                sig_cache[v] = match_signatures(v)
            patterns += [p for p in sig_cache[v] if p not in patterns]
        share = (t - s) / total_vals
        for a in algos:
            p = float(np.max(probs[a][s:t])) if t > s else 0.0
            label = "Malicious" if p >= C.MALICIOUS_T else "Benign"
            action = decide_action(p)
            category = categorize(patterns) if (label == "Malicious" or action != "Allow") else None
            if label == "Malicious" and patterns:
                expl = "Suspicious patterns detected: " + "; ".join(patterns)
            elif label == "Malicious":
                expl = "No known signature matched; the model flagged statistical token/character patterns typical of injection."
            elif action == "Flag":
                expl = "Borderline score - input partially resembles injection patterns" + (": " + "; ".join(patterns) if patterns else ".")
            else:
                expl = "No suspicious SQL injection patterns found."
            results.append({
                "request_id": rid, "timestamp": ts,
                "endpoint": it.get("endpoint") or e or "/api/query",
                "method": it.get("method") or m or "POST",
                "payload": str(it.get("payload", ""))[:5000],
                "algorithm": a, "algorithm_name": C.ALGOS[a],
                "predicted_label": label, "probability": round(p, 6),
                "confidence": round(max(p, 1 - p), 6), "action": action,
                "attack_category": category, "patterns": "|".join(patterns),
                "exec_ms": round((feat_ms + model_ms[a]) * share, 3),
                "true_label": it.get("true_label"), "true_category": it.get("true_category"),
                "source": it.get("source") or "api", "explanation": expl,
            })
    if log:
        database.insert_logs(results)
    return results
