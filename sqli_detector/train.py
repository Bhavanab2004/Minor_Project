"""Train SVM, XGBoost and MLP on the sqli_by_type dataset.

Usage:
    python train.py                 # default: ~120k malicious + 120k benign samples
    python train.py --quick         # fast smoke run (~15k rows)
    python train.py --per-type 40000 --benign 240000   # bigger run

The full dataset has ~9.6M rows, so we draw a stratified random sample from every CSV
(reading them in chunks so memory stays low).
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

import config as C
import augment
from features import FeatureBuilder, normalize


def count_rows(path):
    n = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 24)
            if not buf:
                break
            n += buf.count(b"\n")
    return max(n - 1, 1)


def sample_csv(path, n, seed):
    total = count_rows(path)
    frac = min(1.0, n * 1.25 / total)
    parts = []
    for chunk in pd.read_csv(path, usecols=["sql_query"], dtype=str, chunksize=500_000):
        chunk = chunk.dropna()
        parts.append(chunk.sample(frac=frac, random_state=seed) if frac < 1 else chunk)
    df = pd.concat(parts).drop_duplicates("sql_query")
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df.rename(columns={"sql_query": "payload"}).reset_index(drop=True)


def load_data(per_type, n_benign, seed, do_augment=True):
    frames = []
    # split each category's quota across the files that map to it
    cat_files = {}
    for fname, cat in C.ATTACK_FILES.items():
        cat_files.setdefault(cat, []).append(fname)
    for fname, cat in C.ATTACK_FILES.items():
        quota = per_type // len(cat_files[cat]) if cat == "Other" else per_type
        print(f"  sampling {quota:>7,} from {fname}")
        d = sample_csv(C.DATA_DIR / fname, quota, seed)
        d["label"], d["attack_category"] = 1, cat
        frames.append(d)
    print(f"  sampling {n_benign:>7,} from {C.BENIGN_FILE}  (large file, please wait)")
    b = sample_csv(C.DATA_DIR / C.BENIGN_FILE, n_benign, seed)
    b["label"], b["attack_category"] = 0, "Benign"
    frames.append(b)
    if do_augment:
        aug = augment.generate(int(n_benign * 0.10), int(per_type * 0.10), seed)
        print(f"  adding {len(aug):,} augmented rows (clean benign inputs + textbook attack payloads)")
        frames.append(aug)
    df = pd.concat(frames, ignore_index=True)
    df["payload"] = df["payload"].map(normalize)
    df = df[df.payload.str.len() > 0]
    # remove exact duplicates, and any payload that appears under both labels (label noise)
    df = df.drop_duplicates(["payload", "label"])
    df = df[~df.payload.duplicated(keep=False)]
    return df.reset_index(drop=True)


def metrics_for(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, prob),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=20000, help="malicious samples per attack type")
    ap.add_argument("--benign", type=int, default=120000, help="benign samples")
    ap.add_argument("--eval-size", type=int, default=6000, help="size of held-out dashboard evaluation set")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-augment", action="store_true", help="train on the raw dataset only")
    ap.add_argument("--quick", action="store_true", help="small, fast run for testing the pipeline")
    a = ap.parse_args()
    if a.quick:
        a.per_type, a.benign, a.eval_size = 2500, 15000, 2000

    if not C.DATA_DIR.exists():
        raise SystemExit(f"Dataset folder not found: {C.DATA_DIR}\nExtract sqli_by_type.zip into the data/ folder.")
    C.MODEL_DIR.mkdir(exist_ok=True)
    t_all = time.time()

    print("[1/5] Loading & sampling dataset")
    df = load_data(a.per_type, a.benign, a.seed, not a.no_augment)
    print(f"      {len(df):,} rows  |  malicious={int(df.label.sum()):,}  benign={int((df.label == 0).sum()):,}")

    print("[2/5] Train/test split (stratified)")
    tr, te = train_test_split(df, test_size=a.test_size, random_state=a.seed, stratify=df.attack_category)

    print("[3/5] Feature extraction (TF-IDF + engineered features)")
    t0 = time.time()
    fb = FeatureBuilder()
    Xtr = fb.fit_transform(tr.payload.tolist())
    Xte = fb.transform(te.payload.tolist())
    ytr, yte = tr.label.values, te.label.values
    print(f"      matrix {Xtr.shape}, {time.time() - t0:.0f}s")
    joblib.dump(fb, C.MODEL_DIR / "features.joblib")

    print("[4/5] Training models")
    models = {
        "svm": CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=3, method="sigmoid"),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
                                 colsample_bytree=0.6, tree_method="hist", n_jobs=-1,
                                 eval_metric="logloss", random_state=a.seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), batch_size=256, early_stopping=True,
                             n_iter_no_change=3, max_iter=25, random_state=a.seed),
    }
    results = {"dataset": {"train_rows": int(len(tr)), "test_rows": int(len(te)),
                           "malicious": int(df.label.sum()), "benign": int((df.label == 0).sum()),
                           "n_features": int(Xtr.shape[1])}, "models": {}}
    for key, model in models.items():
        print(f"      training {C.ALGOS[key]} ...", flush=True)
        t0 = time.time()
        model.fit(Xtr, ytr)
        train_s = time.time() - t0
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= C.MALICIOUS_T).astype(int)
        m = metrics_for(yte, pred, prob)
        m["train_seconds"] = round(train_s, 1)
        # recall per attack category (+ specificity for benign)
        per_cat = {}
        for cat in C.CATEGORIES + ["Benign"]:
            mask = (te.attack_category == cat).values
            if mask.any():
                per_cat[cat] = float((pred[mask] == (0 if cat == "Benign" else 1)).mean())
        m["per_category"] = per_cat
        results["models"][key] = m
        joblib.dump(model, C.MODEL_DIR / f"{key}.joblib", compress=3)
        print(f"        acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['roc_auc']:.4f}  ({train_s:.0f}s)")

    print("[5/5] Saving metrics + held-out evaluation set for the dashboard")
    C.METRICS_JSON.write_text(json.dumps(results, indent=2))
    rng = np.random.default_rng(a.seed)
    half = a.eval_size // 2
    mal = te[te.label == 1]
    per = max(1, half // len(C.CATEGORIES))
    ev_mal = pd.concat([g.sample(n=min(len(g), per), random_state=a.seed) for _, g in mal.groupby("attack_category")])
    ev_ben = te[te.label == 0].sample(n=min(half, int((te.label == 0).sum())), random_state=a.seed)
    ev = pd.concat([ev_mal, ev_ben]).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    ev["endpoint"] = rng.choice(C.ENDPOINTS, len(ev))
    ev["method"] = rng.choice(C.METHODS, len(ev), p=[0.35, 0.45, 0.1, 0.1])
    ev["label"] = np.where(ev.label == 1, "Malicious", "Benign")
    ev[["payload", "label", "attack_category", "endpoint", "method"]].to_csv(C.EVAL_SET, index=False)
    print(f"      wrote {len(ev):,} evaluation rows -> {C.EVAL_SET}")
    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min. Next: start the API, then the dashboard.")


if __name__ == "__main__":
    main()
