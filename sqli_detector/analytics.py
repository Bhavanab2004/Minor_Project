"""Metric / aggregation helpers shared by the dashboard and the report generator."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

import config as C

CONSENSUS = "All (consensus)"


def algo_options():
    return [CONSENSUS] + list(C.ALGOS.values())


def view_df(raw, algo_label):
    """One row per request. For a single algorithm: its rows. For consensus: mean probability of the 3 models."""
    if raw.empty:
        return raw.copy()
    if algo_label != CONSENSUS:
        key = [k for k, v in C.ALGOS.items() if v == algo_label][0]
        return raw[raw.algorithm == key].copy()
    top = raw.sort_values("probability").drop_duplicates("request_id", keep="last").set_index("request_id")
    top["probability"] = raw.groupby("request_id")["probability"].mean().reindex(top.index)
    top["exec_ms"] = raw.groupby("request_id")["exec_ms"].sum().reindex(top.index)
    out = top.reset_index()
    out["predicted_label"] = np.where(out.probability >= C.MALICIOUS_T, "Malicious", "Benign")
    out["action"] = np.select([out.probability >= C.BLOCK_T, out.probability >= C.FLAG_T], ["Block", "Flag"], "Allow")
    out["algorithm"] = "consensus"
    out["attack_category"] = out["attack_category"].where(out.predicted_label == "Malicious", None)
    out.loc[(out.predicted_label == "Malicious") & out.attack_category.isna(), "attack_category"] = "Other"
    return out


def kpis(df):
    if df.empty:
        return dict(total=0, malicious=0, benign=0, blocked=0, flagged=0)
    return dict(total=len(df), malicious=int((df.predicted_label == "Malicious").sum()),
                benign=int((df.predicted_label == "Benign").sum()),
                blocked=int((df.action == "Block").sum()), flagged=int((df.action == "Flag").sum()))


def eval_rows(raw):
    return raw[raw.true_label.notna()] if not raw.empty else raw


def model_metrics(raw):
    """Metrics computed from logged predictions on labelled (evaluation) requests."""
    ev = eval_rows(raw)
    rows, cms, rocs = [], {}, {}
    from sklearn.metrics import roc_curve
    for key, name in C.ALGOS.items():
        d = ev[ev.algorithm == key]
        if d.empty:
            continue
        y = (d.true_label == "Malicious").astype(int).values
        p = (d.predicted_label == "Malicious").astype(int).values
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        div = lambda a, b: a / b if b else 0.0
        both = len(set(y)) == 2
        prec, rec = div(tp, tp + fp), div(tp, tp + fn)
        rows.append({"Algorithm": name, "Samples": len(d), "Accuracy": div(tp + tn, len(d)),
                     "Precision": prec, "Recall": rec, "F1-Score": div(2 * prec * rec, prec + rec),
                     "ROC-AUC": roc_auc_score(y, d.probability) if both else np.nan,
                     "False Positive Rate": div(fp, fp + tn), "False Negative Rate": div(fn, fn + tp),
                     "Avg Latency (ms)": d.exec_ms.mean()})
        cms[key] = np.array([[tn, fp], [fn, tp]])
        if both:
            fpr, tpr, _ = roc_curve(y, d.probability)
            rocs[key] = (fpr, tpr)
    return pd.DataFrame(rows), cms, rocs


def category_recall(raw):
    """Rows = true attack category (+Benign specificity), cols = algorithm, value = fraction classified correctly."""
    ev = eval_rows(raw)
    if ev.empty:
        return pd.DataFrame()
    ev = ev.copy()
    ev["correct"] = (ev.true_label == ev.predicted_label).astype(float)
    ev["cat"] = np.where(ev.true_label == "Benign", "Benign", ev.true_category.fillna("Other"))
    t = ev.pivot_table(index="cat", columns="algorithm", values="correct", aggfunc="mean")
    t = t.rename(columns=C.ALGOS)
    order = [c for c in C.CATEGORIES + ["Benign"] if c in t.index]
    return t.loc[order]


def top_patterns(df, n=10):
    mal = df[(df.predicted_label == "Malicious") & df.patterns.notna() & (df.patterns != "")]
    if mal.empty:
        return pd.DataFrame(columns=["Pattern", "Count"])
    s = mal.patterns.str.split("|").explode()
    out = s.value_counts().head(n).rename_axis("Pattern").reset_index(name="Count")
    return out


def fp_fn(raw_or_view):
    ev = eval_rows(raw_or_view)
    fp = ev[(ev.true_label == "Benign") & (ev.predicted_label == "Malicious")]
    fn = ev[(ev.true_label == "Malicious") & (ev.predicted_label == "Benign")]
    return fp, fn


def load_offline_metrics():
    if C.METRICS_JSON.exists():
        return json.loads(C.METRICS_JSON.read_text())
    return None


def offline_table(js):
    rows = []
    for k, m in js["models"].items():
        rows.append({"Algorithm": C.ALGOS[k], "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1-Score": m["f1"], "ROC-AUC": m["roc_auc"],
                     "Training time (s)": m["train_seconds"]})
    return pd.DataFrame(rows)
