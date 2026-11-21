"""Security report generation: self-contained HTML (with charts) and PDF (tables)."""
import io
from datetime import datetime

import pandas as pd

import analytics as A
import charts as CH
import config as C


def _fmt_metrics(mm):
    if mm.empty:
        return mm
    d = mm.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1-Score", "False Positive Rate", "False Negative Rate"]:
        d[c] = (d[c] * 100).map("{:.2f}%".format)
    d["ROC-AUC"] = d["ROC-AUC"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.4f}")
    d["Avg Latency (ms)"] = d["Avg Latency (ms)"].map("{:.2f}".format)
    return d


def _recommendations(k, mm, fp, fn):
    out = []
    if k["total"] == 0:
        return ["No traffic has been analysed yet."]
    rate = k["malicious"] / k["total"]
    out.append(f"{rate:.1%} of analysed requests were classified as malicious; "
               f"{k['blocked']:,} were blocked and {k['flagged']:,} flagged for review.")
    if not mm.empty:
        best = mm.sort_values(["F1-Score", "Accuracy"], ascending=False).iloc[0]
        out.append(f"Best-performing model in this evaluation: {best['Algorithm']} "
                   f"(F1 {best['F1-Score']:.2%}, accuracy {best['Accuracy']:.2%}).")
        fast = mm.sort_values("Avg Latency (ms)").iloc[0]
        out.append(f"Fastest model: {fast['Algorithm']} ({fast['Avg Latency (ms)']:.2f} ms average per request).")
    if len(fn):
        out.append(f"{len(fn):,} false negatives (missed attacks) were recorded - review them and consider "
                   f"lowering the Flag threshold or retraining with these samples.")
    if len(fp):
        out.append(f"{len(fp):,} false positives (benign requests flagged) were recorded - review them to tune "
                   f"the Block threshold and avoid blocking legitimate users.")
    out.append("Use the ensemble (consensus) view for production decisions; single models are best for benchmarking.")
    return out


def build_html(raw, view, algo_label):
    k = A.kpis(view)
    mm, cms, rocs = A.model_metrics(raw)
    mal = view[view.predicted_label == "Malicious"] if not view.empty else view
    fp, fn = A.fp_fn(view)
    tp = A.top_patterns(view)
    cat_counts = mal.attack_category.fillna("Other").value_counts().reindex(C.CATEGORIES, fill_value=0) if len(mal) else pd.Series(dtype=int)

    first = True

    def fig(f):
        nonlocal first
        h = f.to_html(full_html=False, include_plotlyjs="cdn" if first else False, config={"displayModeBar": False})
        first = False
        return h

    body = []
    body.append(f"<h1>SQL Injection Security Report</h1><p class='muted'>Generated {datetime.now():%Y-%m-%d %H:%M} "
                f"&middot; Algorithm view: <b>{algo_label}</b></p>")
    body.append("<div class='kpis'>" + "".join(
        f"<div class='kpi'><div class='l'>{l}</div><div class='v' style='color:{c}'>{v:,}</div></div>"
        for l, v, c in [("Total requests", k["total"], "#3b82f6"), ("Malicious", k["malicious"], "#ef4444"),
                        ("Benign", k["benign"], "#22c55e"), ("Blocked", k["blocked"], "#ef4444"),
                        ("Flagged", k["flagged"], "#f59e0b")]) + "</div>")
    body.append("<h2>Executive summary</h2><ul>" + "".join(f"<li>{r}</li>" for r in _recommendations(k, mm, fp, fn)) + "</ul>")
    if not mm.empty:
        body.append("<h2>Model performance</h2>" + _fmt_metrics(mm).to_html(index=False, border=0, classes="t"))
        body.append(fig(CH.accuracy_bar(mm)))
        body.append("<div class='row'>" + "".join(fig(CH.confusion(cms[k_], C.ALGOS[k_])) for k_ in cms) + "</div>")
        body.append(fig(CH.category_recall_heatmap(A.category_recall(raw))))
    off = A.load_offline_metrics()
    if off:
        body.append("<h2>Offline benchmark (held-out test split from training)</h2>")
        o = A.offline_table(off)
        for c in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            o[c] = (o[c] * 100).map("{:.2f}%".format)
        o["ROC-AUC"] = o["ROC-AUC"].map("{:.4f}".format)
        body.append(o.to_html(index=False, border=0, classes="t"))
    body.append("<h2>Attack analytics</h2>")
    if len(mal):
        body.append("<div class='row'>" + fig(CH.category_pie(mal)) + fig(CH.category_bar(mal)) + "</div>")
        body.append(fig(CH.category_timeline(mal)))
        body.append(fig(CH.top_patterns_bar(tp)))
    else:
        body.append("<p class='muted'>No malicious requests recorded.</p>")
    body.append(f"<h2>Misclassifications</h2><p>False positives: <b>{len(fp):,}</b> &middot; False negatives: <b>{len(fn):,}</b></p>")
    for title, d in [("Sample false negatives (missed attacks)", fn), ("Sample false positives (benign flagged)", fp)]:
        if len(d):
            s = d.head(10)[["timestamp", "endpoint", "payload", "probability"]].copy()
            s["payload"] = s.payload.str.slice(0, 120)
            body.append(f"<h3>{title}</h3>" + s.to_html(index=False, border=0, classes="t"))

    css = """body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;color:#0f172a}
    h1{color:#1e3a8a}h2{border-bottom:2px solid #e2e8f0;padding-bottom:4px;margin-top:34px}.muted{color:#64748b}
    .kpis{display:flex;gap:12px}.kpi{flex:1;background:#f1f5f9;border-radius:10px;padding:12px 16px}
    .kpi .l{font-size:12px;color:#64748b;text-transform:uppercase}.kpi .v{font-size:26px;font-weight:700}
    table.t{border-collapse:collapse;width:100%;font-size:13px}.t th{background:#0f172a;color:#fff;padding:6px;text-align:left}
    .t td{padding:6px;border-bottom:1px solid #e2e8f0}.row{display:flex;gap:10px;flex-wrap:wrap}.row>div{flex:1;min-width:300px}"""
    return f"<!doctype html><html><head><meta charset='utf-8'><title>SQLi Security Report</title><style>{css}</style></head><body>{''.join(body)}</body></html>"


def build_pdf(raw, view, algo_label):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    k = A.kpis(view)
    mm, cms, _ = A.model_metrics(raw)
    mal = view[view.predicted_label == "Malicious"] if not view.empty else view
    fp, fn = A.fp_fn(view)
    st = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=30)

    def table(df, widths=None):
        data = [list(df.columns)] + df.astype(str).values.tolist()
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]))
        return t

    s = [Paragraph("SQL Injection Security Report", st["Title"]),
         Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M} | Algorithm view: {algo_label}", st["Normal"]), Spacer(1, 12),
         table(pd.DataFrame([{"Total requests": k["total"], "Malicious": k["malicious"], "Benign": k["benign"],
                              "Blocked": k["blocked"], "Flagged": k["flagged"]}])),
         Spacer(1, 12), Paragraph("Executive summary", st["Heading2"])]
    for r in _recommendations(k, mm, fp, fn):
        s.append(Paragraph("&bull; " + r, st["Normal"]))
    if not mm.empty:
        s += [Spacer(1, 10), Paragraph("Model performance (evaluation run)", st["Heading2"]), table(_fmt_metrics(mm)),
              Spacer(1, 10), Paragraph("Confusion matrices (rows = actual, columns = predicted)", st["Heading2"])]
        cm_rows = []
        for key, cm in cms.items():
            cm_rows.append({"Algorithm": C.ALGOS[key], "TN": cm[0][0], "FP": cm[0][1], "FN": cm[1][0], "TP": cm[1][1]})
        s.append(table(pd.DataFrame(cm_rows)))
    if len(mal):
        cc = mal.attack_category.fillna("Other").value_counts().reindex(C.CATEGORIES, fill_value=0)
        s += [Spacer(1, 10), Paragraph("Attacks by category", st["Heading2"]),
              table(pd.DataFrame({"Category": cc.index, "Attacks": cc.values}))]
        tp = A.top_patterns(view)
        if len(tp):
            s += [Spacer(1, 10), Paragraph("Top detected attack patterns", st["Heading2"]), table(tp)]
    s += [Spacer(1, 10), Paragraph(f"False positives: {len(fp):,} | False negatives: {len(fn):,}", st["Heading2"])]
    doc.build(s)
    return buf.getvalue()
