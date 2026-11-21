"""SQL Injection Security Dashboard (Streamlit).   Run:  streamlit run dashboard.py"""
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

import analytics as A
import charts as CH
import config as C
import database
import detector
import report
from features import SIG_TO_CATEGORY

st.set_page_config(page_title="SQLi Security Dashboard", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 20% 0%,#111b3a 0%,#0a0f1e 45%)}
section[data-testid=stSidebar]{background:#070b16;border-right:1px solid #1e293b}
.kpi{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:14px 18px}
.kpi-l{color:#94a3b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}
.kpi-v{font-size:1.9rem;font-weight:700;line-height:1.2}
.kpi-s{color:#64748b;font-size:.72rem}
.card{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px 18px;height:100%}
.pill{display:inline-block;padding:3px 12px;border-radius:999px;font-weight:600;font-size:.85rem}
h1,h2,h3{letter-spacing:-.01em}
</style>""", unsafe_allow_html=True)

PAGES = ["📊 Security Overview", "🤖 Model Comparison", "🎯 Attack Analytics", "🧪 Live API Testing",
         "📜 Request Logs", "⚙️ Automated Evaluation", "📄 Security Report"]
ACTION_COLOR = {"Allow": CH.GREEN, "Flag": CH.ORANGE, "Block": CH.RED}
LABEL_COLOR = {"Benign": CH.GREEN, "Malicious": CH.RED}


# ----------------------------------------------------------------------------- helpers
def kpi(col, label, value, color, sub=""):
    v = f"{value:,}" if isinstance(value, (int, np.integer)) else value
    col.markdown(f"<div class='kpi' style='border-left:4px solid {color}'><div class='kpi-l'>{label}</div>"
                 f"<div class='kpi-v' style='color:{color}'>{v}</div><div class='kpi-s'>{sub}</div></div>",
                 unsafe_allow_html=True)


def pill(text, color):
    return f"<span class='pill' style='background:{color}22;color:{color};border:1px solid {color}'>{text}</span>"


def api_online():
    try:
        r = requests.get(f"{C.API_URL}/health", timeout=0.7)
        return r.ok and r.json().get("models_ready", False)
    except Exception:
        return False


def call_detection(items, log=True):
    """Send items to the FastAPI detection endpoint; fall back to the in-process detector if the API is down."""
    for i in items:
        i["log"] = log
    if api_online():
        out = []
        for k in range(0, len(items), 300):
            r = requests.post(f"{C.API_URL}/detect/batch", json={"items": items[k:k + 300]}, timeout=300)
            r.raise_for_status()
            out += r.json()
        return out, "FastAPI endpoint"
    return detector.run_detection(items, log=log), "local engine (API not running)"


def window_filter(df, window):
    if df.empty or window == "All time":
        return df
    delta = {"Last hour": timedelta(hours=1), "Last 24 hours": timedelta(hours=24), "Last 7 days": timedelta(days=7)}[window]
    return df[df.timestamp >= datetime.now() - delta]


def no_data():
    st.info("No request data yet. Open **⚙️ Automated Evaluation** and click *Run evaluation* "
            "(or test a payload in **🧪 Live API Testing**).")


# ----------------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🛡️ SQLi Shield")
st.sidebar.caption("ML-based SQL injection detection for APIs")
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
st.sidebar.divider()
algo_view = st.sidebar.selectbox("Algorithm view", A.algo_options(),
                                 help="Consensus averages the probabilities of all 3 models for each request.")
window = st.sidebar.selectbox("Time window", ["All time", "Last hour", "Last 24 hours", "Last 7 days"])
if st.sidebar.button("🔄 Refresh data"):
    st.rerun()
st.sidebar.divider()
if not detector.models_ready():
    st.sidebar.error("Models not trained yet.\nRun `python train.py`")
else:
    st.sidebar.markdown("API endpoint: " + ("🟢 online" if api_online() else "🟠 offline (using local engine)"))

raw_all = database.read_logs()
raw = window_filter(raw_all, window)
view = A.view_df(raw, algo_view)
mal = view[view.predicted_label == "Malicious"] if not view.empty else view
st.title(page.split(" ", 1)[1])

# ----------------------------------------------------------------------------- 1. overview
if page == PAGES[0]:
    k = A.kpis(view)
    c = st.columns(5)
    kpi(c[0], "Total API requests", k["total"], CH.BLUE, algo_view)
    kpi(c[1], "Malicious requests", k["malicious"], CH.RED, f"{k['malicious'] / max(1, k['total']):.1%} of traffic")
    kpi(c[2], "Benign requests", k["benign"], CH.GREEN, f"{k['benign'] / max(1, k['total']):.1%} of traffic")
    kpi(c[3], "Blocked requests", k["blocked"], CH.RED, "action = Block")
    kpi(c[4], "Flagged / suspicious", k["flagged"], CH.ORANGE, "action = Flag")
    st.write("")
    if view.empty:
        no_data()
    else:
        a, b = st.columns([2, 1])
        a.plotly_chart(CH.timeline(view), key="ov_tl")
        b.plotly_chart(CH.classification_donut(view), key="ov_don")
        a, b, c3 = st.columns(3)
        a.plotly_chart(CH.category_bar(mal), key="ov_cat")
        b.plotly_chart(CH.accuracy_ring(view), key="ov_ring")
        c3.plotly_chart(CH.attack_heatmap(mal), key="ov_heat")
        st.subheader("Latest alerts")
        al = view[view.action != "Allow"].sort_values("timestamp", ascending=False).head(12)
        if al.empty:
            st.success("No alerts in the selected window.")
        else:
            st.dataframe(al[["timestamp", "endpoint", "method", "payload", "predicted_label", "probability",
                             "action", "attack_category"]], hide_index=True)

# ----------------------------------------------------------------------------- 2. model comparison
elif page == PAGES[1]:
    mm, cms, rocs = A.model_metrics(raw)
    if mm.empty:
        st.info("No labelled evaluation data yet - run **⚙️ Automated Evaluation** first. "
                "Showing the offline benchmark from training below.")
    else:
        a, b = st.columns(2)
        a.plotly_chart(CH.accuracy_bar(mm), key="mc_acc")
        b.plotly_chart(CH.metrics_grouped(mm), key="mc_grp")
        st.subheader("Model performance comparison table")
        fmt = {c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "False Positive Rate", "False Negative Rate"]}
        fmt.update({"ROC-AUC": "{:.4f}", "Avg Latency (ms)": "{:.2f}"})
        st.dataframe(mm.style.format(fmt).highlight_max(subset=["Accuracy", "F1-Score"], color="#1e3a8a"), hide_index=True)
        st.subheader("Confusion matrix heatmaps")
        cols = st.columns(3)
        for col, key in zip(cols, C.ALGOS):
            if key in cms:
                col.plotly_chart(CH.confusion(cms[key], C.ALGOS[key]), key=f"cm_{key}")
        a, b = st.columns(2)
        a.plotly_chart(CH.category_recall_heatmap(A.category_recall(raw)), key="mc_catheat")
        b.plotly_chart(CH.roc(rocs), key="mc_roc")
        st.plotly_chart(CH.latency_bar(mm), key="mc_lat")
        st.caption("Latency = amortised per-request time (feature extraction + inference) during the batched evaluation. "
                   "Single-request latency is shown in the Live API Testing panel.")
    off = A.load_offline_metrics()
    if off:
        with st.expander("Offline benchmark - held-out test split created by train.py", expanded=mm.empty):
            d = off["dataset"]
            st.caption(f"{d['train_rows']:,} training rows / {d['test_rows']:,} test rows / {d['n_features']:,} features")
            ot = A.offline_table(off)
            st.dataframe(ot.style.format({c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "F1-Score"]}
                                         | {"ROC-AUC": "{:.4f}"}), hide_index=True)
            cols = st.columns(3)
            for col, (key, m) in zip(cols, off["models"].items()):
                col.plotly_chart(CH.confusion(m["confusion_matrix"], C.ALGOS[key] + " (test split)"), key=f"off_{key}")

# ----------------------------------------------------------------------------- 3. attack analytics
elif page == PAGES[2]:
    if mal.empty:
        no_data()
    else:
        a, b = st.columns(2)
        a.plotly_chart(CH.category_pie(mal), key="aa_pie")
        b.plotly_chart(CH.category_bar(mal), key="aa_bar")
        st.plotly_chart(CH.category_timeline(mal), key="aa_line")
        a, b = st.columns([3, 2])
        tp = A.top_patterns(view)
        a.plotly_chart(CH.top_patterns_bar(tp), key="aa_top")
        b.markdown("##### Pattern table")
        b.dataframe(tp, hide_index=True)
        st.plotly_chart(CH.attack_heatmap(mal), key="aa_heat")
    st.subheader("Attack categories reference")
    st.table(pd.DataFrame({
        "Attack category": ["Union-based", "Boolean-based", "Error-based", "Time-based", "Other"],
        "What to display": ["UNION SELECT patterns", "OR 1=1 and logical conditions", "Database error-inducing payloads",
                            "SLEEP() and delay-based patterns", "Other suspicious SQL patterns (stacked queries, metadata, comments)"]}))
    st.caption("Categories are assigned by matching signatures in the payload of requests the ML models classified as malicious.")

# ----------------------------------------------------------------------------- 4. live testing
elif page == PAGES[3]:
    PRESETS = {
        "Boolean tautology": "SELECT * FROM users WHERE username = 'admin' OR '1'='1'",
        "UNION SELECT": "SELECT id, email FROM users WHERE id = 1 UNION SELECT username, password FROM admin--",
        "Time-based (SLEEP)": "SELECT * FROM orders WHERE id = 1 AND (SELECT * FROM (SELECT(SLEEP(5)))a)#",
        "Error-based": "SELECT * FROM users WHERE id = 1 AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT version())))",
        "Stacked query": "SELECT * FROM products WHERE id = 5; DROP TABLE users;--",
        "Benign query": "SELECT id, email FROM customers WHERE created_at LIKE '2024-01-01' AND status = 'active'",
        "HTTP request (GET)": "GET /api/v1/search?q=shoes&id=1%20OR%201%3D1-- HTTP/1.1\nHost: shop.example.com\nUser-Agent: curl/8.0",
        "JSON body": '{"username": "admin\' OR \'1\'=\'1", "password": "x"}',
    }
    if "payload" not in st.session_state:
        st.session_state.payload = PRESETS["Boolean tautology"]

    def _preset():
        st.session_state.payload = PRESETS[st.session_state.preset]

    st.selectbox("Load an example", list(PRESETS), key="preset", on_change=_preset)
    st.text_area("Enter API input / SQL payload (raw SQL, JSON body, or a full HTTP request)", key="payload", height=130)
    c1, c2, c3 = st.columns([2, 1, 1])
    choice = c1.radio("Select ML algorithm", ["Support Vector Machine", "XGBoost", "Multilayer Perceptron", "Compare all three"],
                      horizontal=True)
    endpoint = c2.text_input("Endpoint (optional)", value="/api/v1/search")
    log_it = c3.checkbox("Save to request logs", value=True)
    if st.button("🔍 Analyze request", type="primary"):
        if not detector.models_ready():
            st.error("Train the models first: `python train.py`")
        elif not st.session_state.payload.strip():
            st.warning("Enter a payload first.")
        else:
            algo = "all" if choice == "Compare all three" else [k for k, v in C.ALGOS.items() if v == choice][0]
            t0 = time.perf_counter()
            res, via = call_detection([{"payload": st.session_state.payload, "algorithm": algo, "endpoint": endpoint,
                                        "source": "live"}], log=log_it)
            rt = (time.perf_counter() - t0) * 1000
            st.caption(f"Served by {via} - round-trip {rt:.0f} ms")
            cols = st.columns(len(res))
            for col, r in zip(cols, res):
                with col:
                    st.markdown(f"<div class='card'><h4 style='margin-top:0'>{r['algorithm_name']}</h4>"
                                f"Classification &nbsp;{pill(r['predicted_label'], LABEL_COLOR[r['predicted_label']])}<br><br>"
                                f"Recommended action &nbsp;{pill(r['action'], ACTION_COLOR[r['action']])}<br><br>"
                                f"Confidence: <b>{r['confidence']:.2%}</b><br>"
                                f"P(malicious): <b>{r['probability']:.2%}</b><br>"
                                f"Execution time: <b>{r['exec_ms']:.2f} ms</b><br>"
                                f"Attack type: <b>{r['attack_category'] or '-'}</b></div>", unsafe_allow_html=True)
                    st.plotly_chart(CH.gauge(r["probability"], "Malicious probability"), key=f"g_{r['algorithm']}")
            st.subheader("Explanation of suspicious input patterns")
            st.write(res[0]["explanation"])
            pats = [p for p in res[0]["patterns"].split("|") if p]
            if pats:
                st.dataframe(pd.DataFrame({"Matched pattern": pats,
                                           "Attack category": [SIG_TO_CATEGORY[p] for p in pats]}),
                             hide_index=True)
    st.caption("Thresholds: Allow < 40% - Flag 40-80% - Block ≥ 80% malicious probability (change in config.py).")

# ----------------------------------------------------------------------------- 5. logs
elif page == PAGES[4]:
    if raw_all.empty:
        no_data()
    else:
        c = st.columns([2, 1, 1, 1])
        q = c[0].text_input("🔎 Search by endpoint or payload")
        verdict = c[1].multiselect("Classification", ["Benign", "Malicious"], default=["Benign", "Malicious"])
        algos = c[2].multiselect("Algorithm", list(C.ALGOS), default=list(C.ALGOS), format_func=C.ALGOS.get)
        action = c[3].multiselect("Action", ["Allow", "Flag", "Block"], default=["Allow", "Flag", "Block"])
        tmin, tmax = raw_all.timestamp.min(), raw_all.timestamp.max()
        d = st.columns(4)
        d1 = d[0].date_input("From date", tmin.date(), min_value=tmin.date(), max_value=tmax.date())
        t1 = d[1].time_input("From time", datetime.min.time())
        d2 = d[2].date_input("To date", tmax.date(), min_value=tmin.date(), max_value=tmax.date())
        t2 = d[3].time_input("To time", datetime.max.time().replace(microsecond=0))
        f = raw_all[raw_all.predicted_label.isin(verdict) & raw_all.algorithm.isin(algos) & raw_all.action.isin(action)]
        f = f[(f.timestamp >= datetime.combine(d1, t1)) & (f.timestamp <= datetime.combine(d2, t2))]
        if q:
            f = f[f.endpoint.str.contains(q, case=False, regex=False, na=False) |
                  f.payload.str.contains(q, case=False, regex=False, na=False)]
        fp, fn = A.fp_fn(f)
        t_all, t_fp, t_fn = st.tabs([f"All logs ({len(f):,})", f"False positives ({len(fp):,})", f"False negatives ({len(fn):,})"])
        cols = ["id", "timestamp", "endpoint", "method", "payload", "algorithm", "predicted_label", "probability",
                "action", "attack_category", "exec_ms", "true_label", "source"]

        def show(df, key):
            if df.empty:
                st.success("Nothing to show here.")
                return
            st.download_button("⬇️ Export logs as CSV", df.to_csv(index=False).encode(), f"sqli_logs_{key}.csv",
                               "text/csv", key=f"dl_{key}")
            sel = st.dataframe(df[cols].head(2000), hide_index=True, on_select="rerun",
                               selection_mode="single-row", key=f"tbl_{key}")
            st.caption("Showing up to 2,000 rows. Click a row (checkbox on the left) to view complete request details.")
            rows = sel.selection.rows if hasattr(sel, "selection") else []
            if rows:
                rec = df.head(2000).iloc[rows[0]].to_dict()
                rec["timestamp"] = str(rec["timestamp"])
                st.markdown("##### Complete request details")
                st.json({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in rec.items()})

        with t_all:
            show(f, "all")
        with t_fp:
            st.caption("Benign requests that the model classified as malicious (labelled evaluation requests only).")
            show(fp, "fp")
        with t_fn:
            st.caption("Malicious requests that the model missed (labelled evaluation requests only).")
            show(fn, "fn")

# ----------------------------------------------------------------------------- 6. automated evaluation
elif page == PAGES[5]:
    st.markdown("""**Pipeline:** 1) load benign + malicious SQL queries → 2) send each to the API detection endpoint →
3) pass through SVM, XGBoost and MLP → 4) collect predictions → 5) compare with actual labels →
6) calculate metrics → 7) show charts & heatmaps on the dashboard.""")
    if not detector.models_ready():
        st.error("Models not found. Run `python train.py` first.")
    else:
        src = st.radio("Query source", ["Built-in held-out test set (from training)", "Upload my own CSV"], horizontal=True)
        data = None
        if src.startswith("Built-in"):
            if C.EVAL_SET.exists():
                data = pd.read_csv(C.EVAL_SET)
            else:
                st.error("eval_set.csv missing - run train.py.")
        else:
            up = st.file_uploader("CSV with a `sql_query` (or `payload`) column and optional `label` column "
                                  "(Malicious/Benign/1/0) or `injection_type`", type="csv")
            if up:
                u = pd.read_csv(up)
                pc = "payload" if "payload" in u.columns else "sql_query"
                if pc not in u.columns:
                    st.error("CSV needs a `sql_query` or `payload` column.")
                else:
                    data = pd.DataFrame({"payload": u[pc].astype(str)})
                    if "label" in u.columns:
                        data["label"] = u["label"].astype(str).str.lower().map(
                            lambda v: "Malicious" if v in ("1", "malicious", "yes", "attack", "true") else "Benign")
                    elif "injection_type" in u.columns:
                        data["label"] = np.where(u["injection_type"].astype(str).str.lower().isin(["none_type", "none", "benign"]),
                                                 "Benign", "Malicious")
                    else:
                        data["label"] = None
                    data["attack_category"] = None
                    rng = np.random.default_rng(1)
                    data["endpoint"] = rng.choice(C.ENDPOINTS, len(data))
                    data["method"] = rng.choice(C.METHODS, len(data))
        if data is not None and len(data):
            c1, c2 = st.columns(2)
            n = c1.slider("Number of queries to test", 50, min(len(data), 10000), min(len(data), 1500), step=50)
            hours = c2.number_input("Spread timestamps over the last N hours (0 = now). Simulates historical traffic "
                                    "so the time charts are meaningful.", 0, 720, 24)
            if st.button("▶️ Run evaluation", type="primary"):
                sample = data.sample(n=n, random_state=int(time.time()) % 10000).reset_index(drop=True)
                rng = np.random.default_rng()
                now = datetime.now()
                span = hours * 3600
                bursts = rng.uniform(0.1, 0.9, 3) * span
                items = []
                for r in sample.itertuples():
                    if span:
                        off = rng.uniform(0, span) if (r.label != "Malicious" or rng.random() < 0.6) \
                            else float(np.clip(rng.normal(rng.choice(bursts), 2400), 0, span))
                        ts = (now - timedelta(seconds=float(off))).isoformat(timespec="seconds")
                    else:
                        ts = now.isoformat(timespec="seconds")
                    items.append({"payload": r.payload, "algorithm": "all", "endpoint": r.endpoint, "method": r.method,
                                  "source": "evaluation", "true_label": r.label if isinstance(r.label, str) else None,
                                  "true_category": r.attack_category if isinstance(r.attack_category, str) else None,
                                  "timestamp": ts})
                bar = st.progress(0.0, text="Sending queries to the detection endpoint...")
                t0 = time.time()
                via = ""
                for i in range(0, len(items), 300):
                    _, via = call_detection(items[i:i + 300], log=True)
                    bar.progress(min(1.0, (i + 300) / len(items)), text=f"Classified {min(i + 300, len(items)):,}/{len(items):,} queries")
                bar.empty()
                st.success(f"Done: {len(items):,} queries × 3 algorithms in {time.time() - t0:.1f}s via {via}.")
                mm, _, _ = A.model_metrics(database.read_logs().query("source == 'evaluation'").tail(len(items) * 3))
                if not mm.empty:
                    st.dataframe(mm[["Algorithm", "Samples", "Accuracy", "Precision", "Recall", "F1-Score"]].style.format(
                        {c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "F1-Score"]}), hide_index=True)
                st.info("Open **Model Comparison** for confusion-matrix heatmaps and **Security Overview** for the full dashboard.")
    st.divider()
    with st.expander("🗑️ Danger zone - clear all logs"):
        if st.checkbox("I understand this permanently deletes all stored request logs") and st.button("Clear logs"):
            database.clear_logs()
            st.success("Logs cleared.")
            st.rerun()

# ----------------------------------------------------------------------------- 7. report
elif page == PAGES[6]:
    st.write("Generate a security report from the currently selected algorithm view and time window.")
    if view.empty:
        no_data()
    else:
        if st.button("📄 Generate report", type="primary"):
            with st.spinner("Building report..."):
                st.session_state.report_html = report.build_html(raw, view, algo_view)
                st.session_state.report_pdf = report.build_pdf(raw, view, algo_view)
        if "report_html" in st.session_state:
            c = st.columns(2)
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            c[0].download_button("⬇️ Download HTML report (with charts)", st.session_state.report_html,
                                 f"sqli_report_{stamp}.html", "text/html")
            c[1].download_button("⬇️ Download PDF report", st.session_state.report_pdf,
                                 f"sqli_report_{stamp}.pdf", "application/pdf")
            with st.expander("Preview", expanded=True):
                st.components.v1.html(st.session_state.report_html, height=700, scrolling=True)
