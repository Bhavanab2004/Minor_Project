"""Plotly figure builders (dark navy theme; green=benign, red=malicious, orange=suspicious, blue/purple=models)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config as C

GREEN, RED, ORANGE = "#22c55e", "#ef4444", "#f59e0b"
BLUE, PURPLE, CYAN = "#3b82f6", "#8b5cf6", "#22d3ee"
MODEL_COLORS = {C.ALGOS["svm"]: BLUE, C.ALGOS["xgboost"]: PURPLE, C.ALGOS["mlp"]: CYAN}
LABEL_COLORS = {"Benign": GREEN, "Malicious": RED, "Suspicious": ORANGE}
CAT_COLORS = {"Union-based": RED, "Boolean-based": ORANGE, "Error-based": PURPLE, "Time-based": BLUE, "Other": "#94a3b8"}
HEAT = [[0, "#0f172a"], [0.5, "#6d28d9"], [1, "#ef4444"]]
HEAT_BLUE = [[0, "#0f172a"], [0.5, "#4338ca"], [1, "#3b82f6"]]
HEAT_ACC = [[0, "#7f1d1d"], [0.5, "#f59e0b"], [1, "#16a34a"]]


def style(fig, h=340, legend=True):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=45, b=10), font=dict(color="#cbd5e1"),
                      title_font=dict(size=15), showlegend=legend,
                      legend=dict(orientation="h", y=-0.18, x=0))
    return fig


def _bucket(ts):
    if ts.empty:
        return pd.Timedelta("1h")
    span = (ts.max() - ts.min()).total_seconds()
    if span <= 1800:
        return pd.Timedelta("1min")
    if span <= 3 * 3600:
        return pd.Timedelta("5min")
    if span <= 3 * 86400:
        return pd.Timedelta("1h")
    return pd.Timedelta("1D")


def empty(msg="No data yet"):
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#64748b"))
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    return style(f, 300, False)


def classification_donut(df):
    if df.empty:
        return empty()
    c = df.predicted_label.value_counts()
    f = go.Figure(go.Pie(labels=c.index, values=c.values, hole=0.62, sort=False,
                         marker=dict(colors=[LABEL_COLORS[l] for l in c.index]), textinfo="label+percent"))
    return style(f.update_layout(title="Request classification"), 320, False)


def timeline(df, title="Attack monitoring - requests over time"):
    if df.empty:
        return empty()
    d = df.copy()
    d["bucket"] = d.timestamp.dt.floor(_bucket(d.timestamp))
    g = d.groupby(["bucket", "predicted_label"]).size().reset_index(name="Requests")
    f = px.line(g, x="bucket", y="Requests", color="predicted_label", color_discrete_map=LABEL_COLORS, markers=True)
    f.update_traces(line_shape="spline")
    f.update_layout(title=title, xaxis_title=None, legend_title=None)
    return style(f, 320)


def accuracy_ring(df):
    """Mockup-style 'Detection Accuracy' ring: TP / TN / FP / FN of labelled requests."""
    ev = df[df.true_label.notna()] if not df.empty else df
    if ev.empty:
        return empty("Run an evaluation to see detection accuracy")
    y, p = ev.true_label == "Malicious", ev.predicted_label == "Malicious"
    vals = {"True positives": int((y & p).sum()), "True negatives": int((~y & ~p).sum()),
            "False positives": int((~y & p).sum()), "False negatives": int((y & ~p).sum())}
    f = go.Figure(go.Pie(labels=list(vals), values=list(vals.values()), hole=0.7, sort=False,
                         marker=dict(colors=[RED, GREEN, ORANGE, PURPLE]), textinfo="none"))
    acc = (vals["True positives"] + vals["True negatives"]) / max(1, len(ev))
    f.add_annotation(text=f"<b>{acc:.2%}</b><br><span style='font-size:11px'>accuracy</span>", showarrow=False,
                     font=dict(size=22))
    return style(f.update_layout(title="Detection accuracy"), 320)


def category_bar(mal):
    if mal.empty:
        return empty()
    c = mal.attack_category.fillna("Other").value_counts().reindex(C.CATEGORIES, fill_value=0).reset_index()
    c.columns = ["Category", "Attacks"]
    f = px.bar(c, x="Category", y="Attacks", color="Category", color_discrete_map=CAT_COLORS, text="Attacks")
    f.update_layout(title="Number of attacks by category", xaxis_title=None)
    return style(f, 340, False)


def category_pie(mal):
    if mal.empty:
        return empty()
    c = mal.attack_category.fillna("Other").value_counts().reset_index()
    c.columns = ["Category", "Attacks"]
    f = px.pie(c, names="Category", values="Attacks", color="Category", color_discrete_map=CAT_COLORS, hole=0.4)
    f.update_layout(title="Attack category distribution")
    return style(f, 340)


def category_timeline(mal):
    if mal.empty:
        return empty()
    d = mal.copy()
    d["bucket"] = d.timestamp.dt.floor(_bucket(d.timestamp))
    d["attack_category"] = d.attack_category.fillna("Other")
    g = d.groupby(["bucket", "attack_category"]).size().reset_index(name="Attacks")
    f = px.line(g, x="bucket", y="Attacks", color="attack_category", color_discrete_map=CAT_COLORS, markers=True)
    f.update_traces(line_shape="spline")
    f.update_layout(title="Attack frequency over time", xaxis_title=None, legend_title=None)
    return style(f, 340)


def top_patterns_bar(tp):
    if tp.empty:
        return empty()
    f = px.bar(tp.iloc[::-1], x="Count", y="Pattern", orientation="h", text="Count")
    f.update_traces(marker_color=PURPLE)
    f.update_layout(title="Top 10 most frequently detected attack patterns", yaxis_title=None)
    return style(f, 380, False)


def attack_heatmap(mal):
    """Heatmap: attack category x time bucket (attack density)."""
    if mal.empty:
        return empty()
    d = mal.copy()
    d["bucket"] = d.timestamp.dt.floor(_bucket(d.timestamp))
    d["attack_category"] = d.attack_category.fillna("Other")
    p = d.pivot_table(index="attack_category", columns="bucket", values="request_id", aggfunc="count", fill_value=0)
    p = p.reindex([c for c in C.CATEGORIES if c in p.index])
    f = go.Figure(go.Heatmap(z=p.values, x=list(p.columns), y=list(p.index), colorscale=HEAT,
                             colorbar=dict(title="Attacks")))
    f.update_layout(title="Attack heatmap - category vs time")
    return style(f, 320, False)


def confusion(cm, title):
    f = px.imshow(np.array(cm), x=["Benign", "Malicious"], y=["Benign", "Malicious"], text_auto=True,
                  color_continuous_scale=HEAT_BLUE, labels=dict(x="Predicted", y="Actual", color="Count"), aspect="equal")
    f.update_layout(title=title, coloraxis_showscale=False)
    f.update_traces(textfont=dict(size=18))
    return style(f, 330, False)


def accuracy_bar(mm):
    if mm.empty:
        return empty("Run an evaluation first")
    d = mm.copy()
    d["Acc %"] = d.Accuracy * 100
    f = px.bar(d, x="Algorithm", y="Acc %", color="Algorithm", color_discrete_map=MODEL_COLORS,
               text=d["Acc %"].map("{:.2f}%".format))
    lo = max(0, float(d["Acc %"].min()) - 3)
    f.update_yaxes(range=[lo, 100.2], title="Accuracy (%)")
    f.update_layout(title="Accuracy comparison", xaxis_title=None)
    return style(f, 340, False)


def metrics_grouped(mm):
    if mm.empty:
        return empty()
    m = mm.melt(id_vars="Algorithm", value_vars=["Accuracy", "Precision", "Recall", "F1-Score"], var_name="Metric")
    f = px.bar(m, x="Metric", y="value", color="Algorithm", barmode="group", color_discrete_map=MODEL_COLORS)
    lo = max(0, float(m.value.min()) - 0.03)
    f.update_yaxes(range=[lo, 1.002])
    f.update_layout(title="Precision / Recall / F1 by model", xaxis_title=None, legend_title=None)
    return style(f, 340)


def category_recall_heatmap(t):
    if t.empty:
        return empty()
    f = go.Figure(go.Heatmap(z=(t.values * 100), x=list(t.columns), y=list(t.index), colorscale=HEAT_ACC,
                             zmin=max(0, float(np.nanmin(t.values * 100)) - 5), zmax=100,
                             text=np.round(t.values * 100, 2), texttemplate="%{text}%",
                             colorbar=dict(title="% correct")))
    f.update_layout(title="Detection rate heatmap - attack category vs model")
    return style(f, 340, False)


def roc(rocs):
    if not rocs:
        return empty()
    f = go.Figure()
    for k, (fpr, tpr) in rocs.items():
        f.add_scatter(x=fpr, y=tpr, mode="lines", name=C.ALGOS[k], line=dict(color=MODEL_COLORS[C.ALGOS[k]], width=3))
    f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#475569"), showlegend=False)
    f.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
    return style(f, 340)


def latency_bar(mm):
    if mm.empty:
        return empty()
    f = px.bar(mm, x="Algorithm", y="Avg Latency (ms)", color="Algorithm", color_discrete_map=MODEL_COLORS,
               text=mm["Avg Latency (ms)"].map("{:.2f}".format))
    f.update_layout(title="Average prediction time (ms)", xaxis_title=None)
    return style(f, 340, False)


def gauge(prob, title):
    f = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(size=30)),
        title=dict(text=title, font=dict(size=13)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#e2e8f0"),
                   steps=[dict(range=[0, C.FLAG_T * 100], color="#14532d"),
                          dict(range=[C.FLAG_T * 100, C.BLOCK_T * 100], color="#92400e"),
                          dict(range=[C.BLOCK_T * 100, 100], color="#7f1d1d")])))
    return style(f, 230, False).update_layout(margin=dict(l=20, r=20, t=50, b=5))
