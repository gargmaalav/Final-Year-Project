"""
Lightweight Plotly panels for the upload path and the forecast projection.

Deliberately simpler than viz/render_window.py's animated 3-panel chart
(that one is subject-specific and left untouched -- it's Rayyan's file, and
still used for the dataset-subject path). These are quick, static figures
for recordings render_window.py doesn't know how to load: an uploaded file's
raw signal + MDF trend, and the forecast trend + confidence/prediction bands.
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    # Reuse Rayyan's vendored plotly.js bundle rather than pulling from a CDN.
    # render_window.py deliberately vendors it so a chart needs no network;
    # loading these two figures from a CDN would have made them the only part
    # of the app that breaks when the demo machine is offline.
    from render_window import _plotly_basic_js
except Exception:                                   # viz/ not on sys.path
    _plotly_basic_js = None


def _to_html(fig) -> str:
    """Figure -> self-contained HTML, vendored plotly.js when available."""
    if _plotly_basic_js is None:
        return fig.to_html(full_html=False, include_plotlyjs="cdn",
                           config={"responsive": True})
    chart = fig.to_html(full_html=False, include_plotlyjs=False,
                        config={"responsive": True})
    return f"<script>{_plotly_basic_js()}</script>" + chart


def raw_and_mdf_figure(seg, mdf_t, mdf_v, title: str = "Uploaded recording") -> str:
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Raw EMG (bandpassed)", "Median frequency over time"))
    fig.add_trace(go.Scatter(x=seg.t, y=seg.data[:, 0], mode="lines",
                             line=dict(width=0.6, color="#00d4ff"),
                             name="EMG"), row=1, col=1)
    fig.add_trace(go.Scatter(x=mdf_t, y=mdf_v, mode="markers+lines",
                             line=dict(color="#58a6ff"),
                             marker=dict(size=5), name="MDF"), row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="EMG (a.u.)", row=1, col=1)
    fig.update_yaxes(title_text="MDF (Hz)", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=560, title=title,
                      showlegend=False, margin=dict(t=60, b=40))
    return _to_html(fig)


def forecast_figure(forecast: dict, title: str = "Fatigue trend forecast") -> str:
    t_future = list(forecast["t_future"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_future + t_future[::-1],
        y=list(forecast["pi_hi"]) + list(forecast["pi_lo"])[::-1],
        fill="toself", fillcolor="rgba(167,139,250,0.15)", line=dict(width=0),
        name="95% prediction band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=t_future + t_future[::-1],
        y=list(forecast["ci_hi"]) + list(forecast["ci_lo"])[::-1],
        fill="toself", fillcolor="rgba(167,139,250,0.35)", line=dict(width=0),
        name="95% confidence band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=list(forecast["t_fit"]), y=list(forecast["y_fit"]),
                             mode="lines", name="observed trend",
                             line=dict(color="#58a6ff", width=2)))
    fig.add_trace(go.Scatter(x=t_future, y=list(forecast["y_future"]),
                             mode="lines", name="forecast",
                             line=dict(color="#a78bfa", width=2, dash="dash")))
    fig.update_layout(template="plotly_dark", height=380, title=title,
                      xaxis_title="Time (s)", yaxis_title="MDF (Hz)",
                      margin=dict(t=50, b=30),
                      legend=dict(orientation="h", y=1.15))
    return _to_html(fig)
