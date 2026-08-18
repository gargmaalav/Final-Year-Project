"""
render_window(subject, t_start, side) -> interactive Plotly HTML string
=========================================================================

The team integration contract (README.md / docs/superpowers/specs/
2026-07-12-chatbot-interactive-viz-design.md):
  Produced by: Rayyan. Consumed by: Aryan's Open WebUI tool
  (models/openwebui_tool_reference.py), embedded via Open WebUI's
  (HTMLResponse, result_context) tuple mechanism.

Content matches viz/signal_viewer.py's single-subject panels (the tool the
supervisor previewed and liked): raw EMG window coloured by fatigue label,
MDF-over-time, FFT of the current window. Interaction model does NOT carry
over -- this renders one static Plotly figure at the query's t_start (native
hover/zoom/pan only), not signal_viewer.py's scrub slider + Play button.
Scrub/playback is an explicit fast-follow, not silently substituted.

Single-subject only. An all-13-subjects overview existed but was dropped
2026-07-13 (13 overlaid MDF lines were an unreadable spaghetti plot); the
single-subject view is the deliverable.

One-time setup:
    pip install plotly
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO_ROOT, "zenodo_biceps"),
           os.path.join(_REPO_ROOT, "convergence_analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import loader  # noqa: E402  load_biceps_segment, load_fatigue_labels, mdf_trend
import core    # noqa: E402  median_frequency

DATA_ROOT = os.path.join(_REPO_ROOT, "zenodo_biceps", "sEMG_data")

# matches models/classify.py's trained config (models/fatigue_model.pt ->
# config['target_fs']) so the chart and the LLM's grounded numbers agree
TARGET_FS = 250
WIN_SEC = 4.0    # authors' MDF window (loader.mdf_trend default)
STEP_SEC = 2.0   # authors' MDF step   (loader.mdf_trend default)

LABEL_COLOR = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
LABEL_NAME = {0: "Fresh", 1: "Transition", 2: "Fatigued"}

ASK_COLOR = "#b388ff"   # persistent "asked: Ns" marker (distinct from the
                        # scrub cursor and the yellow FFT-MDF line)
SEL_COLOR = "#9aa5b1"   # shaded select-to-inspect span on the MDF panel; a
                        # neutral slate (not blue) so the transient span never
                        # blends with the always-on #58a6ff fatigue-trend line

# This chart always rendered plotly_dark, with a "white" scrub cursor line
# that only makes sense against a dark plot background -- so a reader on the
# UI's light theme (viz/chatbot_ui.html's STATE.theme) got a black chart in a
# white card, and would have gotten an invisible white-on-white cursor too had
# the template alone been swapped. Both are threaded together per theme so
# swapping one can't leave the other stranded.
_THEME = {
    "dark":  {"template": "plotly_dark",  "cursor": "#ffffff",
             "key_bg": "#161616", "key_fg": "#ccc", "key_border": "#333"},
    "light": {"template": "plotly_white", "cursor": "#1a1a1a",
             "key_bg": "#f4f4f4", "key_fg": "#333", "key_border": "#ddd"},
}


def _theme(theme: str) -> dict:
    return _THEME.get(theme, _THEME["dark"])

ALL_SUBJECTS = list(range(1, 14))

# Paper-y for the Play/Pause/Jump/Reset-zoom button rows. The 150px top margin
# holds three things, and they have to stack without touching:
#   ~px 23-45   the figure title, pinned to the container top
#   ~px 79-109  this button row (y=1.10 of a 710px plot region)
#   ~px 128-146 row 1's subplot title, sitting just above the plot area
# Lower than this and the buttons cover the subplot title; higher and they
# cover the figure title. Both rows share the constant so they stay on one
# line -- change it here rather than at either call site.
BUTTON_Y = 1.10

# Open WebUI embeds tool HTML in a sandboxed iframe with no `allow-same-origin`
# (open_webui frontend: FullHeightIframe.svelte), so its own same-origin
# content.scrollHeight measurement always throws and silently no-ops - the
# chart renders at the iframe's default (tiny) height. The only resize path
# that works from inside a sandboxed iframe is postMessage; Open WebUI's own
# listener there checks `data.type === 'iframe:height'` (same file). This
# snippet reports the real height once Plotly finishes drawing
# (plotly_afterplot, not a timing guess), and adds a fullscreen button (the
# embed iframe already carries `allowfullscreen`).
#
# NOTE for the animated (scrub/playback) chart: plotly_afterplot fires on
# EVERY frame during playback (~100+ times). postHeight() dedupes on the last
# posted height and debounces, so animation never floods Open WebUI's resize
# listener - the figure height is fixed, so only the first draw and genuine
# resize/fullscreen changes post.
_IFRAME_CHROME = """
<button id="__viz_fs_btn" style="position:fixed;top:8px;right:8px;z-index:9999;
  padding:6px 10px;background:#222;color:#eee;border:1px solid #555;
  border-radius:6px;cursor:pointer;font:12px sans-serif;opacity:0.85;">
  ⛶ Fullscreen
</button>
<script>
(function () {
  var _lastH = -1, _timer = null;
  function _doPost() {
    var h = Math.max(
      document.documentElement.scrollHeight,
      document.body ? document.body.scrollHeight : 0
    );
    if (h === _lastH) return;      // unchanged -> don't spam the parent
    _lastH = h;
    window.parent.postMessage({type: 'iframe:height', height: h}, '*');
  }
  function postHeight() {           // debounced: collapses a burst of
    if (_timer) clearTimeout(_timer);  // afterplot events into one post
    _timer = setTimeout(_doPost, 120);
  }
  document.addEventListener('DOMContentLoaded', function () {
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    if (gd && gd.on) { gd.on('plotly_afterplot', postHeight); }
  });
  window.addEventListener('load', function () { setTimeout(postHeight, 300); });
  window.addEventListener('resize', function () { setTimeout(postHeight, 100); });

  var btn = document.getElementById('__viz_fs_btn');
  btn.addEventListener('click', function () {
    var el = document.documentElement;
    (el.requestFullscreen || el.webkitRequestFullscreen || function () {}).call(el);
  });
})();
</script>
"""


_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
_PLOTLY_BASIC_JS = os.path.join(_VENDOR_DIR, "plotly-basic-3.7.0.min.js")
_basic_bundle_cache = None


def _plotly_basic_js() -> str:
    """Vendored plotly.js BASIC bundle (v3.7.0 - matches plotly.py 6.9.0's own
    bundled plotly.js, so the figure JSON and the runtime never version-skew).
    Inlined in place of the full bundle: every trace here is go.Scatter (which
    basic covers) and frames/sliders/animation are core (in all bundles), so
    this cuts ~3.5MB off every chart with no feature loss. Self-contained (no
    CDN) so the sandboxed iframe needs no network and no CSP allowance. Read
    once, cached."""
    global _basic_bundle_cache
    if _basic_bundle_cache is None:
        with open(_PLOTLY_BASIC_JS, encoding="utf-8") as f:
            _basic_bundle_cache = f.read()
    return _basic_bundle_cache


def _wrap_for_iframe(plotly_html: str) -> str:
    # basic bundle first so `Plotly` is defined before to_html's newPlot script
    # (both to_html calls pass include_plotlyjs=False - the lib is added here).
    return f"<script>{_plotly_basic_js()}</script>" + plotly_html + _IFRAME_CHROME


# Select-to-inspect + linked navigator on the MDF panel (the only whole-
# recording-time axis). Box-select a time span there and the chart (a) reads
# out the dominant fatigue state + MDF min/mean/max for the span, (b) shades
# the span so it stays visible, and (c) jumps the EMG/FFT detail panels + scrub
# cursor to that span's centre - the detail view "reflects" the selection.
# Clicking a single MDF point jumps the detail there too. The detail panels
# show ONE 4 s window by construction, so they navigate TO a point in the
# selection; they do not stretch to span it.
#
# The MDF panel is subplot row 2, x-axis 'x2', so plotly_selected's range is
# keyed 'x2' and clicked MDF points carry data.xaxis === 'x2'. Selections on
# the EMG window (row 1, 'x', a 0-4 s axis) or the FFT (row 3, 'x3', frequency)
# are NOT recording-time and are never mis-mapped - they show a hint instead.
# Navigation drives Plotly.animate to the nearest baked frame (same mechanism
# as the slider); the asked-marker + span rect are layout shapes (indices 0/1)
# that frames never rewrite, so they survive playback. (Verified in the sandbox.)
_SELECT_INSPECT = """
<div id="__viz_readout" style="font:13px/1.5 -apple-system,sans-serif;
  color:#ddd;background:#161616;border-top:1px solid #333;padding:10px 14px;">
  <span style="color:#888">Tip: pick the <b>Box Select</b> tool (top-right) and
  drag across the middle MDF panel to inspect a span - the readout and the
  EMG/FFT panels jump to it. Click a point on that panel to jump there.</span>
</div>
<script>
(function () {
  var D = __VIZ_DATA__;
  var SEL_SHAPE = 1;                 // layout.shapes[1] = the select-span rect
  var box = document.getElementById('__viz_readout');
  var TIP = '<span style="color:#888">Tip: pick the <b>Box Select</b> tool '
          + '(top-right) and drag across the middle MDF panel to inspect a span '
          + '- the readout and the EMG/FFT panels jump to it. Click a point on '
          + 'that panel to jump there.</span>';
  var gd = null;
  function hint(msg) { box.innerHTML = '<span style="color:#e0a030">' + msg + '</span>'; }

  function nearestK(x) {
    var best = 0, bd = Infinity;
    for (var i = 0; i < D.t.length; i++) {
      var d = Math.abs(D.t[i] - x); if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  function jumpTo(x) {                // move detail panels + cursor to frame @x
    if (!gd || !window.Plotly) return;
    Plotly.animate(gd, [String(nearestK(x))], {mode: 'immediate',
      frame: {duration: 0, redraw: true}, transition: {duration: 0}});
  }
  function relayoutShape(fields) {
    if (!gd || !window.Plotly) return;
    var u = {};
    for (var k in fields) u['shapes[' + SEL_SHAPE + '].' + k] = fields[k];
    Plotly.relayout(gd, u);
  }
  function shadeSpan(lo, hi) { relayoutShape({x0: lo, x1: hi, opacity: 0.16}); }
  function clearShade() { relayoutShape({opacity: 0.0}); }

  function inspect(t0, t1) {
    var lo = Math.min(t0, t1), hi = Math.max(t0, t1);
    var idx = [];
    for (var i = 0; i < D.t.length; i++) if (D.t[i] >= lo && D.t[i] <= hi) idx.push(i);
    if (!idx.length) { hint('No MDF windows between ' + lo.toFixed(0) + 's and '
                            + hi.toFixed(0) + 's.'); return; }
    var mn = Infinity, mx = -Infinity, sum = 0, cnt = {};
    for (var j = 0; j < idx.length; j++) {
      var v = D.v[idx[j]];
      if (v < mn) mn = v; if (v > mx) mx = v; sum += v;
      var l = D.lab[idx[j]]; cnt[l] = (cnt[l] || 0) + 1;
    }
    var mean = sum / idx.length;
    var dom = null, bestN = -1;
    for (var k in cnt) if (cnt[k] > bestN) { bestN = cnt[k]; dom = k; }
    var name = (D.names[dom] !== undefined) ? D.names[dom] : ('label ' + dom);
    var col = D.colors[dom] || '#888';
    box.innerHTML =
      '<b>' + lo.toFixed(0) + '-' + hi.toFixed(0) + ' s</b> &nbsp; '
      + '<span style="background:' + col + ';color:#111;padding:1px 7px;'
      + 'border-radius:10px;font-weight:600">' + name + '</span> '
      + '<span style="color:#888">(' + bestN + '/' + idx.length + ' windows)</span>'
      + ' &nbsp;|&nbsp; MDF '
      + '<b>' + mn.toFixed(1) + '</b> / <b>' + mean.toFixed(1) + '</b> / '
      + '<b>' + mx.toFixed(1) + '</b> Hz <span style="color:#888">(min / mean / max)</span>';
  }

  function onSelect(ev) {
    if (!ev || !ev.range) return;
    var r = ev.range;
    if (r.x2) {
      var lo = Math.min(r.x2[0], r.x2[1]), hi = Math.max(r.x2[0], r.x2[1]);
      inspect(lo, hi);           // stats for the span
      shadeSpan(lo, hi);         // keep the span visible on the timeline
      jumpTo((lo + hi) / 2);     // detail panels reflect the selection (its centre)
    } else {
      hint('Time spans are read off the middle MDF panel (whole-recording '
         + 'time). That box was on the EMG-window or FFT panel, which are not '
         + 'recording time.');
    }
  }
  function onClick(ev) {
    if (!ev || !ev.points || !ev.points.length) return;
    var pt = ev.points[0];
    if (pt.data && pt.data.xaxis === 'x2') jumpTo(pt.x);   // click a timeline point
  }
  function wire() {
    gd = document.getElementsByClassName('plotly-graph-div')[0];
    if (!gd || !gd.on) { setTimeout(wire, 120); return; }
    gd.on('plotly_selected', onSelect);
    gd.on('plotly_click', onClick);
    gd.on('plotly_deselect', function () { box.innerHTML = TIP; clearShade(); });
  }
  document.addEventListener('DOMContentLoaded', wire);
  wire();
})();
</script>
"""


def _select_inspect_html(mdf_t, mdf_v, mdf_labels) -> str:
    data = {
        "t": [round(float(v), 2) for v in mdf_t],
        "v": [round(float(v), 2) for v in mdf_v],
        "lab": [int(l) for l in mdf_labels],
        "names": {str(k): v for k, v in LABEL_NAME.items()},
        "colors": {str(k): v for k, v in LABEL_COLOR.items()},
    }
    payload = json.dumps(data).replace("</", "<\\/")
    return _SELECT_INSPECT.replace("__VIZ_DATA__", payload)


def _dominant_label(t_center: float, lab_t, lab_v, half: float = WIN_SEC / 2):
    """Majority fatigue label in [t_center-half, t_center+half]. -1 if no labels.

    Returns the -1 sentinel (not None) when the trial has no fatigue-label CSV, so
    the labels-absent path stays a plain int array: `int(-1)` serialises fine and
    LABEL_COLOR.get(-1)/LABEL_NAME.get(-1) fall through to the grey "no labels"
    styling. Returning None here used to crash `int(None)` in frame_label /
    _select_inspect_html before the advertised grey-fallback trace could render.
    """
    if lab_t is None or lab_t.size == 0:
        return -1
    mask = (lab_t >= t_center - half) & (lab_t <= t_center + half)
    if not mask.any():
        idx = int(np.argmin(np.abs(lab_t - t_center)))
        return int(lab_v[idx])
    vals, counts = np.unique(lab_v[mask], return_counts=True)
    return int(vals[np.argmax(counts)])


def _validate_side(side: str) -> str:
    side = (side or "R").upper()
    if side not in ("R", "L"):
        raise ValueError(f"side must be 'R' or 'L', got {side!r}")
    return side


def _load_subject(subject: int, side: str):
    seg = loader.load_biceps_segment(DATA_ROOT, subject, side,
                                     target_fs=TARGET_FS, bandpass=True)
    fs = int(getattr(seg, "eff_fs", TARGET_FS))
    lab_t, lab_v = loader.load_fatigue_labels(DATA_ROOT, subject, side)
    return seg, fs, lab_t, lab_v


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Plotly's own legend used to sit in the figure's top margin, horizontally,
# right above the row-1 subplot title. That's fine at full width, but this
# chart is usually embedded in a narrower container (the chatbot's collapsed
# "Show the signal" expander) -- there, the legend's ~5 entries wrapped onto a
# second line and landed directly on top of the title text below it. A plain
# HTML strip has no such failure mode: it wraps like any other paragraph, so
# it can never overlap plot content regardless of the embedding width.
def _key_html(theme: str = "dark") -> str:
    th = _theme(theme)

    def _dot(color: str, label: str) -> str:
        return (f'<span style="display:inline-flex;align-items:center;'
                f'margin:2px 14px 2px 0;white-space:nowrap">'
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{color};margin-right:5px;'
                f'flex:none"></span>{label}</span>')

    def _line(color: str, label: str) -> str:
        return (f'<span style="display:inline-flex;align-items:center;'
                f'margin:2px 14px 2px 0;white-space:nowrap">'
                f'<span style="display:inline-block;width:16px;height:0;'
                f'border-top:2px dashed {color};margin-right:5px;'
                f'flex:none"></span>{label}</span>')

    items = (
        _dot(LABEL_COLOR[0], LABEL_NAME[0])
        + _dot(LABEL_COLOR[1], LABEL_NAME[1])
        + _dot(LABEL_COLOR[2], LABEL_NAME[2])
        + _line("#58a6ff", "fatigue trend")
        + _line(ASK_COLOR, "time asked about")
    )
    return (f'<div style="font:12px -apple-system,sans-serif;color:{th["key_fg"]};'
            f'background:{th["key_bg"]};padding:8px 14px;display:flex;'
            f'flex-wrap:wrap;border-bottom:1px solid {th["key_border"]}">{items}</div>')


def _single_subject_html(subject: int, t_start: float, side: str,
                         model_pred: dict | None = None,
                         theme: str = "dark") -> str:
    """Interactive 3-panel single-subject chart with scrub + playback.

    Deliberately reproduces viz/signal_viewer.py's supervisor-liked layout
    (the tool the supervisor previewed) as an in-chat Plotly chart:
      Panel 1  raw EMG of the CURRENT 4 s window (x = 0..4 s), tinted by the
               window's fatigue state - the detailed waveform view.
      Panel 2  median frequency (MDF) over the WHOLE recording, scatter
               coloured by fatigue label, with a moving scrub cursor.
      Panel 3  FFT spectrum of the current window + its median-frequency
               marker.
    A Play button + Time slider scrub the window across the recording (2 s
    step), the frames/slider reproduction of signal_viewer.py's slider+Play.
    Every per-frame value reuses loader.mdf_trend windows so the panels and
    the LLM's classify()-grounded numbers stay in agreement (no JS FFT).
    """
    if subject not in ALL_SUBJECTS:
        raise ValueError(f"subject must be 1-13, got {subject}")

    th = _theme(theme)
    seg, fs, lab_t, lab_v = _load_subject(subject, side)
    x = seg.data[:, 0].astype(float)
    t = seg.t.astype(float)

    win = int(round(WIN_SEC * fs))
    if x.size < win:
        raise ValueError(
            f"subject {subject} recording ({x.size / fs:.1f}s) is shorter "
            f"than one {WIN_SEC:.0f}s window")

    t_start = min(max(float(t_start), 0.0), float(t[-1]))

    # aligned frame source: fs-correct windows (loader.mdf_trend, not a
    # hand-rolled reimpl). t_centers/mdf_v drive every per-frame value so the
    # FFT marker, MDF cursor and window all refer to the same window.
    mdf_t, mdf_v, _ = loader.mdf_trend(seg, fs=fs, win_sec=WIN_SEC, step_sec=STEP_SEC)
    if mdf_t.size:
        mdf_labels = np.array([_dominant_label(tc, lab_t, lab_v) for tc in mdf_t])
    else:
        # degenerate case: every window gapped, so mdf_trend found none.
        # Synthesise one window at t_start so the chart still renders.
        s0 = int(np.clip(np.searchsorted(t, t_start), 0, x.size - win))
        tc0 = float(t[s0] + WIN_SEC / 2.0)
        mdf_t = np.array([tc0])
        mdf_v = np.array([core.median_frequency(x[s0:s0 + win], fs=fs)])
        mdf_labels = np.array([_dominant_label(tc0, lab_t, lab_v)])

    freqs = np.fft.rfftfreq(win, 1.0 / fs)
    fmax = min(500.0, fs / 2.0)
    fband = freqs <= fmax
    freqs_band = freqs[fband]
    tw = (np.arange(win) / fs).tolist()   # per-window time axis, 0..4 s (const)

    def _window_start(tc: float) -> int:
        start = int(np.searchsorted(t, tc - WIN_SEC / 2.0))
        return int(min(max(start, 0), x.size - win))

    # bake every frame's payload once, server-side (same np.fft/hanning as the
    # legacy static path - no JS FFT, so nothing can diverge from classify()).
    # Round baked floats so the serialized frame payload carries short numbers
    # (a big chunk of the HTML size) with no visible change: the FFT trace is
    # already normalised 0..1 so 4 dp is exact enough; raw EMG is ~1e-3, so a
    # flat 4 dp would staircase the waveform - round to keep ~4 significant
    # figures at the signal's own scale instead.
    _amp = float(np.max(np.abs(x))) or 1e-12
    _emg_dp = int(np.clip(4 - np.floor(np.log10(_amp)), 4, 10))
    frame_emg, frame_spec = [], []
    for tc in mdf_t:
        s = _window_start(float(tc))
        w = x[s:s + win]
        spec = np.abs(np.fft.rfft(w * np.hanning(win))) ** 2
        spec = spec / (spec.max() + 1e-12)
        frame_emg.append(np.round(w, _emg_dp).tolist())
        frame_spec.append(np.round(spec[fband], 4))
    frame_mdf = [round(float(v), 2) for v in mdf_v]
    frame_label = [int(l) for l in mdf_labels]

    # opening frame = the one nearest the query's t_start, so the chart agrees
    # with the LLM's t_start-grounded text (not frame 0 at t=0).
    k0 = int(np.argmin(np.abs(mdf_t - t_start)))
    n_frames = len(mdf_t)
    animate = n_frames >= 2

    # fixed EMG y-range across all frames (so amplitude is comparable over
    # time, exactly as signal_viewer.py pins ax_sig ylim to the whole signal).
    xlo, xhi = float(np.min(x)), float(np.max(x))
    ypad = 0.10 * (xhi - xlo + 1e-12)
    ylo, yhi = xlo - ypad, xhi + ypad

    mlo = float(np.min(mdf_v)) if mdf_v.size else 0.0
    mhi = float(np.max(mdf_v)) if mdf_v.size else 1.0
    mpad = 0.10 * (mhi - mlo + 1e-9)
    mlo, mhi = mlo - mpad, mhi + mpad

    def _tint(k):
        return _rgba(LABEL_COLOR.get(frame_label[k], "#888888"), 0.12)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"S{subject} {side} biceps - raw EMG of the current 4 s window (bandpass 20-450 Hz)",
            "Median frequency (MDF) over the whole recording - fatigue marker + scrub cursor",
            "FFT spectrum of the current window",
        ),
        vertical_spacing=0.1,
    )

    # --- static: MDF-over-time, split by fatigue label (panel 2) ---
    if lab_t is None:
        fig.add_trace(go.Scatter(x=mdf_t, y=mdf_v, mode="markers+lines",
                                 name="MDF (no ground-truth labels)",
                                 marker=dict(size=5, color="#888")), row=2, col=1)
    else:
        for lbl in (0, 1, 2):
            mask = mdf_labels == lbl
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=mdf_t[mask], y=mdf_v[mask], mode="markers",
                    name=LABEL_NAME[lbl],
                    marker=dict(size=6, color=LABEL_COLOR[lbl])), row=2, col=1)

    # --- static: fitted MDF decline line + slope (panel 2) ---
    # A least-squares line through the whole MDF trend; the slope in Hz/min is
    # the quantitative fatigue signature (median frequency falls as the muscle
    # fatigues). Static overlay - it does NOT change per frame, so it lives with
    # the base traces (added before `base` below) and never enters fig.frames.
    # hoverinfo=skip so it does not steal hover from the coloured MDF dots; the
    # slope value rides in the legend label. Needs >=2 windows to fit.
    if mdf_v.size >= 2:
        _slope_hz_s, _mdf_icpt = np.polyfit(mdf_t, mdf_v, 1)
        _slope_hz_min = _slope_hz_s * 60.0
        # guard |slope| < 0.05 so {:+.1f} never renders a bare "-0.0 Hz/min"
        _trend_lbl = ("fatigue trend ~0 Hz/min (flat)" if abs(_slope_hz_min) < 0.05
                      else f"fatigue trend {_slope_hz_min:+.1f} Hz/min")
        fig.add_trace(go.Scatter(
            x=mdf_t, y=_slope_hz_s * mdf_t + _mdf_icpt, mode="lines",
            name=_trend_lbl,
            line=dict(color="#58a6ff", width=2.5, dash="dash"),
            hoverinfo="skip"), row=2, col=1)

    # --- animated traces (fixed indices from here): [tint, emg, cursor,
    #     fft line, fft marker], each rewritten by every frame by position ---
    base = len(fig.data)
    fig.add_trace(go.Scatter(                                   # base+0 window tint
        x=[0, WIN_SEC, WIN_SEC, 0], y=[ylo, ylo, yhi, yhi],
        fill="toself", mode="lines", fillcolor=_tint(k0),
        line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(                                   # base+1 EMG window
        x=tw, y=frame_emg[k0], mode="lines", name="EMG (current 4 s window)",
        line=dict(width=0.7, color="#00d4ff")), row=1, col=1)
    fig.add_trace(go.Scatter(                                   # base+2 scrub cursor
        x=[float(mdf_t[k0]), float(mdf_t[k0])], y=[mlo, mhi], mode="lines",
        line=dict(color=th["cursor"], dash="dash", width=1.2),
        hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(                                   # base+3 FFT power
        x=freqs_band, y=frame_spec[k0], mode="lines", name="FFT power",
        line=dict(color="#ff6b6b")), row=3, col=1)
    fig.add_trace(go.Scatter(                                   # base+4 window MDF
        x=[frame_mdf[k0], frame_mdf[k0]], y=[0.0, 1.05], mode="lines",
        line=dict(color="yellow", dash="dash", width=1.5),
        name="window MDF", hoverinfo="skip", showlegend=False), row=3, col=1)
    anim_idx = [base, base + 1, base + 2, base + 3, base + 4]

    def _title(k):
        # Plain title: subject, queried time, and the window's median frequency.
        # No per-window fatigue-state label here -- the fatigue stage is shown by
        # the dot colours (Fresh/Transition/Fatigued), and the model's own verdict
        # is delivered in the chatbot's text answer, not on the chart.
        return (f"EMG Fatigue Progression - Subject {subject} ({side} Biceps) | "
                f"t={mdf_t[k]:.0f}s, window MDF={frame_mdf[k]:.1f} Hz")

    if animate:
        frames = []
        for k in range(n_frames):
            # Each frame carries ONLY the attributes that change per frame.
            # Plotly partial-merges a frame trace onto its base trace (same
            # index), so the CONSTANT x-axes (tw, freqs_band) and the constant
            # y-extents inherit from the base traces instead of being re-baked
            # into all ~N frames. That redundancy (constant x repeated per
            # frame) was the bulk of the payload; dropping it is the biggest
            # single size win with an identical rendered figure.
            frames.append(go.Frame(name=str(k), traces=anim_idx, data=[
                go.Scatter(fillcolor=_tint(k)),                    # tint: fillcolor only
                go.Scatter(y=frame_emg[k]),                        # EMG: y only (x=tw inherits)
                go.Scatter(x=[float(mdf_t[k]), float(mdf_t[k])]),  # cursor: x only
                go.Scatter(y=frame_spec[k]),                       # FFT: y only (x=freqs inherits)
                go.Scatter(x=[frame_mdf[k], frame_mdf[k]]),        # window MDF: x only
            ], layout=go.Layout(title=dict(text=_title(k)))))
        fig.frames = frames

        # "Jump to asked time" reuses the exact same animate call the slider
        # steps use (just aimed at k0), so it always lands on the frame the
        # chatbot's text answer is actually describing -- the fast way back
        # after scrubbing or playing away from it.
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", direction="left", showactive=False,
                x=0.02, y=BUTTON_Y, xanchor="left", yanchor="top", pad=dict(r=10),
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 120, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate", "transition": {"duration": 0}}]),
                    dict(label="⏮ Jump to asked time", method="animate",
                         args=[[str(k0)], {"frame": {"duration": 0, "redraw": True},
                                           "mode": "immediate", "transition": {"duration": 0}}]),
                ])],
            sliders=[dict(
                active=k0, x=0.02, y=0, len=0.96, xanchor="left", yanchor="top",
                pad=dict(t=40, b=10), currentvalue=dict(prefix="Time: ", font=dict(size=12)),
                # No per-step text label: with a couple hundred 2 s steps
                # across a long recording, Plotly stacked every one of those
                # labels along the track and they overlapped into an
                # unreadable smear of digits. currentvalue above already
                # shows the selected time, so the steps only need their tick.
                steps=[dict(method="animate", label="",
                            args=[[str(k)], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate", "transition": {"duration": 0}}])
                       for k in range(n_frames)])])

    # Reset-zoom restores every panel's original axis ranges in one click --
    # the scroll-zoom + box-select drag mode makes it easy to zoom into a
    # panel and then have no obvious way back short of a page reload.
    fig.update_layout(
        updatemenus=list(fig.layout.updatemenus) + [dict(
            type="buttons", direction="left", showactive=False,
            x=0.98, y=BUTTON_Y, xanchor="right", yanchor="top",
            buttons=[dict(label="⤾ Reset zoom", method="relayout", args=[{
                "xaxis.range": [0, WIN_SEC], "yaxis.range": [ylo, yhi],
                "xaxis2.range": [float(t[0]), float(t[-1])], "yaxis2.range": [mlo, mhi],
                "xaxis3.range": [0, fmax], "yaxis3.range": [0, 1.05],
            }])])])

    fig.update_layout(
        template=th["template"], height=920 if animate else 820, showlegend=False,
        # Title pinned to the top of the *figure* (yref="container"), not
        # centred in the top margin as Plotly defaults to. Centred, it landed
        # in the same band as the Play/Pause/Jump/Reset-zoom updatemenus above
        # -- which is why those buttons had to be hidden until hover in the
        # first place (viz's _HIDE_UPDATEMENU_CSS). The margin is now deep
        # enough to stack title above buttons above plot, so revealing the
        # buttons no longer covers the title telling you what you're looking at.
        title=dict(text=_title(k0), yref="container", y=0.975, yanchor="top"),
        margin=dict(t=150 if animate else 90, b=60),
        # box-select defaults to a horizontal (time) band for select-to-inspect
        # on the MDF panel; scroll-zoom stays available so select mode does not
        # cost the user zoom.
        dragmode="select", selectdirection="h",
    )
    # Axis titles spell out the technical shorthand (MDF, a.u., norm.) rather
    # than assuming the reader already knows it -- the panel titles above still
    # carry the precise terms for anyone who wants them.
    fig.update_xaxes(title_text="Time in window (s)", range=[0, WIN_SEC], row=1, col=1)
    fig.update_yaxes(title_text="Signal strength (a.u.)", range=[ylo, yhi], row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", range=[float(t[0]), float(t[-1])], row=2, col=1)
    fig.update_yaxes(title_text="Median frequency (Hz)", range=[mlo, mhi], row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, fmax], row=3, col=1)
    fig.update_yaxes(title_text="Signal strength (normalised)", range=[0, 1.05], row=3, col=1)

    # Persistent "asked: t_start" marker on the MDF panel: a fixed vertical line
    # + label at the exact time the user asked about. Added as layout shapes/
    # annotations, which frames never rewrite (frames only set layout.title), so
    # it stays put while playback/scrub moves the white cursor away - the chart
    # always shows the moment that was actually queried. shapes[0] = this line.
    fig.add_vline(x=t_start, row=2, col=1, line=dict(color=ASK_COLOR, width=2))
    # shapes[1] = the select-to-inspect span rect, reserved here (invisible)
    # and shown/moved by the JS on box-select. Kept below the data points.
    fig.add_shape(type="rect", xref="x2", yref="y2",
                  x0=t_start, x1=t_start, y0=mlo, y1=mhi,
                  fillcolor=SEL_COLOR, opacity=0.0, line=dict(width=0),
                  layer="below")
    # Keep the asked-marker labels inside the plot near the right edge: a
    # left-anchored label at a late t_start overhangs the axis, so flip to
    # right-anchored (text runs back toward the marker) in the last quarter.
    _t_end = float(t[-1]) if t.size else t_start
    _late = t_start > 0.75 * _t_end
    _lab_anchor = "right" if _late else "left"
    _lab_shift = -3 if _late else 3

    fig.add_annotation(x=t_start, y=mhi, text=f"asked: {t_start:.0f}s",
                       showarrow=False, xanchor=_lab_anchor, yanchor="top",
                       xshift=_lab_shift, yshift=-2,
                       font=dict(color=ASK_COLOR, size=11), row=2, col=1)

    # auto_play=False: rest at the opening frame (nearest t_start) until the
    # user hits Play. Plotly's to_html defaults auto_play=True, which fires
    # .animate() on load and scrolls the chart away from t_start immediately.
    chart = fig.to_html(full_html=False, auto_play=False, include_plotlyjs=False,
                        config={"responsive": True, "scrollZoom": True})
    return (_key_html(theme) + _wrap_for_iframe(chart)
            + _select_inspect_html(mdf_t, mdf_v, mdf_labels))


def render_window(subject: int, t_start: float, side: str = "R",
                  model_pred: dict | None = None, theme: str = "dark") -> str:
    """Return an interactive Plotly chart as an HTML fragment (no full_html wrapper).

    3-panel single-subject view (raw EMG / MDF / FFT) at t_start, the same
    content as viz/signal_viewer.py's build_viewer single-subject mode.

    model_pred: accepted for backward compatibility with models/serve.py, which
    still passes the classify() result, but no longer drawn. The on-chart model
    chip was removed 2026-07-13 (it duplicated the chatbot's text answer and the
    model-vs-ground-truth wording clashed at transition windows); the model's
    verdict is delivered in the chatbot text, and the chart shows the signal plus
    the ground-truth fatigue colours.
    """
    side = _validate_side(side)
    if subject is None:
        raise ValueError("subject is required (the all-subjects overview was removed)")
    return _single_subject_html(int(subject), t_start, side,
                                model_pred=model_pred, theme=theme)


if __name__ == "__main__":
    # smoke test -- needs the Zenodo dataset present at DATA_ROOT.
    html = render_window(13, 120.0, "R")
    print(f"single-subject OK, {len(html)} chars")
