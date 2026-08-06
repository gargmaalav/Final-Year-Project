"""
render_window(subject, t_start, side) -> interactive Plotly HTML string
=========================================================================

The team integration contract (README.md / docs/superpowers/specs/
2026-07-12-chatbot-interactive-viz-design.md):
  Produced by: Rayyan. Consumed by: Aryan's Open WebUI tool
  (models/openwebui_tool_reference.py), embedded via Open WebUI's
  (HTMLResponse, result_context) tuple mechanism.

Two panels, simplified for a non-technical chatbot audience (2026-08-06
redesign): (1) fatigue over the whole session - the hero chart, ground-truth
dots + the deployed model's own guess, (2) a small "signal right now"
snapshot. Originally matched viz/signal_viewer.py's 3-panel layout (raw EMG /
MDF-over-time / FFT); the FFT panel was dropped as too technical to read at a
glance, and the remaining two got a plain-language static header/legend
instead of Plotly's own in-canvas title/legend (which had no room in a
narrow chat embed and reliably overlapped itself).

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
from reliability_tiers import reliability_tier  # noqa: E402

DATA_ROOT = os.path.join(_REPO_ROOT, "zenodo_biceps", "sEMG_data")

# matches models/classify.py's trained config (models/fatigue_model.pt ->
# config['target_fs']) so the chart and the LLM's grounded numbers agree
TARGET_FS = 250
WIN_SEC = 4.0    # authors' MDF window (loader.mdf_trend default)
STEP_SEC = 2.0   # authors' MDF step   (loader.mdf_trend default)

LABEL_COLOR = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
# Plain-language names for a non-technical reader. The dataset's canonical
# 3-class scheme is fresh / transition / fatigued; "Getting tired" is the
# reader-facing wording for the transition class (same class, plainer word).
LABEL_NAME = {0: "Fresh", 1: "Getting tired", 2: "Fatigued"}

ASK_COLOR = "#b388ff"   # persistent "asked: Ns" marker (distinct from the
                        # white scrub cursor and the yellow FFT-MDF line)
SEL_COLOR = "#9aa5b1"   # shaded select-to-inspect span on the MDF panel; a
                        # neutral slate (not blue) so the transient span never
                        # blends with the always-on #58a6ff fatigue-trend line

ALL_SUBJECTS = list(range(1, 14))

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


# Select-to-inspect + linked navigator on the fatigue panel (the only whole-
# recording-time axis). Box-select a time span there and the chart (a) reads
# out the dominant fatigue state + frequency min/mean/max for the span, (b)
# shades the span so it stays visible, and (c) jumps the small signal-snapshot
# panel + scrub cursor to that span's centre. Clicking a single point jumps
# there too. The snapshot panel shows ONE 4 s window by construction, so it
# navigates TO a point in the selection; it does not stretch to span it.
#
# The fatigue panel is subplot row 1, x-axis 'x', so plotly_selected's range
# is keyed 'x' and clicked points carry data.xaxis === 'x'. Selections on the
# signal-snapshot panel (row 2, 'x2', a 0-4 s axis) are NOT recording-time and
# are never mis-mapped - they show a hint instead. Navigation drives
# Plotly.animate to the nearest baked frame (same mechanism as the slider);
# the asked-marker + span rect are layout shapes (indices 0/1) that frames
# never rewrite, so they survive playback. (Verified in the sandbox.)
_SELECT_INSPECT = """
<div id="__viz_readout" style="font:13px/1.5 -apple-system,sans-serif;
  color:#ddd;background:#161616;border-top:1px solid #333;padding:10px 14px;">
  <span style="color:#888">Tip: to look closer at part of the session, pick the
  <b>Box Select</b> tool (top-right) and drag across the big chart above. You
  get the fatigue state for that stretch, and the small chart below jumps to
  it. Click any dot to jump there too.</span>
</div>
<script>
(function () {
  var D = __VIZ_DATA__;
  var SEL_SHAPE = 1;                 // layout.shapes[1] = the select-span rect
  var box = document.getElementById('__viz_readout');
  var TIP = '<span style="color:#888">Tip: to look closer at part of the session, '
          + 'pick the <b>Box Select</b> tool (top-right) and drag across the big '
          + 'chart above. You get the fatigue state for that stretch, and the '
          + 'small chart below jumps to it. Click any dot to jump there too.</span>';
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
    if (!idx.length) { hint('No data between ' + lo.toFixed(0) + 's and '
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
    var disagreeN = 0;
    for (var m = 0; m < idx.length; m++) {
      var mv = D.model[idx[m]];
      if (mv !== -1 && mv !== D.lab[idx[m]]) disagreeN++;
    }
    var disagreeMsg = '';
    if (disagreeN > 0) {
      disagreeMsg = " &nbsp;|&nbsp; <span style='color:#e0a030'>the tool's own "
        + "guess disagreed with what the person reported for " + disagreeN
        + " of " + idx.length + " moments here</span>";
    }
    box.innerHTML =
      '<b>' + lo.toFixed(0) + '-' + hi.toFixed(0) + ' s</b> &nbsp; '
      + '<span style="background:' + col + ';color:#111;padding:1px 7px;'
      + 'border-radius:10px;font-weight:600">' + name + '</span> '
      + '<span style="color:#888">(' + bestN + '/' + idx.length + ' moments)</span>'
      + ' &nbsp;|&nbsp; frequency '
      + '<b>' + mn.toFixed(1) + '</b> / <b>' + mean.toFixed(1) + '</b> / '
      + '<b>' + mx.toFixed(1) + '</b> Hz <span style="color:#888">(lowest / average / highest)</span>'
      + disagreeMsg;
  }

  function onSelect(ev) {
    if (!ev || !ev.range) return;
    var r = ev.range;
    if (r.x) {
      var lo = Math.min(r.x[0], r.x[1]), hi = Math.max(r.x[0], r.x[1]);
      inspect(lo, hi);           // stats for the span
      shadeSpan(lo, hi);         // keep the span visible on the timeline
      jumpTo((lo + hi) / 2);     // detail panel reflects the selection (its centre)
    } else {
      hint('Time spans are read off the big chart above (the whole session). '
         + 'That box was on the small close-up chart below, which is not the '
         + 'session timeline - try dragging across the big chart instead.');
    }
  }
  function onClick(ev) {
    if (!ev || !ev.points || !ev.points.length) return;
    var pt = ev.points[0];
    if (pt.data && pt.data.xaxis === 'x') jumpTo(pt.x);   // click a timeline point
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


def _select_inspect_html(mdf_t, mdf_v, mdf_labels, model_lab=None) -> str:
    data = {
        "t": [round(float(v), 2) for v in mdf_t],
        "v": [round(float(v), 2) for v in mdf_v],
        "lab": [int(l) for l in mdf_labels],
        # -1 sentinel = no model prediction for that window (same convention
        # as the ground-truth labels-absent fallback, see _dominant_label).
        "model": [int(l) if l is not None else -1
                  for l in (model_lab if model_lab is not None else [None] * len(mdf_t))],
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


def _meta_header_html(title_prefix: str, tier_line: str | None, trend_sentence: str) -> str:
    """Plain static HTML above the chart: subject + reliability tier + trend.

    Kept OUT of the Plotly canvas on purpose -- a Plotly title/legend has a
    fixed narrow width to fight in the chat embed and reliably overlapped
    itself there. Plain HTML wraps normally at any container width instead.
    """
    tier_html = (f'<span style="color:#8ab4f8;">{tier_line}</span>'
                f'<span style="color:#555;"> &nbsp;&middot;&nbsp; </span>' if tier_line else "")
    return (
        '<div style="font:14px/1.4 -apple-system,sans-serif;color:#eee;'
        'background:#161616;padding:12px 14px 8px;">'
        f'<div style="font-weight:600;font-size:15px;">{title_prefix}</div>'
        f'<div style="margin-top:3px;font-size:12.5px;color:#aaa;">'
        f'{tier_html}{trend_sentence}</div>'
        '</div>'
    )


def _legend_row_html(show_model: bool) -> str:
    """Static colour-key row, replacing Plotly's own in-canvas legend (which
    has no room to lay out horizontally in a narrow embed without colliding
    with the title)."""
    items = [(LABEL_COLOR[0], "Fresh"), (LABEL_COLOR[1], "Getting tired"),
            (LABEL_COLOR[2], "Fatigued")]
    swatches = "".join(
        '<span style="display:inline-flex;align-items:center;gap:5px;'
        'margin-right:16px;white-space:nowrap;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{c};'
        'display:inline-block;"></span>' + name + '</span>'
        for c, name in items
    )
    if show_model:
        swatches += (
            '<span style="display:inline-flex;align-items:center;gap:5px;'
            'margin-right:16px;white-space:nowrap;">'
            '<span style="font-weight:800;color:#888;">&#10005;</span>'
            "Tool's own guess <span style=\"color:#666;\">"
            "(green = agrees, red = disagrees)</span></span>"
        )
    return (
        '<div style="font:12px -apple-system,sans-serif;color:#ccc;'
        'background:#161616;padding:0 14px 10px;display:flex;flex-wrap:wrap;">'
        + swatches + '</div>'
    )


def _chart_html(seg, fs: int, t_start: float,
                length_tag: str, title_prefix: str,
                lab_t=None, lab_v=None, model_preds=None, subject=None) -> str:
    """Interactive 3-panel chart with scrub + playback, over any core.Segment.

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

    Shared by render_window() (a dataset subject, ground-truth labels) and
    render_segment() (an uploaded recording, no labels -- lab_t/lab_v=None
    already renders the grey "no fatigue labels" fallback below).
    length_tag/title_prefix carry the caller-specific wording
    ("Subject 13 (R biceps)" vs "your uploaded recording") into error
    messages and panel titles without duplicating this ~250-line function.

    model_preds: optional {window_centre_time: predicted_label} from the
    DEPLOYED model (models/classify.py), one entry per MDF window. When
    given, renders as a second series ("Tool's own guess") distinct from
    the ground-truth dots -- the chart previously only ever showed ground
    truth, which a non-technical reader could mistake for the model's own
    call. subject: used only to look up reliability_tier() for the title;
    None (the uploaded-recording path) omits the tier.
    """
    x = seg.data[:, 0].astype(float)
    t = seg.t.astype(float)

    win = int(round(WIN_SEC * fs))
    if x.size < win:
        raise ValueError(
            f"{length_tag} ({x.size / fs:.1f}s) is shorter "
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

    tw = (np.arange(win) / fs).tolist()   # per-window time axis, 0..4 s (const)

    def _window_start(tc: float) -> int:
        start = int(np.searchsorted(t, tc - WIN_SEC / 2.0))
        return int(min(max(start, 0), x.size - win))

    # bake every frame's EMG snippet once, server-side. Round to keep ~4
    # significant figures at the signal's own scale (raw EMG is ~1e-3, so a
    # flat 4 dp would staircase the waveform) - a big chunk of payload size.
    _amp = float(np.max(np.abs(x))) or 1e-12
    _emg_dp = int(np.clip(4 - np.floor(np.log10(_amp)), 4, 10))
    frame_emg = []
    for tc in mdf_t:
        s = _window_start(float(tc))
        frame_emg.append(np.round(x[s:s + win], _emg_dp).tolist())
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

    # --- model's own prediction, as a distinct series from ground truth ---
    # Small X markers (not filled dots) so it reads visually as "a guess",
    # never mistaken for the same kind of mark as the ground-truth dots.
    # Also builds model_lab, aligned 1:1 with mdf_t, for the select-inspect
    # disagreement callout (-1 sentinel = no matching prediction).
    model_lab = [-1] * len(mdf_t)
    mp_t, mp_v, mp_lab = [], [], []
    if model_preds:
        for i, tc in enumerate(mdf_t):
            key = min(model_preds.keys(), key=lambda k: abs(k - float(tc)))
            if abs(key - float(tc)) <= STEP_SEC:  # only plot a genuine match
                mp_t.append(float(tc)); mp_v.append(float(mdf_v[i]))
                lbl = int(model_preds[key])
                mp_lab.append(lbl)
                model_lab[i] = lbl

    # --- fitted MDF trend line + plain-language direction ---
    # A least-squares line through the whole MDF trend; needs >=2 windows.
    trend_sentence = "Not enough of the session yet for a trend."
    _trend_x = _trend_y = None
    if mdf_v.size >= 2:
        _slope_hz_s, _mdf_icpt = np.polyfit(mdf_t, mdf_v, 1)
        _slope_hz_min = _slope_hz_s * 60.0
        # guard |slope| < 0.05 so wording never flips on noise near zero
        _direction = ("staying about the same" if abs(_slope_hz_min) < 0.05
                      else "slowing down over time" if _slope_hz_min < 0
                      else "speeding up over time")
        trend_sentence = f"Overall, the signal is {_direction}."
        _trend_x, _trend_y = mdf_t, _slope_hz_s * mdf_t + _mdf_icpt

    tier_line = reliability_tier(subject) if subject is not None else None

    # --- two panels: (1) fatigue over the whole session - the hero chart a
    # non-technical reader actually needs, (2) a small signal snapshot for
    # context. No third (FFT) panel, no in-canvas title/legend - both moved to
    # plain static HTML above the chart so nothing overlaps in a narrow embed.
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.62, 0.38],
        subplot_titles=(
            "Is the muscle tiring? (lower = more tired)",
            "What the signal looks like right now",
        ),
        vertical_spacing=0.26,
    )
    # Shrink the subplot titles (default 16px) so they fit the gap alongside
    # panel 1's x-axis without colliding with it or panel 2's plot area. Only
    # the two subplot-title annotations exist at this point (the "asked"
    # annotation is added later), so this can't touch anything else.
    fig.update_annotations(font_size=13)

    if lab_t is None:
        fig.add_trace(go.Scatter(x=mdf_t, y=mdf_v, mode="markers+lines",
                                 name="Signal frequency (no fatigue labels for this trial)",
                                 marker=dict(size=5, color="#888"),
                                 showlegend=False), row=1, col=1)
    else:
        for lbl in (0, 1, 2):
            mask = mdf_labels == lbl
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=mdf_t[mask], y=mdf_v[mask], mode="markers",
                    name=LABEL_NAME[lbl], showlegend=False,
                    marker=dict(size=7, color=LABEL_COLOR[lbl])), row=1, col=1)

    if mp_t:
        mp_colors = ["#e74c3c" if l else "#2ecc71" for l in mp_lab]
        fig.add_trace(go.Scatter(
            x=mp_t, y=mp_v, mode="markers", name="Tool's own guess",
            showlegend=False,
            marker=dict(size=9, symbol="x", color=mp_colors,
                        line=dict(width=1.5, color=mp_colors))),
            row=1, col=1)

    if _trend_x is not None:
        fig.add_trace(go.Scatter(
            x=_trend_x, y=_trend_y, mode="lines", name=trend_sentence,
            showlegend=False,
            line=dict(color="#58a6ff", width=2.5, dash="dash"),
            hoverinfo="skip"), row=1, col=1)

    # --- animated traces (fixed indices from here): [tint, emg, cursor],
    #     each rewritten by every frame by position ---
    base = len(fig.data)
    fig.add_trace(go.Scatter(                                   # base+0 window tint
        x=[0, WIN_SEC, WIN_SEC, 0], y=[ylo, ylo, yhi, yhi],
        fill="toself", mode="lines", fillcolor=_tint(k0),
        line=dict(width=0), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(                                   # base+1 EMG window
        x=tw, y=frame_emg[k0], mode="lines", name="Muscle signal (this moment)",
        showlegend=False,
        line=dict(width=0.7, color="#00d4ff")), row=2, col=1)
    fig.add_trace(go.Scatter(                                   # base+2 scrub cursor
        x=[float(mdf_t[k0]), float(mdf_t[k0])], y=[mlo, mhi], mode="lines",
        line=dict(color="white", dash="dash", width=1.2),
        hoverinfo="skip", showlegend=False), row=1, col=1)
    anim_idx = [base, base + 1, base + 2]

    if animate:
        frames = []
        for k in range(n_frames):
            # Each frame carries ONLY the attributes that change per frame.
            # Plotly partial-merges a frame trace onto its base trace (same
            # index), so the CONSTANT x-axis (tw) inherits from the base trace
            # instead of being re-baked into all ~N frames.
            frames.append(go.Frame(name=str(k), traces=anim_idx, data=[
                go.Scatter(fillcolor=_tint(k)),                    # tint: fillcolor only
                go.Scatter(y=frame_emg[k]),                        # EMG: y only (x=tw inherits)
                go.Scatter(x=[float(mdf_t[k]), float(mdf_t[k])]),  # cursor: x only
            ]))
        fig.frames = frames

        fig.update_layout(
            updatemenus=[dict(
                type="buttons", direction="left", showactive=False,
                x=0.0, y=1.16, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 120, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate", "transition": {"duration": 0}}]),
                ])],
            sliders=[dict(
                active=k0, x=0.0, y=0, len=1.0, xanchor="left", yanchor="top",
                pad=dict(t=36, b=10), currentvalue=dict(prefix="Time: ", font=dict(size=12)),
                steps=[dict(method="animate", label=f"{mdf_t[k]:.0f}s",
                            args=[[str(k)], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate", "transition": {"duration": 0}}])
                       for k in range(n_frames)])])

    fig.update_layout(
        template="plotly_dark", height=640 if animate else 580, showlegend=False,
        margin=dict(t=40 if not animate else 56, b=56, l=52, r=16),
        # box-select defaults to a horizontal (time) band for select-to-inspect
        # on the fatigue panel; scroll-zoom stays available so select mode does
        # not cost the user zoom.
        dragmode="select", selectdirection="h",
    )
    # No x-axis title on panel 1 (redundant with the header above and frees
    # room in the gap between panel 1's ticks and panel 2's subplot title).
    fig.update_xaxes(range=[float(t[0]), float(t[-1])], row=1, col=1)
    fig.update_yaxes(title_text="Signal frequency (Hz)", range=[mlo, mhi], row=1, col=1)
    fig.update_xaxes(title_text="Time within this snapshot (s)", range=[0, WIN_SEC], row=2, col=1)
    fig.update_yaxes(title_text="Signal strength", range=[ylo, yhi], row=2, col=1)

    # Persistent "asked: t_start" marker on the fatigue panel: a fixed vertical
    # line + label at the exact time the user asked about. Added as layout
    # shapes/annotations, which frames never rewrite, so it stays put while
    # playback/scrub moves the white cursor away. shapes[0] = this line.
    fig.add_vline(x=t_start, row=1, col=1, line=dict(color=ASK_COLOR, width=2))
    # shapes[1] = the select-to-inspect span rect, reserved here (invisible)
    # and shown/moved by the JS on box-select. Kept below the data points.
    fig.add_shape(type="rect", xref="x", yref="y",
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
                       font=dict(color=ASK_COLOR, size=11), row=1, col=1)

    # auto_play=False: rest at the opening frame (nearest t_start) until the
    # user hits Play. Plotly's to_html defaults auto_play=True, which fires
    # .animate() on load and scrolls the chart away from t_start immediately.
    chart = fig.to_html(full_html=False, auto_play=False, include_plotlyjs=False,
                        config={"responsive": True, "scrollZoom": True})
    header = _meta_header_html(title_prefix, tier_line, trend_sentence)
    legend = _legend_row_html(show_model=bool(mp_t))
    return (header + legend + _wrap_for_iframe(chart)
            + _select_inspect_html(mdf_t, mdf_v, mdf_labels, model_lab))


def render_window(subject: int, t_start: float, side: str = "R",
                  model_pred: dict | None = None,
                  model_preds: dict[float, int] | None = None) -> str:
    """Return an interactive Plotly chart as an HTML fragment (no full_html wrapper).

    3-panel single-subject view (raw EMG / MDF / FFT) at t_start, the same
    content as viz/signal_viewer.py's build_viewer single-subject mode.

    model_pred: unused, kept only so existing callers passing it don't break.
    model_preds: optional {window_centre_time: predicted_label} from the
    DEPLOYED model (models/classify.py), one entry per MDF window. When given,
    renders as a second series ("Tool's own guess") distinct from the
    ground-truth dots, so the chart shows where the model agrees/disagrees
    with the subject's self-report instead of only ever showing ground truth
    (an earlier on-chart model chip was removed 2026-07-13 for duplicating the
    chatbot's text answer -- this is a different, per-window design that
    surfaces disagreement explicitly rather than repeating a single verdict).
    """
    side = _validate_side(side)
    if subject is None:
        raise ValueError("subject is required (the all-subjects overview was removed)")
    if subject not in ALL_SUBJECTS:
        raise ValueError(f"subject must be 1-13, got {subject}")

    seg, fs, lab_t, lab_v = _load_subject(subject, side)
    t_start = min(max(float(t_start), 0.0), float(seg.t[-1]))
    return _chart_html(
        seg, fs, t_start,
        length_tag=f"subject {subject} recording",
        title_prefix=f"Subject {subject} ({side} biceps)",
        lab_t=lab_t, lab_v=lab_v, model_preds=model_preds, subject=subject)


def render_segment(seg, fs: int, t_start: float,
                   model_pred: dict | None = None,
                   model_preds: dict[float, int] | None = None) -> str:
    """Same 3-panel chart as render_window(), for an UPLOADED recording.

    No ground-truth fatigue labels exist for an upload, so panel 2's dots
    render in the grey "no fatigue labels for this trial" fallback that
    already existed for dataset trials missing a labels CSV -- this is not a
    new code path, just the existing lab_t=None branch reached from a new
    caller. `seg`/`fs` come from loader.to_segment() on the uploaded CSV
    (models/serve.py's /classify_upload, /render_upload). No subject id
    exists for an upload, so the title carries no reliability tier.

    model_pred: unused, kept for callers passing it.
    model_preds: see render_window().
    """
    t_start = min(max(float(t_start), 0.0), float(seg.t[-1]))
    return _chart_html(
        seg, fs, t_start,
        length_tag="the uploaded recording",
        title_prefix="uploaded recording", model_preds=model_preds)


if __name__ == "__main__":
    # smoke test -- needs the Zenodo dataset present at DATA_ROOT.
    html = render_window(13, 120.0, "R")
    print(f"single-subject OK, {len(html)} chars")
