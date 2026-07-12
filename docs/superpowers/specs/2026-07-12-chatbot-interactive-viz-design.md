# Chatbot interactive visualization - design

Date: 2026-07-12
Owner: Rayyan
Branch: `feat/rayyan-chatbot-viz` (off `feat/aryan-classify`)

## Background

Ray's original task ("visualize all 13 subjects in one video") turned out to be
the wrong deliverable: `viz/signal_viewer.py` on `feat/rayyan-visualization` is
an interactive matplotlib GUI, never exported to an actual video file (no
`.save()`/ffmpeg/moviepy call anywhere in project history), unmerged, and
requires the viewer to run Python plus a GUI plus the dataset locally, so it
isn't shareable.

The project's own integration contract (`README.md`, agreed before building)
already specifies the real deliverable:

```python
def render_window(subject: int, t_start: float, side: str = "R") -> str:
    """Return interactive Plotly chart as an HTML string (no full_html wrapper)."""
```

Produced by Rayyan, meant to be shown side-by-side with Aryan's `classify()`
grounded text answer. The original contract assumed a dedicated frontend
(Maalav's) as the consumer; that frontend doesn't exist as a branch yet.
Aryan's Discord update (2026-07-11) shipped a working chatbot on
`feat/aryan-classify`: Open WebUI plus Ollama (`llama3.2:3b`) plus a
tool-calling bridge (`models/serve.py`, `models/openwebui_tool_reference.py`,
`models/classify.py`), which is the only working UI right now. This design
folds `render_window()` into that existing chatbot instead of building a
separate frontend.

## Decision: fold into the chatbot, not a standalone video

Confirmed end-to-end in a live test: the chatbot already answers fatigue
questions correctly via `classify()`, but purely as paraphrased text, no
visual. The gap isn't "no video," it's "no visual in the one working UI."

## Constraint carried over: Legacy function-calling only

Open WebUI's **Native** function-calling mode hallucinates instead of calling
the tool on this stack (Ollama 0.30.10, confirmed both by prior-session memory
and by this session's own testing). The chatbot must stay on **Legacy** mode.
Any mechanism this design relies on has to work under Legacy - this ruled out
one draft (see "Rejected approaches" below) and had to be verified against the
installed Open WebUI source, not just its docs.

## Mechanism: Open WebUI's `(HTMLResponse, result_context)` tuple

Open WebUI (installed: v0.10.2, `.venv/lib/python3.11/site-packages/open_webui`)
supports tools returning an `HTMLResponse` with `headers={"Content-Disposition":
"inline"}`. The response body renders as an interactive iframe embedded
directly in the chat message, independent of the LLM. A tool can instead
return a 2-tuple `(HTMLResponse, result_context)`: the `HTMLResponse` renders
as the embed, and `result_context` (a str/dict/list) is what the LLM actually
sees and grounds its text answer in.

Verified directly against the installed source, not assumed from docs:

- `process_tool_result()` (`open_webui/utils/middleware.py:817-870`) contains
  the tuple-unpack and `Content-Disposition: inline` to embed logic. No
  Legacy/Native branching inside this function.
- It is called from **both**:
  - `chat_completion_tools_handler` (`middleware.py:1212`), which is only
    invoked when `metadata['params']['function_calling'] == 'legacy'`
    (`middleware.py:2771-2776`), the Legacy path.
  - `streaming_chat_response_handler` (`middleware.py:4750`), the native
    streamed tool-call loop, the Native path.

Both paths funnel into the same mode-agnostic function. The rich-UI embed
mechanism works under Legacy. This was the design's one real blocking risk;
it's resolved by reading the source, not by assumption.

## Architecture

```
User query
    -> Open WebUI (llama3.2:3b, Legacy function calling)
    -> tool method in openwebui_tool_reference.py
         -> GET /classify  (existing, Aryan's)   -> structured features
         -> GET /render    (new, this design)    -> Plotly HTML fragment
    -> tool returns (HTMLResponse(render_html, inline), grounding_text)
    -> chat message shows: interactive chart iframe + LLM's grounded text
```

This matches the original README diagram's intent, text answer plus visual
panel together, implemented via Open WebUI's native embed mechanism instead
of a separate frontend.

## Components

### 1. `viz/render_window.py` (new file)

```python
def render_window(subject: int | None, t_start: float, side: str = "R") -> str:
    """Return an interactive Plotly chart as an HTML fragment
    (fig.to_html(full_html=False)). subject=None returns an all-13-subjects
    overview (one figure, one trace per subject, hover + legend-toggle)."""
```

- `subject=N`: matches `viz/signal_viewer.py`'s single-subject mode
  (`build_viewer`, `--subject N`) panel-for-panel: raw EMG window colored by
  fatigue label, MDF-over-time, live FFT of the current window. This is a
  deliberate match, not just reuse of convenient code: confirmed with Ray
  (2026-07-12) that `signal_viewer.py`'s single-subject mode is the actual
  tool his supervisor previewed and liked (not `convergence_analysis/gui.py`,
  which is a different pipeline entirely: OpenBCI self-collected data,
  FS=250Hz, no filtering, drift/convergence-detection, unrelated to the
  Zenodo biceps fatigue classification this chatbot runs on). `render_window()`
  is meant to read as that same tool now embedded in the chatbot, not a
  divergent new feature, so all three panels carry over unchanged. Rebuilt
  with `plotly.graph_objects` instead of matplotlib so it's interactive
  (hover, zoom, pan) instead of requiring a local GUI window.
- `subject=None`: one interactive figure with 13 MDF-over-time traces
  (one per subject, distinct colors, toggle via legend, hover for exact
  value), the interactive replacement for the static
  `zenodo_biceps/out/team_summary.png`, and the actual answer to "all 13
  subjects in one view."
- Imports `loader` from `zenodo_biceps/loader.py` and `core` from
  `convergence_analysis/core.py` (both already exist on `feat/aryan-classify`,
  so there's no branch-reachability problem). Does **not** copy
  `signal_viewer.py`'s `sys.path.insert(dirname(__file__))` pattern, since
  that only works if the script is run with cwd = `zenodo_biceps/`, which
  contradicts where the file actually lives (`viz/`), and is broken as
  committed. `render_window.py` uses explicit repo-root-relative `sys.path`
  inserts for both `zenodo_biceps/` and `convergence_analysis/` so it works
  regardless of cwd.

### 2. `models/serve.py` (extend, Aryan's file)

One new route, same FastAPI app as `/classify`:

```python
@app.get("/render")
def render_endpoint(subject: int | None = None, t_start: float = 0, side: str = "R"):
    return {"html": render_window(subject, t_start, side)}
```

### 3. `models/openwebui_tool_reference.py` (extend, Aryan's file)

Fold the visualization into the existing `get_fatigue` method rather than
adding a second one; every subject-level query gets a chart, per the
requirement that all outputs have an interactable visualization:

```python
def get_fatigue(self, subject: int, t_start: float, side: str = "R") -> str | tuple:
    # existing: call /classify, build grounding text
    # new: call /render, wrap in HTMLResponse(headers={"Content-Disposition": "inline"})
    # return (html_response, grounding_text) on success
    # on any /render failure: return grounding_text alone (str), same as today
```

## Error handling

If `/render` is unreachable, times out, or the subject/time-window has no
data, the tool drops the `HTMLResponse` and returns the grounding text alone,
identical to current behavior. The chart is additive; its failure never
blocks the existing text answer.

## Testing

Live Open WebUI session (already running, tool already registered, Legacy
mode already set): re-ask the same test question ("Is subject 13 fatigued at
60 seconds on the right side?") after the change and confirm the chart
renders inline alongside the text. Before that, a throwaway single-method
test (`return HTMLResponse("<b>test</b>", headers=...)`, ask for it in the
live Legacy-mode chat) is the fast way to confirm the embed mechanism fires
before wiring up the real Plotly logic.

## Ownership / branch handling

`viz/render_window.py` is a new file, no conflict with Aryan's work. The
edits to `models/serve.py` and `models/openwebui_tool_reference.py` are
additive (new route, new return shape on the existing method) but they are
Aryan's files: branch off `feat/aryan-classify` as `feat/rayyan-chatbot-viz`,
and flag the change to Aryan before merging.

## Rejected approaches

- **Markdown image URL** (`![...](http://localhost:8000/plot?...)` returned
  as plain tool text): rejected because a tool's plain-text return goes to
  the LLM, not the renderer. The live test in this session showed
  `llama3.2:3b` paraphrasing tool output into prose, with the raw content
  buried in a collapsed "1 Source" toggle. A markdown image tag would likely
  be reworded or dropped, not rendered.
- **Base64 data-URI embed** in the tool's text return: same underlying
  problem as the markdown URL (still routes through the LLM's context), plus
  it bloats context with a large base64 string on an already-small local
  model.
- **Standalone mp4 export** of `signal_viewer.py`: dropped per the original
  brainstorm. Not shareable, not integrated into the one working UI, and the
  interactive Plotly version already fulfills the "all 13 subjects"
  requirement inside the chatbot.
