# Handoff — `testing-changes` branch

Written 2026-08-18. Paste this into a fresh session to pick up where things left off.

## Where the code is

| Branch | HEAD | State |
|---|---|---|
| `feat/merge-chatbot-ui` | `8d1cd74` | Pushed. Matches origin. The UI/port work. |
| `testing-changes` | `e592311` | **Local only, never pushed.** 9 commits of speed + UX work on top. |

Uncommitted and unrelated (dirty before any of this started, leave alone):
`.gitignore`, `zenodo_biceps/out/*.csv|.md|.json`

## Run it

```bash
python models/serve.py          # UI at http://localhost:8000
```

Wait for `[warm-up] ready in ~25s` in the terminal before asking anything. A
question asked during warm-up **queues behind it** (Ollama serialises) and comes
back slower than if there were no warm-up at all.

Check readiness programmatically:

```bash
curl -s localhost:8000/health
# {"status":"ok","warm":true,"warm_error":null,"token_cap":320,"answers_truncated":0}
```

Tests (all passing): `frontend/test_answers.py` 69, `frontend/test_understanding.py` 124,
`viz/test_render_window.py` 9.

**`render_window.py` and `turn.py` are imported at server start** — chart or backend
changes need a `serve.py` restart. `chatbot_ui.html` reloads on its own (no-store header).

## The 9 commits on `testing-changes`

```
e592311  Keep the figures panel closed until a graph is asked for
fab9960  Raise the token ceiling clear of the longest real answer
d22729a  Report when the token cap cuts an answer instead of hiding it
f22301e  Never show a sentence the token cap cut off half-finished
a4aa568  Say when the server is warm instead of leaving it to guesswork
9221cda  Draw the reading chart only when the reader asks for it
f3a4388  Start the chart and the segment load alongside classify instead of after it
dde5e9c  Go back to 3b, overlap the chart render with the model call, and warm the prompt cache at startup
b883ddb  Cut answer latency by reusing the prompt prefix, keeping the model loaded, and caching charts
```

## Speed work — what was done and why

Baseline was **26-27s** per answer. Now **12-20s**, typically 12-15s.

1. **Prompt reordered so Ollama can cache it** (`prompt.py`) — the biggest win, ~13s.
   The prompt is 3279 chars of which only ~148 are the measured data; the variable
   parts used to sit in the middle so nothing could be reused. All fixed instructions
   now come first, per-question data last. Measured: a shared 572-token prefix took
   13.71s the first time, **0.30s** the second; the same text placed *after* the
   variable part took the full 13.21s again. **Do not move a variable block back
   above a static one** — it silently costs seconds with nothing in the output to show it.
2. **`keep_alive: "30m"`** (`llm.py`) — Ollama unloads after 5 min idle and reloading
   costs 6.7s. Stays warm through 30 min of inactivity.
3. **Chart + segment load run alongside `classify`** (`turn.py`, `_IO_POOL`).
4. **Warm-up at startup** (`turn.warm_up`, called from `serve.py`) — moves the
   first-question cost (~20s) onto boot.
5. **`_cached_chart` lru_cache** — figure rebuild was 2.2s even with the data cached.

### Measured and rejected (don't redo these)
- **Thread count does nothing.** 7.4-8.0 tok/s at 6, 8, 12 or default — memory-bandwidth
  bound on the Ryzen 5 5625U.
- **Ollama serialises requests.** Two concurrent calls = 15.87s vs 16.89s sequential.
  So overlapping the recommendation call with the answer call buys nothing.
- **GPU is unavailable.** `api/ps` shows `vram=0.0GB`; integrated AMD Radeon.
- **llama3.2:1b rejected on accuracy** (see below), not speed.

### What's left
`classify` (~1.6s) + generation (~8-13s). Generation is ~85% and irreducible at
8 tok/s. The only real remaining lever is **streaming** — text appearing in ~3s
instead of a blank wait. Doesn't reduce total time, changes the felt latency more
than anything else would.

## Charts are now drawn on demand

`POST /turn` returns a `chart_ref` (`{source, subject, t_start, side}`), not HTML.
Each answer gets a **Show graph** button; clicking it calls the new `POST /chart`,
which is the only place a reading chart is built.

Why: rendering per-answer cost ~2s wall (10.21s vs 8.27s measured, and the model
call itself slowed 7.25→7.84s from CPU contention) and a **3.3 MB** payload that
most readers never looked at. Response is now ~1 KB.

Forecast charts still come with the answer — only built when a horizon was asked
for, and 380px rather than 920px.

The figures panel is **closed by default** and only opens on that click (like
Claude's artifact panel). The header figures button appears only once something
is in the panel. Window resizing can narrow the panel but never open it.

## Answer quality work

- Verdict no longer quotes raw hertz (they're in the provenance line below it).
- `interpret.trim_sentences` cuts model prose to 2 sentences (3 for analysis answers)
  **in code**, because asking for brevity didn't work — asked for "1-3 sentences" it
  wrote four; asked for "ONE or TWO" it wrote three.
- `interpret.plain_words` swaps jargon the prompt bans but the model writes anyway
  ("indicating" → "which means"). A twelve-item banned list failed outright; a 3B
  model holds two or three prohibitions, not twelve.
- The prompt used to describe the old Streamlit expander and invite the model to
  point at it, costing a wasted sentence on every answer. Fixed.

## Token cap (`llm.NUM_PREDICT = 320`)

A ceiling that should never be reached, **not** a length target. Sized from the
longest path measured uncapped: reading 62 tok, follow-up "why?" 106 tok,
**recommendation 152 tok**. The recommendation is the one to size for — unlike
readings it is *not* trimmed afterwards, so every token shows. At 200 it was at
76% of the ceiling.

Removing the cap means *unlimited*, not a safe default: uncapped, an open-ended
prompt generated 765 tokens in 76s. Past `TIMEOUT_SEC` (120) the request fails and
the prose is lost entirely, falling back to bullet points — worse than a trimmed tail.

Safety net: `trim_sentences` drops anything after the last full stop (an unfinished
sentence), keeping it only if nothing complete survived anywhere. `/health` reports
`answers_truncated`, which should stay 0.

## Known issues / open items

- **`turn.py` lost ~100 comments in the extraction from `app.py`** (112 → 9 comment
  lines, 10% → 1%). Guards all still run but nothing explains why they exist. Worth
  restoring before `app.py` is deleted.
- **No Regenerate button** — `app.py` had `_regenerate()`; no equivalent exists.
- **`frontend/recommend.py`** never ported to this frontend (flagged undecided in
  `docs/decisions.md`, still undecided).
- **Charts are always `plotly_dark`** — a black chart in a white card in light mode.
  Needs the theme threaded into `render_window()` and `charts.py`.
- **Inline verdict echoes survive.** `strip_verdict_echo` works on paragraphs and
  never removes the last one, so an echo written inline ("This means they are not
  showing signs of fatigue") gets through. Deliberate, tested, documented.
- **The model sometimes volunteers advice** ("they should take a break") on a plain
  reading question — beyond what was measured, and duplicates the recommendation path.
- **Google Fonts 404s** (Archivo Narrow) — the UI needs internet to render as designed.
- **llama3.2:1b is installed but rejected**: it opened all four sampled readings with
  "The person is not fatigued", including two the classifier called FATIGUED. Nothing
  wrong reached the screen (verdict is Python-rendered, echo stripped) but it was one
  phrasing from contradicting itself. `llm.py` records this.
- **Disk is nearly full**: 234 of 237 GB, mostly a 175 GB OneDrive folder. The first
  `ollama pull` failed at 0 bytes free.

## Not done

`testing-changes` has **never been pushed**. Decide whether to merge it into
`feat/merge-chatbot-ui` or keep it separate for A/B testing.
