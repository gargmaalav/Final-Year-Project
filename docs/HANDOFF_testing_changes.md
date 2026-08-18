# Handoff — `testing-changes` branch

Last updated 2026-08-18 after the follow-up and open-ended answer fixes. Paste the
"start here" block below into a fresh session to pick up where things left off.

## Where the code is

| Branch | HEAD | State |
|---|---|---|
| `feat/merge-chatbot-ui` | `8d1cd74` | Pushed. Matches origin. The UI/port work. |
| `testing-changes` | `2069e1d` | Pushed, in sync with origin. Speed + UX work, then the answer-quality fixes below. |

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

Tests (all passing): `frontend/test_answers.py` 97, `frontend/test_understanding.py` 190,
`viz/test_render_window.py` 9. None need Ollama or the dataset except
`test_understanding.py --live`, which is skipped by default.

**`render_window.py` and `turn.py` are imported at server start** — chart or backend
changes need a `serve.py` restart. `chatbot_ui.html` reloads on its own (no-store header).

## The commits on `testing-changes`

```
2069e1d  Stop readings addressing the reader as the subject and volunteering advice
98549fb  Answer an open-ended question instead of reprinting the last reading
3c24857  Record the follow-up fixes and that the branch is pushed
824b249  Make a follow-up add something instead of restating the answer it explains
397873f  Recognise "so what does this mean" as being about the last answer
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

- **Confidence is called "certainty" in follow-ups.** `prompt.py` strips the
  confidence line out of the *reading* prompt (rule 3 forbids the word outright),
  but `build_followup_prompt` passes the full fact lines through unfiltered, so a
  follow-up wrote "the model's 95% certainty in its fatigued/not-fatigued call".
  It is the model's confidence in its own call, not a measure of correctness --
  `interpret.plain_lines` says so in the line itself. Same filter plus a test is
  the fix; **~10 minutes, and the cheapest real win on this list.**
- **The model invents reasoning around real numbers** in follow-ups: "it will
  take approximately 12% of the total recording time before they reach a point
  where fatigue is likely". No such projection was measured. Narrower than the
  failures this architecture was built for and no wrong figure reaches the
  screen, but it is the last soft spot in the follow-up path.
- **`turn.py` lost ~100 comments in the extraction from `app.py`** (112 → 9 comment
  lines, 10% → 1%). Guards all still run but nothing explains why they exist. Worth
  restoring before `app.py` is deleted.
- **No Regenerate button** — `app.py` had `_regenerate()`; no equivalent exists.
- **Charts are always `plotly_dark`** — a black chart in a white card in light mode.
  Needs the theme threaded into `render_window()` and `charts.py`.
- **Inline verdict echoes survive.** `strip_verdict_echo` works on paragraphs and
  never removes the last one, so an echo written inline ("This means they are not
  showing signs of fatigue") gets through. Deliberate, tested, documented.
- **Google Fonts 404s** (Archivo Narrow) — the UI needs internet to render as designed.
- **llama3.2:1b is installed but rejected**: it opened all four sampled readings with
  "The person is not fatigued", including two the classifier called FATIGUED. Nothing
  wrong reached the screen (verdict is Python-rendered, echo stripped) but it was one
  phrasing from contradicting itself. `llm.py` records this.
- **Disk is nearly full**: 234 of 237 GB, mostly a 175 GB OneDrive folder. The first
  `ollama pull` failed at 0 bytes free.

## Follow-up questions ("why?", "so what does this mean?")

Asked "so what does this mean" under a reading, the reply was a byte-identical
copy of the answer being asked about. Two separate causes:

1. **It never reached the follow-up path.** `_FOLLOWUP_RE` listed "that" but
   not "this", and tolerated no leading "so"/"ok"/"and". It fell through to
   READING, which re-classified the same window and re-rendered the same
   verdict. Widened, and `intent.followup_kind()` now splits follow-ups four
   ways — WHY / MEANING / SIMPLER / MORE — because one generic "explain the
   reasoning" instruction produced one generic answer to all four.
2. **The follow-up prompt invited a restatement.** It handed the model the
   previous answer and never said what to ADD. Each kind now names the material
   it should reach for, and `interpret.drop_repeated_sentences()` removes what
   the prompt does not (content-word overlap, so paraphrase is caught too).

Three more defects the live replay turned up, all fixed:

- **"why?" under a NOT-fatigued verdict recited the mechanism of fatigue as the
  reason for its absence.** The chain is now asked for in the conditional, and
  the "not far enough here" qualifier goes **first** — asked for last, the
  length cap cut it off and left the contradiction standing.
- **"your muscle signal" for subject 11.** A dataset subject is a third party;
  an upload is the reader's own. `prompt._addressing()` says which.
- **"66.8 Hz, which is still above the fresh level of 70.0 Hz."** Both figures
  real, the relation backwards. `interpret.drop_hertz_comparisons()` removes
  the one sentence shape that can be wrong while every number in it is right.

Also: follow-ups are capped at `turn.FOLLOWUP_MAX_SENTENCES` (4) across the
whole answer, not per paragraph — asked for 2-4 it returned nine; and
`plain_words` no longer turns "start indicating fatigue" into "start which
means fatigue" (it maps bare "indicating" to "showing" now, and keeps "which
means" for the comma-anchored connective sense).

Tests: `frontend/test_answers.py` 97, `frontend/test_understanding.py` 190.

## Open-ended questions, and the catch-all behind them

Reported separately, same shape as the follow-up bug: **"what can u tell me
about the data"** reprinted the reading already on screen, word for word. It
named no subject, no time and no side, so it resolved entirely from the
previous turn and re-ran the classifier on the same window.

Two fixes, and the second matters more:

1. `intent.asks_about_the_data()` recognises the open-ended ask. With a
   recording under discussion it becomes an OVERVIEW of that recording; with
   none, the catalogue.
2. **`extract.named_nothing_new()` is the catch-all.** If a message resolved to
   exactly the window already answered *and* named nothing itself, re-measuring
   can only reprint -- so it goes to the follow-up path instead. Judged on the
   resolved window, not on wording, so phrasings nobody has thought of land
   there too. This is the general fix; the router will keep missing phrasings
   (there is no finite list of ways to ask a vague question) and every miss now
   lands somewhere useful.

`turn.py` checks for a recommendation or a forecast request **before** that
guard. Both resolve to the same window and are caught by it, and sent to the
follow-up path they lost the very thing they asked for. Pinned in
`test_understanding.py`.

## Reading-path defects found while testing the above

None of these were the reported bug; all three were live in every reading.

- **It addressed the reader as the subject.** "You may need to adjust your grip
  or technique" -- about subject 11, to someone who is not subject 11. The
  prompt literally said *"say it out loud to the person who did the exercise"*,
  which is true of an upload and false of a dataset subject. Now
  `prompt._person_rule(window)`.
- **It volunteered coaching nobody asked for** ("they should take regular
  breaks to rest and recover") on a plain reading. Nothing in an EMG window
  measures whether that is warranted. Forbidden in the prompt and enforced by
  `interpret.drop_advice()`, which is skipped when a recommendation *was*
  asked for so it cannot gut the real one.
- **"This indicates they are fatigued" survived as a whole paragraph** of pure
  restatement -- the echo pattern matched "is fatigued" but not "are fatigued".

Also: `recommend.wants_recommendation()` matched "recommend a training plan"
but not **"what should I do"**, "any advice" or "should they take a break", so
the plainest phrasings never produced a recommendation at all. Widened, while
"should I be worried?" correctly stays a follow-up -- that asks what the
reading means, not what to do about it.

## Not done / where to pick up

Roughly in order of value:

1. **The confidence-as-certainty filter** (above). Small, self-contained, real.
2. **Streaming** — the last latency lever. Text in ~3s instead of a blank wait;
   does not cut total time, changes felt latency more than anything else would.
3. **Chart theming** — always `plotly_dark`, so a black chart sits in a white
   card in light mode. Needs the theme threaded through `render_window()` and
   `charts.py`.
4. **Restore `turn.py`'s comments** before `app.py` is deleted.
5. **The merge decision** — fold `testing-changes` into `feat/merge-chatbot-ui`,
   or keep it separate for A/B testing. Still undecided.

## How to check it still works

Restart the server first — `turn.py`, `prompt.py`, `interpret.py` and
`extract.py` are all imported at start, so edits to them need a restart. If
port 8000 is still held by an old process, uvicorn fails to bind and exits, and
the browser keeps talking to the stale one:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  Select-Object -ExpandProperty OwningProcess |
  ForEach-Object { Stop-Process -Id $_ -Force }
python models/serve.py
```

Then, in one chat:

| Ask | Expect |
|---|---|
| `Is subject 11's right side fatigued at 60 s?` | Verdict + 1-2 sentences. Never "you"/"your". No advice. |
| `so what does this mean` | Something new. **No `Reading:` line, no Show graph button** |
| `what can u tell me about the data` | Whole-recording summary of subject 11, not the 60s reading again |
| `why?` | Opens "has not moved far enough…", fatigue in the conditional |
| `and at 90 seconds?` | A fresh reading **at 90s** |
| `what should they do about it?` | A Recommendation block ending in the disclaimer |
| `will they get more tired over the next minute?` | A forecast chart |

**The `Reading:` line under a follow-up is the reliable tell that the server is
running stale code** — a follow-up never touches the classifier, so it has no
window to cite and no chart to offer. The wording changes run to run; that line
does not.
