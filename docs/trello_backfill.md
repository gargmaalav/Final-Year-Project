# Trello Board — Audit and Backfill Plan (v2)

Based on the board export `AgLGIp95 - signal-driven-ai-systems (4).json` (board
name on export: **Signal-Driven AI Systems**). This supersedes the v1 audit
below — the board has moved on from the old hypothetical
Backlog/To Do/Doing/Done structure that `docs/trello_board_plan.md` was
written against. The real board now uses **Sprint 1/2/3 To Do / Doing /
Done**, logbook lists per person, a team "Semester 2 Logs" list, Meeting
Minutes, and an "Other" list. Use this file as the current source of truth;
treat `trello_board_plan.md` only as a source of already-written card text to
reuse (names below point at the matching section there).

Good news first: a lot of v1's findings are already fixed. The `Note to self
aryan` list is archived (closed). Minutes for 18/05, 25/05, 02/06 now exist as
cards. The 15/06 and 23/07 gap is gone.

---

## 1. Current structure (as exported)

**Lists**, open unless marked:

| List | Cards | State |
|---|---|---|
| Sprint 1 To Do | 0 | **empty** |
| Sprint 1 Doing | 0 | **empty** |
| Sprint 1 Done (2 Mar – 26 Apr) | 6 | all bare titles |
| Sprint 2 To Do | 0 | **empty** |
| Sprint 2 Doing | 0 | **empty** |
| Sprint 2 Done | 11 | all bare titles |
| Sprint 2 Backlog | 0 | **empty, closed** |
| Sprint 3 To Do | 0 | **empty** |
| Sprint 3 Doing | 3 | bare titles |
| Sprint 3 Done | 6 | bare titles |
| Aryan Logbook Semester 1 | 14 | 2 empty, 1 missing (Week 1) |
| Maalav Logbook Semester 1 | 13 | 1 empty, 1 missing (Week 1) |
| Rayyan Logbook Semester 1 | 16 | 1 duplicate card, 2 empty |
| Semester 2 Logs (Done as a team) | 5 | 3 PDF-link-only, 2 empty |
| Meeting Minutes | 13 | 7 are PDF-link-only, no summary |
| Other | 6 | 3 empty (RQ x2, Project Timeline) |
| Other *(closed duplicate)* | 0 | harmless, leave archived |
| Note to self aryan | 1 | **archived — good, leave it** |

**Labels** — only 3 of 6 colours are named: green `Admin + Other`, orange
`ML/DL + pipeline`, yellow `Frontend`. Blue, purple, red are unnamed and
presumably unused.

**Members** — two duplicate accounts still exist: `Maalav Garg` (qcn72331) /
`Malav Garg` (malavgarg), and `Wei Qi Yan` appears twice (`weiqiyan` /
`weiqiyan1`). **Zero cards have a member assigned anywhere on the board.**

**Checklists / due dates** — none exist on any of the 94 open cards.

---

## 2. Fixes, in priority order

### 2a. All 23 Sprint Done cards are bare titles (highest priority)

Every card in Sprint 1 Done, Sprint 2 Done, Sprint 3 Done has no description,
no member, no due date — just a title and one label. A marker clicking any of
these sees nothing. `docs/trello_board_plan.md` §2 already has written,
evidence-linked descriptions for the equivalent work — map them across by
title. Direct mapping (old plan title → current card title, same content
applies):

- "Understand EMG acquisition and the OpenBCI GUI" → **Understanding EMG and
  OpenBCI GUI**
- "Initial literature review" → **Initial Literature Review**
- "Initial wireframes and Figma designs for the chatbot" → **Creating
  initial wireframes and designs on Figma...**
- "Classical fatigue classifiers — RF / SVM / KNN" → **Train initial fatigue
  classifiers**
- "EMG fatigue pipeline — Zenodo biceps dataset" → **Rebuild pipeline on
  chosen dataset**
- "Dataset selection research and direction decision" → **Finding suitable
  dataset**
- "Fix wrong sample rate..." + "Channel convergence detection..." →
  **Finding channel convergence, and replaying cleaned sections** (merge
  both write-ups into this one card's description — it is the single card
  that covers both)
- "Harden pipeline validation" → fold into **Build the signal preprocessing
  pipeline**
- "LSTM sequence classifier..." → **Add LSTM classifier and testing
  prediction**
- "Transformer sequence classifier" → **Add transformer classifier and
  comparing all models**
- "Progress summary for supervisor and team" → **Summarizing results for
  showing supervisor our progress**
- "Progress Report submitted" → **Submit progress report**
- "Evaluate Open WebUI + Ollama tool-calling — and reject it" → **Allowing
  the ollama local models to call the trained model** *(the title in the
  current board actually says the opposite of what happened — see 2b)*
- "Streamlit chatbot frontend" → **Building the chatbot frontend**
- "Merge chatbot work into the integration branch" → **Merging to main,
  cleaning up repo**
- Remaining current-board cards with no v1 writeup yet (write these fresh,
  short, evidence-linked): **Set up the OpenBCI hardware...**, **Create
  Presentation PPT**, **Presentation Preparation**, **Create visualisations
  of the recorded data**, **Build real-time video playback of the raw
  signals**, **Testing the whole system end to end**, **Fixing bugs on
  chatbot frontend...**, **Final touches on chatbot**

Also assign a member to each (Aryan/Maalav/Rayyan/all three, per the old
plan's attributions) — right now literally zero cards have an owner.

### 2b. Card title vs. what actually happened

**"Allowing the ollama local models to call the trained model"** (Sprint 3
Done) reads as if native Ollama tool-calling was built and works. What
actually happened is the opposite: tool-calling was tried, `llama3.2:3b`
hallucinated fake tool calls, and the team built a dedicated Streamlit
frontend specifically so the LLM never calls anything — Python calls
`classify()` directly and the LLM only phrases the answer. Either rename the
card to **"Evaluate Ollama tool-calling — rejected, built direct-call
frontend instead"** or keep the title and make the description state the
rejection clearly. This is the same slides/build inconsistency flagged in
the Research Question card (2d) — fix both the same way, as a documented
decision, not a contradiction.

### 2c. Empty sprint columns

Sprint 1 To Do, Sprint 1 Doing, Sprint 2 To Do, Sprint 2 Doing, Sprint 3 To Do
are all empty. For a project that is clearly mid-Sprint-3 (Doing has 3 live
cards: final report, demo video, poster/presentation), empty To Do/Doing
columns on the earlier sprints read correctly as "finished and cleared" —
that's fine, **leave Sprint 1/2 To Do and Doing empty**, no action needed.
What needs action is **Sprint 3 To Do**, which should hold the near-term
not-yet-started work from `trello_board_plan.md` §4 (To Do list):

- Resolve the Claude API vs Ollama inconsistency across all documents
- Improve 3-class accuracy or justify a 2-class deployment model
- Extend unit test coverage
- Chatbot polish items (confidence shown, friendlier no-subject/time
  handling, short-upload handling)

### 2d. "Other" list — three empty cards

- **Project Timeline** — paste the table from `trello_board_plan.md` v1
  draft (§3a there) — Semester 2 phase table against actual status.
- **Maalav Garg - Research Question** and **Rayyan Abzal - Research
  Question** — both empty. Drafts exist in the old plan (§3b/3c) but are
  marked "unconfirmed — ask Maalav/Rayyan to check the wording." Do not
  paste those in as if they're final without asking first.
- **Aryan Shah - Research Question** — has content but needs two fixes: the
  Figma bullet is duplicated (delete one copy), and "Integrate Claude API"
  should become "Integrate a local LLM via Ollama" with the one-line
  rationale (cost/privacy/offline) — exact replacement text is in the old
  plan §3d. This is the single most visible slides-vs-build inconsistency on
  the whole board; fix it here first, then match the slides and report.

### 2e. Meeting Minutes — 7 cards are a bare PDF link, no summary text

`docs/meeting_minutes/build_minutes.py` already has the full structured
content for 5 of these (pulled directly from the script — ready to paste,
drop the HTML tags, keep the headings/bullets):

- **18/05/2026** — progress review, pipeline/video overlap roadblock,
  request to supervisor for ~3 rest/fatigued data sets
- **25/05/2026** — plot cleaned period dynamically, then predict future
  frequency from the pattern
- **02/06/2026** — deep-learning forecasting on converged-channel data
  (MATLAB pointer), label/cut the video, LSTM this break then Transformer
  next semester
- **06/08/2026** — chatbot demo (server-access blocker: Ollama not
  installed, email sent to Ngai), ~85%+ model confidence confirmed good by
  supervisor, usability goals, timeline to week 6
- **20/08/2026** — file-format support beyond CSV, optional speech-to-text,
  demo video is the main focus for the poster

For **15/06/2026** and **23/07/2026**, no equivalent structured source file
exists in the repo (`trello_board_plan.md` has a short draft for 15/06 only;
23/07 can be reconstructed from the "review of actions from 23 July" section
of the 6 August minutes above — chat history, upload-format research, more
detailed output, multi-input decision, perfect-the-video). I can draft both
from what's available, but if the original PDFs are at hand it's worth
pulling the real text instead of reconstructing it.

### 2f. Logbooks

- **Rayyan — duplicate "Week 7 Logbook" card.** One has real content, one is
  empty. Delete the empty one (don't merge — the content card already covers
  it).
- **Aryan & Maalav — both missing a Week 1 card.** Everyone else's lists
  start at Week 1 or Week 2 consistently; check whether Week 1 was folded
  into Week 2's text for these two, or genuinely never logged.
- **Aryan — Week 12 and Week 14 cards exist but are empty.** Week 13 has
  content. Content for these weeks is already written up elsewhere (LSTM/
  Transformer work, per the week-map table in §3 below) — just needs
  copying in.
- **Rayyan — Week 12 and Week 13 cards exist but are empty.**
- **Maalav — Week 12 card exists but is empty.**
- Nobody's logbook extends into Sprint 3 (final report / demo / poster
  weeks) — only the team "Semester 2 Logs" list attempts later weeks, and
  that one is also incomplete (next point).

### 2g. "Semester 2 Logs (Done as a team)" — the weekly team log

`docs/weekly_logs/` already has Weeks 1–4 built (pdf/docx/html), matching the
`build_logs.py` source. The Trello cards for these are currently **PDF-link
only, no summary text** (same problem as the meeting minutes) — pull the
same structured content into the card description the way §2e describes.
**Week 5 has no card content and no local source file yet** — this one is
actually unwritten, not just un-pasted; it needs drafting from scratch once
the week-5 work is settled, probably folding in as part of the final-report
push.

### 2h. Labels and members

- Name the 3 unused label colours or delete them — `Blue`, `Purple`, `Red`
  currently carry no meaning. If keeping the richer v1 label set
  (`Pipeline`, `Supervisor Request`, `Blocked / Risk`, `Data Collection`
  alongside the current `Admin + Other` / `ML/DL + pipeline` / `Frontend`),
  that's 3 more names to add; otherwise just delete the 3 spares.
- Merge `Malav Garg` (malavgarg) into `Maalav Garg` (qcn72331), or at least
  stop using the wrong-spelling account going forward — new assignments
  should go to one account consistently.
- Merge/remove the duplicate `Wei Qi Yan` account (`weiqiyan1`).
- **Assign a member to every Sprint Done/Doing card** — right now not one
  card on the whole board has an owner, which undercuts the per-person
  logbook evidence sitting right next to it.

---

## 3. Week → evidence map (unchanged from v1, still accurate)

Useful for filling the remaining empty logbook weeks (2f) and the Semester 2
team weeks (2g). Week numbering follows Maalav's own dated cards.

| Week | Dates | What happened | Evidence | Whose logbook |
|---|---|---|---|---|
| W13 | 01/06–07/06 | Sample-rate bug (1000→250 Hz), 11% packet loss found. Convergence detection + replay GUI built. | `convergence_analysis/FINDINGS.md` | Rayyan |
| W14 | 08/06–14/06 | Dataset direction locked (hybrid: public now, forearm later). Zenodo pipeline built + hardened. LSTM classifier + 2s-ahead prediction. Progress summary written. | commits `b78b149`,`e44f268`,`5adb98c` | Rayyan, Maalav, Aryan |
| W15 | 15/06–21/06 | Integration contract agreed. signal_viewer + PlantUML. Cross-subject generalisation note. Fatigue pattern graph delivered. | commits `fb04106`,`c3010b7`,`c0962eb` | Rayyan, Aryan |
| W17 | 29/06–05/07 | Transformer classifier built (originally "next semester"). `classify()` + training + serving + eval. | commits `94fb158`,`7e4e4af`,`fb8415f` | Aryan |
| W18 | 06/07–12/07 | Design doc for chatbot visualisation, reviewed before building. | commits `7a16978`,`02eb157`,`25ea98f` | Rayyan |
| W19 | 13/07–19/07 | `render_window()` Plotly chart + MDF trend line/slope. Tests. requirements.txt. | commits `cad2e9c`,`eb754b1`,`0f55561` | Rayyan |
| W21 | 27/07–02/08 | Streamlit frontend, `classify_upload()`, MDF forecast, teammate testing guide, merge to integration branch. | commits `213878d`,`77122e2`,`5aec879`,`3a76d92` | Maalav, Aryan |

---

## 4. Two checks before calling the board done

1. **Every Done/Doing card has a description, evidence, and an owner.** Right
   now that's true of zero of the 23 Sprint Done cards and the 3 Sprint 3
   Doing cards — this is the single biggest gap left.
2. **Research Question + chatbot-architecture cards agree with the slides
   and report on Claude API vs Ollama.** One documented decision, repeated
   consistently everywhere, beats three different stories.
