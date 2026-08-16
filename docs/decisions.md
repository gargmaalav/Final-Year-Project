# Decisions

Architecture and scope decisions for this project, with the why. Referenced from root
CLAUDE.md. Add new decisions at the top.

## 2026-08-06: Standalone vanilla-JS chatbot UI is primary, not Maalav's Streamlit frontend

Two working chatbot frontends now exist for this project:
- `viz/chatbot_ui.html` + `models/serve.py` (this branch, `feat/rayyan-chatbot-ui-implement`):
  single-file vanilla JS, no build step, calls `classify()`/`render_window()`/
  `forecast_fatigue()` over HTTP, real Ollama-generated answers.
- `frontend/` (Maalav's `maalav-chatbot` branch, merged into `feat/chatbot-integration`):
  Streamlit app calling the same Python functions directly (no HTTP hop), also has upload
  calibration, MDF forecast, and sport/plan recommendations (`frontend/recommend.py`).

Ray chose to keep the vanilla-JS UI as the primary frontend and drop the Streamlit one
("we want to drop maalavs and use my new one as his isn't the best").

**Why:** Ray's own product/design judgment on this session's build, after being shown both
existed - not a technical tie-breaker either way found in code.

**How to apply:** don't build against or recommend `frontend/`/Streamlit for this project's
chatbot UI going forward unless Ray reverses this. `models/fatigue_forecast.py` and
`models/forecast_lstm.py` were pulled from `feat/chatbot-integration` into this branch
(same forecast code, not a rewrite) since those are real backend capability, independent of
which frontend uses them. `frontend/recommend.py` (sport/plan recommendations) has NOT been
ported over yet - open question, not decided either way.
