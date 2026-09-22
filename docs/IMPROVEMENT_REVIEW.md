# UI, performance, and result-quality review

Reviewed 22 September 2026 on `codex/ui-performance-quality`.
Fetched all remotes with pruning and checked the branch against `origin/main`
(`a0914b0`); it was already current. The remaining remote feature branch has
no commits outside main.

## Changes implemented

- Upload graphs carry a recording ID through the API. Opening an older answer
  cannot silently display the latest upload. Missing/expired references fail
  rather than substitute a different recording.
- Upload cache identity includes the supplied sample rate. Correcting that rate
  now recalculates the signal and calibration instead of reusing stale results.
- CSV input rejects infinities, duplicate/backward timestamps, and ambiguous
  extra columns. Upload processing reads at most 20 MB plus one detection byte;
  larger files receive an actionable error. This is a processing limit, not a
  pre-multipart network ingress limit.
- Chart initialization uses the same start sample as classification, including
  off-grid queries and end-of-recording clamping. Previously a 60 s request
  selected the window centered at 60 s (starting at 58 s). A browser check
  exposed 59.5 Hz in the graph versus 60.8 Hz in the answer. Chart titles now
  explicitly name the window start.
- Browser chart responses use a versioned, cacheable local Plotly asset, with
  gzip response compression. Standalone generated HTML stays self-contained.
- Model initialization is serialized and publishes weights only after loading
  completes, preventing concurrent requests from seeing a partially loaded model.
- Loading feedback includes elapsed time; keyboard focus returns to the composer.
  Buttons use real disabled states, IME Enter is respected, and chat navigation
  is locked during a turn so its answer cannot appear in another conversation.
  Late graph responses are ignored after leaving their originating chat.
- Added the missing `python-multipart` runtime dependency and corrected upload
  guidance (the default reading is at the end, not the start).

## Performance evidence

A local subject-13, 60 s chart measured 3,115,895 bytes before API conversion,
1,996,009 bytes after removing the repeated runtime (36% smaller), and 518,271
bytes after gzip level 4 (83% smaller than the original HTML). This measurement
preceded the exact-window alignment fix; sizes will vary slightly with frames.
The shared runtime is a separate cacheable download on first use. These are
HTML byte sizes, not complete HTTP/JSON sizes or end-to-end latency claims.
Rendering still took about 2.44 s cold; this change primarily saves transfer,
serialization, and repeated runtime storage, not figure construction time.

## Remaining priorities

1. **Reproducible quality evaluation.** Restore the missing subject-11 trial-5
   recording and run the complete classifier/calibration suite. Score the actual
   deployed weights on held-out subjects with `models/eval_deployed.py`; do not
   substitute research LOSO scores from a different model/configuration. Add
   per-subject recall, false-positive rates, and confidence calibration before
   treating softmax confidence as probability of a correct fatigue verdict.
2. **Answer latency.** Record timings for parsing, loading, features, inference,
   forecast, and LLM wording. Benchmark cold and warm p50/p95 across representative
   questions. Consider an immediate deterministic reading followed by optional
   model explanation. That needs an explicit staged-response API and request
   IDs so retries cannot apply follow-up context twice.
3. **Chart cost.** Replace precomputed waveform animation frames with small
   on-demand window requests. Keep a compact overview; load detailed samples
   only when a user opens or scrubs the graph. Verify displayed MDF against
   classification at off-grid times before adopting any downsampling.
4. **Session lifecycle.** Bound retained sessions/uploads and introduce expiry
   with explicit UI messaging. Current sessions and locks live indefinitely,
   and browser history survives server restarts while analysis context does not.
   Eviction must never remove an active session or substitute another upload.
5. **Upload onboarding.** Add a preview with explicit time/channel selection,
   units, sampling rate, and duration before analysis. Timestamp regularity and
   gap handling need a documented acceptance policy; monotonicity alone does
   not establish a uniformly sampled recording.
6. **UI polish.** Make detailed figures easier to read on narrow screens, add
   a return-to-latest-message control, and show server readiness/expired context
   alongside saved chats. Keep the measured verdict and its provenance visible.
7. **Repeatable installs and CI.** Pin a tested dependency set, separate legacy
   Streamlit requirements from the API runtime, and run dataset-independent tests
   in CI. Keep dataset-backed evaluation in a separately provisioned job.

## Validation

- `python frontend/test_answers.py`: 102 passed.
- `python frontend/test_understanding.py`: 190 passed; optional live LLM
  adversarial checks were not requested by that command.
- `python models/test_classify.py --fast`: 11 checks passed.
- `python -m unittest viz.test_render_window -v`: 10 tests passed.
- `python -m unittest discover -s tests -v`: 8 tests passed; regression coverage for input
  validation, upload identity/rate changes, exact chart windows, asset delivery,
  and oversize uploads.
- JavaScript syntax checked with Node; Python modules compiled; diff checked.
- Browser smoke test: real subject-13 answer, loading/disabled states, restored
  focus, and successfully rendered chart using the shared runtime. After the
  alignment fix, the browser confirmed both graph and answer report 60.8 Hz.
- Full classifier test stopped on missing subject-11 trial-5 data. No claim of
  improved model accuracy or a complete dataset-backed validation is made.

## Follow-up: missing dataset recordings

The subject-7 summary exposed an uncaught missing-file exception. Restored 12
missing biceps trial CSVs from the existing local dataset archive (raw data
remains gitignored). Added a regression test for missing summary recordings and
replaced the misleading upload-settings fallback for HTTP 500 responses.
The full classifier suite now passes, including 94.7% agreement between fresh
upload calibration and stored calibration across subjects 11–13. Forecast-model
checks that require trained forecast weights still report their own skips.
All 9 API/upload/chart regression tests pass.
