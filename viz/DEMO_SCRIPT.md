# EMG Fatigue Chatbot - Demo Recording Guide

For the video to the supervisor + group. Everything below is already set up and
proven working in the browser tab that is open now (a fresh chat with two turns
already answered). You can record that chat as-is, or clear it and re-type the
prompts live. Aim for ~2-3 minutes.

## Before you hit record (all already done, just confirm)

- Three things must be running (they are, right now):
  - Ollama (`ollama serve`)
  - The model API on port 8000 (`.venv/bin/python models/serve.py`)
  - Open WebUI on port 8080
- The open chat is configured: model **llama3.2:3b**, Function Calling set to
  **Legacy**, and the **EMG Fatigue Classifier** tool enabled. If you start a
  brand-new chat these reset, so re-set them: Controls -> Function Calling ->
  Legacy, and the wrench/Integrations menu -> enable EMG Fatigue Classifier.
- Recording on macOS: press **Cmd+Shift+5**, choose "Record Selected Portion",
  drag over the browser window, click Record. Stop from the menu bar. The file
  lands on your Desktop.

## What to say + show (scene by scene)

**1. Opening (10s).** "This is our EMG fatigue classifier wired into a chatbot.
You ask about a subject in plain English, the deep-learning model runs, and it
answers with an interactive chart inline - the link between our model and the
chatbot frontend."

**2. First question - not fatigued (40s).** Prompt already there:
> Is subject 13 fatigued at 120 seconds?

Point out, top to bottom:
- The text answer is the model's verdict: it says subject 13 is not fatigued,
  grounded in the model's real output, and it cites the tool as its source.
- Three linked panels: raw EMG of that 4-second window, median frequency (MDF)
  over the whole recording with each window coloured by its fatigue stage
  (green Fresh, orange Transition, red Fatigued), and the FFT spectrum.
- On the middle MDF panel, the blue dashed "fatigue trend" line is a fitted
  decline through the whole recording; its label reads the slope in Hz/min
  (about -2.7 Hz/min here) - that falling median frequency is the quantitative
  fatigue signature, not just the colour of the dots.
- The violet "asked: 120s" marker pins the exact moment you asked about, so you
  do not lose it when the window scrubs during playback.

**3. Interactivity (30s).** On that same chart:
- Click **Play** (top-left of the chart) - the window scrubs across the whole
  recording and you can watch the waveform and spectrum change over time. The
  "asked: 120s" marker stays pinned so you never lose the queried moment.
- Pick the **Box Select** tool (top-right of the chart) and drag across the
  middle MDF panel - it reads out the dominant fatigue state and MDF range for
  that span, and jumps the detail panels to it.

**4. Second question - fatigued (30s).** Prompt already there:
> Is subject 13 fatigued at 200 seconds?

"Same subject, later in the recording." Point out:
- The answer is now fatigued, and the model agrees at 100%.
- The MDF has dropped from ~61 Hz to ~53 Hz - that decline is the physiological
  fatigue signature, and you can see the dots go green -> orange -> red across
  the recording.

**5. Close (10s).** "So the model, the classification, and an interactive view of
the signal are all in one chatbot answer. The charts are self-contained and
lightweight, so they load fast inline."

## Keep it clean on camera (avoid these)

- **Stick to the right arm or don't mention an arm.** The small 3B model
  reliably picks up the subject and time, but sometimes misses "left arm" and
  defaults to right. All the prompts above avoid that. If you want a left-arm
  example, double-check the chart title says "L Biceps" before trusting it.
- Don't ask "show me all subjects" - the all-subjects overview was removed
  (it was an unreadable 13-line tangle), so there's no chart for that.

## Good backup prompts (all right-arm, all verified)

- Not fatigued: subject 3 at 90 seconds; subject 7 at 200 seconds.
- Fatigued: subject 2 at 180 seconds; subject 1 at 150 seconds; subject 5 at
  200 seconds.
