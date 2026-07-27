# Draft message to supervisor + group

Copy-paste and adjust the tone to your group's norm. Record the video first
(see DEMO_SCRIPT.md), then share it alongside this note.

---

Hi all,

Quick update on the visualization + chatbot integration. I have the EMG fatigue
model wired into the chatbot end to end, and it is working.

You ask about a subject in plain English (for example "is subject 13 fatigued at
120 seconds?"), the deep-learning model runs on that window, and the chatbot
answers with the result plus an interactive chart embedded right in the reply.
This is the link between our model and the chatbot frontend.

What the chart shows, for the queried subject:
- the raw EMG of that 4-second window,
- median frequency over the whole recording, coloured by fatigue state, with a
  fitted trend line reading the fatigue rate in Hz/min,
- the FFT spectrum of the window.

You can play it back to scrub the window across the whole recording, and
box-select a time span to read out its fatigue state and frequency range. The
model's own verdict comes through in the chatbot's text answer, and the chart
colours each window of the recording by the dataset's ground-truth fatigue
stage, so you can see the progression from fresh to fatigued.

The short video walks through two questions on subject 13: at 120 seconds the
model says not fatigued, and at 200 seconds it says fatigued, with the median
frequency visibly dropping in between, which is the fatigue signature we expect.

Happy to demo it live or talk through any of it.

Thanks,
Rayyan

---

Attachments to include when you send (do not write "attached" in the body, just
add the files):
- the screen recording you make from DEMO_SCRIPT.md
