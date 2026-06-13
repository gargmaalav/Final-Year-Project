# Dataset Candidates - sEMG Fatigue / Spectrum-Shape Forecasting

Research date: 2026-06-12. Target use: predict future FFT spectrum SHAPE (normalized power-per-bin %) over a sustained or repeated contraction; MDF decline as fatigue marker; LSTM train-70%/predict-30% backtest. Need high FS (clean FFT), multi-subject, raw signal, open license.

## DIRECTION RESOLVED 2026-06-12 (forearm, hybrid, both targets)
- Slides ALWAYS planned FOREARM FLEXOR GRIP (slides 4, 12, Maalav's), 3 sets Fresh/Moderate/Fatigued, sustained isometric to failure. Bicep was the DEVIATION (OpenBCI railed -> fell back to Zenodo bicep). Returning to forearm = back to spec, not a pivot.
- "Forearm easier" = COLLECTION ease only (fatigue in 30-60s, strong signal, cheap grip trainer). It makes PUBLIC DATA harder: clean bicep fatigue sets exist, forearm-grip-to-failure ~ none.
- Final self-collected data = OpenBCI Cyton 250 Hz, bandpass 20-120 Hz (low-FS BY DESIGN). So high-FS public sets are LESS representative of the real inference target. BRIDGE RULE: downsample all public training data to ~250 Hz before training so features/models transfer to OpenBCI.
- Decisions (Ray, 2026-06-12): data source = HYBRID (public stopgap now, self-collect forearm later); ML target = BOTH classifier (slides, 3-class) AND spectrum forecast (supervisor) as separate deliverables.

### Plan per deliverable
- Phase 1 classifier (forearm-bound): **Zenodo 5189275** - only public forearm + sustained-fatigue set; segment 120s hold into Fresh/Mod/Fatigued; 200 Hz ~ 250 Hz target (representative, not a downgrade); caveat 8-bit + load-hold not grip-squeeze.
- Phase 1 forecast: KEEP **Zenodo 14182446 biceps** (validated 89.3%, do not churn); optionally also forecast 5189275's forearm trajectory downsampled to 250 Hz.
- Phase 2 (self-collect, slide plan): 250 Hz OpenBCI forearm flexor grip, 3 sets; serves classifier (set labels) + forecast (trajectory); retrain both. Needs hardware fix + ethics.
- DROPPED under forearm direction: figshare 24770868 (prime mover = deltoid on frontal raise; keep only as optional high-FS bicep cross-check); Nature HD-sEMG PMC7670452 (no fatigue protocol).

## What OURS measures (anchor)
- Muscle: biceps brachii, UPPER ARM only (Zenodo trials 5/6; other trials are deltoids; OpenBCI electrodes on bicep). NO forearm.
- FS: 1259 Hz (Zenodo) / true 250 Hz (OpenBCI). Single biceps channel after averaging.

## Comparison vs ours + forearm coverage

| Source | FS | Mode | Muscles | Forearm? | Fit vs ours |
|---|---|---|---|---|---|
| Ours (Zenodo 14182446 + OpenBCI) | 1259 / 250 Hz | dynamic curls / sustained hold | biceps + deltoids | No | baseline |
| figshare 24770868 | 2148 Hz | sustained isometric 210 s | biceps, triceps, brachioradialis, ant deltoid, infraspinatus, 4x trapezius/cervical | Yes (brachioradialis) | strongest overall |
| Nature PMC7670452 HD-sEMG | 2048 Hz | brief 10 s MVC (no fatigue) | biceps, triceps, anconeus, brachioradialis, pronator teres | Yes (2-3 forearm) | most forearm, but no fatigue trajectory |
| Zenodo 5189275 | 200 Hz 8-bit | sustained 6 kg elbow-flex 120 s | 8 channels on the FOREARM (forearm-fatigue study) | Yes (forearm-dedicated) | only forearm fatigue set, low quality |

Forearm note: brachioradialis is anatomically forearm but acts as an elbow flexor (behaves like biceps). True wrist-driver forearm muscles (flexor/extensor carpi, pronator teres) appear only in the Nature HD-sEMG set. No single source gives BOTH forearm coverage AND a clean high-FS fatigue trajectory.

## Alignment axis that matters: contraction MODE
- Dynamic (rep-rest bursts) = current Zenodo primary (dumbbell curls).
- Sustained isometric (continuous hold) = classic monotonic MDF-decline paradigm, cleanest spectral compression, single continuous trajectory maps directly onto the 70/30 backtest.

Pipeline + 89.3% backtest + supervisor GUI were validated on DYNAMIC data. Treat new sets as COMPLEMENTARY, not drop-in swaps.

## Ranked shortlist

### 1. figshare 24770868 - Self-report + Palpation + sEMG, isometric (BEST NEW MATCH)
- DOI 10.6084/m9.figshare.24770868. Paper PMC10869346 (Scientific Data, Nature). CC BY 4.0.
- 30 male subjects (29 full), 2148 Hz, 9 channels including Biceps Brachii plus Brachioradialis, Triceps, deltoid, trapezius.
- Protocol: sustained isometric dumbbell frontal raise (1 kg) held 210 s to fatigue. One continuous trajectory per subject.
- Files (VERIFIED via figshare API): 30x `Subject##.xlsx` raw sEMG (~1.4 GB total, ~48 MB each) plus 30x self-reported fatigue plus 30x palpation scores (every 30 s) plus readme. Public, no login.
- Why: high FS = clean FFT; sustained hold = textbook MDF decline; single 210 s series = direct 70/30 backtest fit; multi-subject for LSTM. Caveat: 30 subjects x 1 trajectory each (vs Zenodo 13 x multiple trials x 2 arms), modest but workable. Raw is .xlsx not .mat.

### 2. Zenodo 14182446 - biceps sEMG + self-perceived fatigue (CURRENT PRIMARY, keep)
- Already downloaded. 13 subjects, 1259 Hz, biceps both arms, dynamic dumbbell curls to fatigue, 0/1/2 labels. Keep as the dynamic-mode source.

### 3. Zenodo 5189275 - Multi-channel sEMG for fatigue (LOW PRIORITY)
- 15 subjects, 200 Hz, 8-bit, 8 ch ON THE FOREARM, sustained 6 kg elbow-flex 90deg, 120 s, raw txt/.mat, CC BY 4.0. Paper: NILES 2020, "Upper Limb Muscle Fatigue Analysis", studies FOREARM fatigue.
- Only forearm-dedicated fatigue set found, but same low-FS / low-res weakness as the dead OpenBCI data. Use only if forearm coverage is required.

### 4. Nature s41597-020-00717-6 (PMC7670452) - HD-sEMG elbow (NOT for fatigue)
- 12 subjects, 2048 Hz, HD arrays, 5 muscles including biceps. But brief 10 s contractions at 10/30/50% MVC with 2 min rest, protocol AVOIDS fatigue. Clean biceps reference only; no spectral-decline trajectory.

## Notes
- PhysioNet surfaced no strong sEMG fatigue set; Ninapro is gesture recognition, not fatigue. Breadth covered: Zenodo, Mendeley, Nature/Sci-Data, figshare, PhysioNet.
- Mendeley 8j2p29hnbv (biceps/triceps "fatigue") = 1 Hz envelope only, useless for FFT. Rejected.
