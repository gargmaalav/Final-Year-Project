"""
Entry point for the 8-channel EMG convergence analysis.

Usage:
    python run.py gui       # interactive converged-sections player (needs a display)
    python run.py detect    # headless: CSV + summary + PNG snapshots of sections
    python run.py forecast  # headless: MDF trend + forecast over long segments
    python run.py all       # detect + forecast, then launch the player

All analysis is restricted to the recording's clean segments (the "certain
period"), uses FS = 250 Hz (proven from the Sample Index wrap rate), and applies
NO noise filtering -- only per-window normalization so the 8 channels can be
compared by shape.

Run this from inside the convergence_analysis/ folder: the data path is the
relative ../data/OpenBCI-RAW-2026-03-21_15-37-11.txt.
"""
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gui"
    if cmd == "detect":
        import detect
        detect.run()
    elif cmd == "forecast":
        import forecast
        forecast.run()
    elif cmd == "gui":
        import gui
        gui.run()
    elif cmd == "plot":
        import plot_overview
        plot_overview.run()
    elif cmd == "all":
        import detect
        detect.run()
        import forecast
        forecast.run()
        import gui
        gui.run()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
