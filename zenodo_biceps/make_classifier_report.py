"""
Create a compact classifier results table from metrics_*.json files.

Run after classify_biceps.py experiments:
    python make_classifier_report.py

Outputs:
  - out/classifier_results_summary.csv
  - out/classifier_results_summary.md
"""
from __future__ import annotations

import csv
import glob
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load_metric(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cfg = payload["config"]
    results = payload["results"]
    best_name = max(results, key=lambda n: results[n]["mean_acc"])
    best = results[best_name]

    return {
        "file": os.path.basename(path),
        "label_mode": cfg["label_mode"],
        "fs": cfg["target_fs"],
        "temporal": cfg["temporal"],
        "transition_margin_sec": cfg["transition_margin_sec"],
        "n_features": cfg["n_features"],
        "best_model": best_name,
        "best_acc": best["mean_acc"],
        "best_macro_f1": best["mean_f1"],
        "best_std_acc": best["std_acc"],
        "rf_acc": results.get("RF", {}).get("mean_acc", ""),
        "svm_acc": results.get("SVM", {}).get("mean_acc", ""),
        "knn_acc": results.get("KNN", {}).get("mean_acc", ""),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Classifier Results Summary\n\n")
        f.write("| Label mode | FS | Temporal | Margin | Features | Best | Acc | Macro-F1 |\n")
        f.write("|---|---:|---:|---:|---:|---|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['label_mode']} | {r['fs']} | {r['temporal']} | "
                f"{r['transition_margin_sec']} | {r['n_features']} | "
                f"{r['best_model']} | {r['best_acc']:.3f} | "
                f"{r['best_macro_f1']:.3f} |\n"
            )


def main() -> None:
    paths = sorted(glob.glob(os.path.join(OUT, "metrics_*.json")))
    rows = [load_metric(p) for p in paths]
    rows.sort(key=lambda r: (
        r["label_mode"],
        r["fs"],
        not r["temporal"],
        r["transition_margin_sec"],
    ))

    csv_path = os.path.join(OUT, "classifier_results_summary.csv")
    md_path = os.path.join(OUT, "classifier_results_summary.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    for r in rows:
        print(
            f"{r['label_mode']:<26} fs={r['fs']:<4} "
            f"temporal={str(r['temporal']):<5} "
            f"margin={r['transition_margin_sec']:<4} "
            f"best={r['best_model']:<3} "
            f"acc={r['best_acc']:.3f} f1={r['best_macro_f1']:.3f}"
        )
    print(f"\nsaved {csv_path}")
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()
