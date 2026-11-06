"""Compute the overall rank R_avg (paper Sec. 3.2) from per-method metric scores.

Each method is ranked on the eight metrics of Tab. 1 (FID, KID, B_consist, H_consist:
lower is better; A_s, F_t, T_score, VLM_c: higher is better). A method without VLM_c
(same-category-only methods) is ranked last on VLM_c, i.e. one position below all
methods with an available score. R_avg is the mean of the eight ranks (lower is better).

Usage:
    python tools/compute_ravg.py --metrics results/table1_metrics.csv --output results/ranking.json
"""
import argparse
import csv
import json

LOWER_BETTER = ["FID", "KID", "B_consist", "H_consist"]
HIGHER_BETTER = ["A_s", "F_t", "T_score", "VLM_c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/table1_metrics.csv")
    ap.add_argument("--output", default="results/ranking.json")
    args = ap.parse_args()

    with open(args.metrics, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ranks = {r["method"]: [] for r in rows}
    for metric in LOWER_BETTER + HIGHER_BETTER:
        avail = [(float(r[metric]), r["method"]) for r in rows if r[metric] != ""]
        avail.sort(reverse=metric in HIGHER_BETTER)
        for pos, (_, m) in enumerate(avail, start=1):
            ranks[m].append(pos)
        for r in rows:
            if r[metric] == "":
                ranks[r["method"]].append(len(avail) + 1)
    ravg = {m: sum(v) / len(v) for m, v in ranks.items()}
    order = sorted(rows, key=lambda r: ravg[r["method"]])
    out = {
        "metadata": {
            "metric": "R_avg",
            "definition": "Mean rank over FID, KID, A_s, F_t, T_score, VLM_c, B_consist, H_consist; "
                          "missing VLM_c is ranked last; lower is better.",
            "models": len(rows),
        },
        "rank": [{"position": i + 1, "Model": r["method"], "paradigm": r["paradigm"],
                  "R_avg": round(ravg[r["method"]], 4)} for i, r in enumerate(order)],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    for e in out["rank"]:
        print(f"{e['position']:>2}  {e['Model']:<16} {e['R_avg']:.3f}")


if __name__ == "__main__":
    main()
