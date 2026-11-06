# -*- coding: utf-8 -*-
"""Image-clustered bootstrap CIs for BT Spearman rho and pair-level agreement.

The resampling unit is the image: one image carries its 7 method scores, all
A/B/tie votes from the 20 questionnaires, and the 21 method-pair outcomes
aggregated by strict plurality. Whole clusters are drawn with replacement.
Images of different dimensions are distinct, so each dimension is resampled
independently (one (B, 20) index matrix per dimension, drawn in the fixed
dimension order from a single seeded generator); within a dimension, rho and
agreement share the same indices. The pooled agreement averages the five
independently resampled dimensions.

Aligned with the repository's authoritative implementations (no bespoke rules):
  - rho: BT win rates + Spearman, identical to metric_bt.py (cross-checked)
  - pair agreement: calls compute_agreement.run(), which implements strict
    plurality, tied top votes -> tie, and explicit tie votes

Frozen settings are recorded in the JSON/MD reports: seed, B, and CI method.

This directory is self-contained (it ships a frozen copy of
compute_agreement.py); machine score TXTs and metric_bt.py are read from
human_preference/, questionnaire CSVs from results_v2/:
  cd VTBench
  python reproducibility/bootstrap_ci.py
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
HUMAN_PREFERENCE_DIR = REPO_DIR / "human_preference"
# append (not insert(0)): compute_agreement must resolve to the frozen copy in
# this directory; only the committed metric_bt.py in human_preference/ is
# resolved from the original directory.
sys.path.append(str(HUMAN_PREFERENCE_DIR))

from metric_bt import (  # noqa: E402
    METHOD_KEYS, CSV_MAP, METRICS_CONFIG, TXT_KEY_MAP, parse_vton_metrics,
)
from compute_agreement import run as compute_agreement_run, RULE_DEFINITION  # noqa: E402

DEFAULT_CSV_DIR = REPO_DIR / "results_v2"
DEFAULT_SCORES_DIR = HUMAN_PREFERENCE_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR

# Frozen configuration; the seed shares its starting point with the
# pre-registered main seed in METHODOLOGY.md Sec. 4 but is used independently.
SEED = 20260730
N_BOOT = 10000
ALPHA = 0.05
N_IMAGES_PER_DIM = 20
N_PAIRS = 21

TXT_FILES = {
    "Aesthetic": "aesthetic_scores.txt",
    "Background": "bg_scores.txt",
    "Cross": "cross_scores.txt",
    "Hand": "hand_scores.txt",
    "Text": "text_scores.txt",
}
DIM_NAME = {
    "Aesthetic": "aesthetic", "Background": "bg", "Cross": "cross",
    "Hand": "hand", "Text": "text",
}

METHOD_INDEX = {method: i for i, method in enumerate(METHOD_KEYS)}
PAIRS = list(combinations(range(len(METHOD_KEYS)), 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-clustered bootstrap CI (rho + pair agreement).")
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS_CONFIG), default=list(METRICS_CONFIG))
    return parser.parse_args()


def compare_score(value: str) -> tuple[int, int, float]:
    if ">" in value:
        winner, loser = value.split(">", 1)
        return CSV_MAP[winner.strip()], CSV_MAP[loser.strip()], 1.0
    if "<" in value:
        loser, winner = value.split("<", 1)
        return CSV_MAP[winner.strip()], CSV_MAP[loser.strip()], 1.0
    if "=" in value:
        first, second = value.split("=", 1)
        return CSV_MAP[first.strip()], CSV_MAP[second.strip()], 0.5
    raise ValueError(f"Cannot parse questionnaire comparison: {value!r}")


def load_machine_wins(scores_dir: Path, metric: str) -> dict[str, np.ndarray]:
    """{image: 7-d machine win vector}; summed over 21 pairwise comparisons per image (unnormalized)."""
    entries = parse_vton_metrics((scores_dir / TXT_FILES[metric]).read_text(encoding="utf-8"))
    images = METRICS_CONFIG[metric]["images"]
    is_positive = METRICS_CONFIG[metric]["is_positive"]
    result: dict[str, np.ndarray] = {}
    for entry in entries:
        image = entry["Image"].rsplit(".", 1)[0]
        if image not in images:
            continue
        vals = np.zeros(len(METHOD_KEYS))
        for raw_key, value in entry.items():
            if raw_key in TXT_KEY_MAP:
                vals[METHOD_INDEX[TXT_KEY_MAP[raw_key]]] = float(value)
        wins = np.zeros(len(METHOD_KEYS))
        for i, j in PAIRS:
            if vals[i] == vals[j]:
                wins[i] += 0.5
                wins[j] += 0.5
            elif (vals[i] > vals[j]) == is_positive:
                wins[i] += 1.0
            else:
                wins[j] += 1.0
        result[image] = wins
    missing = set(images) - set(result)
    if missing:
        raise RuntimeError(f"{metric}: machine scores missing for {len(missing)} images")
    return result


def load_human_wins(csv_dir: Path, metric: str) -> tuple[dict[str, np.ndarray], int]:
    """{image: 7-d human win vector summed over all raters}; number of raters."""
    images = set(METRICS_CONFIG[metric]["images"])
    start, end = METRICS_CONFIG[metric]["row_range"]
    wins_by_image = {image: np.zeros(len(METHOD_KEYS)) for image in images}
    csv_files = sorted(csv_dir.glob("result_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No result_*.csv found in {csv_dir}")
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, sep="\t")
        columns = list(df.columns[1:])  # first column = sample identifier
        for index, row in df.iterrows():
            if not start <= index <= end:
                continue
            image = Path(str(row.iloc[0])).stem  # sample column, e.g. "texture/FTF-image_0553"
            if image not in images:
                continue
            wins = wins_by_image[image]
            for column in columns:
                value = str(row[column]).strip()
                if not value:
                    continue
                winner, loser, weight = compare_score(value)
                wins[METHOD_INDEX[winner]] += weight
                wins[METHOD_INDEX[loser]] += 1.0 - weight
    return wins_by_image, len(csv_files)


def win_rates(total_wins: np.ndarray, n_images: int, n_raters: int = 1) -> np.ndarray:
    return total_wins / (n_images * n_raters * (len(METHOD_KEYS) - 1))


def rho_from_wins(machine: np.ndarray, human: np.ndarray) -> float:
    rho = spearmanr(machine, human)[0]
    return float(rho) if np.isfinite(rho) else float("nan")


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load per-image sufficient statistics for both statistics ----
    rho_data = {}
    agreement_by_dim = {}
    for metric in args.metrics:
        machine_wins = load_machine_wins(args.scores_dir, metric)
        human_wins, n_raters = load_human_wins(args.csv_dir, metric)
        # Frozen cluster order = METRICS_CONFIG image list order (the order used for the
        # reported CIs); do not re-sort, as the bootstrap indices are tied to this order.
        images = [image for image in METRICS_CONFIG[metric]["images"] if image in machine_wins]
        if len(images) != N_IMAGES_PER_DIM or set(images) != set(human_wins):
            raise RuntimeError(f"{metric}: per-image data incomplete or inconsistent")
        rho_data[metric] = {
            "machine": np.stack([machine_wins[image] for image in images]),
            "human": np.stack([human_wins[image] for image in images]),
            "n_raters": n_raters,
            "images": images,
        }

    sample_rows, agreement_summary = compute_agreement_run(str(args.csv_dir), str(args.scores_dir))
    for metric in args.metrics:
        dim = DIM_NAME[metric]
        counts = {row["image"]: row["exact_agreement_count"] for row in sample_rows
                  if row["category"] == dim}
        images = rho_data[metric]["images"]
        if set(counts) != set(images):
            raise RuntimeError(f"{metric}: image sets differ between agreement and rho data")
        agreement_by_dim[metric] = np.asarray([counts[image] for image in images], dtype=np.float64)

    # ---- Frozen random stream: an independent (B, 20) image-index matrix per dimension
    #      (drawn in args.metrics order); shared by rho and agreement within a dimension ----
    rng = np.random.default_rng(args.seed)
    draws_by_dim = {m: rng.integers(0, N_IMAGES_PER_DIM, size=(args.n_boot, N_IMAGES_PER_DIM))
                    for m in args.metrics}

    results = {}
    overall_agree_boot = np.zeros(args.n_boot)
    for metric in args.metrics:
        data = rho_data[metric]
        machine_matrix, human_matrix = data["machine"], data["human"]
        n_raters = data["n_raters"]
        agree_counts = agreement_by_dim[metric]

        point_rho = rho_from_wins(
            win_rates(machine_matrix.sum(axis=0), N_IMAGES_PER_DIM),
            win_rates(human_matrix.sum(axis=0), N_IMAGES_PER_DIM, n_raters),
        )
        point_agree = float(agree_counts.sum() / (N_IMAGES_PER_DIM * N_PAIRS))

        boot_rho = np.empty(args.n_boot)
        boot_agree = np.empty(args.n_boot)
        for b in range(args.n_boot):
            idx = draws_by_dim[metric][b]
            boot_rho[b] = rho_from_wins(
                win_rates(machine_matrix[idx].sum(axis=0), N_IMAGES_PER_DIM),
                win_rates(human_matrix[idx].sum(axis=0), N_IMAGES_PER_DIM, n_raters),
            )
            boot_agree[b] = agree_counts[idx].sum() / (N_IMAGES_PER_DIM * N_PAIRS)
        overall_agree_boot += boot_agree / len(args.metrics)

        valid_rho = boot_rho[np.isfinite(boot_rho)]
        if len(valid_rho) < 0.95 * args.n_boot:
            raise RuntimeError(f"{metric}: too many non-finite bootstrap rho values")
        rho_ci = percentile_ci(valid_rho)
        agree_ci = percentile_ci(boot_agree)
        results[metric] = {
            "n_images": N_IMAGES_PER_DIM,
            "n_raters": n_raters,
            "rho": point_rho, "rho_ci_low": rho_ci[0], "rho_ci_high": rho_ci[1],
            "rho_boot_mean": float(valid_rho.mean()),
            "pair_agreement": point_agree,
            "pair_agreement_ci_low": agree_ci[0], "pair_agreement_ci_high": agree_ci[1],
            "pair_agreement_count": int(agree_counts.sum()),
            "pair_total": N_IMAGES_PER_DIM * N_PAIRS,
            "n_valid_rho_boot": int(len(valid_rho)),
        }
        r = results[metric]
        print(f"{metric:11s} rho={r['rho']:.4f} CI=[{r['rho_ci_low']:.4f}, {r['rho_ci_high']:.4f}]  "
              f"agree={r['pair_agreement']:.4f} CI=[{r['pair_agreement_ci_low']:.4f}, "
              f"{r['pair_agreement_ci_high']:.4f}]")

    overall_point = float(sum(r["pair_agreement"] for r in results.values()) / len(results))
    overall_ci = percentile_ci(overall_agree_boot)
    overall = {
        "pair_agreement": overall_point,
        "ci_low": overall_ci[0], "ci_high": overall_ci[1],
        "exact_agreement_count": agreement_summary["overall"]["exact_agreement_count"],
        "aggregated_pair_count": agreement_summary["overall"]["aggregated_pair_count"],
        "definition": RULE_DEFINITION,
    }
    print(f"overall     agree={overall_point:.4f} CI=[{overall_ci[0]:.4f}, {overall_ci[1]:.4f}]  "
          f"({overall['exact_agreement_count']}/{overall['aggregated_pair_count']})")

    # ---- Frozen metadata ----
    metadata = {
        "resampling_unit": "image (all 7 methods, all 20 questionnaires and 21 pairs kept together)",
        "draws": "independent (B, 20) image-index matrix per dimension (drawn in dimension order); shared by rho and agreement within a dimension",
        "ci_method": "percentile bootstrap (2.5/97.5)",
        "seed": args.seed,
        "n_boot": args.n_boot,
        "ci_level": 1 - ALPHA,
        "rng": "numpy.random.default_rng (PCG64)",
        "agreement_source": "compute_agreement.run() (strict plurality, tied top votes -> tie)",
        "rho_source": "metric_bt.py BT win rates + Spearman (cross-checked)",
        "item_accounting": {
            "pairs_per_image": N_PAIRS,
            "images_per_dimension": N_IMAGES_PER_DIM,
            "items_per_dimension": N_IMAGES_PER_DIM * N_PAIRS,
            "dimensions": len(args.metrics),
            "items_total": N_IMAGES_PER_DIM * N_PAIRS * len(args.metrics),
            "image_clusters_total": N_IMAGES_PER_DIM * len(args.metrics),
            "votes_per_item": 20,
            "excluded_items": 0,
            "note": "2100 items are NOT independent bootstrap units; resampling is over image clusters.",
        },
    }
    payload = {"metadata": metadata, "results": results, "overall_pair_agreement": overall}
    json_path = args.output_dir / "bootstrap_ci.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    n_dims = len(args.metrics)
    items_per_dim = N_IMAGES_PER_DIM * N_PAIRS
    lines = [
        "# Image-clustered bootstrap CI (frozen)", "",
        f"- Resampling unit: **image**; B = {args.n_boot}; seed = **{args.seed}** (numpy PCG64)",
        f"- Independent (B, 20) draw per dimension ({n_dims} dimensions, drawn in fixed order); rho and agreement share it within a dimension",
        f"- CI: {1 - ALPHA:.0%} percentile (2.5/97.5)",
        "- rho: BT win rates, identical to `metric_bt.py`",
        "- Agreement: `compute_agreement.run()` strict plurality; tied top votes -> tie",
        f"- Item accounting: 21 unique method pairs × 20 images = **{items_per_dim} items per dimension**; "
        f"{n_dims} dimensions = **{items_per_dim * n_dims} items in total** "
        f"({overall['exact_agreement_count']} exact agreements, no item excluded, 20 votes per item)",
        f"- These {items_per_dim * n_dims} items are **not** independent bootstrap units: "
        f"resampling is over {N_IMAGES_PER_DIM * n_dims} image clusters "
        f"({N_IMAGES_PER_DIM} per dimension), each carrying its full set of 21 pairs and 20 questionnaires",
        "- Reproduce from the repository root: `python reproducibility/bootstrap_ci.py`",
        "", "| Dimension | rho | rho 95% CI | exact agreement | pair agreement | agreement 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for metric, r in results.items():
        lines.append(
            f"| {metric} | {r['rho']:.4f} | [{r['rho_ci_low']:.4f}, {r['rho_ci_high']:.4f}] | "
            f"{r['pair_agreement_count']}/{items_per_dim} | {r['pair_agreement']:.4f} | "
            f"[{r['pair_agreement_ci_low']:.4f}, {r['pair_agreement_ci_high']:.4f}] |"
        )
    lines += [
        f"| **Overall** | — | — | **{overall['exact_agreement_count']}/{items_per_dim * n_dims}** | "
        f"**{overall_point:.4f}** | **[{overall_ci[0]:.4f}, {overall_ci[1]:.4f}]** |", "",
        "Full settings (seed, B, CI method): `bootstrap_ci.json`.", "",
    ]
    md_path = args.output_dir / "bootstrap_ci_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {json_path}\nsaved: {md_path}")


if __name__ == "__main__":
    main()
