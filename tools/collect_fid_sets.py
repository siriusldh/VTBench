"""Collect the FID/KID image sets used in the paper (Tab. 1).

Reference set : all 2,896 person (model) images of the four test sets (CBC, FTF, HOC, CCP).
Generated set : all results of one method on the four test sets (methods without
                cross-category support have no CCP results; failed generations are absent).

Files from different test sets can share names (e.g. CBC and HOC both use 0001.png), so every
image is linked into a flat folder with a subset prefix. Then run:
    python eval_fid_kid.py --gt_folder <out>/real --pred_folder <out>/<method>

Usage:
    python tools/collect_fid_sets.py --data-root ./data \
        --result-dir ./data/results/<method> --method <method> --out ./outputs/fid_sets
"""
import argparse
import csv
import os
import shutil

# result sub-folder of each test set inside an extracted methods_result/<method>.zip
RESULT_SUBDIR = {"CBC": "complex", "FTF": "text", "HOC": "hand", "CCP": "cross"}


def link(src, dst, copy):
    if os.path.lexists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.getenv("VTBENCH_DATA_ROOT", "./data"))
    ap.add_argument("--result-dir", default=None, help="extracted <method>/ folder with complex/ text/ hand/ cross/")
    ap.add_argument("--method", default=None)
    ap.add_argument("--out", default="./outputs/fid_sets")
    ap.add_argument("--copy", action="store_true", help="copy files instead of creating symlinks")
    args = ap.parse_args()

    with open(os.path.join(args.data_root, "pairs.tsv"), newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f, delimiter="\t"))

    real_dir = os.path.join(args.out, "real")
    os.makedirs(real_dir, exist_ok=True)
    n_real, missing_real = 0, []
    for p in pairs:
        src = os.path.join(args.data_root, p["person_image"])
        if not os.path.exists(src):
            missing_real.append(p["person_image"])
            continue
        link(src, os.path.join(real_dir, f"{p['subset']}_{os.path.basename(src)}"), args.copy)
        n_real += 1
    print(f"reference set: {n_real}/{len(pairs)} person images -> {real_dir}")
    if missing_real:
        print(f"[warn] {len(missing_real)} person images missing (restore the link-only images first), "
              f"e.g. {missing_real[:3]}")

    if args.result_dir:
        method = args.method or os.path.basename(os.path.normpath(args.result_dir))
        pred_dir = os.path.join(args.out, method)
        os.makedirs(pred_dir, exist_ok=True)
        counts = {}
        for p in pairs:
            folder = os.path.join(args.result_dir, RESULT_SUBDIR[p["subset"]])
            stem = os.path.splitext(os.path.basename(p["person_image"]))[0]
            for ext in (".png", ".jpg", ".jpeg"):
                src = os.path.join(folder, stem + ext)
                if os.path.exists(src):
                    link(src, os.path.join(pred_dir, f"{p['subset']}_{stem}{ext}"), args.copy)
                    counts[p["subset"]] = counts.get(p["subset"], 0) + 1
                    break
        print(f"generated set ({method}): {sum(counts.values())} images {counts} -> {pred_dir}")


if __name__ == "__main__":
    main()
