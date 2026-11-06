#!/usr/bin/env bash
# Generate background segmentation masks for the complex-background evaluation.
#
# Usage:
#   bash demo_one_key.sh /path/to/images /path/to/masks
# or:
#   INPUT_DIR=/path/to/images RESULT_DIR=/path/to/masks bash demo_one_key.sh
#
# Defaults assume VTBENCH_DATA_ROOT (or ./data) holds the benchmark data.
set -euo pipefail

DATA_ROOT="${VTBENCH_DATA_ROOT:-./data}"
INPUT_DIR="${1:-${INPUT_DIR:-$DATA_ROOT/results/complex}}"
RESULT_DIR="${2:-${RESULT_DIR:-$DATA_ROOT/results/complex_seg}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/infer_hrocr_one_key.py" \
    --input_dir "$INPUT_DIR" \
    --result_dir "$RESULT_DIR" \
    --output_comp --background_bgr 255 255 255 --output_mask --guide_filter
