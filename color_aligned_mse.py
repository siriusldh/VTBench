"""Color-aligned MSE for background preservation.

Unlike the plain background MSE in ``eval_complex_bg.py``, this metric first
fits a global affine RGB mapping from the generated image to the reference
image using only background pixels, then computes MSE on that background.
Global brightness or color shifts are therefore removed before the structural
comparison. Lower is better.

The metric functions take float32 RGB arrays in [0, 1] and a boolean mask in
which True marks background pixels.

The CLI evaluates whole directories, like ``eval_complex_bg.py``: it matches
images by filename in the original/generated folders and averages the scores.
The original foreground mask defines the background; an optional generated
mask can be supplied to exclude foreground on both sides.

Example:
    python color_aligned_mse.py \
        --img-path-be /path/to/original/images \
        --mask-path-be /path/to/original/masks \
        --img-path-af /path/to/generated/images \
        --mask-path-af /path/to/generated/masks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an RGB image as float32 in [0, 1], optionally resizing it."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if target_size is not None and image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def load_background_mask(path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """Load a foreground mask and return a boolean background mask (True = background)."""
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.NEAREST)
        foreground = np.asarray(image, dtype=np.uint8) > 127
    background = ~foreground
    if not np.any(background):
        raise ValueError(f"Mask contains no background pixels: {path}")
    return background


def color_align_to_reference(
    generated: np.ndarray, reference: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """Fit a global affine RGB mapping from generated to reference on background pixels."""
    source_pixels = generated[background]
    target_pixels = reference[background]
    if len(source_pixels) < 4:
        raise ValueError("Not enough background pixels for color alignment")

    design = np.concatenate(
        (source_pixels.astype(np.float64), np.ones((len(source_pixels), 1))), axis=1
    )
    coefficients, *_ = np.linalg.lstsq(design, target_pixels.astype(np.float64), rcond=None)
    aligned = generated.astype(np.float64) @ coefficients[:3] + coefficients[3]
    return np.clip(aligned, 0.0, 1.0).astype(np.float32)


def color_aligned_mse(
    generated: np.ndarray, reference: np.ndarray, background: np.ndarray
) -> float:
    """Color-aligned background MSE; lower is better."""
    aligned = color_align_to_reference(generated, reference, background)
    return float(np.mean(np.square(aligned[background] - reference[background])))


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def evaluate_directory(
    img_path_be: Path, mask_path_be: Path, img_path_af: Path,
    mask_path_af: Path | None = None,
) -> list[dict]:
    """Score every matching image pair in two directories."""
    img_path_be, mask_path_be, img_path_af = (
        Path(img_path_be), Path(mask_path_be), Path(img_path_af)
    )
    mask_path_af = Path(mask_path_af) if mask_path_af is not None else None

    results = []
    reference_paths = sorted(
        p for p in img_path_be.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    for reference_path in reference_paths:
        generated_path = img_path_af / reference_path.name
        mask_path = mask_path_be / reference_path.name
        missing = [p for p in (generated_path, mask_path) if not p.is_file()]
        if missing:
            print(f"[skip] {reference_path.name}: missing {', '.join(str(p) for p in missing)}")
            continue

        reference = load_rgb(reference_path)
        target_size = (reference.shape[1], reference.shape[0])
        generated = load_rgb(generated_path, target_size)
        background = load_background_mask(mask_path, target_size)

        if mask_path_af is not None:
            generated_mask_path = mask_path_af / reference_path.name
            if generated_mask_path.is_file():
                background &= load_background_mask(generated_mask_path, target_size)
            else:
                print(f"[warn] {reference_path.name}: generated mask not found, "
                      "using the original mask only")

        results.append({
            "image": reference_path.name,
            "color_aligned_mse": color_aligned_mse(generated, reference, background),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch color-aligned background MSE evaluation."
    )
    parser.add_argument("--img-path-be", default="./data/benchmark_complex_bg/image",
                        help="Original images before try-on.")
    parser.add_argument("--mask-path-be", default="./data/benchmark_complex_bg/image_bg_seg/mask",
                        help="Foreground masks of the original images.")
    parser.add_argument("--img-path-af", default="./data/results/complex",
                        help="Generated images after try-on.")
    parser.add_argument("--mask-path-af", default=None,
                        help="Optional foreground masks of the generated images; "
                             "excludes foreground on both sides when provided.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON file for per-image scores.")
    args = parser.parse_args()

    results = evaluate_directory(
        args.img_path_be, args.mask_path_be, args.img_path_af, args.mask_path_af
    )
    if not results:
        raise SystemExit("No valid image pairs found; check the input directories")

    for item in results:
        print(f"{item['image']}: {item['color_aligned_mse']:.6f}")
    mean = float(np.mean([item["color_aligned_mse"] for item in results]))
    print(f"Average Color-aligned MSE: {mean:.6f} ({len(results)} images)")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "per_image_scores": results,
            "average_color_aligned_mse": mean,
            "image_count": len(results),
        }, indent=2), encoding="utf-8")
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
