"""Crop the garment region of generated FTF results for CLIP-I and DINO (T_score).

`json/ftf_garment_bbox.json` gives, for every FTF person image, the garment bounding
box [x0, y0, x1, y1] in the 768x1024 person-image frame. The box is rescaled to the
size of each generated result, and the crop is saved under the same filename.

Usage:
    python tools/crop_ftf_garment.py \
        --result-dir ./data/results/<method>/text \
        --output-dir ./data/results/<method>/text_garment_crops
"""
import argparse
import json
import os

from PIL import Image

REF_W, REF_H = 768, 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bbox-json", default=os.path.join(os.path.dirname(__file__), "..", "json", "ftf_garment_bbox.json"))
    args = ap.parse_args()

    with open(args.bbox_json, encoding="utf-8") as f:
        boxes = {os.path.splitext(r["image"])[0]: r["bbox"] for r in json.load(f)}
    os.makedirs(args.output_dir, exist_ok=True)
    done = missing = 0
    for name in sorted(os.listdir(args.result_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg") or stem not in boxes:
            continue
        im = Image.open(os.path.join(args.result_dir, name)).convert("RGB")
        sx, sy = im.width / REF_W, im.height / REF_H
        x0, y0, x1, y1 = boxes[stem]
        im.crop((round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))).save(
            os.path.join(args.output_dir, name))
        done += 1
    missing = len(boxes) - done
    print(f"Cropped {done} results; {missing} FTF samples without a result in {args.result_dir}")


if __name__ == "__main__":
    main()
