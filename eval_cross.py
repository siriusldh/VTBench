"""Cross-Category Plausibility (VLM_c) evaluation on the CCP test set.

For every CCP pair (person image P1, target garment P2, generated result P3), a
three-image composite is judged by Qwen-VL-Max as "Reasonable" or
"Unreasonable". VLM_c is the percentage of the 420 CCP pairs judged
"Reasonable"; a missing result counts as "Unreasonable".

By default the main protocol prompt (paper Tab. 1 / Appendix A) is used.
`--strict` switches to the stricter identity-aware variant reported in the
appendix.

Example:
    python eval_cross.py \
        --img-paths /path/to/<method>/cross \
        --img-base-path /path/to/VTBench/benchmark_crossvton \
        --output ./outputs/<method>_vlm_c.json
"""
import argparse
import base64
import json
import os
import time

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

PROMPT_MAIN = (
    "I used a virtual try-on algorithm to replace the clothing of the model in the left image with the "
    "clothing in the middle image, resulting in the output on the right. (When replacing clothing, it is "
    "considered reasonable as long as the overall appearance of the model in the right image makes sense and "
    "remains consistent with the middle clothing in terms of type and style. For example, when replacing a top "
    "with a dress, the upper body part must remain consistent, while the skirt part can be adjusted freely; if "
    "replacing pants with a skirt, as long as the skirt matches the middle image, other parts can be "
    "appropriately adjusted. If the clothing in the output image is essentially the same as the input model's "
    "original clothing, the output is considered unreasonable). You only need to judge whether it is "
    "reasonable. If reasonable, reply ONLY with \"Reasonable\"; if unreasonable, reply ONLY with \"Unreasonable\"."
)

PROMPT_STRICT = (
    "The input is a horizontal three-image composite: P1 is the original model image, P2 is the target garment "
    "image, and P3 is the cross-category try-on result. Judge strictly whether P3 is reasonable. The garment in "
    "P3 that corresponds to P2 must be the very same garment as P2: not only the garment type, but also design "
    "details such as color, pattern, texture, logo, neckline, sleeve shape, and hem must match P2; mere "
    "similarity in style or type without an identical appearance counts as a mismatch. For cross-category "
    "replacement, only the garment part corresponding to P2 must be identical (e.g., when a dress replaces a top, "
    "the top portion must match P2); the remaining parts may be adjusted sensibly. The way the garment is worn "
    "must be reasonable: no obvious remnants of P1's original clothing, no mismatched top/bottom combinations, "
    "and no clothing-body interpenetration. The transitions at the neckline, cuffs, waist, and hem must be "
    "natural, with no visible stitching, misalignment, or blending artifacts. If P3 is essentially still P1's "
    "original garment, it is also unreasonable. Failure to satisfy any single criterion means unreasonable. "
    "Reply ONLY with \"Reasonable\" or \"Unreasonable\"."
)

PAIR_FILES = ["test_pairs_ds2lower.txt", "test_pairs_ds2up.txt",
              "test_pairs_lower2ds.txt", "test_pairs_up2ds.txt"]


def parse_args():
    p = argparse.ArgumentParser(description="Cross-category plausibility (VLM_c) evaluation.")
    p.add_argument("--img-paths", default="./data/results/cross",
                   help="Directory with generated CCP results, named by the person image (<method>/cross/).")
    p.add_argument("--img-base-path", default="./data/benchmark_crossvton",
                   help="CCP root containing test_pairs_*.txt (<person> <cloth>, relative paths).")
    p.add_argument("--save-base", default="./outputs/cross_concat",
                   help="Directory for the three-image composites sent to the VLM.")
    p.add_argument("--output", default="./outputs/vlm_c.json",
                   help="JSON file with per-sample judgments and the final VLM_c (resumable).")
    p.add_argument("--strict", action="store_true",
                   help="Use the stricter identity-aware variant instead of the main protocol.")
    p.add_argument("--model", default="qwen-vl-max")
    return p.parse_args()


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_pairs(root):
    pairs = []
    for name in PAIR_FILES:
        with open(os.path.join(root, name), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    return pairs


def compose(person, cloth, result, out_path):
    ims = [Image.open(p).convert("RGB").resize((384, 512)) for p in (person, cloth, result)]
    canvas = Image.new("RGB", (384 * 3, 512))
    for i, im in enumerate(ims):
        canvas.paste(im, (384 * i, 0))
    canvas.save(out_path)


def judge(client, model, prompt, image_path, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=10,
            )
            text = resp.choices[0].message.content.strip()
            return "Unreasonable" if "Unreasonable" in text else ("Reasonable" if "Reasonable" in text else text)
        except Exception as e:  # network / rate limit
            print(f"[retry {attempt + 1}] {os.path.basename(image_path)}: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def main():
    args = parse_args()
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    prompt = PROMPT_STRICT if args.strict else PROMPT_MAIN
    os.makedirs(args.save_base, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    state = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            state = json.load(f).get("per_sample", {})

    pairs = load_pairs(args.img_base_path)
    for person_rel, cloth_rel in tqdm(pairs, desc="VLM_c"):
        sample = os.path.splitext(os.path.basename(person_rel))[0]
        if state.get(sample) in ("Reasonable", "Unreasonable", "missing"):
            continue
        result = os.path.join(args.img_paths, os.path.basename(person_rel))
        if not os.path.exists(result):
            state[sample] = "missing"  # counted as Unreasonable
            continue
        concat = os.path.join(args.save_base, os.path.basename(person_rel))
        compose(os.path.join(args.img_base_path, person_rel),
                os.path.join(args.img_base_path, cloth_rel), result, concat)
        label = judge(client, args.model, prompt, concat)
        if label is not None:
            state[sample] = label
        time.sleep(0.5)

    judged = [state.get(os.path.splitext(os.path.basename(p))[0]) for p, _ in pairs]
    reasonable = sum(v == "Reasonable" for v in judged)
    summary = {
        "protocol": "strict" if args.strict else "main",
        "num_pairs": len(pairs),
        "reasonable": reasonable,
        "missing": sum(v == "missing" for v in judged),
        "unjudged": sum(v not in ("Reasonable", "Unreasonable", "missing") for v in judged),
        "VLM_c": 100.0 * reasonable / len(pairs),
        "per_sample": state,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"VLM_c = {summary['VLM_c']:.4f} ({reasonable}/{len(pairs)}; missing {summary['missing']}, "
          f"unjudged {summary['unjudged']})")


if __name__ == "__main__":
    main()
