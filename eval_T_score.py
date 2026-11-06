"""
Virtual Try-On (VTON) Texture Dimension Evaluation and Scoring Integration Code (V3)
Included Metrics: CLIP-I, DINO, CLIP-T, VLM (Qwen-VL-Max)
Comprehensive Scoring: each of the 4 method-level metric scores is min-max normalized with the frozen
VTBench-v1 calibration anchors (minimum/maximum over the original 19 benchmarked methods), clipped to [0, 1],
and the four normalized scores are averaged with equal weight. The anchors are never updated, so adding a
method does not change existing scores.
"""
import os
import json
import base64
import time
import concurrent.futures
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from tqdm import tqdm

import torch
import clip

import torch
import clip
from PIL import Image
from scipy import spatial
from torchvision.transforms import transforms
from openai import OpenAI

# ================= 1. Historical Benchmark Data Entry =================
RAW_DATA = [
    {"Model": "ITA", "VLM_t": 0.8341, "CLIP_I": 0.8129, "DINO": 0.716, "CLIP_T": 0.3041},
    {"Model": "IMAGDressing", "VLM_t": 0.7857, "CLIP_I": 0.7781, "DINO": 0.6411, "CLIP_T": 0.3099},
    {"Model": "2021 PF-AFN", "VLM_t": 0.684, "CLIP_I": 0.7723, "DINO": 0.6551, "CLIP_T": 0.3195},
    {"Model": "2022 FS-VTON", "VLM_t": 0.6871, "CLIP_I": 0.7766, "DINO": 0.6681, "CLIP_T": 0.3117},
    {"Model": "2022 DA-FLOW", "VLM_t": 0.5617, "CLIP_I": 0.7466, "DINO": 0.5824, "CLIP_T": 0.2929},
    {"Model": "2022 HR-VITON", "VLM_t": 0.6671, "CLIP_I": 0.7781, "DINO": 0.6321, "CLIP_T": 0.3037},
    {"Model": "2024 SD-VITON", "VLM_t": 0.7464, "CLIP_I": 0.8194, "DINO": 0.7293, "CLIP_T": 0.3112},
    {"Model": "2023 LaDI-VTON", "VLM_t": 0.685, "CLIP_I": 0.7239, "DINO": 0.6483, "CLIP_T": 0.2771},
    {"Model": "2024 StableVTON", "VLM_t": 0.6868, "CLIP_I": 0.8017, "DINO": 0.7011, "CLIP_T": 0.3015},
    {"Model": "2024 TPD", "VLM_t": 0.856, "CLIP_I": 0.8197, "DINO": 0.6935, "CLIP_T": 0.318},
    {"Model": "2024 CAT-DM", "VLM_t": 0.6964, "CLIP_I": 0.7211, "DINO": 0.6515, "CLIP_T": 0.2725},
    {"Model": "2024 IDM-VTON", "VLM_t": 0.8439, "CLIP_I": 0.839, "DINO": 0.7469, "CLIP_T": 0.3165},
    {"Model": "2024 OOTD", "VLM_t": 0.6021, "CLIP_I": 0.7319, "DINO": 0.4905, "CLIP_T": 0.2699},
    {"Model": "2024 CatVTON", "VLM_t": 0.865, "CLIP_I": 0.8425, "DINO": 0.7168, "CLIP_T": 0.3175},
    {"Model": "2024 VTON-HandFit", "VLM_t": 0.8459, "CLIP_I": 0.8443, "DINO": 0.7028, "CLIP_T": 0.318},
    {"Model": "2024 FitDit", "VLM_t": 0.9065, "CLIP_I": 0.8705, "DINO": 0.766, "CLIP_T": 0.3267},
    {"Model": "2025 CrossVTON", "VLM_t": 0.895, "CLIP_I": 0.8611, "DINO": 0.7418, "CLIP_T": 0.3255},
    {"Model": "2025 Leffa", "VLM_t": 0.8919, "CLIP_I": 0.8607, "DINO": 0.7377, "CLIP_T": 0.3259},
    {"Model": "2025 Qwen-Edit", "VLM_t": 0.9212, "CLIP_I": 0.8699, "DINO": 0.7539, "CLIP_T": 0.3323},
    {"Model": "2025 Nano Banana", "VLM_t": 0.8836, "CLIP_I": 0.8439, "DINO": 0.7146, "CLIP_T": 0.3187},
    {"Model": "2025 Seedream", "VLM_t": 0.9162, "CLIP_I": 0.8592, "DINO": 0.7226, "CLIP_T": 0.3352},
]

# Frozen VTBench-v1 calibration anchors (paper appendix): method-level min/max over the
# original 19 benchmarked methods. Do NOT recompute these when adding methods.
ANCHORS = {
    "CLIP_I": (0.7211, 0.8705),
    "DINO":   (0.4905, 0.7660),
    "CLIP_T": (0.2699, 0.3352),
    "VLM_t":  (0.5617, 0.9212),
}

METRICS_MAP = {"clip_i": "CLIP_I", "dino": "DINO", "clip_t": "CLIP_T", "vlm": "VLM_t"}
METRICS = list(METRICS_MAP.values())

# ================= 2. Evaluation Functions =================

def eval_clip_i(args, image_pairs, model, transform, metric='clip_i'):
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            if metric == 'clip_i':
                image_features = model.encode_image(image_input).detach().cpu().float()
            elif metric == 'dino':
                image_features = model(image_input).detach().cpu().float()
        return image_features

    eval_score = 0
    for img_pair in tqdm(image_pairs, desc=f"Evaluating {metric.upper()}"):
        generated_features = encode(Image.open(img_pair[0]).convert('RGB'), model, transform)
        gt_features = encode(Image.open(img_pair[1]).convert('RGB'), model, transform)
        
        similarity = 1 - spatial.distance.cosine(
            generated_features.view(generated_features.shape[1]),
            gt_features.view(gt_features.shape[1])
        )
        eval_score += similarity
        
    return eval_score / len(image_pairs) if image_pairs else 0.0

def eval_clip_t_images(args, image_path_caption_pairs, model, transform, caption_dict):
    clip_t_score = 0
    for img_path, base_name in tqdm(image_path_caption_pairs, desc="Evaluating CLIP-T"):
        image = Image.open(img_path).convert("RGB")
        image_features = model.encode_image(transform(image).unsqueeze(0).to(args.device)).detach().cpu().float()
        
        caption = caption_dict.get(base_name, "")
        text_tokens = clip.tokenize(caption, truncate=True).to(args.device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).detach().cpu().float()
            
        similarity = 1 - spatial.distance.cosine(
            image_features.view(image_features.shape[1]),
            text_features.view(text_features.shape[1])
        )
        clip_t_score += similarity
        
    return clip_t_score / len(image_path_caption_pairs) if image_path_caption_pairs else 0.0

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def prepare_image(model_img_path, cloth_img_path, res_img_path, save_path):
    model_image = Image.open(model_img_path).resize((384, 512))
    cloth_image = Image.open(cloth_img_path).resize((384, 512))
    res_image = Image.open(res_img_path).resize((384, 512))
    
    new_image = Image.new('RGB', (384*3, 512))
    new_image.paste(model_image, (0,0))
    new_image.paste(cloth_image, (384,0))
    new_image.paste(res_image, (384*2,0))
    new_image.save(save_path)
    return save_path

# VLM_t prompt (paper appendix A, "Texture Fidelity (VLM_t)").
VLM_T_PROMPT = (
    "I used a virtual try-on algorithm to replace the clothing of the model in the left image (P1) with the "
    "clothing in the middle image (P2), resulting in the output on the right (P3). Please evaluate the try-on "
    "effect based on the following criteria:\n"
    "(1) Is the text on the clothing in P3 consistent with P2? Is it clear, complete, and recognizable?\n"
    "(2) Is the color of the clothing in P3 consistent with P2?\n"
    "(3) Are the texture details of the clothing in P3 consistent with P2?\n"
    "(4) Are the clothing wrinkles in P3 natural?\n"
    "Finally, please provide a comprehensive score (1-100), where a higher score indicates a better try-on "
    "effect, and a lower score indicates a poorer effect. Please output only a numerical score without any "
    "other text description."
)


def query_score_qwen(base64_image, client):
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content":[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": VLM_T_PROMPT}
                ]
            }
        ]
    )
    content = completion.choices[0].message.content
    try:
        return float(content.strip())
    except:
        return None

def process_one_vlm(pair_info, client):
    model_img_path, cloth_img_path, gen_whole_path, save_path = pair_info
    try:
        out_image_path = prepare_image(model_img_path, cloth_img_path, gen_whole_path, save_path)
        res_image_base64 = encode_image(out_image_path)
        score = query_score_qwen(res_image_base64, client)
        time.sleep(0.05)
        return score
    except Exception as e:
        print(f"[ERROR VLM] {e}")
        return None

def eval_vlm_texture(vlm_pairs):
    if not vlm_pairs: return 0.0
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    all_scores =[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one_vlm, pair, client): pair for pair in vlm_pairs}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(vlm_pairs), desc="Evaluating VLM"):
            score = f.result()
            if score is not None: all_scores.append(score)
    return (sum(all_scores) / len(all_scores)) / 100.0 if all_scores else 0.0

# ================= 4. Data Loading =================

def load_all_data(args):
    pairs_crop_gt, pairs_whole_text, pairs_vlm = [], [], []
    if not (args.generated_crop_path and args.generated_whole_path):
        return pairs_crop_gt, pairs_whole_text, pairs_vlm

    # test_pairs.txt: "<person> <cloth>" per line; results are named after the person image.
    pair_map = {}
    with open(args.txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            pair_map[os.path.basename(parts[0])] = (parts[0], parts[1])

    def lookup(img_name):
        # tolerate a different result extension (e.g. .jpg result for a .png person image)
        if img_name in pair_map:
            return pair_map[img_name]
        stem = os.path.splitext(img_name)[0]
        for ext in ('.png', '.jpg', '.jpeg'):
            if stem + ext in pair_map:
                return pair_map[stem + ext]
        return None

    gen_crop_list = sorted(f for f in os.listdir(args.generated_crop_path) if f.endswith(('.png', '.jpg')))
    gen_whole_list = sorted(f for f in os.listdir(args.generated_whole_path) if f.endswith(('.png', '.jpg')))
    unmatched, missing_inputs = [], []

    for img_name in gen_crop_list:
        pair = lookup(img_name)
        if pair is None:
            unmatched.append(img_name); continue
        cloth_img_path = os.path.join(args.base_img_path, 'cloth', pair[1])
        if os.path.exists(cloth_img_path):
            pairs_crop_gt.append((os.path.join(args.generated_crop_path, img_name), cloth_img_path))
        else:
            missing_inputs.append(cloth_img_path)

    pairs_whole_text = [(os.path.join(args.generated_whole_path, f), f) for f in gen_whole_list]

    if not os.path.exists(args.vlm_save_base): os.makedirs(args.vlm_save_base)
    for img_name in gen_whole_list:
        pair = lookup(img_name)
        if pair is None:
            unmatched.append(img_name); continue
        model_img_path = os.path.join(args.base_img_path, 'image', pair[0])
        cloth_img_path = os.path.join(args.base_img_path, 'cloth', pair[1])
        if os.path.exists(model_img_path) and os.path.exists(cloth_img_path):
            pairs_vlm.append((model_img_path, cloth_img_path, os.path.join(args.generated_whole_path, img_name), os.path.join(args.vlm_save_base, img_name)))
        else:
            missing_inputs.extend(p for p in (model_img_path, cloth_img_path) if not os.path.exists(p))

    print(f"CLIP-I/DINO pairs: {len(pairs_crop_gt)}, CLIP-T images: {len(pairs_whole_text)}, VLM_t pairs: {len(pairs_vlm)}")
    if unmatched:
        print(f"[warn] {len(unmatched)} result files are not listed in {args.txt_path}, e.g. {unmatched[:3]}")
    if missing_inputs:
        print(f"[warn] {len(set(missing_inputs))} input images are missing (restore the link-only images first), "
              f"e.g. {sorted(set(missing_inputs))[:3]}")
    return pairs_crop_gt, pairs_whole_text, pairs_vlm

# ================= 5. Main Control =================

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--generated_crop_path', type=str, default=None)
    parser.add_argument('--generated_whole_path', type=str, default=None)
    parser.add_argument('--base_img_path', type=str, default='./data/benchmark_texture')
    parser.add_argument('--txt_path', type=str, default='./data/benchmark_texture/test_pairs.txt')
    parser.add_argument('--caption_path', type=str, default='./json/ftf_descriptions.json')
    parser.add_argument('--vlm_save_base', type=str, default='vlm_stitch_cache')
    parser.add_argument('--model_name_override', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='results/t_score',
                        help='T_score leaderboard output directory (kept separate '
                             'from the overall R_avg ranking in results/ranking.json)')
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluated_scores = {}
    model_name = args.model_name_override or "Current_Model"

    if args.generated_crop_path and args.generated_whole_path:
        print(f">>> Evaluating model: {model_name}")
        pairs_crop_gt, pairs_whole_text, pairs_vlm = load_all_data(args)
        
        caption_dict = {}
        if os.path.exists(args.caption_path):
            with open(args.caption_path, 'r') as f:
                caption_dict = {os.path.basename(item.get("image") or item["image_path"]): item["text"] for item in json.load(f)}

        evaluated_scores['VLM_t'] = eval_vlm_texture(pairs_vlm)
        
        clip_model, clip_transform = clip.load("ViT-B/32", args.device)
        evaluated_scores['CLIP_I'] = float(eval_clip_i(args, pairs_crop_gt, clip_model, clip_transform, 'clip_i'))
        evaluated_scores['CLIP_T'] = float(eval_clip_t_images(args, pairs_whole_text, clip_model, clip_transform, caption_dict))
        
        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').eval().to(args.device)
        dino_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        evaluated_scores['DINO'] = float(eval_clip_i(args, pairs_crop_gt, dino_model, dino_transform, 'dino'))
        
        evaluated_scores['Model'] = model_name
        RAW_DATA.append(evaluated_scores)
    else:
        print(">>> No paths provided. Calculating leaderboard using RAW_DATA only.")

    # Normalization with frozen VTBench-v1 anchors (values outside the anchors are clipped and flagged)
    results_rank = []
    for item in RAW_DATA:
        norm_scores, flags = [], []
        for m in METRICS:
            lo, hi = ANCHORS[m]
            z = (item[m] - lo) / (hi - lo)
            if z < 0 or z > 1:
                flags.append(m)
            norm_scores.append(min(1.0, max(0.0, z)))
        entry = {"Model": item["Model"], "Score": sum(norm_scores) / len(norm_scores)}
        if flags:
            entry["out_of_anchor_range"] = flags
            print(f"[flag] {item['Model']}: {', '.join(flags)} outside VTBench-v1 anchors (clipped in T_score)")
        results_rank.append(entry)

    results_rank.sort(key=lambda x: x["Score"], reverse=True)
    
    print("\n" + "="*30 + " LEADERBOARD " + "="*30)
    for i, res in enumerate(results_rank):
        tag = " <--- NEW" if res['Model'] == model_name else ""
        print(f"{i+1:<4} {res['Model']:<25} {res['Score']:.4f}{tag}")

    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, "ranking.json"), 'w') as f:
        json.dump({"rank": results_rank, "current": evaluated_scores}, f, indent=4)
