import os
import json
import base64
import re
import time
import io
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ================= 1. API Configuration =================
api_key = os.getenv("DASHSCOPE_API_KEY", "")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ================= 2. Path Configuration =================
# All dataset/result locations are resolved from VTBENCH_DATA_ROOT (default ./data).
# Override it instead of editing absolute paths in the source.

DATA_ROOT = os.getenv("VTBENCH_DATA_ROOT", "./data")
BASE_RESULT_DIR = os.getenv("VTBENCH_RESULT_DIR", os.path.join(DATA_ROOT, "results"))  # extracted <method>/ archive
CROSS_VTON_ROOT = os.path.join(DATA_ROOT, "benchmark_crossvton")

DATASET_CONFIG = {
    'complex_bg': {
        'type': 'normal',
        'result_dir': os.path.join(BASE_RESULT_DIR, "complex"),
        'pairs_file': os.path.join(DATA_ROOT, "benchmark_complex_bg", "test_pairs.txt"),
        'model_dir': os.path.join(DATA_ROOT, "benchmark_complex_bg", "image"),
        'cloth_dir': os.path.join(DATA_ROOT, "benchmark_complex_bg", "cloth")
    },
    'cross': {
        'type': 'relative',
        'result_dir': os.path.join(BASE_RESULT_DIR, "cross"),
        'pairs_file': [
            os.path.join(CROSS_VTON_ROOT, 'test_pairs_ds2lower.txt'),
            os.path.join(CROSS_VTON_ROOT, 'test_pairs_ds2up.txt'),
            os.path.join(CROSS_VTON_ROOT, 'test_pairs_lower2ds.txt'),
            os.path.join(CROSS_VTON_ROOT, 'test_pairs_up2ds.txt')
        ],
        'root_dir': CROSS_VTON_ROOT
    },
    'hand': {
        'type': 'normal',
        'result_dir': os.path.join(BASE_RESULT_DIR, "hand"),
        'pairs_file': os.path.join(DATA_ROOT, "benchmark_hand", "test_pairs.txt"),
        'model_dir': os.path.join(DATA_ROOT, "benchmark_hand", "image"),
        'cloth_dir': os.path.join(DATA_ROOT, "benchmark_hand", "cloth")
    },
    'sum': {
        'type': 'normal',
        'result_dir': os.path.join(BASE_RESULT_DIR, "text"),
        'pairs_file': os.path.join(DATA_ROOT, "benchmark_texture", "test_pairs.txt"),
        'model_dir': os.path.join(DATA_ROOT, "benchmark_texture", "image"),
        'cloth_dir': os.path.join(DATA_ROOT, "benchmark_texture", "cloth")
    }
}

SAVE_JSON_PATH = os.path.join(BASE_RESULT_DIR, 'aesthetic_qwen_score_final.json')
MAX_WORKERS = 8
SAVE_INTERVAL = 20

PAIRS_MAPPING = {}

# ================= 3. Pair Parsing Logic =================

def load_pairs_mapping():
    print("Loading pair mapping list...")
    global PAIRS_MAPPING
    
    for key, config in DATASET_CONFIG.items():
        PAIRS_MAPPING[key] = {}
        files = config['pairs_file']
        if isinstance(files, str): files = [files]
            
        for fpath in files:
            if not os.path.exists(fpath):
                print(f"[Warning] File does not exist: {fpath}")
                continue
            
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    
                    model_str = parts[0]
                    cloth_str = parts[1]
                    result_filename = os.path.basename(model_str)
                    
                    PAIRS_MAPPING[key][result_filename] = {
                        'model_info': model_str,
                        'cloth_info': cloth_str
                    }
        print(f"[{key}] Loaded pairs: {len(PAIRS_MAPPING[key])} groups")

# ================= 4. Image Stitching and Saving Logic =================

def get_source_paths(dataset_key, filename):
    config = DATASET_CONFIG[dataset_key]
    mapping = PAIRS_MAPPING[dataset_key].get(filename)
    
    if not mapping:
        return None, None
    
    model_info = mapping['model_info']
    cloth_info = mapping['cloth_info']
    
    if config['type'] == 'relative':
        model_path = os.path.join(config['root_dir'], model_info)
        cloth_path = os.path.join(config['root_dir'], cloth_info)
    else:
        model_path = os.path.join(config['model_dir'], model_info)
        cloth_path = os.path.join(config['cloth_dir'], cloth_info)
        
    return model_path, cloth_path

def stitch_and_save_image(dataset_key, filename):
    config = DATASET_CONFIG[dataset_key]
    result_dir = config['result_dir']
    
    save_dir_path = result_dir + "_qwen"
    os.makedirs(save_dir_path, exist_ok=True)
    
    save_file_path = os.path.join(save_dir_path, filename)

    # === Cache Check ===
    if os.path.exists(save_file_path):
        try:
            return Image.open(save_file_path).convert('RGB')
        except Exception:
            pass 

    # === Stitching Logic ===
    result_img_path = os.path.join(result_dir, filename)
    if not os.path.exists(result_img_path): return None
    try:
        res_img = Image.open(result_img_path).convert('RGB')
    except:
        return None

    model_path, cloth_path = get_source_paths(dataset_key, filename)

    TARGET_H = 512
    def resize_h(img):
        w, h = img.size
        new_w = int(w * (TARGET_H / h))
        return img.resize((new_w, TARGET_H))

    res_img = resize_h(res_img)
    final_img = res_img
    
    if model_path and cloth_path and os.path.exists(model_path) and os.path.exists(cloth_path):
        try:
            model_img = resize_h(Image.open(model_path).convert('RGB'))
            cloth_img = resize_h(Image.open(cloth_path).convert('RGB'))
            
            total_w = model_img.width + cloth_img.width + res_img.width
            canvas = Image.new('RGB', (total_w, TARGET_H))
            canvas.paste(model_img, (0, 0))
            canvas.paste(cloth_img, (model_img.width, 0))
            canvas.paste(res_img, (model_img.width + cloth_img.width, 0))
            
            final_img = canvas
        except Exception as e:
            print(f"Stitching failed for {filename}: {e}")
    
    try:
        final_img.save(save_file_path, quality=95)
    except Exception as e:
        print(f"Failed to save stitched image: {e}")

    return final_img

# ================= 5. API Call =================

def image_to_base64(image_obj):
    buffered = io.BytesIO()
    image_obj.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def query_score(image_obj, max_retries=3):
    if image_obj is None: return 0
    b64_img = image_to_base64(image_obj)
    
    prompt = """Task: Aesthetic Quality Scoring of Virtual Try-On Generations

You are presented with a horizontally concatenated image of three sub-images.
Evaluate ONLY the rightmost virtual try-on result.

Act as a professional expert in aesthetics and image quality assessment, and
score only the aesthetic quality and visual naturalness of that generated image
itself. Scoring range: 1.0 to 100.0 (decimals allowed, e.g., 85.5).

Evaluation dimensions (in order of importance):
1. Image Quality: clarity, detail rendition, no blur, no obvious artifacts,
   and natural edges.
2. Human Naturalness: natural and coherent pose, skin tone and body shape;
   no deformities or distortions in the face, hands, or limbs.
3. Clothing Visual Appeal: the garment is complete and fits the body, with
   natural cut, folds, drape and boundaries, coordinated with the human pose.
4. Overall Visual Harmony: the person, clothing, background, composition and
   lighting form a coherent whole, with no visible stitching or generation
   artifacts.

Score strictly according to the aesthetic criteria above.

Output format:
Output ONLY a single number (e.g., 90.5), with no explanations, prefixes,
or suffixes."""

    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{
                    "role": "user", 
                    "content":[
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.01,
                max_tokens=10
            )
            content = resp.choices[0].message.content.strip()
            match = re.search(r'\d+(\.\d+)?', content)
            if match:
                return float(match.group())
        except Exception as e:
            if "429" in str(e): time.sleep(2)
            else: print(f"API Error: {e}")
        time.sleep(1)
    return 0

# ================= 6. Main Logic =================

def process_task(dataset_key, filename):
    img_obj = stitch_and_save_image(dataset_key, filename)
    score = query_score(img_obj)
    return {
        "dataset": dataset_key,
        "filename": filename,
        "score": score
    }

def calculate_macro_average(stats_acc):
    """Calculate macro average: (Avg1 + Avg2 + Avg3 + Avg4) / 4"""
    dataset_averages =[]
    
    for k, v in stats_acc.items():
        if v["count"] > 0:
            avg = v["sum"] / v["count"]
            dataset_averages.append(avg)
            
    if len(dataset_averages) > 0:
        return sum(dataset_averages) / len(dataset_averages)
    else:
        return 0

def save_json(filepath, details, stats_acc):
    final_stats = {}
    for k, v in stats_acc.items():
        avg = v["sum"] / v["count"] if v["count"] else 0
        final_stats[k] = {"average": round(avg, 2), "count": v["count"]}
    
    # Macro average
    macro_avg = calculate_macro_average(stats_acc)
    final_stats["global_average"] = round(macro_avg, 2)
    
    data = {"stats": final_stats, "details": details}
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save JSON: {e}")

def load_existing_results(filepath):
    """Load existing JSON data to resume from a breakpoint"""
    if os.path.exists(filepath):
        try:
            print(f"Historical records detected: {filepath}, loading...")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("details", {})
        except Exception as e:
            print(f"Failed to read historical file: {e}, will start over.")
            return {}
    return {}

def main():
    load_pairs_mapping()
    
    # 1. Load historical records
    results_details = load_existing_results(SAVE_JSON_PATH)
    
    # 2. Rebuild statistical data (stats_acc) based on historical records
    #    This ensures that the statistics include previously processed data
    stats_acc = {k: {"sum": 0, "count": 0} for k in DATASET_CONFIG.keys()}
    
    print("Rebuilding historical statistical data...")
    for key_id, score in results_details.items():
        # key_id format: dataset_filename (e.g., complex_bg_001.jpg)
        # We need to determine which dataset it belongs to
        matched_ds = None
        for ds_key in DATASET_CONFIG.keys():
            if key_id.startswith(ds_key + "_"):
                matched_ds = ds_key
                break
        
        if matched_ds and score > 0:
            stats_acc[matched_ds]["sum"] += score
            stats_acc[matched_ds]["count"] += 1
            
    current_macro = calculate_macro_average(stats_acc)
    print(f"Historical data loading complete. Processed: {len(results_details)} images, current average score: {current_macro:.2f}")

    # 3. Scan and filter tasks
    tasks =[]
    print("Scanning and filtering tasks...")
    
    for key, config in DATASET_CONFIG.items():
        rdir = config['result_dir']
        if not os.path.exists(rdir): continue
        files =[f for f in os.listdir(rdir) if f.lower().endswith(('.jpg', '.png'))]
        
        for f in files:
            # Check if it already exists
            unique_key = f"{key}_{f}"
            if unique_key in results_details:
                continue # Skip processed ones
            
            tasks.append((key, f))
            
    print(f"New pending tasks: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All tasks have been completed, no processing needed.")
        save_json(SAVE_JSON_PATH, results_details, stats_acc)
        return

    # 4. Execute remaining tasks
    processed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(process_task, k, f): (k, f) for k, f in tasks}
        
        pbar = tqdm(as_completed(future_map), total=len(tasks), desc="Evaluating")
        for future in pbar:
            try:
                res = future.result()
                ds = res['dataset']
                score = res['score']
                fname = res['filename']
                
                # Record
                key_id = f"{ds}_{fname}"
                results_details[key_id] = score
                
                if score > 0:
                    stats_acc[ds]["sum"] += score
                    stats_acc[ds]["count"] += 1
                
                # Update progress bar
                curr_macro_avg = calculate_macro_average(stats_acc)
                pbar.set_postfix({"Avg": f"{curr_macro_avg:.1f}"})
                
                # Save periodically
                processed_count += 1
                if processed_count % SAVE_INTERVAL == 0:
                    save_json(SAVE_JSON_PATH, results_details, stats_acc)
                    
            except Exception as e:
                print(f"Error: {e}")
                
    save_json(SAVE_JSON_PATH, results_details, stats_acc)
    
    final_macro_avg = calculate_macro_average(stats_acc)
    print(f"\nAll tasks completed.")
    print(f"Final average score (Macro Avg): {final_macro_avg:.2f}")
    print(f"Results saved to: {SAVE_JSON_PATH}")

if __name__ == "__main__":
    main()
