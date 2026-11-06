import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment

# Ensure hamer_depth is in your PYTHONPATH or same directory
from hamer_depth import get_depth_info


# ==============================================================================
# 1. Evaluation Functions (Hungarian Matching & MPJPE-2D)
# ==============================================================================

def hungarian_matching_with_penalty(preds, gts, penalty=50.0):
    """
    Strict one-to-one Hungarian matching algorithm.
    :param preds: List of predicted poses (M, 21, 2)
    :param gts: List of ground truth poses (N, 21, 2)
    :param penalty: Penalty value for unmatched poses
    :return: List of errors for all poses in this image (including penalties)
    """
    M, N = len(preds), len(gts)
    if M == 0 and N == 0:
        return[]
    
    # Create an extended cost matrix (max(M,N) x max(M,N))
    size = max(M, N)
    cost_matrix = np.full((size, size), penalty)
    
    # Fill in the actual costs
    for i in range(M):
        for j in range(N):
            # Calculate the average error for 21 keypoints
            error = np.linalg.norm(preds[i] - gts[j], axis=1).mean()
            cost_matrix[i, j] = error
            
    # Execute Hungarian algorithm
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    
    # Collect valid matching errors
    errors =[]
    matched_gt = set()
    for r, c in zip(row_idx, col_idx):
        if r < M and c < N:  # Exclude virtual matches
            errors.append(cost_matrix[r, c])
            matched_gt.add(c)
    
    # Add penalty for unmatched ground truth poses
    for j in range(N):
        if j not in matched_gt:
            errors.append(penalty)
    
    return errors

def calculate_mpjpe2d_robust(preds_dict, gts_dict, penalty=100.0):
    """
    Robust MPJPE-2D calculation function.
    :param preds_dict: Prediction dictionary {image_id: [pose_list]}
    :param gts_dict: Ground truth dictionary {image_id: [pose_list]}
    :param penalty: Penalty value for missed detections
    :return: Average error (pixels)
    """
    all_errors =[]
    
    # Iterate through all image IDs in the ground truth
    for image_id in gts_dict:
        gts = np.array(gts_dict[image_id])
        num_gts = len(gts)
        
        # Handle missing predictions
        if image_id not in preds_dict:
            if num_gts > 0:
                all_errors.extend([penalty] * num_gts)
            continue
        
        preds = np.array(preds_dict[image_id])
        num_preds = len(preds)
        
        # Handle empty predictions
        if num_preds == 0:
            if num_gts > 0:
                all_errors.extend([penalty] * num_gts)
            continue
        
        # Execute Hungarian matching
        errors = hungarian_matching_with_penalty(preds, gts, penalty)
        all_errors.extend(errors)
    
    # Calculate global average error
    return np.mean(all_errors) if all_errors else penalty

def scale_coordinates(data_dict, scale_factor=2):
    """Utility function to scale coordinates if needed"""
    scaled_dict = {}
    for img_id, poses in data_dict.items():
        np_poses = np.array(poses) * scale_factor  # Vectorized operation
        scaled_dict[img_id] = np_poses.tolist()
    return scaled_dict


# ==============================================================================
# 2. Main Processing Logic (Prediction -> Save -> Evaluation)
# ==============================================================================

if __name__ == "__main__":
    import argparse

    # --- Paths Configuration ---
    parser = argparse.ArgumentParser(description="Hand pose consistency evaluation (MPJPE-2D).")
    parser.add_argument("--img-dir", default="./data/results/hand",
                        help="Directory containing generated try-on images.")
    parser.add_argument("--pred-json-output", default="./outputs/hand_all_pos2d.json",
                        help="Where to save predicted 2D poses.")
    parser.add_argument("--gt-json-file", default="./outputs/hoc_reference_pos2d.json",
                        help="2D hand poses of the reference (person) images. If the file does not exist "
                             "and --ref-dir is given, it is created by running the same detector on --ref-dir.")
    parser.add_argument("--ref-dir", default="./data/benchmark_hand/image",
                        help="HOC person images used to build the reference poses.")
    args = parser.parse_args()

    img_dir = args.img_dir
    pred_json_output = args.pred_json_output
    gt_json_file = args.gt_json_file
    

    def predict_dir(directory):
        positions = {}
        names = sorted(f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        for name in tqdm(names, desc=f"Extracting 2D Poses ({os.path.basename(os.path.normpath(directory))})"):
            path = os.path.join(directory, name)
            try:
                _, result_list, _ = get_depth_info(Image.open(path).convert('RGB'), path)
                pos_2d = result_list[1]
                positions[os.path.splitext(name)[0]] = pos_2d.tolist() if hasattr(pos_2d, 'tolist') else pos_2d
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
        return positions

    # ================= Phase 0: Reference poses =================
    if not os.path.exists(gt_json_file):
        print(f"Reference poses not found; detecting hands on {args.ref_dir}")
        ref_positions = predict_dir(args.ref_dir)
        # A reference without a detected hand is invalid and excluded.
        ref_positions = {k: v for k, v in ref_positions.items() if v is not None and len(v) > 0}
        os.makedirs(os.path.dirname(os.path.abspath(gt_json_file)), exist_ok=True)
        with open(gt_json_file, 'w') as f:
            json.dump(ref_positions, f, indent=2)
        print(f"Saved reference poses for {len(ref_positions)} images to: {gt_json_file}")

    # ================= Phase 1: Predict and Save =================
    print("\n" + "="*50)
    print("Phase 1: Predicting 2D Poses and Saving to JSON")
    print("="*50)
    
    all_positions = {}
    vton_imgs =[f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    vton_imgs.sort()
    
    print(f"Total images found: {len(vton_imgs)}")
    
    for img in tqdm(vton_imgs, desc="Extracting 2D Poses"):
        vton_img_path = os.path.join(img_dir, img)
        try:
            # Process image
            vton_img = Image.open(vton_img_path).convert('RGB')
            depth_img, result_list, crop_imgs = get_depth_info(vton_img, vton_img_path)
            
            # Extract 2D keypoints
            pos_2d = result_list[1]
            base_name = os.path.splitext(img)[0]
            
            # Convert to Python native types (compatible with numpy arrays)
            if hasattr(pos_2d, 'tolist'):
                all_positions[base_name] = pos_2d.tolist()
            else:
                all_positions[base_name] = pos_2d
                
        except Exception as e:
            print(f"Error processing {img}: {str(e)}")

    # Save all predicted data into JSON
    with open(pred_json_output, 'w') as f:
        json.dump(all_positions, f, indent=2)
    print(f"Successfully saved all 2D positions to: {pred_json_output}")


    # ================= Phase 2: Calculate Metrics =================
    print("\n" + "="*50)
    print("Phase 2: Calculating Hand Metrics (Robust MPJPE-2D)")
    print("="*50)
    
    # Load Ground Truth data
    with open(gt_json_file, 'r') as f:
        gts_dict = json.load(f)
        
    print(f"Loaded {len(all_positions)} predictions and {len(gts_dict)} ground truths.")

    # (Optional) If you need to scale your predictions, uncomment the next line:
    # all_positions = scale_coordinates(all_positions, scale_factor=2)

    # Align keys (assuming we evaluate based on the images present in GT)
    image_ids = sorted(gts_dict.keys())
    eval_gts_dict = {k: gts_dict[k] for k in image_ids}
    
    # Handle missing predictions gracefully by inserting empty lists
    eval_preds_dict = {k: all_positions.get(k,[]) for k in image_ids}
    
    # Calculate metrics
    mpjpe = calculate_mpjpe2d_robust(eval_preds_dict, eval_gts_dict, penalty=100.0)
    
    print(f"\n[Final Result]")
    print(f"Robust MPJPE-2D = {mpjpe:.5f} pixels")
    print("="*50 + "\n")