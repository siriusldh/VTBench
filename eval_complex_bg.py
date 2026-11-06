import cv2
import numpy as np
import os
from tqdm import tqdm

def calculate_background_mse(img_before, img_after, mask_before, mask_after, kernel_size=5):
    """
    Calculate the Mean Squared Error (MSE) of the background region before and after virtual try-on.
    :param img_before: Image before try-on (BGR format)
    :param img_after: Image after try-on (BGR format)
    :param mask_before: Mask before try-on (Binary image, 255 indicates the model area)
    :param mask_after: Mask after try-on (Binary image, 255 indicates the model area)
    :param kernel_size: Kernel size for morphological operations (odd number)
    :return: MSE value of the background region
    """
    # Convert to uint8 for processing
    mask_before = mask_before.astype(np.uint8)
    mask_after = mask_after.astype(np.uint8)

    # 1. Calculate the union of masks
    union_mask = cv2.bitwise_or(mask_before, mask_after)

    # 2. Morphological operations (erosion followed by dilation)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # eroded = cv2.erode(union_mask, kernel, iterations=1)
    # processed_mask = cv2.dilate(eroded, kernel, iterations=1)

    # 3. Get the background region (invert the mask)
    background_mask = cv2.bitwise_not(union_mask)

    # # Save the mask
    # cv2.imwrite('background_mask.png', background_mask)

    # 4. Convert images to float type
    img_before = img_before.astype(np.float32)
    img_after = img_after.astype(np.float32)

    # 5. Calculate MSE (background region only)
    diff = img_before - img_after
    squared_diff = np.square(diff)

    # Apply mask to all channels
    if len(img_before.shape) == 3:  # Color image
        mask_3d = (background_mask / 255.0).astype(np.float32)[:, :, np.newaxis]
        # import pdb; pdb.set_trace()
        masked_diff = squared_diff * mask_3d
        valid_pixels = np.count_nonzero(mask_3d) * 3
    else:  # Grayscale image
        masked_diff = squared_diff * background_mask
        valid_pixels = np.count_nonzero(background_mask)

    # Avoid division by zero
    if valid_pixels == 0:
        return 0.0

    mse = np.sum(masked_diff) / valid_pixels
    return mse

# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Background preservation MSE evaluation.")
    parser.add_argument("--img-path-be", default="./data/benchmark_complex_bg/image",
                        help="Original images before try-on.")
    parser.add_argument("--mask-path-be", default="./data/benchmark_complex_bg/image_bg_seg/mask",
                        help="Background masks of the original images.")
    parser.add_argument("--img-path-af", default="./data/results/complex",
                        help="Generated images after try-on.")
    parser.add_argument("--mask-path-af", default="./data/results/complex_seg/mask",
                        help="Background masks of the generated images.")
    args = parser.parse_args()

    img_path_be = args.img_path_be
    mask_path_be = args.mask_path_be
    img_path_af = args.img_path_af
    mask_path_af = args.mask_path_af

    
    print(f"img_path_after: {img_path_af}")
    img_names = sorted(f for f in os.listdir(img_path_be) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    mse_avg = 0
    mse_count = 0
    skipped = []  # failed generations / missing files are excluded from the average (paper appendix I)
    for img_name in tqdm(img_names):
        img_path_before = f"{img_path_be}/{img_name}"
        img_path_after = f"{img_path_af}/{img_name}"
        mask_path_before = f"{mask_path_be}/{img_name}"
        mask_path_after = f"{mask_path_af}/{img_name}"

        img_before = cv2.imread(img_path_before)
        img_after = cv2.imread(img_path_after)
        mask_before = cv2.imread(mask_path_before, cv2.IMREAD_GRAYSCALE)
        mask_after = cv2.imread(mask_path_after, cv2.IMREAD_GRAYSCALE)
        if img_before is None or img_after is None or mask_before is None or mask_after is None:
            skipped.append(img_name)
            continue
        
        # Resize to the same dimensions
        # img_before = cv2.resize(img_before, (img_after.shape[1], img_after.shape[0]))
        # mask_before = cv2.resize(mask_before, (img_after.shape[1], img_after.shape[0]))

        img_after = cv2.resize(img_after, (img_before.shape[1], img_before.shape[0]))
        mask_after = cv2.resize(mask_after, (img_before.shape[1], img_before.shape[0]))
        # print(f"img_before: {img_before.shape}, img_after: {img_after.shape}, mask_before: {mask_before.shape}, mask_after: {mask_after.shape}")
        
        # Binarize the mask (ensure values are 0 and 255)
        _, mask_before = cv2.threshold(mask_before, 127, 255, cv2.THRESH_BINARY)
        _, mask_after = cv2.threshold(mask_after, 127, 255, cv2.THRESH_BINARY)

        # Calculate MSE
        mse = calculate_background_mse(img_before, img_after, mask_before, mask_after)
        # print(f"Background MSE: {mse:.2f}")
        mse_avg += mse
        mse_count += 1
        
    if skipped:
        print(f"[warn] {len(skipped)} of {len(img_names)} samples skipped (missing result, input, or mask), "
              f"e.g. {skipped[:3]}")
    mse_avg /= mse_count
    print(f"Average Background MSE over {mse_count}/{len(img_names)} samples: {mse_avg:.4f}")