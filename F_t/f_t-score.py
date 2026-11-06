"""
Specify two paths, use PP-OCR to perform OCR on the corresponding images in the two paths,
then compare and evaluate using the OCR results.
"""

import os
import shutil
from tqdm import tqdm

# Inputs follow the Hugging Face layout under VTBENCH_DATA_ROOT (default ./data);
# generated FTF results are read from VTBENCH_RESULT_DIR/text (an extracted
# methods_result/<method>.zip, i.e. <method>/text/).
DATA_ROOT = os.getenv("VTBENCH_DATA_ROOT", "./data")
METHOD = os.getenv("VTBENCH_METHOD", "method")
RESULT_DIR = os.getenv("VTBENCH_RESULT_DIR", os.path.join(DATA_ROOT, "results", METHOD))
from PIL import Image
import numpy as np

from wwm_ocr import core, get_args, NumpyEncoder
from wwm_ocr_eval import cloth2clothes

import cv2
import base64
import time
import hashlib
import json
import pandas as pd
import math

def cat_image(pp1, pp2):
    im1, im2 = None, None
    if pp1:
        im1 = Image.open(pp1)
    if pp2:
        im2 = Image.open(pp2)
        im2 = im2.resize((1024, 1024))
    
    # Combine two images into one: 1 row, 2 columns
    total_width = 2 * max(im1.width, im2.width)
    max_height = max(im1.height, im2.height)

    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste(im1, (0, 0))
    if im2:
        new_im.paste(im2, (im1.width, 0))
    return new_im

######################################################### Comparison
cloth_paths = os.path.join(DATA_ROOT, 'benchmark_texture', 'cloth')

# Read test_pairs.txt
with open(os.path.join(DATA_ROOT, 'benchmark_texture', 'test_pairs.txt'), 'r') as f:  # <person> <cloth>
    pairs = f.readlines()

# Parse each line to get filenames of try-on images and cloth images
image_cloth_pairs = []
for line in pairs:
    line = line.strip()
    if line:
        image_name, cloth_name = line.split()
        image_cloth_pairs.append((image_name, cloth_name))

# Create a dictionary: key = try-on image filename, value = corresponding cloth image filename
image_cloth_dict = {pair[0]: pair[1] for pair in image_cloth_pairs}

args = get_args()

name_list = []
score_list = []
clothes_texts_list = []
img_texts_list = []

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Create result directories
result_dir = os.path.join(DATA_ROOT, 'ocr_results')
os.makedirs(result_dir, exist_ok=True)
concat_dir = os.path.join(DATA_ROOT, 'concat_images')
os.makedirs(concat_dir, exist_ok=True)

image_path = [
    os.path.join(RESULT_DIR, 'text'),
]

for image_dir in tqdm(image_path, ncols=80):
    # Use model name as subfolder name
    model_name = METHOD
    model_result_dir = os.path.join(result_dir, model_name)
    model_concat_dir = os.path.join(concat_dir, model_name)
    os.makedirs(model_result_dir, exist_ok=True)
    os.makedirs(model_concat_dir, exist_ok=True)
    
    for n in os.listdir(image_dir):
        if 'TPD' in image_dir:
            cloth = os.path.join(cloth_paths, image_cloth_dict[n.replace(".jpg", ".png")])
        else:
            cloth = os.path.join(cloth_paths, image_cloth_dict[n])

        image = os.path.join(image_dir, n)

        # Concatenate images and save
        concat_image = cat_image(image, cloth)
        concat_save_path = os.path.join(model_concat_dir, f"{os.path.splitext(n)[0]}_concat.jpg")
        concat_image.save(concat_save_path)

        img_result = ocr.ocr(image, cls=True)[0]
        clothes_result = ocr.ocr(cloth, cls=True)[0]

        if clothes_result is None or img_result is None:
            clothes_texts = []
            img_texts = []
        else:
            clothes_texts = [{'text': clothes_result[idx][-1][0]} for idx in range(len(clothes_result))]
            img_texts = [{'text': img_result[idx][-1][0]} for idx in range(len(img_result))]
        
        if len(clothes_texts) == 0 or len(img_texts) == 0:
            avg_metrics = 0
        else:
            avg_metrics, detail_metrics = cloth2clothes(clothes_texts, img_texts)
            
        # Save results to txt file
        result_file = os.path.join(model_result_dir, f"{os.path.splitext(n)[0]}.txt")
        with open(result_file, 'w') as f:
            if avg_metrics != 0:
                f.write(f"{avg_metrics} | {detail_metrics}\n")
            else:
                f.write(f"{avg_metrics}\n")