# VTBench: Disentangled and Human-Aligned Evaluation for Image-Based Virtual Try-on

Official evaluation code for **VTBench** (NeurIPS 2026, Evaluations and Datasets Track).
Data, generated results of all 21 benchmarked methods, and human-evaluation data are on
[Hugging Face](https://huggingface.co/datasets/vtbenchmark/VTBench).

VTBench evaluates image-based virtual try-on along six dimensions:

| Dimension | Metric | Test set | Script |
|---|---|---|---|
| Distributional Fidelity | FID / KID ↓ | all generated results | `eval_fid_kid.py` |
| Aesthetics | $A_s$ ↑ (Qwen-VL-Max) | all generated results | `eval_aethesic.py` |
| Texture Fidelity | $F_t$ ↑ (OCR), $T_{score}$ ↑ | FTF | `F_t/`, `eval_T_score.py` |
| Cross-Category Plausibility | $\text{VLM}_c$ ↑ (Qwen-VL-Max) | CCP | `eval_cross.py` |
| Background Consistency | $B_{consist}$ ↓ (+ Color-aligned MSE) | CBC | `eval_complex_bg.py`, `color_aligned_mse.py` |
| Hand Consistency | $H_{consist}$ ↓ (MPJPE-2D) | HOC | `eval_hand.py` |

The overall rank $R_{avg}$ is computed by `tools/compute_ravg.py`.

## 0. Data

```bash
git clone https://github.com/siriusldh/VTBench.git
cd VTBench
```

Download the dataset (add `--exclude "methods_result/*"` to skip the 31.8 GB of method results):

```bash
hf download vtbenchmark/VTBench --repo-type dataset --local-dir data
```

912 web images are provided only as source links (`VTBench_image_source_list_en.csv`); see the dataset card for how to restore them.
The scripts assume the Hugging Face layout under `./data` (or `$VTBENCH_DATA_ROOT`):

```
data/
├── benchmark_complex_bg/   # CBC: image/, cloth/, test_pairs.txt
├── benchmark_texture/      # FTF: image/, cloth/, test_pairs.txt
├── benchmark_hand/         # HOC: image/, cloth/, test_pairs.txt
├── benchmark_crossvton/    # CCP: dress/, upper/, lower/, test_pairs_*.txt
├── pairs.tsv               # all 2,896 pairs with explicit person/cloth columns
└── results/<method>/       # generated results of the method to evaluate
    ├── complex/  text/  hand/  cross/
```

All `test_pairs*.txt` files list `<person> <cloth>` per line. Generated results are named after the person image of each pair. To evaluate a released method, extract `methods_result/<method>.zip` into `data/results/`.

For the VLM-based metrics, export a DashScope API key:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

## 1. Texture Fidelity

* **$F_t$** (text-level OCR fidelity): equal-weight mean of term-frequency similarity, precision, recall, and edit-distance similarity between PP-OCRv3 outputs on the garment and on the try-on result.

  ```bash
  export VTBENCH_DATA_ROOT=./data VTBENCH_METHOD=<method> VTBENCH_RESULT_DIR=./data/results/<method>
  python F_t/f_t-score.py      # OCR + per-sample scores -> $VTBENCH_DATA_ROOT/ocr_results/<method>/
  python F_t/cal_benchmark.py  # averages per method
  ```

* **$T_{score}$**: composite of CLIP-I, DINO, CLIP-T, and $\text{VLM}_t$. Each method-level score is min-max normalized with the **frozen VTBench-v1 calibration anchors** (minimum/maximum over the original 19 methods; see `ANCHORS` in `eval_T_score.py`), clipped to [0, 1], and averaged with equal weight. Values outside the anchors are flagged. The anchors are never recomputed, so adding a method does not change existing scores.

  CLIP-I and DINO compare the garment region of the result with the target garment; crop it with the released boxes (`json/ftf_garment_bbox.json`). CLIP-T uses the released garment descriptions (`json/ftf_descriptions.json`).

  ```bash
  python tools/crop_ftf_garment.py --result-dir ./data/results/<method>/text \
      --output-dir ./data/results/<method>/text_garment_crops
  python eval_T_score.py \
      --generated_crop_path ./data/results/<method>/text_garment_crops \
      --generated_whole_path ./data/results/<method>/text \
      --base_img_path ./data/benchmark_texture \
      --txt_path ./data/benchmark_texture/test_pairs.txt \
      --caption_path ./json/ftf_descriptions.json \
      --model_name_override <method>
  ```

## 2. Distributional Fidelity (FID / KID)

Protocol of Tab. 1: the **reference set** is all 2,896 person (model) images of the four test sets (CBC, FTF, HOC, CCP); the **generated set** is all results of the method on the four test sets (methods without cross-category support have no CCP results; failed generations are absent). FID and KID (reported ×1000) are computed with `clean-fid`. Collect both sets into flat folders (file names are prefixed with the subset to avoid collisions), then evaluate:

```bash
python tools/collect_fid_sets.py --data-root ./data --result-dir ./data/results/<method> \
    --method <method> --out ./outputs/fid_sets
python eval_fid_kid.py --gt_folder ./outputs/fid_sets/real --pred_folder ./outputs/fid_sets/<method> \
    --batch_size 16 --num_workers 8
```

## 3. Hand Consistency

MPJPE-2D between hand keypoints detected (HaMeR) on the HOC person image and on the try-on result. A reference without a detected hand is excluded; if the reference hand is detected but the output hand is not, the output is counted as a failure (fixed penalty of 100 px per unmatched reference hand; predicted and reference hands are matched with the Hungarian algorithm). Detector outputs are used directly, without manual correction.

### Installation

```
conda create -n handfit python=3.10
conda activate handfit
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.23.post1
pip install transformers==4.39.3 accelerate==0.29.2 diffusers==0.24.0 deepspeed==0.14.0
cd hamer && pip install -e .[all] && cd ..
pip install scikit-image omegaconf kornia h5py datasets einops opencv-python gradio onnxruntime
conda install menpo::osmesa
pip install --upgrade pyopengl==3.1.4
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch --no-build-isolation
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
cd hamer/third-party/ViTPose && pip install -v -e . && cd ../../..
```

### Run

```bash
python eval_hand.py \
    --img-dir ./data/results/<method>/hand \
    --ref-dir ./data/benchmark_hand/image \
    --gt-json-file ./outputs/hoc_reference_pos2d.json \
    --pred-json-output ./outputs/<method>_hand_pos2d.json
```

Reference poses are detected once on `--ref-dir` and cached in `--gt-json-file`.

## 4. Cross-Category Plausibility

Each CCP pair (person, target garment, result) is judged by Qwen-VL-Max as Reasonable/Unreasonable. $\text{VLM}_c$ is the percentage of the 420 pairs judged Reasonable; a missing result counts as Unreasonable. The default prompt is the main protocol of the paper; `--strict` uses the stricter identity-aware variant reported in the appendix.

```bash
python eval_cross.py \
    --img-paths ./data/results/<method>/cross \
    --img-base-path ./data/benchmark_crossvton \
    --output ./outputs/<method>_vlm_c.json          # add --strict for the identity-aware variant
```

## 5. Background Consistency

### Step 1: human segmentation masks

The same external HRNet-OCR human segmentation model (2 classes: person / background; architecture from [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)) is applied to every person image and every generated result (`inferseg/`). Download the weights to `inferseg/ckpt/final_state.pth`:

* **Weights**: [`final_state.pth` (Google Drive)](https://drive.google.com/file/d/1QIgATgW7QpvGTZLGuixDnu4XIGc-17Uw/view?usp=drive_link)

```bash
pip install gdown
mkdir -p inferseg/ckpt
gdown 1QIgATgW7QpvGTZLGuixDnu4XIGc-17Uw -O inferseg/ckpt/final_state.pth
```

```bash
bash inferseg/demo_one_key.sh ./data/benchmark_complex_bg/image ./data/benchmark_complex_bg/image_bg_seg
bash inferseg/demo_one_key.sh ./data/results/<method>/complex ./data/results/<method>/complex_seg
```

### Step 2: $B_{consist}$ (MSE on the background, 0–255 scale)

```bash
python eval_complex_bg.py \
    --img-path-be ./data/benchmark_complex_bg/image \
    --mask-path-be ./data/benchmark_complex_bg/image_bg_seg/mask \
    --img-path-af ./data/results/<method>/complex \
    --mask-path-af ./data/results/<method>/complex_seg/mask
```

### Color-aligned MSE (complementary)

Fits a global affine RGB mapping from the generated image to the person image on background pixels (person-image mask), then computes MSE on the same pixels ([0, 1] scale). It tolerates global brightness/color shifts but can also absorb channel permutations or inversions, so it complements rather than replaces $B_{consist}$.

```bash
python color_aligned_mse.py \
    --img-path-be ./data/benchmark_complex_bg/image \
    --mask-path-be ./data/benchmark_complex_bg/image_bg_seg/mask \
    --img-path-af ./data/results/<method>/complex \
    --output ./outputs/<method>_camse.json
```

## 6. Aesthetics

Qwen-VL-Max scores only the generated (rightmost) image from 1 to 100 on image quality, human naturalness, garment visual appeal, and overall harmony; it does not assess fidelity to the target garment. The prompt is given verbatim in the paper appendix.

$A_s$ is a macro average: the scores are first averaged within each test set (CBC, FTF, HOC, CCP) and the per-set means are then averaged with equal weight (methods without CCP results average over three sets). Calls that return no valid score are recorded as 0 and excluded from the averages. Scores are cached in `aesthetic_qwen_score_final.json`, so an interrupted run resumes.

```bash
VTBENCH_DATA_ROOT=./data VTBENCH_RESULT_DIR=./data/results/<method> python eval_aethesic.py
```

## 7. Overall rank $R_{avg}$

`results/table1_metrics.csv` holds the Tab. 1 scores of the 21 methods. `tools/compute_ravg.py` ranks every method on the eight metrics (a method without $\text{VLM}_c$ is ranked last on it) and averages the ranks; `results/ranking.json` is its output.

```bash
python tools/compute_ravg.py --metrics results/table1_metrics.csv --output results/ranking.json
```

To add a method, append its eight scores to the CSV and rerun.

## 8. Human-alignment analysis

`results_v2/` (20 annotators' pairwise votes) and `human_preference/` (per-sample scores) reproduce the human-alignment statistics (Fig. 4, Tab. 2, and the bootstrap CIs in the appendix). They are identical to `human_evaluation/annotations/` and `human_evaluation/per_sample_scores/` on Hugging Face and use the dataset's sample IDs (`CBC-*`, `FTF-*`, `HOC-*`, `CCP-*`, `VITONHD-*`; see `human_evaluation/sample_map.csv`).

```bash
python human_preference/metric_bt.py                 # Spearman rho per dimension
python reproducibility/compute_agreement.py          # exact A/B/tie agreement
python reproducibility/bootstrap_ci.py               # 95% image-clustered bootstrap CIs (B = 10,000, seed 20260730)
```

To run on the Hugging Face copy instead, pass `--csv-dir <hf>/human_evaluation/annotations --scores-dir <hf>/human_evaluation/per_sample_scores`.

## Method names

Released archives keep internal spellings: `ITA` = ITA-MDT, `IMAGDressing` = IMAGDressing-v1, `FitDit` = FitDiT, `OOTD` = OOTDiffusion, `StableVTON` = StableVITON, `Seedream` = Seedream 4.0.

## License

The evaluation code is released under **CC BY-NC 4.0** (see [LICENSE](LICENSE)). The dataset has its own terms (see the dataset card). Third-party components keep their licenses:

| Path | License |
|---|---|
| `hamer/` | MIT (`hamer/LICENSE.md`) |
| `hamer/third-party/ViTPose/` | upstream license in that directory |
| `F_t/wwm_ocr.py` | Apache-2.0 header retained from upstream |

## Citation

```bibtex
@inproceedings{luo2026vtbench,
  title     = {{VTBench}: Disentangled and Human-Aligned Evaluation for Image-Based Virtual Try-on},
  author    = {Luo, Donghao and Li, Caoshuo and Liang, Yujie and Hu, Xiaobin and Wu, Zuxuan and Fu, Yanwei},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Track on Evaluations and Datasets},
  year      = {2026}
}
```
