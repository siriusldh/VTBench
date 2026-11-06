import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['mathtext.rm'] = 'Times New Roman'
from scipy.stats import spearmanr

# 7 standard model names for unified evaluation
METHOD_KEYS =['SD', 'IDM', 'CatVTON', 'HandFit', 'FitDiT', 'CrossVTON', 'Qwen-Edit']

# Mapping from CSV letters to models for human scores
CSV_MAP = {'A': 'SD', 'B': 'IDM', 'C': 'CatVTON', 'D': 'HandFit', 'E': 'FitDiT', 'F': 'CrossVTON', 'G': 'Qwen-Edit'}

# Mapping from various TXT aliases to standard models (OOTD maps to SD for CrossVTON)
TXT_KEY_MAP = {
    'SD': 'SD', 'SD-VITON': 'SD', 'OOTD': 'SD',
    'IDM': 'IDM', 'idm': 'IDM',
    'CatVTON': 'CatVTON',
    'FitDiT': 'FitDiT', 'FitDit': 'FitDiT',
    'CrossVTON': 'CrossVTON', 'cross_vton': 'CrossVTON',
    'HandFit': 'HandFit', 'vton-handfit': 'HandFit', 'vton_handfit': 'HandFit',
    'Qwen-Edit': 'Qwen-Edit'
}

# Fixed list of 20 test images per metric (Hugging Face sample IDs; see human_evaluation/sample_map.csv)
IMAGE_LISTS = {
    'Aesthetic': ['VITONHD-00737_00', 'VITONHD-02726_00', 'VITONHD-02765_00', 'VITONHD-02849_00', 'VITONHD-05412_00', 'VITONHD-05562_00', 'VITONHD-05922_00', 'VITONHD-05935_00', 'VITONHD-06570_00', 'VITONHD-06640_00', 'VITONHD-08098_00', 'VITONHD-08157_00', 'VITONHD-08973_00', 'VITONHD-09396_00', 'VITONHD-10078_00', 'VITONHD-10256_00', 'VITONHD-10712_00', 'VITONHD-10721_00', 'VITONHD-12384_00', 'VITONHD-12749_00'],
    'Background': ['CBC-0021', 'CBC-0073', 'CBC-0113', 'CBC-0134', 'CBC-0146', 'CBC-0157', 'CBC-0187', 'CBC-0201', 'CBC-0211', 'CBC-0219', 'CBC-0248', 'CBC-0252', 'CBC-0262', 'CBC-0265', 'CBC-0302', 'CBC-0390', 'CBC-0392', 'CBC-0405', 'CBC-0428', 'CBC-0436'],
    'Cross': ['CCP-1-042-0', 'CCP-1-089-0', 'CCP-1-102-0', 'CCP-1-145-0', 'CCP-1-151-0', 'CCP-1-162-0', 'CCP-1-172-0', 'CCP-1-180-0', 'CCP-2-027-0', 'CCP-2-062-0', 'CCP-3-015-0', 'CCP-3-148-0', 'CCP-4-037-0', 'CCP-4-147-0', 'CCP-6-002-0', 'CCP-6-040-0', 'CCP-7-095-0', 'CCP-7-146-0', 'CCP-8-075-0', 'CCP-8-163-0'],
    'Hand': ['HOC-0082', 'HOC-0194', 'HOC-0285', 'HOC-0286', 'HOC-0447', 'HOC-0513', 'HOC-0530', 'HOC-0577', 'HOC-0621', 'HOC-0733', 'HOC-0788', 'HOC-0829', 'HOC-0861', 'HOC-0976', 'HOC-0995', 'HOC-1033', 'HOC-1047', 'HOC-1090', 'HOC-1194', 'HOC-1266'],
    'Text': ['FTF-image_0344', 'FTF-image_0117', 'FTF-image_0103', 'FTF-image_0597', 'FTF-image_0078', 'FTF-image_0475', 'FTF-image_0330', 'FTF-image_0186', 'FTF-image_0564', 'FTF-image_0528', 'FTF-image_0232', 'FTF-image_0553', 'FTF-image_0297', 'FTF-image_0319', 'FTF-image_0154', 'FTF-image_0116', 'FTF-image_0410', 'FTF-image_0560', 'FTF-image_0296', 'FTF-image_0012']
}

# Metric configuration: 
# 'is_positive': True means higher score is better (positive correlation).
# 'row_range': the corresponding row indices in the CSV files.
METRICS_CONFIG = {
    'Aesthetic': {'is_positive': True,  'images': IMAGE_LISTS['Aesthetic'], 'row_range': (0, 19)},
    'Background': {'is_positive': False, 'images': IMAGE_LISTS['Background'], 'row_range': (20, 39)},
    'Cross':     {'is_positive': True,  'images': IMAGE_LISTS['Cross'],     'row_range': (40, 59)},
    'Hand':      {'is_positive': False, 'images': IMAGE_LISTS['Hand'],      'row_range': (60, 79)},
    'Text':      {'is_positive': True,  'images': IMAGE_LISTS['Text'],      'row_range': (80, 99)}
}

def parse_vton_metrics(text_content):
    """
    Parse the comma-separated text file containing metric scores.
    Expects format:
    sample_id,IDM,CatVTON,FitDiT,CrossVTON,SD,HandFit,Qwen-Edit
    HOC-0788,109.32,408.79,10.06,165.11,47.35,57.98,...
    """
    entries = []
    lines =[line.strip() for line in text_content.strip().split('\n') if line.strip()]
    if not lines:
        return entries

    # Parse headers from the first line
    headers =[h.strip() for h in lines[0].split(',')]

    for line in lines[1:]:
        # Skip potential divider lines
        if line.startswith('====') or line.startswith('----'):
            continue
            
        row_data = [d.strip() for d in line.split(',')]
        if not row_data or len(row_data) < len(headers):
            continue
            
        entry = {}
        for i, col_name in enumerate(headers):
            if i == 0:  # sample identifier column ('sample_id'; legacy files used 'Image')
                entry['Image'] = row_data[i]
            else:
                try:
                    entry[col_name] = float(row_data[i])
                except ValueError:
                    entry[col_name] = 0.0
        entries.append(entry)
        
    return entries

def compute_machine_scores(txt_path, metric_config):
    """
    Compute machine scores from the TXT file.
    Automatically allocates points based on the positive/negative correlation of the metric.
    """
    if not os.path.exists(txt_path):
        print(f"Warning: txt path not found - {txt_path}")
        return {k: 0 for k in METHOD_KEYS}

    with open(txt_path, 'r', encoding='utf-8') as f:
        parsed_data = parse_vton_metrics(f.read())
        
    is_positive = metric_config['is_positive']
    image_list = metric_config['images']
    
    total_scores = {k: 0.0 for k in METHOD_KEYS}
    valid_images = 0

    for entry in parsed_data:
        # Extract the image name without the extension
        img_name = entry['Image'].rsplit('.', 1)[0]
        if img_name not in image_list:
            continue
            
        valid_images += 1
        current_vals = {k: 0.0 for k in METHOD_KEYS} # Default to 0

        # Map different model names in the file to the standard model names
        for raw_key, val in entry.items():
            if raw_key in TXT_KEY_MAP:
                current_vals[TXT_KEY_MAP[raw_key]] = val
        
        # Pairwise comparison mechanism
        for m1 in METHOD_KEYS:
            for m2 in METHOD_KEYS:
                if m1 == m2: 
                    continue
                v1, v2 = current_vals[m1], current_vals[m2]
                
                # Positive correlation: higher score gets 1 point
                # Negative correlation: lower score gets 1 point
                # Tie: each gets 0.5 points
                if is_positive:
                    if v1 > v2: total_scores[m1] += 1
                    elif v1 < v2: total_scores[m2] += 1
                    else: 
                        total_scores[m1] += 0.5
                        total_scores[m2] += 0.5
                else:
                    if v1 < v2: total_scores[m1] += 1
                    elif v1 > v2: total_scores[m2] += 1
                    else:
                        total_scores[m1] += 0.5
                        total_scores[m2] += 0.5

    # Normalization: divide by the total possible matchup points
    max_score = valid_images * (len(METHOD_KEYS) - 1) * 2
    if max_score > 0:
        return {k: v / max_score for k, v in total_scores.items()}
    return total_scores

def compute_human_scores(csv_dir):
    """
    Extract human subjective scores from multiple CSV files.
    """
    csv_files =[os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    # Calculate scores separately based on different metrics
    total_scores = {m: {k: 0.0 for k in METHOD_KEYS} for m in METRICS_CONFIG.keys()}
    image_counts = {m: 0 for m in METRICS_CONFIG.keys()}
    
    for file in csv_files:
        df = pd.read_csv(file, sep='\t')
        compare_columns = df.columns[1:]
        
        for index, row in df.iterrows():
            # Determine which evaluation metric this row belongs to based on the row index
            target_metric = None
            for m_name, m_info in METRICS_CONFIG.items():
                start, end = m_info['row_range']
                if start <= index <= end:
                    target_metric = m_name
                    break
            
            if target_metric is None:
                continue
                
            img_name = row[df.columns[0]].split('/')[-1].rsplit('.', 1)[0]
            if img_name not in METRICS_CONFIG[target_metric]['images']:
                continue
                
            image_counts[target_metric] += 1
            
            for col in compare_columns:
                comparison = str(row[col])
                if '>' in comparison:
                    winner = comparison.split('>')[0]
                    total_scores[target_metric][CSV_MAP[winner]] += 1
                elif '<' in comparison:
                    winner = comparison.split('<')[1]
                    total_scores[target_metric][CSV_MAP[winner]] += 1
                elif '=' in comparison:
                    w, l = comparison.split('=')
                    total_scores[target_metric][CSV_MAP[w]] += 0.5
                    total_scores[target_metric][CSV_MAP[l]] += 0.5

    # Normalize the human scores
    average_scores = {}
    for metric_name, scores in total_scores.items():
        if image_counts[metric_name] > 0:
            # Divide by the frequency of comparisons per row to standardize
            average_scores[metric_name] = {k: v / (image_counts[metric_name] * 6) for k, v in scores.items()}
        else:
            average_scores[metric_name] = scores
            
    return average_scores

def plt_figure(metric_name, machine_scores, human_scores, title, output_dir):
    """x = human win ratio, y = machine win ratio. 每轴 7 方法之和应为 3.5."""
    x = [human_scores[m] for m in METHOD_KEYS]
    y = [machine_scores[m] for m in METHOD_KEYS]
    labels = METHOD_KEYS

    coefficients = np.polyfit(x, y, 1)
    regression_line = np.poly1d(coefficients)
    rho, p_value = spearmanr(x, y)
    print(f"[{title}] Spearman correlation coefficient: {rho:.4f}, p = {p_value:.4g}")
    print(f"[{title}] sum(x/human)={sum(x):.4f}, sum(y/machine)={sum(y):.4f} (应为3.5)")

    x_fit = np.linspace(0.05, 0.95, 100)
    y_fit = regression_line(x_fit)

    plt.figure(figsize=(5, 4), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, len(x)))
    markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', '<', '>']

    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=[colors[i]], marker=markers[i % len(markers)], s=100, zorder=5)

    plt.plot(x_fit, y_fit, linestyle='-', color='#f07167', linewidth=4)
    textstr = f'ρ = {rho:.4f}\n'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=24, verticalalignment='top', color='#f07167')
    plt.title(title, fontsize=24, pad=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{title}.png'), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Compute BT machine-human Spearman correlations.')
    parser.add_argument('--csv-dir', default=os.path.abspath(os.path.join(script_dir, '..', 'results_v2')),
                        help='Directory containing human-preference CSV files.')
    parser.add_argument('--metrics', nargs='+', choices=METRICS_CONFIG.keys(),
                        default=['Aesthetic', 'Background', 'Cross', 'Hand', 'Text'],
                        help='Metrics to process; defaults to all five metrics including Aesthetic.')
    parser.add_argument('--scores-dir', default=script_dir,
                        help='Directory containing the unified machine-score TXT files.')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for correlation plots; defaults to <scores-dir>/output_plots.')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.scores_dir, 'output_plots')
    txt_paths = {
        'Aesthetic': os.path.join(args.scores_dir, 'aesthetic_scores.txt'),
        'Background': os.path.join(args.scores_dir, 'bg_scores.txt'),
        'Cross': os.path.join(args.scores_dir, 'cross_scores.txt'),
        'Hand': os.path.join(args.scores_dir, 'hand_scores.txt'),
        'Text': os.path.join(args.scores_dir, 'text_scores.txt'),
    }
    all_human_scores = compute_human_scores(args.csv_dir)

    for metric_name in args.metrics:
        print(f"--- Processing {metric_name} ---")
        machine_scores = compute_machine_scores(txt_paths[metric_name], METRICS_CONFIG[metric_name])
        human_scores = all_human_scores.get(metric_name, {k: 0 for k in METHOD_KEYS})
        print("Machine Scores:", {k: round(v, 4) for k, v in machine_scores.items()})
        print("Human Scores:  ", {k: round(v, 4) for k, v in human_scores.items()})
        plt_figure(metric_name, machine_scores, human_scores, title=f'{metric_name} Consistency', output_dir=output_dir)
