# -*- coding: utf-8 -*-
"""Authoritative pair-level metric-vs-human agreement computation.

Pipeline (single source of truth, supersedes all previous ad-hoc summaries):

1. Human aggregation: for each (dimension, image, method-pair) item, the 20
   questionnaires' votes over {left-win, tie, right-win} are aggregated by
   strict plurality. If two or more outcomes share the top vote count, the
   aggregated human outcome is a TIE (no unique plurality winner).
2. Machine outcome: from the unified score TXT files in --scores-dir (the
   exact same files used by metric_bt.py for the Spearman correlations).
   Positive metrics (aesthetic/cross/text): higher score wins. Negative
   metrics (bg/hand): lower score wins. Equal scores -> tie.
3. Agreement: machine outcome == human outcome counts 1 (same winner, or
   both tie), otherwise 0.
"""
import argparse
import json
import os
from collections import Counter

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

CSV_MAP = {'A': 'SD', 'B': 'IDM', 'C': 'CatVTON', 'D': 'HandFit', 'E': 'FitDiT', 'F': 'CrossVTON', 'G': 'Qwen-Edit'}
TXT_KEY_MAP = {
    'SD': 'SD', 'SD-VITON': 'SD', 'OOTD': 'SD',
    'IDM': 'IDM', 'idm': 'IDM',
    'CatVTON': 'CatVTON',
    'FitDiT': 'FitDiT', 'FitDit': 'FitDiT',
    'CrossVTON': 'CrossVTON', 'cross_vton': 'CrossVTON',
    'HandFit': 'HandFit', 'vton-handfit': 'HandFit', 'vton_handfit': 'HandFit',
    'Qwen-Edit': 'Qwen-Edit',
}
PAIR_ORDER = (
    ('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('A', 'F'),
    ('B', 'C'), ('B', 'D'), ('B', 'E'), ('B', 'F'), ('C', 'D'),
    ('C', 'E'), ('C', 'F'), ('D', 'E'), ('D', 'F'), ('E', 'F'),
    ('A', 'G'), ('B', 'G'), ('C', 'G'), ('D', 'G'), ('E', 'G'), ('F', 'G'),
)
# dimension -> (row range, score txt, is_positive)
DIMENSIONS = {
    'aesthetic': (range(0, 20), 'aesthetic_scores.txt', True),
    'bg': (range(20, 40), 'bg_scores.txt', False),
    'cross': (range(40, 60), 'cross_scores.txt', True),
    'hand': (range(60, 80), 'hand_scores.txt', False),
    'text': (range(80, 100), 'text_scores.txt', True),
}

RULE_DEFINITION = (
    'For each image, the votes over {left-win, tie, right-win} from all questionnaires '
    'are aggregated per method pair by strict plurality (the most voted outcome wins; '
    'if two or more outcomes share the top count, the aggregated outcome is a tie). '
    'The machine outcome is then compared with the aggregated human outcome for the '
    '21 unique method pairs; the same winner or a tie on both sides counts 1, '
    'otherwise 0.'
)


def parse_left_score(value, left, right):
    value = str(value)
    if '>' in value:
        winner, loser = value.split('>')
        if {winner, loser} != {left, right}:
            raise ValueError(f'Unexpected pair {value}; expected {left}/{right}')
        return 1.0 if winner == left else 0.0
    if '<' in value:
        loser, winner = value.split('<')
        if {winner, loser} != {left, right}:
            raise ValueError(f'Unexpected pair {value}; expected {left}/{right}')
        return 1.0 if winner == left else 0.0
    first, second = value.split('=')
    if {first, second} != {left, right}:
        raise ValueError(f'Unexpected pair {value}; expected {left}/{right}')
    return 0.5


def load_machine_scores(txt_path):
    scores = {}
    with open(txt_path, encoding='utf-8') as fh:
        lines = [l.strip() for l in fh if l.strip()]
    headers = [h.strip() for h in lines[0].split(',')]
    for line in lines[1:]:
        if line.startswith(('====', '----')):
            continue
        cells = [c.strip() for c in line.split(',')]
        if len(cells) < len(headers):
            continue
        img = cells[0].rsplit('.', 1)[0]
        entry = {}
        for i, col in enumerate(headers[1:], start=1):
            std = TXT_KEY_MAP.get(col)
            if std:
                try:
                    entry[std] = float(cells[i])
                except ValueError:
                    entry[std] = 0.0
        scores[img] = entry
    return scores


def human_outcome(frames, row, col, left, right):
    """Strict plurality over the 20 votes; shared top count -> 'tie'."""
    counts = Counter()
    for fr in frames:
        s = parse_left_score(fr.iloc[row][col], left, right)
        counts['left' if s == 1.0 else 'tie' if s == 0.5 else 'right'] += 1
    top = max(counts.values())
    winners = [k for k, v in counts.items() if v == top]
    if len(winners) > 1:
        return 'tie', dict(counts)
    return winners[0], dict(counts)


def machine_outcome(vals, left, right, positive):
    v1, v2 = vals.get(CSV_MAP[left], 0.0), vals.get(CSV_MAP[right], 0.0)
    if v1 == v2:
        return 'tie'
    if positive:
        return 'left' if v1 > v2 else 'right'
    return 'left' if v1 < v2 else 'right'


def run(csv_dir, scores_dir):
    files = sorted(f for f in os.listdir(csv_dir) if f.endswith('.csv'))
    csv_paths = [os.path.join(csv_dir, f) for f in files]
    frames = [pd.read_csv(p, sep='\t') for p in csv_paths]
    columns = list(frames[0].columns[1:])

    sample_rows = []
    categories = []
    grand_agree = grand_total = 0
    for dim, (rows, txt, positive) in DIMENSIONS.items():
        txt_path = os.path.join(scores_dir, txt)
        machine = load_machine_scores(txt_path)
        dim_agree = dim_total = 0
        for row in rows:
            img = frames[0].iloc[row, 0].split('/')[-1].rsplit('.', 1)[0]
            vals = machine.get(img)
            invalid = int(vals is None)
            vals = vals or {}
            sample_agree = 0
            for ci, (left, right) in enumerate(PAIR_ORDER):
                h_out, _ = human_outcome(frames, row, columns[ci], left, right)
                m_out = machine_outcome(vals, left, right, positive)
                sample_agree += int(h_out == m_out)
            sample_rows.append({
                'category': dim, 'image': img, 'respondent_count': len(frames),
                'aggregated_pair_count': len(PAIR_ORDER), 'exact_agreement_count': sample_agree,
                'pair_agreement_rate': round(sample_agree / len(PAIR_ORDER), 6),
                'invalid_comparison_count': invalid * len(PAIR_ORDER),
            })
            dim_agree += sample_agree
            dim_total += len(PAIR_ORDER)
        categories.append({
            'category': dim, 'evaluated_samples': len(rows),
            'aggregated_pair_count': dim_total, 'exact_agreement_count': dim_agree,
            'overall_pair_agreement_rate': round(dim_agree / dim_total, 6),
            'mean_sample_agreement_rate': round(
                sum(r['pair_agreement_rate'] for r in sample_rows if r['category'] == dim) / len(rows), 6),
        })
        grand_agree += dim_agree
        grand_total += dim_total
    summary = {
        'definition': {
            'pair_agreement_rate': RULE_DEFINITION,
            'aggregated_pair_count': 'Each complete image has 21 pairs, i.e. all pairwise combinations of 7 methods.',
            'score_direction': 'Higher is better for aesthetic/text/cross; lower is better for bg/hand.',
            'human_tie_rule': 'If two or more outcomes share the top vote count (e.g. 7/7/6, 10/10/0), '
                              'the aggregated outcome is a tie; a one-vote margin (e.g. 8/7/5) is a '
                              'decisive plurality win, not a tie.',
        },
        'categories': categories,
        'overall': {
            'aggregated_pair_count': grand_total,
            'exact_agreement_count': grand_agree,
            'overall_pair_agreement_rate': round(grand_agree / grand_total, 6),
        },
        'skipped_samples': [],
    }
    return sample_rows, summary


def main():
    parser = argparse.ArgumentParser(description='Authoritative metric-vs-human pair agreement.')
    parser.add_argument('--csv-dir', default=os.path.abspath(os.path.join(BASE, '..', 'results_v2')))
    parser.add_argument('--scores-dir', default=os.path.abspath(os.path.join(BASE, '..', 'human_preference')))
    parser.add_argument('--out-dir', default=None,
                        help='Write sample_agreement.csv and agreement_summary.json here.')
    args = parser.parse_args()

    sample_rows, summary = run(args.csv_dir, args.scores_dir)
    for cat in summary['categories']:
        print(f"[{cat['category']}] {cat['exact_agreement_count']}/{cat['aggregated_pair_count']} "
              f"= {cat['overall_pair_agreement_rate']:.4f}")
    ov = summary['overall']
    print(f"[overall] {ov['exact_agreement_count']}/{ov['aggregated_pair_count']} "
          f"= {ov['overall_pair_agreement_rate']:.4f}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        df = pd.DataFrame(sample_rows, columns=['category', 'image', 'respondent_count',
                                                'aggregated_pair_count', 'exact_agreement_count',
                                                'pair_agreement_rate', 'invalid_comparison_count'])
        df.to_csv(os.path.join(args.out_dir, 'sample_agreement.csv'), index=False)
        with open(os.path.join(args.out_dir, 'agreement_summary.json'), 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f'Wrote {args.out_dir}')


if __name__ == '__main__':
    main()
