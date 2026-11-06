import warnings
warnings.filterwarnings('ignore')

import jieba
import nltk
import numpy as np
import json

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.linalg import norm

def contain_chinese_string(s):
    """Check if the string contains Chinese characters"""
    for c in s:
        if '\u4e00' <= c <= '\u9fff':
            return True
    return False

def precision(reference, hypothesis):
    """Calculate precision between reference and hypothesis"""
    tp = len(set(reference) & set(hypothesis))
    fp = len(set(hypothesis) - set(reference))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall(reference, hypothesis):
    """Calculate recall between reference and hypothesis"""
    tp = len(set(reference) & set(hypothesis))
    fn = len(set(reference) - set(hypothesis))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def tf_similarity(s1, s2):
    """Calculate cosine similarity based on term frequency (TF)"""
    def add_space(s):
        return ' '.join(list(s))

    s1, s2 = add_space(s1), add_space(s2)
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

def get_metrics(pred, gt, minlen=1, heavy_mode: int = 2):
    """
    Compute multiple text similarity metrics:
    tf_similarity: Cosine similarity between word vectors
    precision: Precision score (when heavy_mode >= 1)
    recall: Recall score (when heavy_mode >= 1)
    edit_dist: Normalized edit distance (when heavy_mode == 2)
    All scores are normalized to [0,1], higher = better
    """
    metrics = {}

    if len(pred) < minlen or len(gt) < minlen:
        return metrics

    tf_score = tf_similarity(pred, gt)
    metrics['tf_similarity'] = tf_score

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()


    reference_set = set(reference)
    hypothesis_set = set(hypothesis)

    if heavy_mode >= 1:
        metrics["precision"] = precision(reference_set, hypothesis_set)
        metrics["recall"] = recall(reference_set, hypothesis_set)
    if heavy_mode == 2:
        metrics["edit_dist"] = (1 - nltk.edit_distance(pred, gt) / max(len(pred), len(gt)))
    return metrics


def addweightSum(metrics: dict, weights: dict = None):
    """Compute weighted sum of all metrics"""
    if weights is None:
        weights = {
            'tf_similarity': 1,
            'precision': 1,
            'recall': 1,
            'edit_dist': 1
        }

    keys = list(weights.keys())
    score = sum([metrics[k] * weights[k] for k in keys]) / len(keys)
    return score


def cloth2clothes(clothes1: list, clothes2: list):
    """
    Compare OCR results between cloth image and try-on result image
    clothes1: OCR result from the cloth image
    clothes2: OCR result from the try-on image
    Returns average metrics and detailed matching results
    """
    text_cloth1 = [i['text'] for i in clothes1]
    text_cloth2 = [i['text'] for i in clothes2]

    result_data = []
    for txt1 in text_cloth1:
        _res = {}
        for txt2 in text_cloth2:
            metrics = get_metrics(txt2, txt1)
            _score = addweightSum(metrics, weights=None)
            _res[_score] = metrics

        max_score = max(_res.keys())
        result_data.append(_res[max_score])

    total_metrics = {
        'tf_similarity': 0,
        'precision': 0,
        'recall': 0,
        'edit_dist': 0
    }

    for metrics in result_data:
        for key in total_metrics:
            if key in metrics:
                total_metrics[key] += metrics[key]

    num_results = len(result_data)
    avg_metrics = {key: value / num_results for key, value in total_metrics.items()}

    return avg_metrics, result_data


if __name__ == "__main__":
    # Minimal example: OCR texts of a garment image vs. a try-on result
    garment_texts = [{'text': 'NIKE AIR'}, {'text': 'JUST DO IT'}]
    result_texts = [{'text': 'NIKE AIR'}, {'text': 'JUST D0 IT'}]
    avg_metrics, _ = cloth2clothes(garment_texts, result_texts)
    print(avg_metrics)