import string
import json
import os
import argparse
from rouge_score import rouge_scorer
from transformers import AutoTokenizer
from typing import List
from nltk.translate.bleu_score import sentence_bleu

default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge(prediction, ground_truth, xlingual=False):
    scorer = default_rouge_scorer
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def bleu4(prediction: str, ground_truths: List[str]):
    from nltk.translate import bleu
    from nltk.translate.bleu_score import SmoothingFunction
    smoothie = SmoothingFunction().method4
    prediction = prediction.split()
    ground_truths = [x.split() for x in ground_truths]
    score = bleu(ground_truths, prediction, smoothing_function=smoothie)
    return score


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths: List[str], xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions: List[str], references: List[List], xlingual=False, zh_bleu=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."

    min_length = min(len((predictions)), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]
    if zh_bleu:
        # process chinese text
        predictions = [" ".join([y for y in x]) for x in predictions]
        references = [[" ".join([y for y in x]) for x in rs] for rs in references]

    em, rougeL = 0, 0
    bleu = 0.
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        em += metric_max_over_ground_truths(
            exact_match, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rouge, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        bleu += bleu4(pred, gold)
    em = 100.0 * em / len(references)
    rougeL = 100.0 * rougeL / len(references)
    bleu = 100.0 * bleu / len(references)
    metrics = {"exact_match": em, "rougeL": rougeL, "bleu": bleu}
    metrics = {k: round(v, 6) for k, v in metrics.items()}
    return metrics


# def compute_grouped_metrics(predictions, references, groups, xlingual=False):
#     assert len(predictions) == len(references) == len(groups)
#
#     examples_by_group = {}
#     for pred, gold, group in zip(predictions, references, groups):
#         if group not in examples_by_group:
#             examples_by_group[group] = []
#         examples_by_group[group].append((pred, gold))
#
#     results = {}
#     for group, group_examples in examples_by_group.items():
#         task_predictions, task_references = zip(*group_examples)
#         group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
#         for metric, value in group_metrics.items():
#             results[f"{metric}_for_{group}"] = value
#     return results


if __name__ == "__main__":
    pass
