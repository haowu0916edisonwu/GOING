"""
Evaluation metrics aligned with the original experimental setup.
"""

import re
import string
from collections import Counter
from typing import List


class Metrics:
    """Provide metric utilities used throughout the project."""

    @staticmethod
    def normalize_answer(s: str) -> str:
        """Normalize an answer using the standard DPR/NQ-style pipeline."""

        def remove_articles(text: str) -> str:
            """Remove English articles from the text."""
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            """Collapse repeated whitespace."""
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            """Remove punctuation characters from the text."""
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            """Lowercase the input text."""
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def compute_span_em(prediction: str, ground_truths: List[str]) -> float:
        """Compute span exact match with simple digit-word normalization."""
        num_map = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
            "10": "ten",
        }

        pred_norm = Metrics.normalize_answer(prediction)
        pred_text_version = pred_norm
        for key, value in num_map.items():
            pred_text_version = re.sub(r"\b" + key + r"\b", value, pred_text_version)

        for gt in ground_truths:
            gt_norm = Metrics.normalize_answer(gt)

            if gt_norm in pred_norm:
                return 1.0

            if gt_norm in pred_text_version:
                return 1.0

            gt_digit_version = gt_norm
            for key, value in num_map.items():
                gt_digit_version = re.sub(r"\b" + value + r"\b", key, gt_digit_version)

            if gt_digit_version in pred_norm:
                return 1.0

        return 0.0

    @staticmethod
    def compute_f1(prediction: str, ground_truths: List[str]) -> float:
        """Compute SQuAD-style token F1."""

        def get_tokens(text: str) -> List[str]:
            """Tokenize text after normalization."""
            return Metrics.normalize_answer(text).split()

        pred_tokens = get_tokens(prediction)
        if not pred_tokens:
            return 0.0

        max_f1 = 0.0
        for gt in ground_truths:
            gt_tokens = get_tokens(gt)
            if not gt_tokens:
                continue

            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)

        return max_f1

    @staticmethod
    def compute_rouge_l(prediction: str, ground_truths: List[str]) -> float:
        """Compute ROUGE-L and fall back to F1 if the package is missing."""
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            max_score = 0.0
            for gt in ground_truths:
                score = scorer.score(gt, prediction)["rougeL"].fmeasure
                max_score = max(max_score, score)
            return max_score
        except ImportError:
            print("Warning: rouge-score is not installed; using F1 as an approximation.")
            return Metrics.compute_f1(prediction, ground_truths)

    @staticmethod
    def compute_accuracy(prediction: str, ground_truths: List[str]) -> float:
        """Compute FactKG accuracy with mutually exclusive label handling."""
        gt_label = ground_truths[0].lower().strip()
        pred_clean = prediction.lower()
        pred_clean = "".join(c for c in pred_clean if c.isalnum() or c.isspace())
        pred_tokens = pred_clean.split()

        if pred_clean.startswith(gt_label):
            return 1.0

        has_true = "true" in pred_tokens
        has_false = "false" in pred_tokens

        if gt_label == "true":
            if has_true and (not has_false or pred_clean.find("true") < pred_clean.find("false")):
                return 1.0
        elif gt_label == "false":
            if has_false and (not has_true or pred_clean.find("false") < pred_clean.find("true")):
                return 1.0

        return 0.0
