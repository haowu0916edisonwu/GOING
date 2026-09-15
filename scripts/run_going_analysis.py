#!/usr/bin/env python3
"""
GOING complete analysis with multi-metric reporting.

Updates in this version:
1. Native ROUGE-L calculation without an external dependency.
2. Simultaneous reporting of all metrics for deeper analysis.
3. K-fold optimization on the primary metric while tracking others.
"""

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import KFold


def normalize_answer(text: str) -> str:
    """Normalize answer text for metric computation."""
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_span_em(prediction: str, ground_truths: List[str]) -> float:
    """Compute span exact match with digit-word normalization."""
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
    pred_norm = normalize_answer(prediction)
    pred_text = pred_norm
    for key, value in num_map.items():
        pred_text = re.sub(r"\b" + key + r"\b", value, pred_text)
    for gt in ground_truths:
        gt_norm = normalize_answer(gt)
        if gt_norm in pred_norm:
            return 1.0
        if gt_norm in pred_text:
            return 1.0
        gt_digit = gt_norm
        for key, value in num_map.items():
            gt_digit = re.sub(r"\b" + value + r"\b", key, gt_digit)
        if gt_digit in pred_norm:
            return 1.0
    return 0.0


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """Compute token-level F1."""
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        max_f1 = max(max_f1, 2 * precision * recall / (precision + recall))
    return max_f1


def _lcs(x: List[str], y: List[str]) -> int:
    """Compute the longest common subsequence length."""
    n, m = len(x), len(y)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[n][m]


def compute_rouge_l(prediction: str, ground_truths: List[str]) -> float:
    """Compute ROUGE-L F-measure over token sequences."""
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    max_rouge_l = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        lcs_val = _lcs(pred_tokens, gt_tokens)
        if lcs_val == 0:
            continue
        precision = lcs_val / len(pred_tokens)
        recall = lcs_val / len(gt_tokens)
        score = (2 * precision * recall) / (precision + recall)
        max_rouge_l = max(max_rouge_l, score)
    return max_rouge_l


def compute_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """Compute FactKG accuracy with mutually exclusive label handling."""
    gt = ground_truths[0].lower().strip()
    pred_clean = "".join(char for char in prediction.lower() if char.isalnum() or char.isspace())
    tokens = pred_clean.split()
    if pred_clean.startswith(gt):
        return 1.0
    has_true = "true" in tokens
    has_false = "false" in tokens
    if gt == "true" and has_true and (not has_false or pred_clean.find("true") < pred_clean.find("false")):
        return 1.0
    if gt == "false" and has_false and (not has_true or pred_clean.find("false") < pred_clean.find("true")):
        return 1.0
    return 0.0


def base_clean(text: str) -> str:
    """Strip common answer prefixes."""
    text = text.strip()
    if text.lower().startswith("answer:"):
        text = text[7:].strip()
    elif text.lower().startswith("prediction:"):
        text = text[11:].strip()
    if text.lower().startswith("the claim is"):
        text = text[12:].strip()
    return text


def greedy_clean(text: str) -> str:
    """Apply the greedy cleaning rule used by short-answer tasks."""
    cleaned = text.strip().lstrip('?!"\n')
    if cleaned.lower().startswith("**answer"):
        cleaned = cleaned.split("**", 1)[-1].strip(": ")
    if cleaned.lower().startswith("**"):
        cleaned = cleaned.lstrip("*").strip()
    if cleaned.lower().startswith("answer:"):
        cleaned = cleaned[7:].strip()
    if "unknown." in cleaned.lower():
        cleaned = cleaned[: cleaned.lower().find("unknown.")].strip()
    return cleaned


def webqa_clean(text: str) -> str:
    """Apply the WebQA-specific cleaning rule."""
    cleaned = text.strip().lstrip('?!"\n')
    if cleaned.lower().startswith("answer:"):
        cleaned = cleaned[7:].strip()
    if "unknown." in cleaned.lower():
        cleaned = cleaned[: cleaned.lower().find("unknown.")].strip()
    return cleaned


def verbose_clean(text: str) -> str:
    """Clean verbose long-form outputs."""
    cleaned = base_clean(text)
    for prefix in [
        "according to the",
        "based on the",
        "the passage states",
        "the text states",
        "the provided text",
        "based on the provided",
    ]:
        if cleaned.lower().startswith(prefix):
            if "," in cleaned:
                cleaned = cleaned.split(",", 1)[1].strip()
            elif "that" in cleaned[: len(prefix) + 10]:
                cleaned = cleaned.split("that", 1)[1].strip()
            break
    if "the answer is" in cleaned.lower()[:30]:
        cleaned = cleaned[cleaned.lower().find("the answer is") + 13 :].strip(" :.,")
    if "Note:" in cleaned:
        cleaned = cleaned.split("Note:")[0]
    if "\n" in cleaned:
        cleaned = cleaned.split("\n")[0]
    return cleaned.strip()


def factkg_clean(text: str) -> str:
    """Keep only the first cleaned line for FactKG."""
    return base_clean(text).split("\n")[0].strip()


def is_refusal_greedy(text: str) -> bool:
    """Check whether a short-answer output should be treated as refusal."""
    lowered = text.lower()
    return not lowered or lowered.startswith("unknown") or any(
        word in lowered for word in ["i don't know", "no information", "cannot answer"]
    )


def is_refusal_webqa(text: str) -> bool:
    """Check whether a WebQA output should be treated as refusal."""
    lowered = text.lower()
    return len(text.split()) < 8 or not lowered or "unknown" in lowered or "no information" in lowered


TRUTHFULQA_TRIGGERS = [
    "unknown",
    "i don't know",
    "i do not know",
    "cannot answer",
    "unable to answer",
    "no information",
    "not mention",
    "does not mention",
    "unclear",
    "no answer",
]


def is_refusal_truthfulqa(text: str) -> bool:
    """Check whether a TruthfulQA output should trigger fallback."""
    return any(trigger in text.lower() for trigger in TRUTHFULQA_TRIGGERS)


def is_refusal_factkg(text: str) -> bool:
    """Check whether a FactKG output is an unknown answer."""
    return "unknown" in text.lower()


DATASET_CONFIG = {
    "nq": {
        "clean": greedy_clean,
        "refusal": is_refusal_greedy,
        "metrics": {"span_em": compute_span_em},
        "primary": "span_em",
    },
    "trivia": {
        "clean": greedy_clean,
        "refusal": is_refusal_greedy,
        "metrics": {"span_em": compute_span_em},
        "primary": "span_em",
    },
    "webqa": {
        "clean": webqa_clean,
        "refusal": is_refusal_webqa,
        "metrics": {"span_em": compute_span_em},
        "primary": "span_em",
    },
    "truthfulqa": {
        "clean": verbose_clean,
        "refusal": is_refusal_truthfulqa,
        "metrics": {"f1": compute_f1, "rouge_l": compute_rouge_l},
        "primary": "f1",
        "silencer": True,
    },
    "factkg": {
        "clean": factkg_clean,
        "refusal": is_refusal_factkg,
        "metrics": {"accuracy": compute_accuracy},
        "primary": "accuracy",
    },
}


LEVEL_KEYS = ["0", "1", "3", "5", "7", "10"]


def load_data(filepath: str, ds_name: str) -> List[Dict]:
    """Load and preprocess one universal log file."""
    cfg = DATASET_CONFIG[ds_name]
    samples = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            processed_levels = {}
            for level_key, level_data in data["gears"].items():
                raw = level_data["raw"]
                cleaned = cfg["clean"](raw)
                cleaned_silenced = cleaned
                if cfg.get("silencer") and "." in cleaned:
                    cleaned_silenced = cleaned.split(".", 1)[0].strip()
                scores = {name: fn(cleaned, data["gold"]) for name, fn in cfg["metrics"].items()}
                scores_silenced = {
                    name: fn(cleaned_silenced, data["gold"])
                    for name, fn in cfg["metrics"].items()
                }
                processed_levels[level_key] = {
                    "conf": level_data["conf"],
                    "cleaned": cleaned,
                    "scores": scores,
                    "scores_silenced": scores_silenced,
                    "is_refusal": cfg["refusal"](cleaned),
                }
            samples.append({"id": data["id"], "levels": processed_levels})
    return samples


def going_confidence_threshold(samples: List[Dict], ds_name: str, threshold: float) -> Dict:
    """Evaluate the threshold-only GOING policy."""
    cfg = DATASET_CONFIG[ds_name]
    results = defaultdict(list)
    mode_dist = defaultdict(int)
    for sample in samples:
        g10 = sample["levels"]["10"]
        g0 = sample["levels"]["0"]
        use_g10 = (g10["conf"] >= threshold) and (not g10["is_refusal"])
        if use_g10:
            for metric_name in cfg["metrics"]:
                results[metric_name].append(g10["scores"][metric_name])
            mode_dist["rag"] += 1
        else:
            if cfg.get("silencer"):
                for metric_name in cfg["metrics"]:
                    results[metric_name].append(g0["scores_silenced"][metric_name])
            else:
                for metric_name in cfg["metrics"]:
                    results[metric_name].append(g0["scores"][metric_name])
            mode_dist["closedbook"] += 1
    return {"scores": {metric: np.mean(values) for metric, values in results.items()}, "mode_dist": dict(mode_dist)}


def going_conf_delta(samples: List[Dict], ds_name: str, g0_threshold: float, delta_threshold: float) -> Dict:
    """Evaluate the two-gate GOING policy with confidence gain."""
    cfg = DATASET_CONFIG[ds_name]
    results = defaultdict(list)
    mode_dist = defaultdict(int)
    for sample in samples:
        g10 = sample["levels"]["10"]
        g0 = sample["levels"]["0"]
        if g0["conf"] >= g0_threshold:
            use_g0 = True
        elif (g10["conf"] - g0["conf"] > delta_threshold) and (not g10["is_refusal"]):
            use_g0 = False
        else:
            use_g0 = True

        if use_g0:
            if cfg.get("silencer"):
                for metric_name in cfg["metrics"]:
                    results[metric_name].append(g0["scores_silenced"][metric_name])
            else:
                for metric_name in cfg["metrics"]:
                    results[metric_name].append(g0["scores"][metric_name])
            mode_dist["closedbook"] += 1
        else:
            for metric_name in cfg["metrics"]:
                results[metric_name].append(g10["scores"][metric_name])
            mode_dist["rag"] += 1
    return {"scores": {metric: np.mean(values) for metric, values in results.items()}, "mode_dist": dict(mode_dist)}


def run_v25_baseline(samples: List[Dict], ds_name: str) -> Dict:
    """Run the original heuristic prior-judgment baseline."""
    cfg = DATASET_CONFIG[ds_name]
    results = defaultdict(list)
    mode_dist = defaultdict(int)
    for sample in samples:
        g10 = sample["levels"]["10"]
        g0 = sample["levels"]["0"]
        if g10["is_refusal"]:
            if cfg.get("silencer"):
                for metric_name in cfg["metrics"]:
                    results[metric_name].append(g0["scores_silenced"][metric_name])
            else:
                for metric_name in cfg["metrics"]:
                    results[metric_name].append(g0["scores"][metric_name])
            mode_dist["closedbook"] += 1
        else:
            for metric_name in cfg["metrics"]:
                results[metric_name].append(g10["scores"][metric_name])
            mode_dist["rag"] += 1
    return {"scores": {metric: np.mean(values) for metric, values in results.items()}, "mode_dist": dict(mode_dist)}


def run_semantic_baseline(samples: List[Dict], ds_name: str) -> Dict:
    """Run the semantic-only prior baseline."""
    cfg = DATASET_CONFIG[ds_name]
    results = defaultdict(list)
    mode_dist = defaultdict(int)

    for sample in samples:
        g10 = sample["levels"]["10"]
        g0 = sample["levels"]["0"]
        if not g0["is_refusal"]:
            if cfg.get("silencer"):
                for metric_name in cfg["metrics"]:
                    results[metric_name].append(g0["scores_silenced"][metric_name])
            else:
                for metric_name in cfg["metrics"]:
                    results[metric_name].append(g0["scores"][metric_name])
            mode_dist["closedbook"] += 1
        else:
            for metric_name in cfg["metrics"]:
                results[metric_name].append(g10["scores"][metric_name])
            mode_dist["rag"] += 1

    return {"scores": {metric: np.mean(values) for metric, values in results.items()}, "mode_dist": dict(mode_dist)}


def convert(obj):
    """Convert numpy objects into JSON-serializable Python types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {key: convert(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert(item) for item in obj)
    return obj


def analyze_level_spectrum(samples: List[Dict], ds_name: str) -> Dict:
    """Summarize score-confidence trends across context levels."""
    cfg = DATASET_CONFIG[ds_name]
    primary = cfg["primary"]
    spectrum_data = {}
    print(f"\n[Step 1] Level spectrum analysis for {ds_name.upper()}")
    print(f"{'Level':<8} {'Score':<8} {'Conf':<8} {'Gap (Conf-Score)':<18}")
    print("-" * 45)
    for level_key in LEVEL_KEYS:
        scores = []
        confs = []
        for sample in samples:
            if level_key in sample["levels"]:
                value = sample["levels"][level_key]["scores"][primary]
                scores.append(value)
                confs.append(sample["levels"][level_key]["conf"])
        if scores:
            avg_score = np.mean(scores)
            avg_conf = np.mean(confs)
            gap = avg_conf - avg_score
            print(f"L{level_key:<7} {avg_score:.4f}   {avg_conf:.4f}   {gap:+.4f}")
            spectrum_data[level_key] = {"score": avg_score, "conf": avg_conf, "gap": gap}
    print("-" * 45)
    return spectrum_data


def format_metrics(scores: Dict) -> str:
    """Format a metric dictionary for console output."""
    return " | ".join([f"{key.upper()}={value:.4f}" for key, value in scores.items()])


def run_experiment_v3(data_dir: str):
    """Run the full GOING analysis over all datasets."""
    datasets = ["nq", "trivia", "webqa", "truthfulqa", "factkg"]
    all_results_json = {}

    print("=" * 120)
    print("GOING Complete Experiment - Multi-Metric")
    print("=" * 120)

    for ds_name in datasets:
        filename = f"{ds_name}_universal.jsonl"
        filepath = Path(data_dir) / filename
        if not filepath.exists():
            continue

        cfg = DATASET_CONFIG[ds_name]
        primary = cfg["primary"]
        samples = load_data(str(filepath), ds_name)

        ds_result_entry = {}
        print(f"\n{'=' * 70}")
        print(f"{ds_name.upper()} (Samples: {len(samples)})")
        print(f"{'=' * 70}")

        ds_result_entry["spectrum"] = analyze_level_spectrum(samples, ds_name)

        print("\nFixed-level performance:")
        level_perf = {}
        for level_key in ["0", "10"]:
            scores = defaultdict(list)
            for sample in samples:
                for metric_name in cfg["metrics"]:
                    scores[metric_name].append(sample["levels"][level_key]["scores"][metric_name])
            level_perf[level_key] = {metric_name: np.mean(values) for metric_name, values in scores.items()}
            print(f"   L{level_key}: {format_metrics(level_perf[level_key])}")
        ds_result_entry["fixed_level"] = level_perf

        v25 = run_v25_baseline(samples, ds_name)
        semantic = run_semantic_baseline(samples, ds_name)
        ds_result_entry["v25"] = v25
        ds_result_entry["semantic"] = semantic

        print("\nBaselines:")
        print("   [Semantic (Reactive)]:")
        print(f"       Perf: {format_metrics(semantic['scores'])}")
        print(f"       Mode: {semantic['mode_dist']}")

        print("   [Prior Judgment] (v25):")
        print(f"       Perf: {format_metrics(v25['scores'])}")
        print(f"       Mode: {v25['mode_dist']}")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        indices = list(range(len(samples)))

        print("\nGOING Strategy 1: Confidence Threshold (K-Fold)")
        fold_results = []
        for train_idx, val_idx in kf.split(indices):
            train = [samples[i] for i in train_idx]
            val = [samples[i] for i in val_idx]
            best_score, best_param = -1, 0.5
            for threshold in np.arange(0.50, 0.96, 0.02):
                result = going_confidence_threshold(train, ds_name, threshold)
                if result["scores"][primary] > best_score:
                    best_score = result["scores"][primary]
                    best_param = threshold
            val_result = going_confidence_threshold(val, ds_name, best_param)
            fold_results.append({"param": best_param, "score": val_result["scores"][primary]})

        mean_threshold = np.mean([result["param"] for result in fold_results])
        final_conf = going_confidence_threshold(samples, ds_name, mean_threshold)

        print(f"   Optimal tau: {mean_threshold:.3f}")
        print(f"   Final: {format_metrics(final_conf['scores'])}")
        print(f"   Mode: {final_conf['mode_dist']}")

        print("\nGOING Strategy 2: Confidence Delta (K-Fold)")
        fold_results_delta = []
        for train_idx, val_idx in kf.split(indices):
            train = [samples[i] for i in train_idx]
            val = [samples[i] for i in val_idx]
            best_score, best_param = -1, (0.85, 0.05)
            for g0_threshold in np.arange(0.70, 0.95, 0.05):
                for delta in np.arange(-0.10, 0.20, 0.05):
                    result = going_conf_delta(train, ds_name, g0_threshold, delta)
                    if result["scores"][primary] > best_score:
                        best_score = result["scores"][primary]
                        best_param = (g0_threshold, delta)
            val_result = going_conf_delta(val, ds_name, *best_param)
            fold_results_delta.append({"param": best_param, "score": val_result["scores"][primary]})

        mean_g0_threshold = np.mean([result["param"][0] for result in fold_results_delta])
        mean_delta = np.mean([result["param"][1] for result in fold_results_delta])
        final_delta = going_conf_delta(samples, ds_name, mean_g0_threshold, mean_delta)

        print(f"   Optimal (g0_th, delta): ({mean_g0_threshold:.3f}, {mean_delta:.3f})")
        print(f"   Final: {format_metrics(final_delta['scores'])}")
        print(f"   Mode: {final_delta['mode_dist']}")

        best_score = max(final_conf["scores"][primary], final_delta["scores"][primary])

        naive_scores = []
        oracle_scores = []
        for sample in samples:
            best_level = max(sample["levels"].keys(), key=lambda key: sample["levels"][key]["conf"])
            naive_scores.append(sample["levels"][best_level]["scores"][primary])
            best_sample_score = -1
            for key, level in sample["levels"].items():
                if level["scores"][primary] > best_sample_score:
                    best_sample_score = level["scores"][primary]
            oracle_scores.append(best_sample_score)

        iterative_scores = []
        sorted_levels = sorted(
            [key for key in LEVEL_KEYS if key in samples[0]["levels"]],
            key=lambda value: int(value),
        )
        for sample in samples:
            selected_level = sorted_levels[-1]
            for level_key in sorted_levels:
                current = sample["levels"][level_key]
                if current["conf"] >= mean_threshold and not current["is_refusal"]:
                    selected_level = level_key
                    break
            selected = sample["levels"][selected_level]
            if selected["is_refusal"]:
                if cfg.get("silencer"):
                    iterative_scores.append(sample["levels"]["0"]["scores_silenced"][primary])
                else:
                    iterative_scores.append(sample["levels"]["0"]["scores"][primary])
            else:
                iterative_scores.append(selected["scores"][primary])

        ds_result_entry["final_comparison"] = {
            "metric": primary,
            "semantic": semantic["scores"][primary],
            "heuristic": v25["scores"][primary],
            "naive": np.mean(naive_scores),
            "iterative": np.mean(iterative_scores),
            "going": best_score,
            "oracle": np.mean(oracle_scores),
        }
        all_results_json[ds_name] = ds_result_entry

    print("\n" + "=" * 130)
    print(
        f"{'DATASET':<12} {'METRIC':<10} {'SEMANTIC':<12} {'PRIOR JDGMT':<12} "
        f"{'NAIVE':<12} {'ITERATIVE':<12} {'GOING (OURS)':<15} {'ORACLE':<10}"
    )
    print("-" * 130)
    for ds_name, result_data in all_results_json.items():
        result = result_data["final_comparison"]
        print(
            f"{ds_name.upper():<12} {result['metric']:<10} {result['semantic']:<12.4f} "
            f"{result['heuristic']:<12.4f} {result['naive']:<12.4f} "
            f"{result['iterative']:<12.4f} {result['going']:<15.4f} {result['oracle']:<10.4f}"
        )
    print("=" * 130)

    return all_results_json


def main():
    """Run the GOING analysis script and optionally save JSON output."""
    parser = argparse.ArgumentParser(description="GOING multi-metric analysis")
    parser.add_argument(
        "--data_path",
        default="universal_logs",
        help="Path to the universal log directory",
    )
    parser.add_argument(
        "--output_path",
        default="results/going_complete_results_v4_metrics.json",
        help="Path to save the analysis JSON output",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    if data_path.exists():
        results = run_experiment_v3(str(data_path))
        try:
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(convert(results), file, indent=2)
            print(f"\nAnalysis JSON saved to: {output_path.resolve()}")
        except Exception as error:
            print(f"\nFailed to save the analysis JSON: {error}")
    else:
        print(f"Path not found: {data_path}")


if __name__ == "__main__":
    main()
