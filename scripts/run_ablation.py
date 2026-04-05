"""
Run the GOING ablation study from universal logs.
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold


def normalize_answer(text: str) -> str:
    """Normalize answer text for exact-match scoring."""
    text = str(text).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def greedy_clean(text: str) -> str:
    """Apply the greedy cleaning rule used for NQ and TriviaQA."""
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
    """Apply the WebQA cleaning rule."""
    cleaned = text.strip().lstrip('?!"\n')
    if cleaned.lower().startswith("answer:"):
        cleaned = cleaned[7:].strip()
    if "unknown." in cleaned.lower():
        cleaned = cleaned[: cleaned.lower().find("unknown.")].strip()
    return cleaned


def is_refusal_greedy(text: str) -> bool:
    """Check whether the output should trigger a refusal fallback."""
    lowered = text.lower()
    return not lowered or lowered.startswith("unknown") or any(
        word in lowered for word in ["i don't know", "no information", "cannot answer"]
    )


def is_refusal_webqa(text: str) -> bool:
    """Check the WebQA-specific refusal condition."""
    lowered = text.lower()
    return len(text.split()) < 8 or not lowered or "unknown" in lowered or "no information" in lowered


def compute_span_em(prediction: str, ground_truths: list) -> float:
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


DATASET_CONFIG = {
    "nq": {"clean": greedy_clean, "metric": compute_span_em, "refusal": is_refusal_greedy},
    "webqa": {"clean": webqa_clean, "metric": compute_span_em, "refusal": is_refusal_webqa},
}


def load_data(dataset_name, base_path):
    """Load per-sample scores from one universal log file."""
    path = Path(base_path) / f"{dataset_name}_universal.jsonl"
    if not path.exists():
        return None, None

    cfg = DATASET_CONFIG.get(dataset_name)
    if not cfg:
        return None, None
    clean_fn = cfg["clean"]
    metric_fn = cfg["metric"]
    refusal_fn = cfg["refusal"]

    samples = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                g0 = data["gears"]["0"]
                rag_key = "10" if "10" in data["gears"] else max(data["gears"].keys(), key=int)
                rag = data["gears"][rag_key]

                s0_pred = clean_fn(g0["pred"])
                rag_pred = clean_fn(rag["pred"])

                g10_is_refusal = refusal_fn(rag_pred)

                score_0 = metric_fn(s0_pred, data["gold"])
                score_rag = metric_fn(rag_pred, data["gold"])

                samples.append(
                    {
                        "s0": g0["conf"],
                        "delta": rag["conf"] - g0["conf"],
                        "score_0": score_0,
                        "score_rag": score_rag,
                        "g10_is_refusal": g10_is_refusal,
                    }
                )
            except Exception:
                continue
    return np.array(samples), metric_fn


def eval_strategy_1(samples, tau, force_retrieve=False):
    """Evaluate the threshold-only strategy."""
    scores = []
    retrieved = 0
    total = len(samples)
    for sample in samples:
        use_g10 = False
        should_retrieve = True if force_retrieve else (sample["s0"] < tau)

        if should_retrieve:
            retrieved += 1
            if not sample["g10_is_refusal"]:
                use_g10 = True

        scores.append(sample["score_rag"] if use_g10 else sample["score_0"])
    return np.mean(scores), retrieved / total * 100


def eval_strategy_2(samples, tau, delta, force_retrieve=False, force_accept=False):
    """Evaluate the threshold-plus-delta strategy."""
    scores = []
    retrieved = 0
    total = len(samples)
    for sample in samples:
        use_g10 = False
        should_retrieve = True if force_retrieve else (sample["s0"] < tau)

        if should_retrieve:
            retrieved += 1
            pass_delta = True if force_accept else (sample["delta"] >= delta)

            if pass_delta and not sample["g10_is_refusal"]:
                use_g10 = True

        scores.append(sample["score_rag"] if use_g10 else sample["score_0"])
    return np.mean(scores), retrieved / total * 100


def run_ablation(dataset_name, samples):
    """Run 5-fold ablations for one dataset."""
    del dataset_name
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    range_s1_tau = np.arange(0.50, 0.96, 0.02)
    range_s2_tau = np.arange(0.50, 0.96, 0.05)
    range_s2_delta = np.concatenate([np.arange(-5.0, -0.1, 0.5), np.arange(-0.10, 0.20, 0.05)])

    results = {key: {"scores": [], "retr": []} for key in ["Full", "No_P1", "No_P3", "Std_RAG"]}
    strategy_counts = Counter()

    for train_idx, val_idx in kf.split(samples):
        train_data, val_data = samples[train_idx], samples[val_idx]

        best_s1, best_t1 = -1, 0.5
        for tau in range_s1_tau:
            score, _ = eval_strategy_1(train_data, tau)
            if score > best_s1:
                best_s1, best_t1 = score, tau

        best_s2, best_p2 = -1, (0.8, 0.0)
        for tau in range_s2_tau:
            for delta in range_s2_delta:
                score, _ = eval_strategy_2(train_data, tau, delta)
                if score > best_s2:
                    best_s2, best_p2 = score, (tau, delta)

        if best_s1 > best_s2 + 0.0001:
            winner = "S1"
            full_s, full_r = eval_strategy_1(val_data, best_t1)
            nop1_s, nop1_r = eval_strategy_1(val_data, -1, force_retrieve=True)
            nop3_s, nop3_r = full_s, full_r
        else:
            winner = "S2"
            full_s, full_r = eval_strategy_2(val_data, best_p2[0], best_p2[1])
            nop1_s, nop1_r = eval_strategy_2(val_data, -1, best_p2[1], force_retrieve=True)
            nop3_s, nop3_r = eval_strategy_2(val_data, best_p2[0], -999, force_accept=True)

        strategy_counts[winner] += 1

        std_s, std_r = eval_strategy_1(val_data, -1, force_retrieve=True)

        results["Full"]["scores"].append(full_s)
        results["Full"]["retr"].append(full_r)
        results["No_P1"]["scores"].append(nop1_s)
        results["No_P1"]["retr"].append(nop1_r)
        results["No_P3"]["scores"].append(nop3_s)
        results["No_P3"]["retr"].append(nop3_r)
        results["Std_RAG"]["scores"].append(std_s)
        results["Std_RAG"]["retr"].append(std_r)

    avg_results = {
        key: {metric: np.mean(values) for metric, values in value.items()}
        for key, value in results.items()
    }
    most_common_strat = strategy_counts.most_common(1)[0][0]
    return avg_results, most_common_strat


def main():
    """Run the ablation study across the supported datasets."""
    parser = argparse.ArgumentParser(description="GOING ablation study")
    parser.add_argument(
        "--data_path",
        default="universal_logs",
        help="Path to the universal logs directory",
    )
    args = parser.parse_args()
    data_path = args.data_path

    datasets = ["nq", "webqa"]

    print(f"{'=' * 100}")
    print(f"{'DATASET':<10} {'MODE':<15} {'SCORE':<10} {'RETR%':<10} {'NOTE'}")
    print(f"{'=' * 100}")

    for dataset in datasets:
        samples, _ = load_data(dataset, data_path)
        if samples is None:
            continue

        results, strategy = run_ablation(dataset, samples)

        full_s = results["Full"]["scores"]
        nop1_s = results["No_P1"]["scores"]
        nop3_s = results["No_P3"]["scores"]
        std_s = results["Std_RAG"]["scores"]

        print(
            f"{dataset.upper():<10} {'Full GOING':<15} {full_s:.4f}     "
            f"{results['Full']['retr']:.1f}%      (Baseline - Used {strategy})"
        )
        print(
            f"{'':<10} {'w/o Phase 1':<15} {nop1_s:.4f}     "
            f"{results['No_P1']['retr']:.1f}%      (Force Retrieve)"
        )

        if strategy == "S1":
            note = "Same (Phase 3 inactive)"
        else:
            diff = nop3_s - full_s
            if diff < -0.0001:
                note = "Drop (P3 improved score)"
            elif diff < -0.01:
                note = "COLLAPSE!"
            else:
                note = "Stable"

        print(
            f"{'':<10} {'w/o Phase 3':<15} {nop3_s:.4f}     "
            f"{results['No_P3']['retr']:.1f}%      {note}"
        )
        print(
            f"{'':<10} {'Standard RAG':<15} {std_s:.4f}     "
            f"{results['Std_RAG']['retr']:.1f}%      (No circuit breaker)"
        )
        print(f"{'-' * 100}")


if __name__ == "__main__":
    main()
