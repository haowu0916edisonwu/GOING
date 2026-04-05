"""
Run paired bootstrap significance tests for GOING and baseline systems.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from run_going_analysis import (
    DATASET_CONFIG,
    load_data,
)


def apply_strategy_1_sample(sample: Dict, ds_name: str, threshold: float) -> float:
    """Score one sample with the threshold-only GOING policy."""
    cfg = DATASET_CONFIG[ds_name]
    g10 = sample["levels"]["10"]
    g0 = sample["levels"]["0"]
    use_g10 = (g10["conf"] >= threshold) and (not g10["is_refusal"])
    if use_g10:
        return g10["scores"][cfg["primary"]]
    if cfg.get("silencer"):
        return g0["scores_silenced"][cfg["primary"]]
    return g0["scores"][cfg["primary"]]


def apply_strategy_2_sample(sample: Dict, ds_name: str, g0_threshold: float, delta_threshold: float) -> float:
    """Score one sample with the two-gate GOING policy."""
    cfg = DATASET_CONFIG[ds_name]
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
            return g0["scores_silenced"][cfg["primary"]]
        return g0["scores"][cfg["primary"]]
    return g10["scores"][cfg["primary"]]


def closed_book_score(sample: Dict, ds_name: str) -> float:
    """Return the fixed closed-book score for one sample."""
    primary = DATASET_CONFIG[ds_name]["primary"]
    return sample["levels"]["0"]["scores"][primary]


def standard_rag_score(sample: Dict, ds_name: str) -> float:
    """Return the fixed top-retrieval score for one sample."""
    primary = DATASET_CONFIG[ds_name]["primary"]
    return sample["levels"]["10"]["scores"][primary]


def prior_judgment_score(sample: Dict, ds_name: str) -> float:
    """Return the original prior-judgment baseline score for one sample."""
    cfg = DATASET_CONFIG[ds_name]
    g10 = sample["levels"]["10"]
    g0 = sample["levels"]["0"]
    if g10["is_refusal"]:
        if cfg.get("silencer"):
            return g0["scores_silenced"][cfg["primary"]]
        return g0["scores"][cfg["primary"]]
    return g10["scores"][cfg["primary"]]


def select_fold_strategy(train_samples: List[Dict], ds_name: str) -> Tuple[str, Tuple[float, float] | float]:
    """Select the better GOING strategy on the training fold."""
    primary = DATASET_CONFIG[ds_name]["primary"]

    best_s1_score = -1.0
    best_s1_tau = 0.5
    for tau in np.arange(0.50, 0.96, 0.02):
        scores = [apply_strategy_1_sample(sample, ds_name, tau) for sample in train_samples]
        mean_score = float(np.mean(scores))
        if mean_score > best_s1_score:
            best_s1_score = mean_score
            best_s1_tau = tau

    best_s2_score = -1.0
    best_s2_params = (0.85, 0.05)
    for g0_threshold in np.arange(0.70, 0.95, 0.05):
        for delta in np.arange(-0.10, 0.20, 0.05):
            scores = [
                apply_strategy_2_sample(sample, ds_name, g0_threshold, delta)
                for sample in train_samples
            ]
            mean_score = float(np.mean(scores))
            if mean_score > best_s2_score:
                best_s2_score = mean_score
                best_s2_params = (g0_threshold, delta)

    del primary
    if best_s1_score >= best_s2_score:
        return "strategy_1", best_s1_tau
    return "strategy_2", best_s2_params


def cross_validated_going_scores(samples: List[Dict], ds_name: str) -> np.ndarray:
    """Produce out-of-fold GOING scores for paired testing."""
    n_splits = min(5, len(samples))
    if n_splits < 2:
        return np.array([apply_strategy_1_sample(sample, ds_name, 0.5) for sample in samples], dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(samples))
    predictions = np.zeros(len(samples), dtype=float)

    for train_idx, val_idx in kf.split(indices):
        train_samples = [samples[i] for i in train_idx]
        strategy_name, params = select_fold_strategy(train_samples, ds_name)

        for sample_idx in val_idx:
            sample = samples[sample_idx]
            if strategy_name == "strategy_1":
                predictions[sample_idx] = apply_strategy_1_sample(sample, ds_name, params)
            else:
                predictions[sample_idx] = apply_strategy_2_sample(sample, ds_name, params[0], params[1])

    return predictions


def paired_bootstrap(going_scores: np.ndarray, baseline_scores: np.ndarray, iterations: int, seed: int) -> Dict:
    """Run a paired bootstrap test on two score vectors."""
    if len(going_scores) != len(baseline_scores):
        raise ValueError("Score vectors must have the same length.")
    if len(going_scores) == 0:
        raise ValueError("Score vectors must not be empty.")

    rng = np.random.default_rng(seed)
    n = len(going_scores)
    observed_diff = float(np.mean(going_scores - baseline_scores))
    bootstrap_diffs = np.empty(iterations, dtype=float)

    for i in range(iterations):
        sample_idx = rng.integers(0, n, size=n)
        sampled_diff = going_scores[sample_idx] - baseline_scores[sample_idx]
        bootstrap_diffs[i] = float(np.mean(sampled_diff))

    ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])
    p_value = 2 * min(
        float(np.mean(bootstrap_diffs <= 0.0)),
        float(np.mean(bootstrap_diffs >= 0.0)),
    )
    p_value = min(p_value, 1.0)

    return {
        "going_mean": float(np.mean(going_scores)),
        "baseline_mean": float(np.mean(baseline_scores)),
        "observed_diff": observed_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
    }


def collect_dataset_scores(samples: List[Dict], ds_name: str) -> Dict[str, np.ndarray]:
    """Collect GOING and baseline per-sample scores for one dataset."""
    return {
        "going": cross_validated_going_scores(samples, ds_name),
        "closed_book": np.array([closed_book_score(sample, ds_name) for sample in samples], dtype=float),
        "std_rag": np.array([standard_rag_score(sample, ds_name) for sample in samples], dtype=float),
        "prior_judgment": np.array([prior_judgment_score(sample, ds_name) for sample in samples], dtype=float),
    }


def summarize_dataset(ds_name: str, score_map: Dict[str, np.ndarray], iterations: int, seed: int) -> Dict:
    """Compute paired bootstrap summaries for one dataset."""
    primary = DATASET_CONFIG[ds_name]["primary"]
    results = {
        "metric": primary,
        "n_samples": int(len(score_map["going"])),
        "means": {
            "going": float(np.mean(score_map["going"])),
            "closed_book": float(np.mean(score_map["closed_book"])),
            "std_rag": float(np.mean(score_map["std_rag"])),
            "prior_judgment": float(np.mean(score_map["prior_judgment"])),
        },
        "comparisons": {},
    }

    for offset, baseline_name in enumerate(["closed_book", "std_rag", "prior_judgment"]):
        results["comparisons"][baseline_name] = paired_bootstrap(
            score_map["going"],
            score_map[baseline_name],
            iterations=iterations,
            seed=seed + offset,
        )

    return results


def print_summary(dataset: str, summary: Dict):
    """Print a readable bootstrap summary for one dataset."""
    print(f"\n{dataset.upper()} ({summary['metric']}, n={summary['n_samples']})")
    print(f"  GOING mean: {summary['means']['going']:.4f}")
    print(f"  Closed Book mean: {summary['means']['closed_book']:.4f}")
    print(f"  Std RAG mean: {summary['means']['std_rag']:.4f}")
    print(f"  Prior Judgment mean: {summary['means']['prior_judgment']:.4f}")

    for baseline_name, comparison in summary["comparisons"].items():
        print(
            "  GOING vs "
            f"{baseline_name}: diff={comparison['observed_diff']:.4f}, "
            f"95% CI=[{comparison['ci_low']:.4f}, {comparison['ci_high']:.4f}], "
            f"p={comparison['p_value']:.4g}"
        )


def save_summary(output_path: Path, results: Dict):
    """Save bootstrap results as JSON."""
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def main():
    """Run paired bootstrap testing for all available datasets."""
    parser = argparse.ArgumentParser(description="GOING paired bootstrap significance test")
    parser.add_argument(
        "--data_path",
        default="universal_logs",
        help="Path to the universal logs directory",
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=10000,
        help="Number of bootstrap iterations",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional path to save the JSON summary",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_path)
    datasets = ["nq", "trivia", "webqa", "truthfulqa", "factkg"]
    results = {}

    print("Running paired bootstrap significance tests")
    print(f"Data path: {data_dir}")
    print(f"Bootstrap iterations: {args.bootstrap_iters}")

    for dataset in datasets:
        file_path = data_dir / f"{dataset}_universal.jsonl"
        if not file_path.exists():
            continue

        samples = load_data(str(file_path), dataset)
        if not samples:
            continue

        available_levels = set(samples[0]["levels"].keys())
        if not {"0", "10"}.issubset(available_levels):
            print(f"Skipping {dataset}: required levels 0 and 10 were not found.")
            continue

        score_map = collect_dataset_scores(samples, dataset)
        summary = summarize_dataset(dataset, score_map, args.bootstrap_iters, args.seed)
        results[dataset] = summary
        print_summary(dataset, summary)

    if args.output_path:
        save_summary(Path(args.output_path), results)
        print(f"\nSaved bootstrap summary to: {args.output_path}")


if __name__ == "__main__":
    main()
