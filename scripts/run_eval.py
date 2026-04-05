#!/usr/bin/env python3
"""
Evaluate the original Priori Judgment baseline on CARE.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import CAREDataLoader
from src.evaluator import PrioriJudgmentEvaluator
from src.metrics import Metrics


def debug_single_sample(data_loader, evaluator, dataset):
    """Run the evaluator on one sample for quick debugging."""
    print("\n" + "=" * 70)
    print(f"DEBUG MODE: {dataset.upper()}")
    print("=" * 70)

    samples = data_loader.load_dataset(dataset)
    sample = samples[0]

    print("\nSample info:")
    print(f"  ID: {sample.id}")
    print(f"  Question: {sample.question[:200]}...")
    print(f"  Answers: {sample.answers}")
    print(f"  Context length: {len(sample.context)} chars")
    print(f"  Context preview: {sample.context[:300]}...")

    print("\nRunning two-stage inference...")
    result = evaluator.evaluate_sample(sample)

    print("\nResults:")
    print(f"  Stage 1 output: {result.priori_output}")
    print(f"  Final answer: {result.prediction}")
    print(f"  Mode: {result.mode}")
    print(f"  Gold answers: {result.gold_answers}")

    task_type = evaluator.TASK_TYPES.get(dataset, "open_qa")

    if task_type == "fact_checking":
        score = Metrics.compute_accuracy(result.prediction, result.gold_answers)
        metric_name = "Accuracy"
    elif task_type == "long_form":
        score = Metrics.compute_f1(result.prediction, result.gold_answers)
        metric_name = "F1"
    else:
        score = Metrics.compute_span_em(result.prediction, result.gold_answers)
        metric_name = "Span EM"

    print(f"\n{metric_name}: {score:.4f}")
    print("=" * 70)


def save_predictions(results, output_dir, dataset):
    """Save detailed predictions as JSONL."""
    filename = f"{dataset}_predictions.jsonl"
    if hasattr(output_dir, "joinpath"):
        output_file = output_dir / filename
    else:
        output_file = os.path.join(output_dir, filename)

    print(f"Saving detailed predictions to: {output_file}")

    with open(output_file, "w", encoding="utf-8") as file:
        for result in results:
            pred_data = {
                "id": getattr(result, "id", "unknown_id"),
                "question": getattr(result, "question", ""),
                "prediction": getattr(result, "prediction", ""),
                "gold_answers": getattr(result, "gold_answers", []),
                "mode": getattr(result, "mode", "unknown"),
                "priori_output": getattr(result, "priori_output", None),
            }

            task_type = getattr(result, "task_type", None)

            if not task_type:
                gold = getattr(result, "gold_answers", [])
                if len(gold) == 1 and str(gold[0]).lower() in ["true", "false"]:
                    task_type = "fact_checking"
                else:
                    task_type = "qa"

            if task_type == "fact_checking":
                pred_data["correct"] = getattr(result, "accuracy", 0) == 1.0
            else:
                pred_data["correct"] = getattr(result, "span_em", 0) == 1.0

            file.write(json.dumps(pred_data, ensure_ascii=False) + "\n")

    print("Detailed prediction export finished.")


def main():
    """Run the Priori Judgment evaluation workflow."""
    parser = argparse.ArgumentParser(
        description="Priori Judgment baseline evaluation on CARE"
    )
    parser.add_argument("--data_root", default="data_care/eval", help="Data directory")
    parser.add_argument(
        "--model_name",
        default="NousResearch/Meta-Llama-3-8B-Instruct",
        help="Model name",
    )
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to evaluate")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for debugging")
    parser.add_argument("--debug_sample", action="store_true", help="Debug one sample per dataset")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose data loading output")
    parser.add_argument("--save_predictions", action="store_true", help="Save detailed predictions")
    args = parser.parse_args()

    all_datasets = ["nq", "trivia", "webqa", "truthfulqa", "factkg"]
    if args.datasets:
        all_datasets = [dataset for dataset in all_datasets if dataset in args.datasets]

    paper_metrics_def = [
        ("nq", "span_em", 0.458),
        ("trivia", "span_em", 0.704),
        ("webqa", "span_em", 0.406),
        ("truthfulqa", "f1", 0.254),
        ("truthfulqa", "rouge_l", 0.231),
        ("factkg", "accuracy", 0.666),
    ]

    targets_map = {
        "nq": {"span_em": 0.458},
        "trivia": {"span_em": 0.704},
        "webqa": {"span_em": 0.406},
        "truthfulqa": {"f1": 0.254, "rouge_l": 0.231},
        "factkg": {"accuracy": 0.666},
    }

    print("=" * 70)
    print("Priori Judgment Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_root}")
    print("Using: test_question_aware.jsonl (Top-10 context)")
    print("Format: answer + topk fields")
    print("=" * 70)

    data_loader = CAREDataLoader(args.data_root, verbose=args.verbose)
    evaluator = PrioriJudgmentEvaluator(args.model_name)

    if args.debug_sample:
        for dataset in all_datasets:
            debug_single_sample(data_loader, evaluator, dataset)
        return

    all_results = {}

    for dataset in all_datasets:
        print(f"\n{'=' * 70}")
        print(f"Evaluating {dataset.upper()}")
        print(f"{'=' * 70}")

        samples = data_loader.load_dataset(dataset)
        if args.max_samples:
            samples = samples[: args.max_samples]
            print(f"Debug mode: using {len(samples)} samples")

        result = evaluator.evaluate_dataset(samples)
        all_results[dataset] = result

        if args.save_predictions:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_predictions(result["results"], output_dir, dataset)

        metrics = result["metrics"]
        print("\nResults:")
        for key, value in metrics.items():
            target_value = targets_map[dataset].get(key, 0)
            status = "PASS" if value >= target_value - 0.005 else "FAIL"
            print(f"  {key.upper()}: {value:.4f} (target: {target_value:.4f}) {status}")

        print("\nMode distribution:")
        for mode, count in result["mode_distribution"].items():
            pct = count / len(samples) * 100
            print(f"  {mode}: {count}/{len(samples)} ({pct:.1f}%)")

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<15} {'Metric':<12} {'Result':<10} {'Target':<10} {'Status'}")
    print(f"{'-' * 70}")

    collected_scores = []

    for ds_name, metric_key, target_val in paper_metrics_def:
        if ds_name in all_results and metric_key in all_results[ds_name]["metrics"]:
            score = all_results[ds_name]["metrics"][metric_key]
            collected_scores.append(score)

            status = "PASS" if score >= target_val - 0.005 else "FAIL"

            print(
                f"{ds_name:<15} "
                f"{metric_key.upper():<12} "
                f"{score:<10.4f} "
                f"{target_val:<10.4f} "
                f"{status}"
            )
        else:
            print(
                f"{ds_name:<15} "
                f"{metric_key.upper():<12} "
                f"{'N/A':<10} "
                f"{target_val:<10.4f} "
                f"SKIP"
            )

    print(f"{'-' * 70}")

    final_avg = 0.0

    if len(collected_scores) == 6:
        final_avg = sum(collected_scores) / 6
        target_avg = 0.453
        status_text = "PASS" if final_avg >= target_avg - 0.005 else "FAIL"

        print(
            f"{'AVERAGE':<15} "
            f"{'':<12} "
            f"{final_avg:<10.4f} "
            f"{target_avg:<10.4f} "
            f"{status_text}"
        )
    elif collected_scores:
        final_avg = sum(collected_scores) / len(collected_scores)
        print(
            f"{'PARTIAL AVG':<15} "
            f"{'(Div/' + str(len(collected_scores)) + ')':<12} "
            f"{final_avg:<10.4f} "
            f"{'0.4530':<10} "
            f"PARTIAL"
        )
    else:
        print("No scores collected.")

    print(f"{'=' * 70}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                dataset: {
                    "metrics": result["metrics"],
                    "mode_distribution": result.get("mode_distribution", {}),
                }
                for dataset, result in all_results.items()
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as file:
        file.write("PRIORI JUDGMENT EVALUATION RESULTS\n")
        file.write("=" * 70 + "\n\n")

        for dataset in all_datasets:
            if dataset not in all_results:
                continue
            file.write(f"{dataset.upper()}\n")
            for key, value in all_results[dataset]["metrics"].items():
                file.write(f"  {key.upper()}: {value:.4f}\n")
            file.write("\n")

        if collected_scores:
            label = "AVERAGE" if len(collected_scores) == 6 else "PARTIAL AVG"
            file.write(f"{label}: {final_avg:.4f}\n")

    print("\nResults saved:")
    print(f"  - {output_dir}/results.json (main result)")
    print(f"  - {output_dir}/summary.txt (text summary)")
    if args.save_predictions:
        print(f"  - {output_dir}/{{dataset}}_predictions.jsonl (detailed predictions)")


if __name__ == "__main__":
    main()
