"""
Stage 1 script for generating universal logs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import CAREDataLoader
from src.evaluator_ccar import CCAREvaluator
from src.metrics import Metrics
from src.prompts import PromptTemplates


def main():
    """Generate universal logs for all datasets and context levels."""
    parser = argparse.ArgumentParser(description="GOING Stage 1: universal logging")
    parser.add_argument("--data_root", default="data_care/eval", help="Data directory")
    parser.add_argument(
        "--model_name",
        default="NousResearch/Meta-Llama-3-8B-Instruct",
        help="Model name",
    )
    parser.add_argument("--output_dir", default="universal_logs", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for debugging")
    args = parser.parse_args()

    levels = [0, 1, 3, 5, 7, 10]
    all_datasets = ["nq", "trivia", "webqa", "truthfulqa", "factkg"]

    evaluator = CCAREvaluator(args.model_name)
    data_loader = CAREDataLoader(args.data_root, verbose=False)
    os.makedirs(args.output_dir, exist_ok=True)

    for ds_name in all_datasets:
        samples = data_loader.load_dataset(ds_name)
        if args.max_samples:
            samples = samples[: args.max_samples]

        task_type = evaluator.TASK_TYPES.get(ds_name, "open_qa")
        output_file = os.path.join(args.output_dir, f"{ds_name}_universal.jsonl")

        print(f"\nStarting universal logging: {ds_name.upper()} ({len(samples)} samples)")

        with open(output_file, "w", encoding="utf-8") as file:
            for sample in tqdm(samples, desc="Processing levels"):
                entry = {
                    "id": sample.id,
                    "question": sample.question,
                    "gold": sample.answers,
                    "gears": {},
                }

                for level in levels:
                    if level == 0:
                        prompt = (
                            PromptTemplates.closedbook_fact(sample.question)
                            if task_type == "fact_checking"
                            else PromptTemplates.closedbook_qa_short(sample.question)
                        )
                    else:
                        current_context = "\n".join(sample.context.split("\n")[:level])
                        if task_type == "fact_checking":
                            prompt = PromptTemplates.priori_judgment_fact(sample.question, current_context)
                        elif task_type == "long_form":
                            prompt = PromptTemplates.priori_judgment_truthful(sample.question, current_context)
                        else:
                            prompt = PromptTemplates.priori_judgment_qa(sample.question, current_context)

                    raw_out, conf, tokens = evaluator.generate_and_score(prompt, ds_name)
                    pred = evaluator.apply_v25_cleaning(raw_out, ds_name)
                    clean_out_tokens = evaluator.tokenizer.encode(pred, add_special_tokens=False)
                    tokens["clean_out"] = len(clean_out_tokens)

                    if task_type == "fact_checking":
                        score = Metrics.compute_accuracy(pred, sample.answers)
                        is_correct = score == 1.0
                    elif task_type == "long_form":
                        score = Metrics.compute_f1(pred, sample.answers)
                        is_correct = score > 0.3
                    else:
                        score = Metrics.compute_span_em(pred, sample.answers)
                        is_correct = score == 1.0

                    entry["gears"][str(level)] = {
                        "conf": conf,
                        "pred": pred,
                        "is_correct": is_correct,
                        "tokens": tokens,
                        "raw": raw_out,
                    }

                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                file.flush()

    print(f"\nAll logs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
