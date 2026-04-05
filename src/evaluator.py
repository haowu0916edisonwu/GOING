"""
Evaluator for the original Priori Judgment baseline.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_loader import Sample
from .metrics import Metrics
from .prompts import PromptTemplates


@dataclass
class EvalResult:
    """Store the output for a single evaluated sample."""

    id: str
    question: str
    prediction: str
    gold_answers: List[str]
    mode: str
    priori_output: str


class PrioriJudgmentEvaluator:
    """Evaluate the original two-stage Priori Judgment policy."""

    TASK_TYPES = {
        "nq": "open_qa",
        "trivia": "open_qa",
        "triviaqa": "open_qa",
        "webqa": "open_qa",
        "truthfulqa": "long_form",
        "factkg": "fact_checking",
    }

    DATASET_MAX_TOKENS = {
        "nq": 45,
        "webqa": 45,
        "triviaqa": 45,
        "factkg": 30,
        "truthfulqa": 30,
    }

    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize the model and tokenizer for evaluation."""
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading model with v25 strategy: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, dataset: str = None) -> str:
        """Generate a deterministic completion for one prompt."""
        max_tokens = self.DATASET_MAX_TOKENS.get(dataset, 30)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def base_clean(self, text: str) -> str:
        """Remove common answer prefixes from model output."""
        text_clean = text.strip()
        text_lower = text_clean.lower()
        if text_lower.startswith("answer:"):
            text_clean = text_clean[7:].strip()
        elif text_lower.startswith("prediction:"):
            text_clean = text_clean[11:].strip()
        if text_clean.lower().startswith("the claim is"):
            text_clean = text_clean[12:].strip()
        return text_clean

    def _handle_greedy(self, sample: Sample, ds_name: str):
        """Run the greedy fallback policy for NQ and TriviaQA."""
        priori_prompt = PromptTemplates.priori_judgment_qa(sample.question, sample.context)
        raw_output = self.generate(priori_prompt, dataset=ds_name)

        def greedy_clean(text: str) -> str:
            """Apply the original greedy cleaning heuristic."""
            clean = text.strip().lstrip('?!"\n')
            lower = clean.lower()
            if lower.startswith("**answer"):
                clean = clean.split("**", 1)[-1].strip(": ")
                lower = clean.lower()
            if lower.startswith("**"):
                clean = clean.lstrip("*").strip()
                lower = clean.lower()
            if lower.startswith("answer:"):
                clean = clean[7:].strip()

            if "unknown." in clean.lower():
                idx = clean.lower().find("unknown.")
                clean = clean[:idx].strip()
            return clean

        final_text = greedy_clean(raw_output)
        lower_text = final_text.lower()
        is_unknown = False

        if not lower_text or lower_text.startswith("unknown"):
            is_unknown = True
        elif any(word in lower_text for word in ["i don't know", "no information", "cannot answer"]):
            is_unknown = True

        if is_unknown:
            cb_prompt = PromptTemplates.closedbook_qa_short(sample.question)
            cb_output = self.generate(cb_prompt, dataset=ds_name)
            return greedy_clean(cb_output), "closedbook", raw_output
        return final_text, "rag", raw_output

    def _handle_webqa(self, sample: Sample, ds_name: str):
        """Run the WebQA-specific fallback policy."""
        priori_prompt = PromptTemplates.priori_judgment_qa(sample.question, sample.context)
        raw_output = self.generate(priori_prompt, dataset=ds_name)

        def webqa_clean(text: str) -> str:
            """Apply the original WebQA cleaning heuristic."""
            clean = text.strip().lstrip('?!"\n')
            if clean.lower().startswith("answer:"):
                clean = clean[7:].strip()
            if "unknown." in clean.lower():
                clean = clean[: clean.lower().find("unknown.")].strip()
            return clean

        final_text = webqa_clean(raw_output)
        lower_text = final_text.lower()
        is_unknown = False

        if len(final_text.split()) < 8:
            is_unknown = True

        if not lower_text or "unknown" in lower_text or "no information" in lower_text:
            is_unknown = True

        if is_unknown:
            cb_prompt = PromptTemplates.closedbook_qa_short(sample.question)
            cb_output = self.generate(cb_prompt, dataset=ds_name)
            return webqa_clean(cb_output), "closedbook", raw_output
        return final_text, "rag", raw_output

    def _handle_truthfulqa(self, sample: Sample, ds_name: str):
        """Run the TruthfulQA policy with refusal handling and silencing."""
        priori_prompt = PromptTemplates.priori_judgment_truthful(sample.question, sample.context)
        raw_output = self.generate(priori_prompt, dataset=ds_name)

        def verbose_clean(text: str) -> str:
            """Apply the original verbose-output cleaning heuristic."""
            clean = self.base_clean(text)
            lower = clean.lower()
            verbose_starts = [
                "according to the",
                "based on the",
                "the passage states",
                "the text states",
                "the provided text",
                "based on the provided",
                "according to the provided",
            ]
            for prefix in verbose_starts:
                if lower.startswith(prefix):
                    if "," in clean:
                        clean = clean.split(",", 1)[1].strip()
                        lower = clean.lower()
                    elif "that" in clean[: len(prefix) + 10]:
                        clean = clean.split("that", 1)[1].strip()
                        lower = clean.lower()
                    break

            if "the answer is" in lower:
                idx = lower.find("the answer is")
                if idx < 20:
                    clean = clean[idx + 13 :].strip(" :.,")

            if "Note:" in clean:
                clean = clean.split("Note:")[0]
            if "\n" in clean:
                clean = clean.split("\n")[0]
            return clean.strip()

        final_text = verbose_clean(raw_output)
        lower_text = final_text.lower()
        is_unknown = False

        refined_triggers = [
            "unknown",
            "i don't know",
            "i do not know",
            "cannot answer",
            "unable to answer",
            "no information",
            "not mention",
            "does not mention",
            "not say",
            "unclear",
            "no answer",
        ]

        if any(trigger in lower_text for trigger in refined_triggers):
            is_unknown = True

        if is_unknown:
            cb_prompt = PromptTemplates.closedbook_qa_short(sample.question)
            cb_output = self.generate(cb_prompt, dataset=ds_name)

            cleaned_cb = verbose_clean(cb_output)

            if "." in cleaned_cb:
                cleaned_cb = cleaned_cb.split(".", 1)[0].strip()

            return cleaned_cb, "closedbook", raw_output
        return final_text, "rag", raw_output

    def _handle_factkg(self, sample: Sample, ds_name: str):
        """Run the FactKG policy with unknown fallback."""
        priori_prompt = PromptTemplates.priori_judgment_fact(sample.question, sample.context)
        raw_output = self.generate(priori_prompt, dataset=ds_name)

        def clean(text: str) -> str:
            """Keep only the first cleaned line."""
            return text.strip().split("\n")[0]

        final = clean(self.base_clean(raw_output))

        if "unknown" in final.lower():
            cb_prompt = PromptTemplates.closedbook_fact(sample.question)
            cb_output = self.generate(cb_prompt, dataset=ds_name)
            return clean(self.base_clean(cb_output)), "closedbook", raw_output
        return final, "rag", raw_output

    def evaluate_sample(self, sample: Sample) -> EvalResult:
        """Evaluate one sample and return the prediction record."""
        ds = sample.dataset.lower()
        if ds == "trivia":
            ds = "triviaqa"

        if ds == "truthfulqa":
            pred, mode, raw = self._handle_truthfulqa(sample, ds)
        elif ds == "webqa":
            pred, mode, raw = self._handle_webqa(sample, ds)
        elif ds in ["nq", "triviaqa"]:
            pred, mode, raw = self._handle_greedy(sample, ds)
        else:
            pred, mode, raw = self._handle_factkg(sample, ds)

        return EvalResult(
            id=sample.id,
            question=sample.question,
            prediction=pred,
            gold_answers=sample.answers,
            mode=mode,
            priori_output=raw,
        )

    def evaluate_dataset(self, samples: List[Sample]) -> Dict:
        """Evaluate a full dataset and aggregate metrics."""
        results = []
        print(f"Evaluating {len(samples)} samples...")
        for sample in tqdm(samples, desc="Processing"):
            results.append(self.evaluate_sample(sample))
        task_type = self.TASK_TYPES.get(samples[0].dataset, "open_qa")
        metrics = self._compute_metrics(results, task_type)
        return {
            "metrics": metrics,
            "results": results,
            "mode_distribution": {
                "rag": sum(1 for result in results if result.mode == "rag"),
                "closedbook": sum(1 for result in results if result.mode == "closedbook"),
            },
        }

    def _compute_metrics(self, results: List[EvalResult], task_type: str) -> Dict[str, float]:
        """Aggregate metrics for a dataset based on task type."""
        if task_type == "fact_checking":
            scores = [Metrics.compute_accuracy(result.prediction, result.gold_answers) for result in results]
            return {"accuracy": sum(scores) / len(scores) if scores else 0}
        if task_type == "long_form":
            f1_scores = [Metrics.compute_f1(result.prediction, result.gold_answers) for result in results]
            rouge_scores = [Metrics.compute_rouge_l(result.prediction, result.gold_answers) for result in results]
            return {
                "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
                "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0,
            }
        scores = [Metrics.compute_span_em(result.prediction, result.gold_answers) for result in results]
        return {"span_em": sum(scores) / len(scores) if scores else 0}
