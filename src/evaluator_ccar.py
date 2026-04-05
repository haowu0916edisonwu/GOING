"""
Evaluator for universal logging with confidence extraction.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CCAREvaluator:
    """Run generation and collect confidence statistics for each setting."""

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
        """Initialize the model and tokenizer for confidence logging."""
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Initializing C-CAR evaluator: {model_name}")
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
        )
        self.model.eval()

    @torch.no_grad()
    def generate_and_score(self, prompt: str, ds_name: str) -> Tuple[str, float, Dict[str, int]]:
        """Generate text, estimate confidence, and count tokens."""
        max_tokens = self.DATASET_MAX_TOKENS.get(ds_name, 30)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )
        probs = torch.exp(transition_scores[0]).cpu().numpy()
        confidence = float(np.mean(probs))

        gen_ids = outputs.sequences[0][input_len:]
        output_len = len(gen_ids)
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return raw_text, confidence, {"in": input_len, "out": output_len}

    def apply_v25_cleaning(self, text: str, ds_name: str) -> str:
        """Apply the original v25 cleaning rules to model output."""
        clean = text.strip()
        lower = clean.lower()

        if lower.startswith("answer:"):
            clean = clean[7:].strip()
        elif lower.startswith("prediction:"):
            clean = clean[11:].strip()
        if clean.lower().startswith("the claim is"):
            clean = clean[12:].strip()

        if ds_name == "truthfulqa":
            for prefix in ["according to the", "based on the", "the passage states"]:
                if lower.startswith(prefix) and "," in clean:
                    clean = clean.split(",", 1)[1].strip()
                    break
            if "." in clean:
                clean = clean.split(".", 1)[0].strip()

        elif ds_name in ["nq", "triviaqa", "trivia", "webqa"]:
            clean = clean.lstrip('?!"\n')
            if clean.lower().startswith("answer:"):
                clean = clean[7:].strip()
            if "unknown." in clean.lower():
                clean = clean[: clean.lower().find("unknown.")].strip()

        elif ds_name == "factkg":
            clean = clean.split("\n")[0].strip()

        return clean.strip()
