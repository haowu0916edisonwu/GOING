"""
CARE data loader supporting all dataset matching modes.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Sample:
    """Store one unified evaluation sample."""

    id: str
    question: str
    answers: List[str]
    context: str
    dataset: str


class CAREDataLoader:
    """Load CARE evaluation data across all supported datasets."""

    def __init__(self, data_root: str = "data_care/eval", verbose: bool = False):
        """Initialize the loader with the evaluation data root."""
        self.data_root = Path(data_root)
        self.verbose = verbose

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")

    def load_dataset(self, dataset_name: str) -> List[Sample]:
        """Load one dataset and merge questions with retrieval results."""
        dataset_map = {
            "nq": "nq",
            "trivia": "triviaqa",
            "triviaqa": "triviaqa",
            "webqa": "webqa",
            "truthfulqa": "truthfulqa",
            "factkg": "factkg",
        }

        folder = dataset_map.get(dataset_name, dataset_name)
        dataset_dir = self.data_root / folder

        question_file = dataset_dir / "test.jsonl"
        retrieval_file = dataset_dir / "retrieval" / "colbertv2" / "test_question_aware.jsonl"

        if self.verbose:
            print(f"\nLoading {dataset_name}:")
            print(f"  Question: {question_file}")
            print(f"  Retrieval: {retrieval_file}")

        if not question_file.exists():
            raise FileNotFoundError(f"Question file not found: {question_file}")
        if not retrieval_file.exists():
            raise FileNotFoundError(f"Retrieval file not found: {retrieval_file}")

        questions = self._load_jsonl(question_file)
        retrievals = self._load_jsonl(retrieval_file)

        if self.verbose:
            print(f"  Loaded: {len(questions)} questions, {len(retrievals)} retrievals")

        match_mode = self._detect_match_mode(retrievals)

        if self.verbose:
            print(f"  Match mode: {match_mode}")

        if match_mode == "ID":
            samples = self._merge_by_id(questions, retrievals, dataset_name)
        elif match_mode == "QUERY":
            samples = self._merge_by_query(questions, retrievals, dataset_name)
        elif match_mode == "QUESTION":
            samples = self._merge_by_question(questions, retrievals, dataset_name)
        else:
            raise ValueError(f"Unknown match mode: {match_mode}")

        print(f"Loaded {len(samples)} valid samples from {dataset_name}")
        return samples

    def _detect_match_mode(self, retrievals: List[Dict]) -> str:
        """Detect whether retrievals match by id, query, or question."""
        if not retrievals:
            raise ValueError("Empty retrievals list")

        first = retrievals[0]

        if "id" in first:
            return "ID"
        if "query" in first:
            return "QUERY"
        if "question" in first:
            return "QUESTION"
        raise ValueError(f"Cannot detect match mode. Available keys: {list(first.keys())}")

    def _merge_by_id(
        self,
        questions: List[Dict],
        retrievals: List[Dict],
        dataset_name: str,
    ) -> List[Sample]:
        """Merge question and retrieval files using id alignment."""
        if len(questions) != len(retrievals):
            print(
                "  Warning: question count "
                f"({len(questions)}) != retrieval count ({len(retrievals)})"
            )

        samples = []
        skipped = 0

        for idx, (question, retrieval) in enumerate(zip(questions, retrievals)):
            q_id = str(question.get("id", ""))
            r_id = str(retrieval.get("id", ""))

            if q_id != r_id:
                if self.verbose and skipped < 3:
                    print(f"  Sample {idx}: ID mismatch (Q={q_id}, R={r_id})")
                skipped += 1
                continue

            sample = self._create_sample(question, retrieval, dataset_name, idx)
            if sample:
                samples.append(sample)

        if skipped > 0:
            print(f"  Skipped {skipped} samples due to ID mismatch")

        return samples

    def _merge_by_query(
        self,
        questions: List[Dict],
        retrievals: List[Dict],
        dataset_name: str,
    ) -> List[Sample]:
        """Merge question and retrieval files using query text."""
        if self.verbose:
            print("  Using query-based matching (r['query'] == q['question'])")

        retrieval_dict = {}
        for retrieval in retrievals:
            query_text = retrieval.get("query", "")
            if query_text:
                retrieval_dict[query_text] = retrieval

        if self.verbose:
            print(f"  Built retrieval dict: {len(retrieval_dict)} entries")

        samples = []
        skipped = 0

        for idx, question in enumerate(questions):
            question_text = question.get("question", "")
            retrieval = retrieval_dict.get(question_text)

            if retrieval is None:
                if self.verbose and skipped < 3:
                    print(f"  Sample {idx}: No retrieval found for: {question_text[:50]}...")
                skipped += 1
                continue

            sample = self._create_sample(question, retrieval, dataset_name, idx)
            if sample:
                samples.append(sample)

        if skipped > 0:
            print(f"  Skipped {skipped} samples (no matching retrieval)")

        return samples

    def _merge_by_question(
        self,
        questions: List[Dict],
        retrievals: List[Dict],
        dataset_name: str,
    ) -> List[Sample]:
        """Merge question and retrieval files using question text."""
        if self.verbose:
            print("  Using question-based matching (r['question'] == q['question'])")

        retrieval_dict = {}
        for retrieval in retrievals:
            question_text = retrieval.get("question", "")
            if question_text:
                retrieval_dict[question_text] = retrieval

        if self.verbose:
            print(f"  Built retrieval dict: {len(retrieval_dict)} entries")

        samples = []
        skipped = 0

        for idx, question in enumerate(questions):
            question_text = question.get("question", "")
            retrieval = retrieval_dict.get(question_text)

            if retrieval is None:
                if self.verbose and skipped < 3:
                    print(f"  Sample {idx}: No retrieval found for: {question_text[:50]}...")
                skipped += 1
                continue

            sample = self._create_sample(question, retrieval, dataset_name, idx)
            if sample:
                samples.append(sample)

        if skipped > 0:
            print(f"  Skipped {skipped} samples (no matching retrieval)")

        return samples

    def _create_sample(
        self,
        question: Dict,
        retrieval: Dict,
        dataset_name: str,
        idx: int,
    ) -> Sample:
        """Create a unified sample object from raw question and retrieval entries."""
        sample_id = str(question.get("id", str(idx)))
        question_text = question.get("question", question.get("claim", ""))

        answers = question.get("answer", question.get("answers", []))
        if not isinstance(answers, list):
            answers = [str(answers)]

        context_text = self._extract_context(retrieval, idx)

        return Sample(
            id=sample_id,
            question=question_text,
            answers=answers,
            context=context_text,
            dataset=dataset_name,
        )

    def _extract_context(self, retrieval: Dict, idx: int) -> str:
        """Extract the top-10 retrieved passages into one context string."""
        del idx
        context_parts = []
        top_k = 10

        if retrieval.get("topk"):
            docs = retrieval["topk"][:top_k]

            for i, doc in enumerate(docs):
                raw_text = doc.get("text", "")
                doc_text = self._extract_document_text(raw_text)

                if doc_text:
                    formatted_text = f"Passage-{i} {doc_text}"
                    context_parts.append(formatted_text)

        return "\n".join(context_parts)

    @staticmethod
    def _extract_document_text(raw_text: str) -> str:
        """Extract the document content from question-aware retrieval text."""
        match = re.search(r"Document:\s*(.*)", raw_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        if "Question:" in raw_text:
            parts = raw_text.split("Document:", 1)
            if len(parts) > 1:
                return parts[1].strip()
            parts = raw_text.split("\n", 1)
            if len(parts) > 1:
                return parts[1].strip()

        return raw_text.strip()

    @staticmethod
    def _load_jsonl(file_path: Path) -> List[Dict]:
        """Read a JSONL file into a list of dictionaries."""
        data = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as error:
                    print(f"Warning: line {line_num}: {error}")
        return data
