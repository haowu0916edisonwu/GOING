"""
Prompt templates for GOING evaluation across CARE datasets.
"""


class PromptTemplates:
    """Provide prompt templates for all evaluation modes."""

    @staticmethod
    def priori_judgment_qa(question: str, context: str) -> str:
        """Build the QA prompt used for NQ, TriviaQA, and WebQA."""
        return (
            f"Given the following information:\n"
            f"{context}\n"
            f"Can you answer the following question based on the given information or your internal knowledge? "
            f"If yes, give a short answer with one or few words. "
            f"If not, answer \"Unknown\".\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    @staticmethod
    def priori_judgment_truthful(question: str, context: str) -> str:
        """Build the long-form prompt used for TruthfulQA."""
        return (
            f"Given the following information: \n"
            f"{context}\n"
            f"Can you answer the following question based on the given information or your internal knowledge, "
            f"if yes, you should give a detailed answer, "
            f"if no, you should answer \"Unknown\".\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    @staticmethod
    def priori_judgment_fact(claim: str, context: str) -> str:
        """Build the fact verification prompt used for FactKG."""
        return (
            f"Given the following information:\n"
            f"{context}\n\n"
            f"Can you verify the following claim based on the given information or your internal knowledge? "
            f"If yes, you should answer True or False, if no, you should answer \"Unknown\".\n\n"
            f"Claim: {claim}\n"
            f"Answer:"
        )

    @staticmethod
    def closedbook_qa_short(question: str) -> str:
        """Build the short closed-book QA prompt."""
        return (
            f"Answer the questions:\n"
            f"Question: {question}?\n"
            f"The answer is:"
        )

    @staticmethod
    def closedbook_qa_long(question: str) -> str:
        """Build the long closed-book QA prompt."""
        return (
            f"Answer the questions:\n"
            f"Question: {question}\n"
            f"The answer is:"
        )

    @staticmethod
    def closedbook_fact(claim: str) -> str:
        """Build the closed-book fact verification prompt."""
        return (
            f"Verify the following claims with \"True\" or \"False\":\n"
            f"Claim: {claim}\n"
            f"The answer is:"
        )
