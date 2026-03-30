"""
Complete Evaluation Pipeline

Evaluates any AI system (RAG, agent, chatbot) against a test dataset.
Scores with automated metrics + LLM-as-judge.

Usage:
    python eval_pipeline.py

Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY for LLM-as-judge
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from metrics import exact_match, f1_token, rouge_l


@dataclass
class EvalCase:
    """A single test case."""
    question: str
    expected_answer: str
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""
    question: str
    expected: str
    actual: str
    scores: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    passed: bool = False


class EvalPipeline:
    """Evaluate an AI system against a test dataset.

    Usage:
        pipeline = EvalPipeline(test_cases)
        results = pipeline.run(my_system_fn)
        pipeline.report(results)
    """

    def __init__(self, test_cases: list[EvalCase], pass_threshold: float = 0.7):
        self.test_cases = test_cases
        self.pass_threshold = pass_threshold

    def run(self, system_fn, verbose: bool = True) -> list[EvalResult]:
        """Run evaluation.

        Args:
            system_fn: callable that takes a question string and returns an answer string
            verbose: print progress

        Returns:
            List of EvalResult objects
        """
        results = []

        for i, case in enumerate(self.test_cases):
            if verbose:
                print(f"  Evaluating {i+1}/{len(self.test_cases)}: {case.question[:50]}...")

            # Time the system
            start = time.time()
            try:
                actual = system_fn(case.question)
            except Exception as e:
                actual = f"ERROR: {e}"
            latency = (time.time() - start) * 1000

            # Score with automated metrics
            scores = {
                "exact_match": exact_match(case.expected_answer, actual),
                "f1_token": f1_token(case.expected_answer, actual),
                "rouge_l": rouge_l(case.expected_answer, actual),
            }

            # LLM-as-judge (if API key available)
            judge_scores = self._llm_judge(case.question, case.expected_answer, actual)
            if judge_scores:
                scores.update(judge_scores)

            # Determine pass/fail (based on F1 token score)
            passed = scores["f1_token"] >= self.pass_threshold

            results.append(EvalResult(
                question=case.question,
                expected=case.expected_answer,
                actual=actual,
                scores=scores,
                latency_ms=latency,
                passed=passed,
            ))

        return results

    def _llm_judge(self, question: str, expected: str, actual: str) -> dict:
        """Use an LLM to judge answer quality."""
        prompt = f"""Rate the following answer on a scale of 1-5 for each criterion.

Question: {question}
Expected Answer: {expected}
Actual Answer: {actual}

Criteria:
- Faithfulness (1-5): Is the answer factually consistent with the expected answer?
- Relevance (1-5): Does it address the question asked?
- Completeness (1-5): Does it cover the key information?

Respond ONLY with JSON: {{"faithfulness": N, "relevance": N, "completeness": N}}"""

        try:
            if os.environ.get("ANTHROPIC_API_KEY"):
                import anthropic
                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                scores = json.loads(text)
                return {
                    "judge_faithfulness": scores["faithfulness"] / 5.0,
                    "judge_relevance": scores["relevance"] / 5.0,
                    "judge_completeness": scores["completeness"] / 5.0,
                }
        except Exception:
            pass

        return {}

    def report(self, results: list[EvalResult]):
        """Print a summary report."""
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)

        # Overall stats
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_latency = sum(r.latency_ms for r in results) / total if total else 0

        print(f"\n  Total cases:    {total}")
        print(f"  Passed:         {passed}/{total} ({100*passed/total:.0f}%)")
        print(f"  Avg latency:    {avg_latency:.0f}ms")

        # Average scores
        all_score_keys = set()
        for r in results:
            all_score_keys.update(r.scores.keys())

        print(f"\n  Average Scores:")
        for key in sorted(all_score_keys):
            values = [r.scores[key] for r in results if key in r.scores]
            avg = sum(values) / len(values) if values else 0
            print(f"    {key:25s}: {avg:.3f}")

        # Worst failures
        failures = [r for r in results if not r.passed]
        if failures:
            print(f"\n  Worst Failures ({len(failures)}):")
            # Sort by F1 score (ascending -- worst first)
            failures.sort(key=lambda r: r.scores.get("f1_token", 0))
            for r in failures[:5]:
                print(f"    Q: {r.question[:60]}")
                print(f"    Expected: {r.expected[:60]}")
                print(f"    Got:      {r.actual[:60]}")
                print(f"    F1: {r.scores.get('f1_token', 0):.2f}")
                print()

    def export(self, results: list[EvalResult], path: str):
        """Save results to JSON."""
        data = [asdict(r) for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results exported to {path}")


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    # Create test cases
    test_cases = [
        EvalCase("What is the capital of France?", "Paris", "geography", "easy"),
        EvalCase("What is 2 + 2?", "4", "math", "easy"),
        EvalCase("Who wrote Romeo and Juliet?", "William Shakespeare", "literature", "easy"),
        EvalCase("What is the boiling point of water in Celsius?", "100 degrees Celsius", "science", "easy"),
        EvalCase("What programming language is PyTorch written in?", "Python and C++", "tech", "medium"),
        EvalCase("What does RAG stand for in AI?", "Retrieval-Augmented Generation", "ai", "medium"),
        EvalCase("What year was the transformer paper published?", "2017", "ai", "medium"),
        EvalCase("What is the time complexity of binary search?", "O(log n)", "cs", "medium"),
        EvalCase("Explain backpropagation in one sentence", "Backpropagation computes gradients of the loss with respect to each weight by applying the chain rule backward through the computational graph.", "ai", "hard"),
        EvalCase("What is the difference between precision and recall?", "Precision measures the fraction of relevant results among retrieved results, while recall measures the fraction of relevant results that were retrieved.", "ml", "hard"),
    ]

    # Define a simple "system" to evaluate (just calls Claude API)
    def simple_system(question: str) -> str:
        """A simple AI system that just answers questions."""
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": f"Answer concisely: {question}"}],
            )
            return response.content[0].text
        else:
            return "(No API key -- cannot generate answer)"

    # Run evaluation
    print("Running evaluation pipeline...")
    pipeline = EvalPipeline(test_cases, pass_threshold=0.5)
    results = pipeline.run(simple_system)
    pipeline.report(results)
