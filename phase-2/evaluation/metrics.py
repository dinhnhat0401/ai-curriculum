"""
Reusable Evaluation Metrics

Functions for measuring the quality of AI system outputs.
Each metric takes (expected, actual) strings and returns a score.

Usage:
    from metrics import exact_match, f1_token, rouge_l
    score = f1_token("The cat sat on the mat", "The cat is on the mat")
"""

from collections import Counter


def exact_match(expected: str, actual: str) -> float:
    """Exact string match (case-insensitive, stripped).

    Returns: 1.0 if strings match, 0.0 otherwise.
    """
    return 1.0 if expected.strip().lower() == actual.strip().lower() else 0.0


def f1_token(expected: str, actual: str) -> float:
    """Token-level F1 score.

    Measures overlap between expected and actual tokens.
    Good for Q&A where exact wording may differ but content overlaps.

    Returns: F1 score between 0.0 and 1.0.
    """
    expected_tokens = expected.lower().split()
    actual_tokens = actual.lower().split()

    if not expected_tokens or not actual_tokens:
        return 1.0 if expected_tokens == actual_tokens else 0.0

    expected_counts = Counter(expected_tokens)
    actual_counts = Counter(actual_tokens)

    # Count matching tokens
    common = sum((expected_counts & actual_counts).values())

    if common == 0:
        return 0.0

    precision = common / len(actual_tokens)
    recall = common / len(expected_tokens)

    return 2 * precision * recall / (precision + recall)


def rouge_l(expected: str, actual: str) -> float:
    """ROUGE-L: Longest Common Subsequence based metric.

    Measures the longest sequence of tokens that appears in both
    expected and actual (not necessarily contiguous).

    Good for summarization evaluation.

    Returns: F1 score based on LCS length, between 0.0 and 1.0.
    """
    expected_tokens = expected.lower().split()
    actual_tokens = actual.lower().split()

    if not expected_tokens or not actual_tokens:
        return 1.0 if expected_tokens == actual_tokens else 0.0

    # Compute LCS length using dynamic programming
    m, n = len(expected_tokens), len(actual_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if expected_tokens[i - 1] == actual_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / n
    recall = lcs_length / m

    return 2 * precision * recall / (precision + recall)


def bleu_simple(expected: str, actual: str, max_n: int = 4) -> float:
    """Simplified BLEU score (unigram to max_n-gram).

    BLEU measures n-gram precision of the actual text against expected.
    Commonly used for translation evaluation.

    Returns: BLEU score between 0.0 and 1.0.
    """
    import math

    expected_tokens = expected.lower().split()
    actual_tokens = actual.lower().split()

    if not actual_tokens:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        # Get n-grams
        expected_ngrams = Counter(
            tuple(expected_tokens[i:i + n]) for i in range(len(expected_tokens) - n + 1)
        )
        actual_ngrams = Counter(
            tuple(actual_tokens[i:i + n]) for i in range(len(actual_tokens) - n + 1)
        )

        if not actual_ngrams:
            scores.append(0)
            continue

        # Clipped count
        clipped = sum((actual_ngrams & expected_ngrams).values())
        total = sum(actual_ngrams.values())

        scores.append(clipped / total if total > 0 else 0)

    # Geometric mean of n-gram precisions
    if any(s == 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / len(scores)

    # Brevity penalty
    bp = 1.0
    if len(actual_tokens) < len(expected_tokens):
        bp = math.exp(1 - len(expected_tokens) / len(actual_tokens))

    return bp * math.exp(log_avg)


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("METRIC DEMONSTRATIONS")
    print("=" * 60)

    test_cases = [
        ("The cat sat on the mat", "The cat sat on the mat"),
        ("The cat sat on the mat", "The cat is on the mat"),
        ("The cat sat on the mat", "A dog ran in the park"),
        ("Paris is the capital of France", "The capital of France is Paris"),
        ("42", "42"),
        ("42", "The answer is 42"),
    ]

    for expected, actual in test_cases:
        print(f"\n  Expected: '{expected}'")
        print(f"  Actual:   '{actual}'")
        print(f"    Exact Match: {exact_match(expected, actual):.2f}")
        print(f"    F1 (token):  {f1_token(expected, actual):.2f}")
        print(f"    ROUGE-L:     {rouge_l(expected, actual):.2f}")
        print(f"    BLEU:        {bleu_simple(expected, actual):.2f}")
