"""Shared QA data loading, sampling, and text-overlap metrics.

Both Qwen and LSTM evaluation use these functions. This module requires only
Python's standard library; it does not load TensorFlow or PyTorch.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import mean


def read_examples(path: Path) -> list[dict]:
    """Read question and reference answer fields from a JSONL file."""
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in {path} must be a JSON object.")

            question = row.get("question")
            answer = row.get("answer")
            question = question.strip() if isinstance(question, str) else ""
            answer = answer.strip() if isinstance(answer, str) else ""
            if not question or not answer:
                raise ValueError(
                    f"Line {line_number} in {path} needs a question and an answer."
                )
            examples.append({
                "source_line": line_number,
                "question": question,
                "reference_answer": answer,
            })

    if not examples:
        raise ValueError(f"No complete question-answer examples found in {path}")
    return examples


def select_examples(examples: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Choose a repeatable random sample for a small pilot run."""
    if limit is None:
        return examples
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    if limit > len(examples):
        raise ValueError(f"--limit is {limit}, but the file has only {len(examples)} rows")

    indexes = sorted(random.Random(seed).sample(range(len(examples)), limit))
    return [examples[index] for index in indexes]


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase word tokens for simple overlap metrics."""
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def token_f1(prediction: str, reference: str) -> float:
    """Calculate token-overlap F1, similar to the SQuAD token F1 metric."""
    predicted_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not predicted_tokens or not reference_tokens:
        return 0.0

    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Calculate ROUGE-L F1 using the longest common token sequence."""
    predicted_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not predicted_tokens or not reference_tokens:
        return 0.0

    # Keep the dynamic-programming row as short as possible.
    if len(predicted_tokens) < len(reference_tokens):
        predicted_tokens, reference_tokens = reference_tokens, predicted_tokens
    previous = [0] * (len(reference_tokens) + 1)
    for predicted_token in predicted_tokens:
        current = [0]
        for index, reference_token in enumerate(reference_tokens, start=1):
            if predicted_token == reference_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    longest_common_sequence = previous[-1]
    if longest_common_sequence == 0:
        return 0.0
    precision = longest_common_sequence / len(predicted_tokens)
    recall = longest_common_sequence / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def calculate_metrics(predictions: list[str], references: list[str]) -> dict:
    """Summarize normalized exact match and two lexical-overlap metrics."""
    exact_matches = []
    token_scores = []
    rouge_scores = []
    empty_predictions = 0
    for prediction, reference in zip(predictions, references, strict=True):
        predicted_tokens = tokenize(prediction)
        reference_tokens = tokenize(reference)
        exact_matches.append(predicted_tokens == reference_tokens)
        token_scores.append(token_f1(prediction, reference))
        rouge_scores.append(rouge_l_f1(prediction, reference))
        empty_predictions += not predicted_tokens

    count = len(predictions)
    return {
        "examples": count,
        "normalized_exact_match": mean(exact_matches),
        "token_f1": mean(token_scores),
        "rouge_l_f1": mean(rouge_scores),
        "empty_predictions": empty_predictions,
    }


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of the evaluation file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write one prediction record per line."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
