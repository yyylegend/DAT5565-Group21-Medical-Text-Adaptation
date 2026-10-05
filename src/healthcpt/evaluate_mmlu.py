"""Prepare and score the full MMLU professional_medicine test subset."""

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen

if __package__:
    from .evaluate_qa import load_model
    from .qa_metrics import file_sha256, write_jsonl
else:
    from evaluate_qa import load_model
    from qa_metrics import file_sha256, write_jsonl

DATASET = "cais/mmlu"
REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
SUBJECT = "professional_medicine"
COUNTS = {"dev": 5, "test": 272}
LETTERS = "ABCD"


def validate_example(row):
    """Reject incomplete questions or invalid gold labels before scoring."""
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("A benchmark question is empty.")
    choices = row.get("choices")
    if not isinstance(choices, list) or len(choices) != 4:
        raise ValueError("Each benchmark question needs four choices.")
    if any(not isinstance(choice, str) or not choice.strip() for choice in choices):
        raise ValueError("A benchmark choice is empty.")
    if type(row.get("answer")) is not int or not 0 <= row["answer"] < 4:
        raise ValueError("The gold answer must be an integer from 0 to 3.")
    if row.get("subject") != SUBJECT:
        raise ValueError("The dataset contains an unexpected subject.")


def format_question(row, include_answer=False):
    """Use the standard MMLU question/options/Answer format."""
    text = row["question"].strip()
    for letter, choice in zip(LETTERS, row["choices"], strict=True):
        text += f"\n{letter}. {choice}"
    text += "\nAnswer:"
    if include_answer:
        text += f" {LETTERS[row['answer']]}\n\n"
    return text


def build_prompt(example, demonstrations):
    """Only dev examples contain answers; the test answer never enters the prompt."""
    text = "The following are multiple choice questions (with answers) about professional medicine.\n\n"
    text += "".join(format_question(row, True) for row in demonstrations)
    return text + format_question(example)


def choice_probabilities(log_probabilities):
    """Normalize across four choices and return the highest-scoring letter."""
    if len(log_probabilities) != 4 or not all(math.isfinite(v) for v in log_probabilities):
        raise ValueError("Expected four finite choice scores.")
    largest = max(log_probabilities)
    values = [math.exp(value - largest) for value in log_probabilities]
    total = sum(values)
    probabilities = [value / total for value in values]
    return LETTERS[max(range(4), key=lambda i: log_probabilities[i])], probabilities


def prepare(output_dir):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose an empty dataset folder, or reuse the prepared files.")
    prepared, source_urls = {}, []
    for split, count in COUNTS.items():
        rows = []
        for offset in range(0, count, 100):
            parameters = urlencode({
                "dataset": DATASET, "config": SUBJECT, "split": split,
                "offset": offset, "length": min(100, count - offset),
            })
            url = f"https://datasets-server.huggingface.co/rows?{parameters}"
            print(f"[MMLU Data] {split}: downloading from row {offset}", flush=True)
            with urlopen(url, timeout=60) as response:
                revision = response.headers.get("x-revision")
                page = json.load(response)
            if revision != REVISION:
                raise ValueError(f"Dataset revision changed: {revision}. Expected {REVISION}.")
            if page.get("partial") or page["num_rows_total"] != count:
                raise ValueError("The public dataset has a different size or incomplete response.")
            for item in page["rows"]:
                if item.get("truncated_cells"):
                    raise ValueError("The dataset server returned a truncated cell.")
                if item["row_idx"] != len(rows):
                    raise ValueError("Dataset rows are missing or out of order.")
                row = item["row"]
                validate_example(row)
                rows.append(dict(row, id=f"{split}:{item['row_idx']}"))
            source_urls.append(url)
        if len(rows) != count:
            raise ValueError(f"Expected {count} {split} examples, received {len(rows)}.")
        prepared[split] = rows
    output_dir.mkdir(parents=True)
    for split, rows in prepared.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    manifest = {
        "dataset": DATASET, "revision": REVISION, "subject": SUBJECT,
        "counts": COUNTS, "dataset_card_license": "MIT",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source_urls": source_urls,
        "sha256": {split: file_sha256(output_dir / f"{split}.jsonl") for split in COUNTS},
        "note": "Dev examples are evaluation demonstrations, not training data. Keep test examples out of training.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[MMLU Data] Ready: 272 test questions and 5 dev demonstrations in {output_dir}")


def read_split(dataset_dir, split, manifest):
    path = dataset_dir / f"{split}.jsonl"
    if file_sha256(path) != manifest["sha256"][split]:
        raise ValueError(f"Prepared benchmark file changed: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != COUNTS[split]:
        raise ValueError(f"Unexpected {split} row count.")
    for row in rows:
        validate_example(row)
    return rows


def evaluate(args):
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["dataset"], manifest["revision"], manifest["subject"]) != (DATASET, REVISION, SUBJECT):
        raise ValueError("The dataset manifest does not match this benchmark.")
    demonstrations = read_split(args.dataset_dir, "dev", manifest)
    examples = read_split(args.dataset_dir, "test", manifest)
    if args.limit is not None:
        if not 1 <= args.limit <= len(examples):
            raise ValueError("--limit must be between 1 and 272.")
        examples = examples[:args.limit]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a new, empty evaluation output folder.")

    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark runner.")
    torch.manual_seed(5565)
    processor = AutoProcessor.from_pretrained(str(args.base_dir), local_files_only=True)
    # For Qwen's tokenizer, each continuation ' A' ... ' D' must be one token.
    # Verify this assumption instead of scoring only part of a multi-token label.
    label_ids = [processor.tokenizer.encode(" " + label, add_special_tokens=False) for label in LETTERS]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("This runner requires single-token, space-prefixed answer letters.")
    first_prompt = build_prompt(examples[0], demonstrations)
    prefix = processor.tokenizer.encode(first_prompt, add_special_tokens=False)
    for label, ids in zip(LETTERS, label_ids, strict=True):
        if processor.tokenizer.encode(first_prompt + " " + label, add_special_tokens=False) != prefix + ids:
            raise ValueError("Answer-label tokenization changes the prompt boundary.")
    if file_sha256(args.base_dir / "tokenizer.json") != file_sha256(args.candidate_dir / "tokenizer.json"):
        raise ValueError("Base and candidate tokenizers differ.")
    args.output_dir.mkdir(parents=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "MMLU professional_medicine", "dataset_revision": REVISION,
        "test_sha256": manifest["sha256"]["test"], "dev_sha256": manifest["sha256"]["dev"],
        "examples": len(examples), "full_subject_test": len(examples) == COUNTS["test"],
        "limit": args.limit, "chance_accuracy": 0.25,
        "num_fewshot": 5, "demonstration_split": "dev", "demonstration_selection": "first_5",
        "scoring": "argmax next-token log probability for space-prefixed A/B/C/D; no answer generation",
        "prompt_template": build_prompt({"question": "{question}", "choices": ["{A}", "{B}", "{C}", "{D}"]}, demonstrations),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "transformers": transformers.__version__, "models": {},
        "note": "This is one MMLU subject, not the full MMLU score. It measures exam-question accuracy, not clinical safety. Base models may already have seen benchmark content in pretraining.",
    }
    for key, name, path in (
        ("base", "Base", args.base_dir),
        ("candidate", args.candidate_name, args.candidate_dir),
    ):
        print(f"[MMLU] Loading {name}: {path}", flush=True)
        model = load_model(path, Qwen3_5ForConditionalGeneration).to("cuda").eval()
        predictions, correct = [], 0
        started = time.perf_counter()
        # Save each completed question so a failure preserves the finished work.
        with (args.output_dir / f"{key}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for index, row in enumerate(examples, start=1):
                inputs = processor(text=[build_prompt(row, demonstrations)], return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    scores = torch.log_softmax(logits, dim=-1)[[ids[0] for ids in label_ids]].cpu().tolist()
                prediction, probabilities = choice_probabilities(scores)
                gold = LETTERS[row["answer"]]
                correct += prediction == gold
                result = {
                    "id": row["id"], "question": row["question"], "choices": row["choices"],
                    "gold": gold, "prediction": prediction, "correct": prediction == gold,
                    "choice_probabilities": dict(zip(LETTERS, probabilities, strict=True)),
                }
                predictions.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                if index % 25 == 0 or index == len(examples):
                    handle.flush()
                    elapsed = time.perf_counter() - started
                    eta = elapsed / index * (len(examples) - index)
                    print(f"[MMLU] {name}: {index}/{len(examples)}, accuracy={correct/index:.3f}, ETA={eta/60:.1f} min", flush=True)
        report["models"][key] = {
            "name": name, "path": str(path.resolve()), "correct": correct,
            "accuracy": correct / len(examples), "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": file_sha256(path / "config.json"),
        }
        del model, inputs, logits
        gc.collect()
        torch.cuda.empty_cache()
    base_predictions = [json.loads(line) for line in (args.output_dir / "base_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    candidate_predictions = predictions
    report["paired_counts"] = {
        "both_correct": sum(b["correct"] and c["correct"] for b, c in zip(base_predictions, candidate_predictions, strict=True)),
        "candidate_only_correct": sum(not b["correct"] and c["correct"] for b, c in zip(base_predictions, candidate_predictions, strict=True)),
        "base_only_correct": sum(b["correct"] and not c["correct"] for b, c in zip(base_predictions, candidate_predictions, strict=True)),
        "both_wrong": sum(not b["correct"] and not c["correct"] for b, c in zip(base_predictions, candidate_predictions, strict=True)),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["models"], indent=2))
    print(f"[MMLU] Complete: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Download only this subject as JSON")
    preparation.add_argument("--output-dir", type=Path, default=Path("data/processed/mmlu-professional-medicine"))
    evaluation = commands.add_parser("evaluate", help="Score Base and CPT+SFT using 5-shot prompts")
    evaluation.add_argument("--dataset-dir", type=Path, default=Path("data/processed/mmlu-professional-medicine"))
    evaluation.add_argument("--base-dir", type=Path, required=True)
    evaluation.add_argument("--candidate-dir", type=Path, required=True)
    evaluation.add_argument("--candidate-name", default="CPT+SFT")
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--limit", type=int, help="First N questions for an execution pilot; omit for all 272")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output_dir)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
