"""Compare a Qwen3.5 Base model and a trained model on QA examples."""

import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter

# Keep direct script execution working in the separate PyTorch environment.
if __package__:
    from .qa_metrics import (
        calculate_metrics, file_sha256, read_examples, select_examples,
        token_f1, tokenize, rouge_l_f1, write_jsonl,
    )
else:
    from qa_metrics import (
        calculate_metrics, file_sha256, read_examples, select_examples,
        token_f1, tokenize, rouge_l_f1, write_jsonl,
    )


def load_model(model_dir: Path, model_class):
    """Load a local model and stop if checkpoint keys do not match."""
    model, loading = model_class.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype="auto",
        output_loading_info=True,
    )
    for category in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(category):
            raise RuntimeError(
                f"Checkpoint loading problem in {model_dir} ({category}): "
                f"{loading[category]}"
            )
    return model


def generate_answers(
    model_dir: Path,
    model_name: str,
    processor,
    examples: list[dict],
    device: str,
    max_new_tokens: int,
    warmup_questions: int,
    torch,
    model_class,
) -> dict:
    """Generate answers and record simple inference resource measurements."""
    print(f"[Eval] Loading {model_name}: {model_dir}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    load_started = perf_counter()
    model = load_model(model_dir, model_class)
    model.to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = perf_counter() - load_started

    def generate_one(example: dict) -> tuple[str, int, float, float]:
        """Generate one answer and time the request and model generation."""
        if device == "cuda":
            torch.cuda.synchronize(device)
        request_started = perf_counter()
        prompt = f"Question: {example['question']}\nAnswer:"
        inputs = processor(text=[prompt], return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize(device)
        generation_started = perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if device == "cuda":
            torch.cuda.synchronize(device)
        generation_seconds = perf_counter() - generation_started
        prompt_length = inputs["input_ids"].shape[1]
        answer_tokens = output[0, prompt_length:].detach().cpu().tolist()
        answer = processor.tokenizer.decode(
            answer_tokens, skip_special_tokens=True
        ).strip()
        request_seconds = perf_counter() - request_started
        return answer, len(answer_tokens), request_seconds, generation_seconds

    warmup_count = min(warmup_questions, len(examples))
    for example in examples[:warmup_count]:
        generate_one(example)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    answers = []
    answer_token_counts = []
    request_latencies = []
    generation_times = []
    for index, example in enumerate(examples, start=1):
        answer, token_count, request_seconds, generation_seconds = generate_one(example)
        answers.append(answer)
        answer_token_counts.append(token_count)
        request_latencies.append(request_seconds)
        generation_times.append(generation_seconds)

        if index % 25 == 0 or index == len(examples):
            print(f"[Eval] {model_name}: {index}/{len(examples)} questions", flush=True)

    ordered_latencies = sorted(request_latencies)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)
    total_generation_seconds = sum(generation_times)
    total_answer_tokens = sum(answer_token_counts)
    infrastructure = {
        "model_load_seconds": round(model_load_seconds, 4),
        "warmup_questions": warmup_count,
        "measured_questions": len(examples),
        "request_latency_seconds": {
            "mean": round(mean(request_latencies), 4),
            "median": round(median(request_latencies), 4),
            "p95_nearest_rank": round(ordered_latencies[p95_index], 4),
        },
        "generated_token_ids": total_answer_tokens,
        "output_tokens_per_second": round(
            total_answer_tokens / total_generation_seconds, 4
        ) if total_generation_seconds else None,
        "average_output_token_ids": round(mean(answer_token_counts), 2),
        "gpu_memory_mib": None,
    }
    if device == "cuda":
        mib = 1024 * 1024
        infrastructure["gpu_memory_mib"] = {
            "peak_allocated": round(torch.cuda.max_memory_allocated(device) / mib, 2),
            "peak_reserved": round(torch.cuda.max_memory_reserved(device) / mib, 2),
        }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"answers": answers, "infrastructure": infrastructure}


def calculate_bertscore(
    base_predictions: list[str],
    candidate_predictions: list[str],
    references: list[str],
    torch,
    device: str,
    bert_score_module,
) -> tuple[list[float], list[float], dict]:
    """Score both model outputs with the same English BERTScore model."""
    if device == "cuda":
        torch.cuda.empty_cache()
    print("[Eval] Calculating optional BERTScore with roberta-large...", flush=True)
    combined_scores, hash_code = bert_score_module.score(
        base_predictions + candidate_predictions,
        references + references,
        model_type="roberta-large",
        lang="en",
        device=device,
        batch_size=4,
        return_hash=True,
    )
    f1_scores = combined_scores[2].detach().cpu().tolist()
    split = len(references)
    return (
        f1_scores[:split],
        f1_scores[split:],
        {
            "model_type": "roberta-large",
            "language": "en",
            "rescale_with_baseline": False,
            "package_version": package_version("bert-score"),
            "hash": hash_code,
        },
    )


def directory_file_size_bytes(directory: Path) -> int:
    """Add the sizes of files inside a model folder."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path, help="Local Base model folder")
    parser.add_argument(
        "--candidate-dir",
        required=True,
        type=Path,
        help="Local trained model folder to compare with Base",
    )
    parser.add_argument("--candidate-name", default="CPT", help="Name shown in the report")
    parser.add_argument("--qa-file", required=True, type=Path, help="QA evaluation JSONL file")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty output folder")
    parser.add_argument("--limit", type=int, help="Optional repeatable sample size for a pilot run")
    parser.add_argument("--seed", type=int, default=5565)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-questions", type=int, default=3)
    parser.add_argument(
        "--bertscore", action="store_true",
        help="Also calculate BERTScore F1; requires bert-score in this environment",
    )
    args = parser.parse_args()

    for path in (args.base_dir, args.candidate_dir, args.qa_file):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.warmup_questions < 0:
        raise ValueError("--warmup-questions cannot be negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {args.output_dir}. Choose a new folder."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = select_examples(read_examples(args.qa_file), args.limit, args.seed)
    try:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    except ImportError as error:
        raise SystemExit(
            "This script needs PyTorch and a Qwen3.5-compatible Transformers install "
            "in the active Python environment."
        ) from error
    bert_score_module = None
    if args.bertscore:
        try:
            import bert_score as bert_score_module
        except ImportError as error:
            raise SystemExit(
                "BERTScore was requested but is not installed. Install bert-score "
                "in the active PyTorch evaluation environment and retry."
            ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[Eval] WARNING: CUDA was not found; this evaluation may be very slow.", flush=True)
    else:
        print(f"[Eval] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        # Initialize CUDA before timing either model's load.
        torch.empty(1, device=device)
        torch.cuda.synchronize(device)

    processor = AutoProcessor.from_pretrained(
        str(args.base_dir), local_files_only=True
    )
    references = [example["reference_answer"] for example in examples]
    base_result = generate_answers(
        args.base_dir,
        "Base",
        processor,
        examples,
        device,
        args.max_new_tokens,
        args.warmup_questions,
        torch,
        Qwen3_5ForConditionalGeneration,
    )
    candidate_result = generate_answers(
        args.candidate_dir,
        args.candidate_name,
        processor,
        examples,
        device,
        args.max_new_tokens,
        args.warmup_questions,
        torch,
        Qwen3_5ForConditionalGeneration,
    )
    base_predictions = base_result["answers"]
    candidate_predictions = candidate_result["answers"]
    base_metrics = calculate_metrics(base_predictions, references)
    candidate_metrics = calculate_metrics(candidate_predictions, references)
    base_bertscore = None
    candidate_bertscore = None
    bertscore_metadata = None
    if args.bertscore:
        (
            base_bertscore,
            candidate_bertscore,
            bertscore_metadata,
        ) = calculate_bertscore(
            base_predictions,
            candidate_predictions,
            references,
            torch,
            device,
            bert_score_module,
        )
        base_metrics["bertscore_f1"] = mean(base_bertscore)
        candidate_metrics["bertscore_f1"] = mean(candidate_bertscore)

    base_infrastructure = base_result["infrastructure"]
    candidate_infrastructure = candidate_result["infrastructure"]
    base_infrastructure["model_files_size_bytes"] = directory_file_size_bytes(args.base_dir)
    candidate_infrastructure["model_files_size_bytes"] = directory_file_size_bytes(
        args.candidate_dir
    )

    prediction_rows = []
    for index, (example, base_answer, candidate_answer) in enumerate(zip(
        examples, base_predictions, candidate_predictions, strict=True
    )):
        row = {
            **example,
            "base_prediction": base_answer,
            "candidate_prediction": candidate_answer,
        }
        if base_bertscore is not None and candidate_bertscore is not None:
            row["base_bertscore_f1"] = base_bertscore[index]
            row["candidate_bertscore_f1"] = candidate_bertscore[index]
        prediction_rows.append(row)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_file": str(args.qa_file.resolve()),
        "evaluation_file_sha256": file_sha256(args.qa_file),
        "evaluation_examples": len(examples),
        "limit": args.limit,
        "sampling_seed": args.seed if args.limit is not None else None,
        "prompt_template": "Question: {question}\nAnswer:",
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "batch_size": 1,
            "warmup_questions": args.warmup_questions,
        },
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "models": {
            "base": {"name": "Base", "path": str(args.base_dir.resolve())},
            "candidate": {
                "name": args.candidate_name,
                "path": str(args.candidate_dir.resolve()),
            },
        },
        "metrics": {
            "note": (
                "Normalized exact match, token F1, and ROUGE-L measure text overlap with a reference. "
                "They do not establish medical correctness; review answers manually."
            ),
            "bertscore": bertscore_metadata,
            "base": base_metrics,
            "candidate": candidate_metrics,
        },
        "infrastructure": {
            "note": (
                "Model load time includes from_pretrained and transfer to the device, but excludes CUDA initialization and the shared processor. "
                "Request latency includes tokenization, generation, and decoding, but excludes model loading. "
                "Generation throughput counts generated token IDs, including stop tokens. "
                "GPU memory uses this process's PyTorch CUDA allocator; BERTScore memory is excluded."
            ),
            "base": base_infrastructure,
            "candidate": candidate_infrastructure,
        },
        "prediction_file": "predictions.jsonl",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[Eval] Complete. Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
