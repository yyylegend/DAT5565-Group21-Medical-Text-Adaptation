"""Compare a Qwen3.5 Base model and a trained model on QA examples."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

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
    torch,
    model_class,
) -> list[str]:
    """Generate one answer per question, using the same prompt each time."""
    print(f"[Eval] Loading {model_name}: {model_dir}", flush=True)
    model = load_model(model_dir, model_class)
    model.to(device).eval()
    answers = []

    for index, example in enumerate(examples, start=1):
        prompt = f"Question: {example['question']}\nAnswer:"
        inputs = processor(text=[prompt], return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        prompt_length = inputs["input_ids"].shape[1]
        answer_tokens = output[0, prompt_length:].detach().cpu().tolist()
        answer = processor.tokenizer.decode(
            answer_tokens, skip_special_tokens=True
        ).strip()
        answers.append(answer)

        if index % 25 == 0 or index == len(examples):
            print(f"[Eval] {model_name}: {index}/{len(examples)} questions", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answers


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
    args = parser.parse_args()

    for path in (args.base_dir, args.candidate_dir, args.qa_file):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[Eval] WARNING: CUDA was not found; this evaluation may be very slow.", flush=True)
    else:
        print(f"[Eval] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    processor = AutoProcessor.from_pretrained(
        str(args.base_dir), local_files_only=True
    )
    references = [example["reference_answer"] for example in examples]
    base_predictions = generate_answers(
        args.base_dir,
        "Base",
        processor,
        examples,
        device,
        args.max_new_tokens,
        torch,
        Qwen3_5ForConditionalGeneration,
    )
    candidate_predictions = generate_answers(
        args.candidate_dir,
        args.candidate_name,
        processor,
        examples,
        device,
        args.max_new_tokens,
        torch,
        Qwen3_5ForConditionalGeneration,
    )

    prediction_rows = []
    for example, base_answer, candidate_answer in zip(
        examples, base_predictions, candidate_predictions, strict=True
    ):
        prediction_rows.append({
            **example,
            "base_prediction": base_answer,
            "candidate_prediction": candidate_answer,
        })
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
            "base": calculate_metrics(base_predictions, references),
            "candidate": calculate_metrics(candidate_predictions, references),
        },
        "prediction_file": "predictions.jsonl",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[Eval] Complete. Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
