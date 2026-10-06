"""Add BERTScore to saved LSTM answers without running answer generation."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
from statistics import mean

if __package__:
    from .qa_metrics import calculate_metrics, file_sha256, write_jsonl
else:
    from qa_metrics import calculate_metrics, file_sha256, write_jsonl

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_inputs(lstm_dir, qwen_dir):
    lstm, qwen = read_json(lstm_dir / "metrics.json"), read_json(qwen_dir / "metrics.json")
    rows, qwen_rows = read_rows(lstm_dir / "predictions.jsonl"), read_rows(qwen_dir / "predictions.jsonl")
    if not rows or len(rows) != len(qwen_rows) or len(rows) != lstm["evaluation_examples"] or len(rows) != qwen["evaluation_examples"]:
        raise ValueError("Prediction counts do not match the evaluation reports.")
    if lstm["evaluation_file_sha256"] != qwen["evaluation_file_sha256"]:
        raise ValueError("LSTM and Qwen used different evaluation files.")
    if any(any(a[k] != b[k] for k in ("source_line", "question", "reference_answer")) for a, b in zip(rows, qwen_rows)):
        raise ValueError("LSTM and Qwen questions/references are misaligned.")
    references = [r["reference_answer"] for r in rows]
    answers = [r["candidate_prediction"] for r in rows]
    if any(not isinstance(answer, str) for answer in answers):
        raise ValueError("Every LSTM prediction must be a string.")
    if calculate_metrics(answers, references) != lstm["metrics"]["LSTM"]:
        raise ValueError("Saved LSTM answers do not reproduce the original metrics.")
    settings = qwen["metrics"]["bertscore"]
    if not settings or (settings["model_type"], settings["language"], settings["rescale_with_baseline"]) != ("roberta-large", "en", False):
        raise ValueError("Expected the final Qwen roberta-large English BERTScore settings.")
    return lstm, qwen, rows, answers, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lstm-dir", type=Path, default=ROOT / "runs/qa_eval_lstm_16epoch_test_200")
    parser.add_argument("--qwen-dir", type=Path, default=ROOT / "runs/qa_eval_cpt_sft_test_200_infra")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/qa_eval_lstm_16epoch_test_200_bertscore")
    parser.add_argument("--dry-run", action="store_true", help="Validate saved data and settings without loading the scoring model.")
    args = parser.parse_args()
    lstm, qwen, rows, answers, references = load_inputs(args.lstm_dir, args.qwen_dir)
    expected = qwen["metrics"]["bertscore"]
    print(f"[BERTScore] Validated {len(rows)} saved LSTM answers; Qwen hash: {expected['hash']}", flush=True)
    if args.dry_run:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a new, empty output folder.")
    try:
        import torch
        import transformers
        import bert_score
    except ImportError as error:
        raise SystemExit("Run in the existing Qwen evaluation environment with torch, transformers and bert-score installed.") from error
    if version("bert-score") != expected["package_version"] or transformers.__version__ != qwen["transformers"]:
        raise ValueError(f"Match the Qwen environment: bert-score=={expected['package_version']}, transformers=={qwen['transformers']}.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BERTScore] Scoring on {device}; no Qwen or LSTM model loading.", flush=True)
    scores, score_hash = bert_score.score(
        answers, references, model_type=expected["model_type"], lang=expected["language"],
        rescale_with_baseline=False, idf=False, device=device, batch_size=4, return_hash=True,
    )
    if score_hash != expected["hash"]:
        raise ValueError(f"Scoring settings differ from Qwen: {score_hash}")
    values = scores[2].detach().cpu().tolist()
    if len(values) != len(rows) or not all(math.isfinite(value) for value in values):
        raise ValueError("BERTScore did not return a finite score for every answer.")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_file_sha256": lstm["evaluation_file_sha256"],
        "evaluation_examples": len(rows), "sampling_seed": lstm["sampling_seed"],
        "model_file_sha256": lstm["model_file_sha256"], "generation": lstm["generation"],
        "source_predictions_sha256": file_sha256(args.lstm_dir / "predictions.jsonl"),
        "qwen_metrics_sha256": file_sha256(args.qwen_dir / "metrics.json"),
        "bertscore": dict(expected, hash=score_hash),
        "torch": torch.__version__, "transformers": transformers.__version__, "device": device,
        "metrics": {"LSTM": dict(lstm["metrics"]["LSTM"], bertscore_f1=mean(values))},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", [dict(row, candidate_bertscore_f1=value) for row, value in zip(rows, values)])
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[BERTScore] LSTM F1: {mean(values):.6f}; results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
