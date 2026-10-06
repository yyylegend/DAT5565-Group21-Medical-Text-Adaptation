"""Build an offline bilingual dashboard from the final local run artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(output):
    sources = {
        "qa": "runs/qa_eval_cpt_sft_test_200_infra/metrics.json",
        "mmlu": "runs/mmlu_professional_medicine_5shot/metrics.json",
        "lstm_eval": "runs/qa_eval_lstm_16epoch_test_200/metrics.json",
        "lstm_train": "runs/lstm_qa_50epochs/run.json",
        "cpt_train": "runs/qwen3_5_2b_cpt_full_1epoch/run.json",
        "sft_train": "runs/qwen3_5_2b_sft_full_1epoch/run.json",
        "training_curves": "runs/training_curves.json",
    }
    data = {key: read_json(ROOT / path) for key, path in sources.items()}
    bertscore_path = ROOT / "runs/qa_eval_lstm_16epoch_test_200_bertscore/metrics.json"
    if bertscore_path.exists():
        scored = read_json(bertscore_path)
        original_path = ROOT / "runs/qa_eval_lstm_16epoch_test_200/predictions.jsonl"
        if scored["source_predictions_sha256"] != hashlib.sha256(original_path.read_bytes()).hexdigest():
            raise ValueError("LSTM BERTScore belongs to different predictions.")
        qwen_path = ROOT / sources["qa"]
        if scored["qwen_metrics_sha256"] != hashlib.sha256(qwen_path.read_bytes()).hexdigest() or scored["bertscore"] != data["qa"]["metrics"]["bertscore"]:
            raise ValueError("LSTM BERTScore does not match the final Qwen scoring settings.")
        data["lstm_eval"]["metrics"]["LSTM"]["bertscore_f1"] = scored["metrics"]["LSTM"]["bertscore_f1"]
        sources["lstm_bertscore"] = bertscore_path.relative_to(ROOT).as_posix()
    for stage in ("cpt", "sft"):
        for source in data["training_curves"][stage]["sources"]:
            if hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("TensorBoard logs changed; rerun scripts/extract_training_curves.py.")
    qa = read_rows(ROOT / "runs/qa_eval_cpt_sft_test_200_infra/predictions.jsonl")
    lstm = read_rows(ROOT / "runs/qa_eval_lstm_16epoch_test_200/predictions.jsonl")
    if len(qa) != data["qa"]["evaluation_examples"] or len(lstm) != data["lstm_eval"]["evaluation_examples"]:
        raise ValueError("QA prediction row counts do not match the reports.")
    fields = ("source_line", "question", "reference_answer")
    if len(qa) != len(lstm) or any(any(a[k] != b[k] for k in fields) for a, b in zip(qa, lstm)):
        raise ValueError("LSTM and final Qwen results do not use the same questions and references.")
    if data["qa"]["evaluation_file_sha256"] != data["lstm_eval"]["evaluation_file_sha256"]:
        raise ValueError("LSTM and Qwen evaluation data hashes differ.")
    benchmark = ROOT / "runs/mmlu_professional_medicine_5shot"
    paired = [read_rows(benchmark / f"{name}_predictions.jsonl") for name in ("base", "candidate")]
    for name, rows in zip(("base", "candidate"), paired):
        report = data["mmlu"]["models"][name]
        if len(rows) != data["mmlu"]["examples"] or sum(r["prediction"] == r["gold"] for r in rows) != report["correct"]:
            raise ValueError(f"MMLU {name} predictions do not match the report.")
    if any(a["id"] != b["id"] or a["gold"] != b["gold"] for a, b in zip(*paired)):
        raise ValueError("MMLU paired predictions are misaligned.")
    data["sources"] = [{"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()} for path in sources.values()]
    serialized = json.dumps(data, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).with_name("results_dashboard.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("{{DATA}}", serialized), encoding="utf-8")
    return output.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/final_results.html")
    parser.add_argument("--open", action="store_true", help="Open the generated page in the default browser.")
    args = parser.parse_args()
    output = build(args.output)
    print(f"Dashboard: {output}")
    if args.open:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
