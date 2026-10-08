"""Guard alignment and provenance when scoring previously saved answers."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from healthcpt import score_lstm_bertscore as scorer


class LSTMBERTScoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.lstm, self.qwen, self.output = root / "lstm", root / "qwen", root / "scored"
        self.lstm.mkdir()
        self.qwen.mkdir()
        self.rows = [
            {"source_line": 1, "question": "What is asthma?", "reference_answer": "Airway disease", "candidate_prediction": "airway disease"},
            {"source_line": 2, "question": "What is flu?", "reference_answer": "Viral infection", "candidate_prediction": "infection"},
        ]
        metrics = scorer.calculate_metrics([r["candidate_prediction"] for r in self.rows], [r["reference_answer"] for r in self.rows])
        self.lstm_report = {"evaluation_examples": 2, "evaluation_file_sha256": "data", "sampling_seed": 5565,
                            "model_file_sha256": "model", "generation": {"max_new_words": 128}, "metrics": {"LSTM": metrics}}
        self.settings = {"model_type": "roberta-large", "language": "en", "rescale_with_baseline": False,
                         "package_version": "0.3.13", "hash": "expected-score-hash"}
        self.qwen_report = {"evaluation_examples": 2, "evaluation_file_sha256": "data", "transformers": "5.17.0", "metrics": {"bertscore": self.settings}}
        for folder, report in [(self.lstm, self.lstm_report), (self.qwen, self.qwen_report)]:
            (folder / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
            scorer.write_jsonl(folder / "predictions.jsonl", self.rows)

    def test_reordered_questions_are_rejected(self):
        scorer.write_jsonl(self.qwen / "predictions.jsonl", self.rows[::-1])
        with self.assertRaisesRegex(ValueError, "misaligned"):
            scorer.load_inputs(self.lstm, self.qwen)

    def test_changed_answers_are_rejected(self):
        changed = [dict(self.rows[0], candidate_prediction="unrelated"), self.rows[1]]
        scorer.write_jsonl(self.lstm / "predictions.jsonl", changed)
        with self.assertRaisesRegex(ValueError, "original metrics"):
            scorer.load_inputs(self.lstm, self.qwen)

    def run_scoring(self, score_hash):
        class Values:
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return [0.8, 0.9]
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), __version__="test"),
            "transformers": SimpleNamespace(__version__="5.17.0"),
            "bert_score": SimpleNamespace(score=lambda *args, **kwargs: ((None, None, Values()), score_hash)),
        }
        argv = ["score", "--lstm-dir", str(self.lstm), "--qwen-dir", str(self.qwen), "--output-dir", str(self.output)]
        with patch.dict(sys.modules, modules), patch.object(scorer, "version", return_value="0.3.13"), patch.object(sys, "argv", argv):
            scorer.main()

    def test_score_hash_mismatch_does_not_publish_results(self):
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.run_scoring("different-hash")
        self.assertFalse(self.output.exists())

    def test_scoring_preserves_inputs_and_saves_per_answer_scores(self):
        before = scorer.file_sha256(self.lstm / "predictions.jsonl")
        self.run_scoring(self.settings["hash"])
        report = scorer.read_json(self.output / "metrics.json")
        self.assertAlmostEqual(report["metrics"]["LSTM"]["bertscore_f1"], 0.85)
        self.assertEqual(report["source_predictions_sha256"], before)
        self.assertEqual(scorer.file_sha256(self.lstm / "predictions.jsonl"), before)
        self.assertEqual([r["candidate_bertscore_f1"] for r in scorer.read_rows(self.output / "predictions.jsonl")], [0.8, 0.9])


if __name__ == "__main__":
    unittest.main()
