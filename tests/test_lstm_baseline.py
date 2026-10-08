"""Check sample alignment without installing TensorFlow or downloading models."""

import sys
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from healthcpt.lstm_baseline import check_cached_answers, training_text
from healthcpt import cli


class CachedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {"source_line": 4, "question": "What is asthma?", "reference_answer": "A lung condition."},
            {"source_line": 8, "question": "What is flu?", "reference_answer": "A viral infection."},
        ]
        self.cached = [dict(row, candidate_prediction="An answer.") for row in self.examples]

    def test_matching_question_reference_and_source_line(self):
        check_cached_answers(self.examples, self.cached)

    def test_reordered_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            check_cached_answers(self.examples, self.cached[::-1])

    def test_changed_reference_is_rejected(self):
        self.cached[0]["reference_answer"] = "Different answer."
        with self.assertRaises(ValueError):
            check_cached_answers(self.examples, self.cached)

    def test_missing_answer_or_wrong_count_is_rejected(self):
        with self.assertRaises(ValueError):
            check_cached_answers(self.examples, self.cached[:1])
        self.cached[0].pop("candidate_prediction")
        with self.assertRaises(ValueError):
            check_cached_answers(self.examples, self.cached)

    def test_prompt_and_decimal_values_are_preserved(self):
        example = {"question": "What about 0.5 mg?", "reference_answer": "Ask a clinician."}
        self.assertEqual(training_text(example), "Question: What about 0.5 mg?\nAnswer: Ask a clinician. [eos]")


class CommandEntryTests(unittest.TestCase):
    def test_cpt_command_and_old_alias_dispatch_identically(self):
        for command in ("cpt-train", "cpt-pilot"):
            with patch.object(sys, "argv", ["healthcpt", command, "train", "validation", "run"]):
                with patch("healthcpt.cpt.train", return_value={}) as train:
                    with redirect_stdout(io.StringIO()):
                        cli.main()
                    self.assertEqual(train.call_args.kwargs["train_path"], Path("train"))
                    self.assertEqual(train.call_args.kwargs["limit_train"], 16)

    def test_lstm_cli_uses_existing_training_defaults(self):
        argv = ["healthcpt", "lstm-train", "--train-file", "train", "--validation-file", "validation", "--output-dir", "run"]
        with patch.object(sys, "argv", argv):
            with patch("healthcpt.lstm_baseline.train") as train:
                cli.main()
                args = train.call_args.args[0]
                self.assertEqual(args.hidden_size, 256)
                self.assertEqual(args.vocab_size, 20000)
                self.assertEqual(args.seed, 5565)


if __name__ == "__main__":
    unittest.main()
