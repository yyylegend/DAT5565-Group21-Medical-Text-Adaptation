"""Check sample alignment without installing TensorFlow or downloading models."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from healthcpt.lstm_baseline import check_cached_answers, training_text


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


if __name__ == "__main__":
    unittest.main()
