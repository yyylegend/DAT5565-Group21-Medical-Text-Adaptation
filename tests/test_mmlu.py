"""Check benchmark prompts, forced-choice scores, and source safeguards."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from healthcpt.evaluate_mmlu import (
    REVISION, SUBJECT, build_prompt, choice_probabilities, prepare, validate_example,
)


def example(answer=0):
    return {"question": "Which option is correct?", "choices": ["one", "two", "three", "four"], "answer": answer, "subject": SUBJECT}


class BenchmarkTests(unittest.TestCase):
    def test_gold_test_label_does_not_enter_prompt(self):
        demonstrations = [example(0)] * 5
        self.assertEqual(build_prompt(example(0), demonstrations), build_prompt(example(3), demonstrations))
        self.assertTrue(build_prompt(example(3), demonstrations).endswith("\nAnswer:"))
        self.assertEqual(build_prompt(example(3), demonstrations).count("Answer: A\n\n"), 5)

    def test_choice_rank_and_probabilities(self):
        prediction, probabilities = choice_probabilities([-1000, -1002, -999, -1001])
        self.assertEqual(prediction, "C")
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(probabilities.index(max(probabilities)), 2)

    def test_invalid_scores_rejected(self):
        for values in ([0, 1], [0, 1, float("nan"), 2], [0, 1, float("inf"), 2]):
            with self.assertRaises(ValueError):
                choice_probabilities(values)

    def test_incomplete_question_choices_or_gold_rejected(self):
        for change in ({"question": ""}, {"choices": ["a"]}, {"answer": True}, {"answer": 4}, {"subject": "other"}):
            with self.assertRaises(ValueError):
                validate_example(dict(example(), **change))

    def test_changed_revision_and_truncated_cells_rejected(self):
        for revision, truncated in (("new_revision", []), (REVISION, ["question"])):
            page = {"partial": False, "num_rows_total": 5, "rows": [{"row_idx": 0, "row": example(), "truncated_cells": truncated}]}
            class Response:
                headers = {"x-revision": revision}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self): return json.dumps(page).encode()
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "benchmark"
                with patch("healthcpt.evaluate_mmlu.urlopen", return_value=Response()):
                    with self.assertRaises(ValueError):
                        prepare(output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
