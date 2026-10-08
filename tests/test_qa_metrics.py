"""Protect the shared metric behavior when reorganizing evaluation code."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from healthcpt import evaluate_qa, qa_metrics


class QAMetricTests(unittest.TestCase):
    def test_token_f1_counts_repeated_words(self):
        self.assertAlmostEqual(qa_metrics.token_f1("a a b", "a b b"), 2 / 3)

    def test_rouge_respects_word_order(self):
        self.assertAlmostEqual(qa_metrics.rouge_l_f1("a b c", "a c b"), 2 / 3)

    def test_disjoint_or_empty_answers_have_zero_overlap(self):
        for function in (qa_metrics.token_f1, qa_metrics.rouge_l_f1):
            self.assertEqual(function("flu", "asthma"), 0)
            self.assertEqual(function("", "asthma"), 0)

    def test_case_and_punctuation_normalization(self):
        result = qa_metrics.calculate_metrics(["ASTHMA!", ""], ["asthma", "flu"])
        self.assertEqual(result["normalized_exact_match"], 0.5)
        self.assertEqual(result["token_f1"], 0.5)
        self.assertEqual(result["rouge_l_f1"], 0.5)
        self.assertEqual(result["empty_predictions"], 1)

    def test_sampling_preserves_file_order_and_seed(self):
        rows = [{"source_line": i} for i in range(30)]
        first = qa_metrics.select_examples(rows, 10, 5565)
        self.assertEqual(first, qa_metrics.select_examples(rows, 10, 5565))
        self.assertEqual(len(first), 10)
        self.assertEqual([r["source_line"] for r in first], sorted(r["source_line"] for r in first))

    def test_old_metric_imports_still_work(self):
        self.assertIs(evaluate_qa.calculate_metrics, qa_metrics.calculate_metrics)
        self.assertIs(evaluate_qa.select_examples, qa_metrics.select_examples)


if __name__ == "__main__":
    unittest.main()
