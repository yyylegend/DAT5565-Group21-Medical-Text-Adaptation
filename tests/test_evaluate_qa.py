import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from healthcpt.evaluate_qa import generation_eos_token_ids


class GenerationEOSTests(unittest.TestCase):
    def test_includes_qwen_turn_end_and_tokenizer_eos(self):
        class Tokenizer:
            eos_token_id = 248044

            @staticmethod
            def convert_tokens_to_ids(token):
                return {"<|im_end|>": 248046}[token]

        self.assertEqual(generation_eos_token_ids(Tokenizer()), [248044, 248046])

    def test_deduplicates_matching_end_ids(self):
        class Tokenizer:
            eos_token_id = 248046

            @staticmethod
            def convert_tokens_to_ids(token):
                return 248046

        self.assertEqual(generation_eos_token_ids(Tokenizer()), [248046])


if __name__ == "__main__":
    unittest.main()
