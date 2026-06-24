import argparse
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch

from src.generate import (
    format_generation_result,
    generate,
    split_generation,
    validate_prompt,
)
from src.model import TinyGPT
from src.tokenizer import CharTokenizer


class GenerateTest(unittest.TestCase):
    def test_prompt_ids_are_prefix_and_continuation_has_requested_token_count(self):
        tokenizer = CharTokenizer("abc ")
        prompt_ids = tokenizer.encode("ab")
        full_ids = prompt_ids + tokenizer.encode("c a")

        input_prompt, continuation = split_generation(full_ids, prompt_ids, tokenizer, "ab")

        self.assertEqual(full_ids[: len(prompt_ids)], prompt_ids)
        self.assertEqual(input_prompt, "ab")
        self.assertEqual(continuation, "c a")
        self.assertEqual(len(tokenizer.encode(continuation)), 3)

    def test_display_labels_prompt_and_continuation_separately(self):
        result = format_generation_result("인공지능", " 금융")

        self.assertIn("=== 입력 프롬프트: 모델 생성 부분이 아님 ===", result)
        self.assertIn("인공지능", result)
        self.assertIn("=== 모델이 새로 생성한 이어쓰기: 3자 ===", result)
        self.assertTrue(result.endswith(" 금융"))

    def test_unsupported_prompt_character_reports_clear_validation_error(self):
        tokenizer = CharTokenizer("인공지능 금융")

        with self.assertRaisesRegex(
            ValueError,
            "Prompt contains characters outside the training vocabulary: 꼭",
        ) as context:
            validate_prompt("꼭", tokenizer)

        self.assertIn("character-level tokenizer", str(context.exception))
        self.assertIn("retraining the checkpoint", str(context.exception))

    def test_generate_uses_temp_paths_and_does_not_overwrite_committed_outputs(self):
        committed_paths = [
            Path("outputs/tiny_gpt.pt"),
            Path("outputs/generated_samples.txt"),
        ]
        before = {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in committed_paths
            if path.exists()
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_path = tmp_path / "corpus.txt"
            checkpoint_path = tmp_path / "tiny.pt"
            output_path = tmp_path / "generated.txt"
            corpus_path.write_text("abc abc abc\n", encoding="utf-8")

            tokenizer = CharTokenizer(corpus_path.read_text(encoding="utf-8"))
            model = TinyGPT(
                vocab_size=tokenizer.vocab_size,
                block_size=8,
                n_embd=8,
                n_head=2,
                n_layer=1,
                dropout=0.0,
            )
            torch.save(
                {
                    "config": {
                        "data_path": str(corpus_path),
                        "vocab_size": tokenizer.vocab_size,
                        "block_size": 8,
                        "n_embd": 8,
                        "n_head": 2,
                        "n_layer": 1,
                        "dropout": 0.0,
                    },
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            args = argparse.Namespace(
                checkpoint=str(checkpoint_path),
                output=str(output_path),
                prompt="ab",
                max_new_tokens=4,
                temperature=1.0,
                top_k=2,
                cpu=True,
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                sample = generate(args)

            saved = output_path.read_text(encoding="utf-8")
            prompt_ids = tokenizer.encode("ab")
            full_ids = tokenizer.encode(sample)
            _, continuation = split_generation(full_ids, prompt_ids, tokenizer, "ab")

            self.assertEqual(full_ids[: len(prompt_ids)], prompt_ids)
            self.assertEqual(len(tokenizer.encode(continuation)), 4)
            self.assertIn("=== 입력 프롬프트: 모델 생성 부분이 아님 ===", saved)
            self.assertIn("=== 모델이 새로 생성한 이어쓰기: 4자 ===", saved)
            self.assertEqual(saved, stdout.getvalue().strip())

        after = {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in committed_paths
            if path.exists()
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
