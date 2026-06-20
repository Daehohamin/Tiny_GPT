import unittest

import torch

from src.model import TinyGPT


class ModelTest(unittest.TestCase):
    def test_model_output_shape(self):
        model = TinyGPT(vocab_size=17, block_size=8, n_embd=16, n_head=4, n_layer=2, dropout=0.0)
        x = torch.randint(0, 17, (3, 8))
        y = torch.randint(0, 17, (3, 8))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (3, 8, 17))
        self.assertEqual(loss.dim(), 0)

    def test_causal_generation_shape(self):
        torch.manual_seed(0)
        model = TinyGPT(vocab_size=11, block_size=6, n_embd=12, n_head=3, n_layer=1, dropout=0.0)
        idx = torch.zeros((2, 4), dtype=torch.long)
        out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=5)
        self.assertEqual(tuple(out.shape), (2, 9))


if __name__ == "__main__":
    unittest.main()
