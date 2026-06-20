import unittest

import torch

from src.baselines import MLPNextChar, SequenceNoAttention, SingleHeadMaskedSelfAttention


class BaselineTest(unittest.TestCase):
    def test_mlp_shapes(self):
        model = MLPNextChar(vocab_size=13, block_size=3, n_embd=4, hidden=8)
        x = torch.randint(0, 13, (5, 3))
        y = torch.randint(0, 13, (5,))
        logits, loss, emb, flat = model(x, y)
        self.assertEqual(tuple(emb.shape), (5, 3, 4))
        self.assertEqual(tuple(flat.shape), (5, 12))
        self.assertEqual(tuple(logits.shape), (5, 13))
        self.assertEqual(loss.dim(), 0)

    def test_sequence_no_attention_shapes(self):
        model = SequenceNoAttention(vocab_size=17, block_size=6, n_embd=8)
        x = torch.randint(0, 17, (2, 6))
        logits, loss = model(x, x)
        self.assertEqual(tuple(logits.shape), (2, 6, 17))
        self.assertEqual(loss.dim(), 0)

    def test_masked_attention_future_probabilities_zero(self):
        torch.manual_seed(0)
        attn = SingleHeadMaskedSelfAttention(n_embd=8, head_size=4, block_size=5)
        x = torch.randn(1, 5, 8)
        _, weights = attn(x)
        future = torch.triu(weights[0], diagonal=1)
        self.assertTrue(torch.allclose(future, torch.zeros_like(future)))


if __name__ == "__main__":
    unittest.main()
