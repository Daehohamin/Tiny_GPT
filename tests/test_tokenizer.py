import unittest

from src.tokenizer import CharTokenizer


class TokenizerTest(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        text = "인공지능과 금융 시장"
        tokenizer = CharTokenizer(text)
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(len(tokenizer.stoi), tokenizer.vocab_size)
        self.assertEqual(len(tokenizer.itos), tokenizer.vocab_size)


if __name__ == "__main__":
    unittest.main()
