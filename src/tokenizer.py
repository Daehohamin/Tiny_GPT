from __future__ import annotations


class CharTokenizer:
    """Character-level tokenizer with explicit stoi and itos tables."""

    def __init__(self, text: str):
        if not text:
            raise ValueError("Tokenizer needs non-empty text.")
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"Unknown character: {exc.args[0]!r}") from exc

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        try:
            return "".join(self.itos[int(i)] for i in ids)
        except KeyError as exc:
            raise ValueError(f"Unknown token id: {exc.args[0]!r}") from exc


def load_text(path: str = "data/korean_finance_corpus.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
