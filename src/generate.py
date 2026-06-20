from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import TinyGPT
from .tokenizer import CharTokenizer, load_text


def generate(args: argparse.Namespace) -> str:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    text = load_text(config.get("data_path", "data/korean_finance_corpus.txt"))
    tokenizer = CharTokenizer(text)

    model = TinyGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt) if args.prompt else [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    sample = tokenizer.decode(out)
    Path(args.output).write_text(sample, encoding="utf-8")
    print(sample)
    return sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Korean text from the trained Tiny GPT.")
    parser.add_argument("--checkpoint", default="outputs/tiny_gpt.pt")
    parser.add_argument("--output", default="outputs/generated_samples.txt")
    parser.add_argument("--prompt", default="인공지능")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    generate(build_parser().parse_args())


if __name__ == "__main__":
    main()
