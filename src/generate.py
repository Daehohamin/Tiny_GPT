from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import TinyGPT
from .tokenizer import CharTokenizer, load_text


def validate_prompt(prompt: str, tokenizer: CharTokenizer) -> None:
    unsupported = list(dict.fromkeys(ch for ch in prompt if ch not in tokenizer.stoi))
    if unsupported:
        chars = "".join(unsupported)
        raise ValueError(
            "Prompt contains characters outside the training vocabulary: "
            f"{chars}\n"
            "This character-level tokenizer only supports characters found in "
            "the training corpus. Expanding the vocabulary requires retraining "
            "the checkpoint."
        )


def split_generation(
    full_token_ids: list[int],
    prompt_token_ids: list[int],
    tokenizer: CharTokenizer,
    original_prompt: str,
) -> tuple[str, str]:
    prompt_length = len(prompt_token_ids)
    if full_token_ids[:prompt_length] != prompt_token_ids:
        raise RuntimeError(
            "Internal generation invariant violated: generated token sequence "
            "does not begin with the original prompt token IDs."
        )
    continuation_ids = full_token_ids[prompt_length:]
    return original_prompt, tokenizer.decode(continuation_ids)


def format_generation_result(input_prompt: str, generated_continuation: str) -> str:
    return (
        "=== 입력 프롬프트: 모델 생성 부분이 아님 ===\n"
        f"{input_prompt}\n\n"
        f"=== 모델이 새로 생성한 이어쓰기: {len(generated_continuation)}자 ===\n"
        f"{generated_continuation}"
    )


def generate(args: argparse.Namespace) -> str:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    text = load_text(config.get("data_path", "data/korean_finance_corpus.txt"))
    tokenizer = CharTokenizer(text)
    validate_prompt(args.prompt, tokenizer)

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
    input_prompt, generated_continuation = split_generation(
        out, prompt_ids, tokenizer, args.prompt
    )
    sample = tokenizer.decode(out)
    labelled_result = format_generation_result(input_prompt, generated_continuation)
    Path(args.output).write_text(labelled_result, encoding="utf-8")
    print(labelled_result)
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
