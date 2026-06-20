from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .dataset import CharSequenceDataset, split_ids
from .model import TinyGPT
from .tokenizer import CharTokenizer, load_text


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def estimate_loss(model, loader, device, max_batches: int = 8) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def save_loss_curve(history: list[dict[str, float]], path: Path) -> None:
    steps = [row["iter"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, train_losses, label="train")
    plt.plot(steps, val_losses, label="val")
    plt.xlabel("iteration")
    plt.ylabel("cross entropy loss")
    plt.title("Tiny Korean GPT loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = load_text(args.data_path)
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)
    train_ids, val_ids = split_ids(ids, train_ratio=0.9)

    train_ds = CharSequenceDataset(train_ids, args.block_size)
    val_ds = CharSequenceDataset(val_ids, min(args.block_size, len(val_ids) - 1))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float]] = []
    train_iter = iter(train_loader)
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0 or step == args.max_iters:
            history.append(
                {
                    "iter": step,
                    "train_loss": estimate_loss(model, train_loader, device),
                    "val_loss": estimate_loss(model, val_loader, device),
                }
            )
            print(
                f"iter {step:04d}: train {history[-1]['train_loss']:.4f}, "
                f"val {history[-1]['val_loss']:.4f}"
            )
        if step == args.max_iters:
            break
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    history_path = out_dir / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    save_loss_curve(history, out_dir / "loss_curve.png")

    config = {
        "data_path": args.data_path,
        "vocab_size": tokenizer.vocab_size,
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "max_iters": args.max_iters,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": device,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "stoi": tokenizer.stoi,
    }
    (out_dir / "model_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, out_dir / "tiny_gpt.pt")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(start, max_new_tokens=args.sample_tokens, temperature=0.8, top_k=20)[0]
    sample = tokenizer.decode(generated.tolist())
    (out_dir / "generated_samples.txt").write_text(sample, encoding="utf-8")
    print(f"saved outputs to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small Korean character-level GPT.")
    parser.add_argument("--data-path", default="data/korean_finance_corpus.txt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=240)
    parser.add_argument("--eval-interval", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--sample-tokens", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true", help="CPU-safe short run for Codespaces.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.quick:
        args.max_iters = min(args.max_iters, 120)
        args.eval_interval = min(args.eval_interval, 20)
        args.batch_size = min(args.batch_size, 32)
        args.cpu = True
    train(args)


if __name__ == "__main__":
    main()
