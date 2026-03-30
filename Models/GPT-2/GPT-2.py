from __future__ import annotations

import argparse
import json
from pathlib import Path

import py7zr

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class MathWordProblemDataset(Dataset):
    def __init__(self, data_file: Path, tokenizer: GPT2TokenizerFast, max_length: int = 256):
        with data_file.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data[idx]
        prompt = f"{row['question']}\nAnswer:"
        answer = str(row["answer"])
        enc = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        tgt = self.tokenizer(answer, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": tgt["input_ids"].squeeze(0),
        }


def resolve_dataset_path(dataset_dir: Path, name: str) -> Path:
    candidate = dataset_dir / f"{name}.json"
    if candidate.exists():
        return candidate

    nested = dataset_dir / name / f"{name}.json"
    if nested.exists():
        return nested

    archive = dataset_dir / f"{name}.7z"
    if archive.exists():
        with py7zr.SevenZipFile(archive, mode="r") as zf:
            zf.extractall(path=dataset_dir)
        if candidate.exists():
            return candidate
        if nested.exists():
            return nested

    raise FileNotFoundError(
        f"Unable to find dataset for '{name}'. Checked {candidate}, {nested}, and archive {archive}."
    )


def make_loaders(dataset: Dataset, batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
    train_size = max(1, int(0.8 * len(dataset)))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size)


def train_epoch(model: GPT2LMHeadModel, loader: DataLoader, device: torch.device, lr: float = 5e-5) -> float:
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_total = 0.0
    for batch in loader:
        optim.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        outputs.loss.backward()
        optim.step()
        loss_total += float(outputs.loss.item())
    return loss_total / max(1, len(loader))


def evaluate_exact_match(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            generated = model.generate(input_ids=input_ids, max_new_tokens=32)
            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            truths = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            y_pred.extend([p.strip().lower() for p in preds])
            y_true.extend([t.strip().lower() for t in truths])

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="weighted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate GPT-2 on a math-word-problem dataset")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[2] / "datasets")
    parser.add_argument("--dataset", default="MATH")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    data_file = resolve_dataset_path(args.dataset_dir, args.dataset)
    dataset = MathWordProblemDataset(data_file, tokenizer)
    if args.max_samples < len(dataset):
        dataset, _ = random_split(dataset, [args.max_samples, len(dataset) - args.max_samples])

    train_loader, test_loader = make_loaders(dataset, batch_size=args.batch_size)
    loss = train_epoch(model, train_loader, device)
    acc, f1 = evaluate_exact_match(model, tokenizer, test_loader, device)
    print(f"{args.dataset}: loss={loss:.4f}, accuracy={acc:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    main()
