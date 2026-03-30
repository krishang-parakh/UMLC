from __future__ import annotations

import argparse
import json
from pathlib import Path

import py7zr

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BartForConditionalGeneration, BartTokenizerFast


class MathWordProblemDataset(Dataset):
    def __init__(self, data_file: Path, tokenizer: BartTokenizerFast, max_length: int = 256):
        with data_file.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data[idx]
        source = self.tokenizer(row["question"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        target = self.tokenizer(str(row["answer"]), padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": target["input_ids"].squeeze(0),
        }


def resolve_dataset_path(dataset_dir: Path, name: str) -> Path:
    p = dataset_dir / f"{name}.json"
    if p.exists():
        return p

    nested = dataset_dir / name / f"{name}.json"
    if nested.exists():
        return nested

    archive = dataset_dir / f"{name}.7z"
    if archive.exists():
        with py7zr.SevenZipFile(archive, mode="r") as zf:
            zf.extractall(path=dataset_dir)
        if p.exists():
            return p
        if nested.exists():
            return nested

    raise FileNotFoundError(
        f"Unable to find dataset for '{name}'. Checked {p}, {nested}, and archive {archive}."
    )


def train_epoch(model: BartForConditionalGeneration, loader: DataLoader, device: torch.device, lr: float = 2e-5) -> float:
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    total = 0.0
    for batch in loader:
        optim.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        outputs.loss.backward()
        optim.step()
        total += float(outputs.loss.item())
    return total / max(1, len(loader))


def evaluate(model: BartForConditionalGeneration, tokenizer: BartTokenizerFast, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            out = model.generate(batch["input_ids"].to(device), max_new_tokens=32)
            y_pred.extend([x.strip().lower() for x in tokenizer.batch_decode(out, skip_special_tokens=True)])
            y_true.extend([x.strip().lower() for x in tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)])
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="weighted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Bart-based BertGen baseline")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[2] / "datasets")
    parser.add_argument("--dataset", default="MATH")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)

    dataset = MathWordProblemDataset(resolve_dataset_path(args.dataset_dir, args.dataset), tokenizer)
    if args.max_samples < len(dataset):
        dataset, _ = random_split(dataset, [args.max_samples, len(dataset) - args.max_samples])

    train_size = max(1, int(0.8 * len(dataset)))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    loss = train_epoch(model, train_loader, device)
    acc, f1 = evaluate(model, tokenizer, test_loader, device)
    print(f"{args.dataset}: loss={loss:.4f}, accuracy={acc:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    main()
