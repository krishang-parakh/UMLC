from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import py7zr
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score


def load_dataset(dataset_file: Path) -> list[dict[str, str]]:
    with dataset_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list records in {dataset_file}")
    return data


def build_prompt(question: str) -> str:
    return f"Solve this math word problem. Return only the final answer.\n\nQuestion: {question}\nAnswer:"


def predict(client: OpenAI, model: str, question: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
    response = client.responses.create(
        model=model,
        input=build_prompt(question),
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    return (response.output_text or "").strip()


def evaluate(client: OpenAI, model: str, data: Iterable[dict[str, str]], limit: int | None = None) -> tuple[float, float]:
    y_true: list[str] = []
    y_pred: list[str] = []

    for idx, item in enumerate(data):
        if limit is not None and idx >= limit:
            break
        question = item["question"]
        truth = str(item["answer"]).strip().lower()
        pred = predict(client, model, question).strip().lower()
        y_true.append(truth)
        y_pred.append(pred)

    if not y_true:
        raise ValueError("No evaluation samples were found")

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="weighted")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenAI model on math-word-problem datasets")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[2] / "datasets")
    parser.add_argument("--datasets", nargs="+", default=["MATH", "GSM8K"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    client = OpenAI(api_key=api_key)

    for name in args.datasets:
        dataset_path = resolve_dataset_path(args.dataset_dir, name)
        data = load_dataset(dataset_path)
        accuracy, f1 = evaluate(client, args.model, data, limit=args.limit)
        print(f"{name}: accuracy={accuracy:.4f}, f1={f1:.4f} (samples={min(len(data), args.limit)})")


if __name__ == "__main__":
    main()
