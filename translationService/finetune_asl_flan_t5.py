import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL_NAME = "google/flan-t5-small"
DEFAULT_TRAIN_FILE = Path("data/asl_gloss_train.jsonl")
DEFAULT_VAL_FILE = Path("data/asl_gloss_val.jsonl")
DEFAULT_OUTPUT_DIR = Path("models/flan-t5-asl-mini")


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "input" not in row or "target" not in row:
            raise ValueError(f"{path}:{line_number} must contain input and target")
        rows.append({
            "input": str(row["input"]),
            "target": str(row["target"]),
        })
    return rows


class AslGlossDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        tokenizer,
        max_input_length: int,
        max_target_length: int,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        source = self.tokenizer(
            row["input"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            row["target"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze(0)
        labels = labels.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def train(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)

    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)

    train_dataset = AslGlossDataset(
        train_rows,
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    val_dataset = AslGlossDataset(
        val_rows,
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Train rows: {len(train_dataset)}")
    print(f"Validation rows: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += float(outputs.loss.item())
            total_batches += 1

        train_loss = total_loss / max(total_batches, 1)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-input-length", type=int, default=64)
    parser.add_argument("--max-target-length", type=int, default=48)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
