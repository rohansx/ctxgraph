#!/usr/bin/env python3
"""
Fine-tune DeBERTa-v3-small for entity-pair relation classification and export to ONNX.

Requirements:
    pip install transformers datasets torch optimum onnxruntime onnx

Training data format (scripts/training_data.json):
    [
      {
        "text": "We [E1]chose[/E1] [E2]React[/E2] for the frontend.",
        "head": "chose",
        "tail": "React",
        "label": "chose",
        "split": "train"
      },
      ...
    ]

Usage:
    python scripts/finetune_relation_classifier.py
    python scripts/finetune_relation_classifier.py --epochs 5 --batch-size 32
    python scripts/finetune_relation_classifier.py --data path/to/data.json --output path/to/output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_NAMES: list[str] = [
    "chose",
    "rejected",
    "replaced",
    "depends_on",
    "fixed",
    "introduced",
    "deprecated",
    "caused",
    "constrained_by",
    "none",
]

LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL: dict[int, str] = {i: name for name, i in LABEL2ID.items()}

ENTITY_MARKERS: list[str] = ["[E1]", "[/E1]", "[E2]", "[/E2]"]

MODEL_NAME = "microsoft/deberta-v3-small"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: Path) -> tuple[list[dict], list[dict]]:
    """Load training data JSON and split into train / eval sets."""
    if not path.exists():
        log.error("Training data not found at %s", path)
        sys.exit(1)

    with open(path) as f:
        records = json.load(f)

    log.info("Loaded %d records from %s", len(records), path)

    train_records = [r for r in records if r.get("split", "train") == "train"]
    eval_records = [r for r in records if r.get("split") in ("eval", "val", "dev", "test")]

    # If no explicit eval split, hold out 15 % of training data
    if not eval_records:
        log.info("No eval split found — holding out 15%% of data for evaluation")
        np.random.seed(42)
        indices = np.random.permutation(len(train_records))
        split_at = max(1, int(len(train_records) * 0.85))
        eval_indices = set(indices[split_at:].tolist())
        eval_records = [r for i, r in enumerate(train_records) if i in eval_indices]
        train_records = [r for i, r in enumerate(train_records) if i not in eval_indices]

    unknown_labels = {r["label"] for r in records if r["label"] not in LABEL2ID}
    if unknown_labels:
        log.warning("Unknown labels found (will be skipped): %s", unknown_labels)
        train_records = [r for r in train_records if r["label"] in LABEL2ID]
        eval_records = [r for r in eval_records if r["label"] in LABEL2ID]

    log.info("Train: %d  |  Eval: %d", len(train_records), len(eval_records))

    # Show class distribution
    train_dist = Counter(r["label"] for r in train_records)
    log.info("Train class distribution: %s", dict(sorted(train_dist.items())))

    return train_records, eval_records


def compute_class_weights(records: list[dict]) -> list[float]:
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = Counter(r["label"] for r in records)
    total = sum(counts.values())
    n_classes = len(LABEL_NAMES)
    weights = []
    for label in LABEL_NAMES:
        count = counts.get(label, 0)
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(total / (n_classes * count))
    return weights


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_datasets(
    train_records: list[dict],
    eval_records: list[dict],
    tokenizer,
    max_length: int = 256,
):
    """Tokenise records and return HF Dataset objects."""
    from datasets import Dataset

    def make_dataset(records: list[dict]) -> Dataset:
        texts = [r["text"] for r in records]
        labels = [LABEL2ID[r["label"]] for r in records]

        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )

        return Dataset.from_dict(
            {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }
        )

    return make_dataset(train_records), make_dataset(eval_records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def make_compute_metrics():
    """Return a compute_metrics function for the Trainer."""
    from sklearn.metrics import f1_score, classification_report

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        per_class_f1 = f1_score(labels, preds, average=None, labels=list(range(len(LABEL_NAMES))), zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

        metrics = {"eval_f1": macro_f1}
        for i, name in enumerate(LABEL_NAMES):
            metrics[f"f1_{name}"] = per_class_f1[i]

        # Print full classification report during evaluation
        report = classification_report(
            labels,
            preds,
            target_names=LABEL_NAMES,
            labels=list(range(len(LABEL_NAMES))),
            zero_division=0,
        )
        log.info("\n%s", report)

        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# Weighted loss trainer
# ---------------------------------------------------------------------------


class WeightedTrainer:
    """Trainer subclass that applies class weights to the loss."""

    @staticmethod
    def create(class_weights: list[float], **kwargs):
        import torch
        from transformers import Trainer

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        class _WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **loss_kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits

                device = logits.device
                w = weight_tensor.to(device)
                loss_fn = torch.nn.CrossEntropyLoss(weight=w)
                loss = loss_fn(logits, labels)

                return (loss, outputs) if return_outputs else loss

        return _WeightedTrainer(**kwargs)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(model_dir: Path, output_dir: Path, tokenizer):
    """Export the trained model to ONNX and quantise to INT8."""
    import torch
    from transformers import AutoModelForSequenceClassification
    from onnxruntime.quantization import quantize_dynamic, QuantType

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    int8_path = output_dir / "model_int8.onnx"

    log.info("Loading best model from %s", model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Create dummy inputs
    dummy = tokenizer(
        "We [E1]chose[/E1] [E2]React[/E2] for the frontend.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
    )

    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    log.info("Exporting to ONNX at %s", onnx_path)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Reload and save as single file (no external data) for quantization compat
    import onnx
    onnx_model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(onnx_model, str(onnx_path), save_as_external_data=False)
    # Remove leftover .data file if any
    data_file = onnx_path.with_suffix(".onnx.data")
    if data_file.exists():
        data_file.unlink()
    for f in output_dir.glob("*.data"):
        f.unlink()
    log.info("ONNX model saved (%s)", _file_size(onnx_path))

    # INT8 quantisation
    log.info("Quantising to INT8 …")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    log.info("INT8 model saved (%s)", _file_size(int8_path))

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    log.info("Tokenizer saved to %s", output_dir)

    # Save label map
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(ID2LABEL, f, indent=2)
    log.info("Label map saved to %s", label_map_path)


def _file_size(path: Path) -> str:
    """Return human-readable file size."""
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DeBERTa-v3-small for relation classification",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="scripts/training_data.json",
        help="Path to training data JSON (default: scripts/training_data.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/relation_classifier",
        help="Output directory for ONNX model (default: models/relation_classifier)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    return parser.parse_args()


def main():
    print("Requirements:")
    print("  pip install transformers datasets torch optimum onnxruntime onnx scikit-learn\n")

    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output)
    checkpoint_dir = Path("checkpoints/relation_classifier")

    # ------------------------------------------------------------------
    # Detect device
    # ------------------------------------------------------------------
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        log.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        log.info("Using Apple MPS")
    else:
        device = "cpu"
        log.info("Using CPU (no GPU detected)")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    train_records, eval_records = load_data(data_path)

    # ------------------------------------------------------------------
    # Tokenizer & model
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    log.info("Loading tokenizer and model: %s", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": ENTITY_MARKERS})
    log.info("Added %d special tokens to tokenizer", num_added)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))

    # ------------------------------------------------------------------
    # Freeze backbone — only train classifier head + last 2 layers
    # ------------------------------------------------------------------
    for name, param in model.named_parameters():
        if "classifier" in name or "pooler" in name:
            param.requires_grad = True
        elif "encoder.layer.5" in name or "encoder.layer.4" in name:
            # Unfreeze last 2 transformer layers
            param.requires_grad = True
        elif "embeddings" in name:
            # Keep embeddings trainable for new special tokens
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Trainable params: %d / %d (%.1f%%)", trainable, total, 100.0 * trainable / total)

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    train_ds, eval_ds = build_datasets(train_records, eval_records, tokenizer)
    log.info("Tokenised — train: %d, eval: %d", len(train_ds), len(eval_ds))

    # ------------------------------------------------------------------
    # Class distribution (no weighting — data is balanced by design)
    # ------------------------------------------------------------------
    train_dist = Counter(r["label"] for r in train_records)
    log.info("Class distribution: %s", dict(sorted(train_dist.items())))

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    from transformers import TrainingArguments, EarlyStoppingCallback

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_steps=10,
        bf16=(device == "cuda"),
        dataloader_num_workers=0 if device == "cpu" else 2,
        report_to="none",
        seed=42,
    )

    # ------------------------------------------------------------------
    # Trainer (standard — data is balanced, no need for weighted loss)
    # ------------------------------------------------------------------
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log.info("Starting training …")
    train_result = trainer.train()
    log.info("Training complete — metrics: %s", train_result.metrics)

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    log.info("Running final evaluation …")
    eval_metrics = trainer.evaluate()
    log.info("Eval metrics: %s", eval_metrics)

    # ------------------------------------------------------------------
    # Save best model for export
    # ------------------------------------------------------------------
    best_model_dir = checkpoint_dir / "best"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    log.info("Best model saved to %s", best_model_dir)

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------
    log.info("Exporting to ONNX …")
    export_onnx(best_model_dir, output_dir, tokenizer)

    log.info("Done! Artifacts saved to %s", output_dir)
    log.info("Files:")
    for p in sorted(output_dir.iterdir()):
        log.info("  %s  (%s)", p.name, _file_size(p))


if __name__ == "__main__":
    main()
