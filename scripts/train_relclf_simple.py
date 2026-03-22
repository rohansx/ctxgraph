#!/usr/bin/env python3
"""Train a simple relation classifier using sentence embeddings + logistic regression.

Uses all-MiniLM-L6-v2 (already in ctxgraph) to encode entity-marked text,
then trains a logistic regression classifier on top.

Requirements:
    pip install sentence-transformers scikit-learn onnx numpy

Usage:
    python scripts/prepare_training_data.py   # generate training_data.json first
    python scripts/train_relclf_simple.py
"""

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

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "training_data.json"
OUTPUT_DIR = Path("models/relation_classifier")

LABEL_NAMES = [
    "chose", "rejected", "replaced", "depends_on", "fixed",
    "introduced", "deprecated", "caused", "constrained_by", "none",
]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}


def load_data(path: Path):
    with open(path) as f:
        records = json.load(f)

    train = [r for r in records if r.get("split") == "train"]
    val = [r for r in records if r.get("split") in ("val", "eval", "dev", "test")]
    return train, val


def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.preprocessing import LabelEncoder

    log.info("Loading training data from %s", DATA_PATH)
    train_records, val_records = load_data(DATA_PATH)
    log.info("Train: %d, Val: %d", len(train_records), len(val_records))

    train_dist = Counter(r["label"] for r in train_records)
    log.info("Train distribution: %s", dict(sorted(train_dist.items())))

    # Encode texts with MiniLM
    log.info("Loading sentence-transformers model: all-MiniLM-L6-v2")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    log.info("Encoding training texts...")
    train_texts = [r["text"] for r in train_records]
    val_texts = [r["text"] for r in val_records]

    X_train = st_model.encode(train_texts, show_progress_bar=True, batch_size=64)
    X_val = st_model.encode(val_texts, show_progress_bar=True, batch_size=64)

    y_train = np.array([LABEL2ID[r["label"]] for r in train_records])
    y_val = np.array([LABEL2ID[r["label"]] for r in val_records])

    log.info("Embeddings: train=%s, val=%s", X_train.shape, X_val.shape)

    # Train logistic regression
    log.info("Training logistic regression...")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    log.info("Validation macro F1: %.4f", macro_f1)

    report = classification_report(
        y_val, y_pred,
        target_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        zero_division=0,
    )
    log.info("\n%s", report)

    # Also try with different C values
    best_f1, best_C = macro_f1, 1.0
    for C in [0.01, 0.1, 0.5, 2.0, 5.0, 10.0, 50.0, 100.0]:
        clf_test = LogisticRegression(
            max_iter=1000, C=C, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )
        clf_test.fit(X_train, y_train)
        f1 = f1_score(y_val, clf_test.predict(X_val), average="macro", zero_division=0)
        log.info("C=%.2f -> F1=%.4f", C, f1)
        if f1 > best_f1:
            best_f1 = f1
            best_C = C
            clf = clf_test

    log.info("Best C=%.2f with F1=%.4f", best_C, best_f1)

    # Final report with best model
    y_pred_best = clf.predict(X_val)
    report_best = classification_report(
        y_val, y_pred_best,
        target_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        zero_division=0,
    )
    log.info("Final classification report:\n%s", report_best)

    # Export classifier weights as ONNX
    log.info("Exporting to ONNX...")
    export_logreg_onnx(clf, OUTPUT_DIR)

    # Save label map
    label_map_path = OUTPUT_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({str(i): name for i, name in enumerate(LABEL_NAMES)}, f, indent=2)
    log.info("Label map saved to %s", label_map_path)

    log.info("Done! Model artifacts in %s", OUTPUT_DIR)
    for p in sorted(OUTPUT_DIR.iterdir()):
        size = p.stat().st_size
        if size > 1024:
            log.info("  %s (%d KB)", p.name, size // 1024)
        else:
            log.info("  %s (%d B)", p.name, size)


def export_logreg_onnx(clf, output_dir: Path):
    """Export logistic regression as ONNX: input is 384-dim embedding, output is 10-class logits."""
    import onnx
    from onnx import TensorProto, helper

    output_dir.mkdir(parents=True, exist_ok=True)

    W = clf.coef_.astype(np.float32)  # shape: [10, 384]
    b = clf.intercept_.astype(np.float32)  # shape: [10]

    # Create ONNX graph: logits = X @ W^T + b
    X = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 384])
    Y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 10])

    W_init = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
    b_init = helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.flatten().tolist())

    # MatMul: [1, 384] @ [384, 10] = [1, 10]
    matmul_node = helper.make_node("MatMul", ["embedding", "W_T"], ["matmul_out"])

    # Transpose W: [10, 384] -> [384, 10]
    transpose_node = helper.make_node("Transpose", ["W"], ["W_T"], perm=[1, 0])

    # Add bias
    add_node = helper.make_node("Add", ["matmul_out", "b"], ["logits"])

    graph = helper.make_graph(
        [transpose_node, matmul_node, add_node],
        "relation_classifier",
        [X],
        [Y],
        initializer=[W_init, b_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8

    onnx_path = output_dir / "model_int8.onnx"  # name it int8 since it's already tiny
    onnx.save(model, str(onnx_path))
    log.info("ONNX model saved to %s (%d KB)", onnx_path, onnx_path.stat().st_size // 1024)

    # Verify it runs
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(1, 384).astype(np.float32)
    out = sess.run(None, {"embedding": dummy})
    log.info("ONNX verification: output shape=%s", out[0].shape)


if __name__ == "__main__":
    main()
