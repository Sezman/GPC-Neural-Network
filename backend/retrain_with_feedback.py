"""
Gmail Priority Classifier - retrain from user feedback
------------------------------------------------------------------------------
This script retrains the priority model on the STARTER dataset PLUS the user
corrections collected in feedback.csv. It's the "model improves from feedback"
step of the workflow:

    1. Use Gmail (badges appear).
    2. Correct any wrong labels (click a badge -> pick the right one).
    3. Run this script:   py retrain_with_feedback.py
    4. Restart the backend: py -m uvicorn main:app --reload
    5. The model now reflects your corrections.

It reuses the SAME embedding model and the SAME PriorityNet architecture as
train_nn.py, and writes to the same model/ files, so the backend (main.py) can
load the result with no changes.

The original train_nn.py is left untouched as the starter training script; use
it for a clean train on just the starter data, and use THIS script to fold in
feedback.

Everything runs locally. No email text (training data OR feedback) ever leaves
your machine.

Run it with:
    cd backend
    py retrain_with_feedback.py
------------------------------------------------------------------------------
"""

import os
import json
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration (kept identical to train_nn.py so the two stay in sync)
# ---------------------------------------------------------------------------
DATA_PATH = "email_training_data.csv"   # starter labeled examples (text,label)
FEEDBACK_PATH = "feedback.csv"          # user corrections (created by the backend)
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "priority_model.pt")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# Where evaluation reports are written after each run. Stays local, next to the
# model files; no data leaves the machine.
REPORTS_DIR = "reports"
REPORT_TXT_PATH = os.path.join(REPORTS_DIR, "evaluation_report.txt")
REPORT_JSON_PATH = os.path.join(REPORTS_DIR, "evaluation_report.json")
CONFUSION_PNG_PATH = os.path.join(REPORTS_DIR, "confusion_matrix.png")

# Same embedding model as train_nn.py: small, fast, 384-dimensional output.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384  # output dimension of all-MiniLM-L6-v2

HIDDEN_SIZE = 128   # neurons in the hidden layer
DROPOUT = 0.3       # randomly zero 30% of neurons during training (reduces overfitting)
EPOCHS = 300        # how many times we loop over the data
LEARNING_RATE = 1e-3

# Only these labels are valid. Anything else in the CSVs is dropped.
VALID_LABELS = ["High", "Medium", "Low"]

# Below this many combined examples we skip the train/test split and just train
# on everything (a held-out test set would be too tiny to be meaningful).
MIN_FOR_SPLIT = 30


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
# IMPORTANT: this MUST match PriorityNet in train_nn.py and main.py so the saved
# weights load correctly. Architecture: Linear -> ReLU -> Dropout -> Linear.
class PriorityNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=HIDDEN_SIZE, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # embedding -> hidden
            nn.ReLU(),                           # non-linearity
            nn.Dropout(dropout),                 # regularization
            nn.Linear(hidden_size, num_classes), # hidden -> 3 class logits
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_base_data():
    """Load the starter dataset using its text,label columns."""
    df = pd.read_csv(DATA_PATH)
    base = df[["text", "label"]].copy()
    return base


def load_feedback_data():
    """Load user corrections, if feedback.csv exists.

    Maps the feedback columns onto the training schema:
      - text          -> text (the email text)
      - correctedLabel -> label (the user-confirmed true label)
    predictedLabel and confidence are intentionally ignored for training.
    Returns an empty frame (with the right columns) when there's no feedback.
    """
    empty = pd.DataFrame(columns=["text", "label"])

    if not os.path.exists(FEEDBACK_PATH):
        print("No feedback.csv found - retraining on starter data only.")
        return empty

    df = pd.read_csv(FEEDBACK_PATH)
    if df.empty or "text" not in df.columns or "correctedLabel" not in df.columns:
        print("feedback.csv present but has no usable rows - using starter data only.")
        return empty

    feedback = pd.DataFrame(
        {
            "text": df["text"],
            "label": df["correctedLabel"],
        }
    )
    return feedback


def clean(df):
    """Apply the shared cleaning rules to a text,label dataframe.

    - drop rows with missing text or label
    - keep only the valid labels (High, Medium, Low)
    - strip whitespace and drop rows whose text is empty after stripping
    - remove exact duplicate (text, label) rows
    """
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df[df["label"].isin(VALID_LABELS)]
    df = df.drop_duplicates(subset=["text", "label"])
    return df


# ---------------------------------------------------------------------------
# Evaluation reports
# ---------------------------------------------------------------------------
def format_confusion_matrix(cm, target_names):
    """Render a confusion matrix as aligned text (rows = true, cols = pred)."""
    lines = []
    header = "          " + "  ".join(f"{name[:6]:>6}" for name in target_names)
    lines.append(header)
    for i, row in enumerate(cm):
        cells = "  ".join(f"{int(v):>6}" for v in row)
        lines.append(f"{target_names[i][:8]:>8}  {cells}")
    return "\n".join(lines)


def build_report_text(metrics):
    """Build the human-readable evaluation_report.txt contents from metrics."""
    lines = []
    lines.append("Gmail Priority Classifier - Evaluation Report")
    lines.append("=" * 50)
    lines.append(f"Timestamp:             {metrics['timestamp']}")
    lines.append(f"Base examples:         {metrics['baseExamples']}")
    lines.append(f"Feedback examples used:{metrics['feedbackExamplesUsed']:>4}")
    lines.append(f"Final dataset size:    {metrics['finalDatasetSize']}")
    lines.append("")
    lines.append("Class distribution:")
    for label, count in metrics["classDistribution"].items():
        lines.append(f"  {label:<7}: {count}")
    lines.append("")

    train_acc = metrics["trainAccuracy"]
    lines.append(f"Train accuracy: {train_acc * 100:.1f}%" if train_acc is not None else "Train accuracy: n/a")

    if metrics["testAccuracy"] is not None:
        lines.append(f"Test accuracy:  {metrics['testAccuracy'] * 100:.1f}%")
    else:
        lines.append("Test accuracy:  n/a (dataset too small for a train/test split)")

    if metrics["confusionMatrix"] is not None:
        lines.append("")
        lines.append("Confusion matrix (rows = true, cols = predicted):")
        import numpy as np  # local import; numpy ships with torch/sklearn

        lines.append(
            format_confusion_matrix(np.array(metrics["confusionMatrix"]), metrics["classes"])
        )

    if metrics["classificationReportText"] is not None:
        lines.append("")
        lines.append("Classification report:")
        lines.append(metrics["classificationReportText"])

    return "\n".join(lines) + "\n"


def save_confusion_matrix_png(metrics):
    """Save a confusion matrix image if matplotlib is installed.

    Returns True if the image was written, False if it was skipped (matplotlib
    missing or no confusion matrix to plot). Never raises on a missing library.
    """
    if metrics["confusionMatrix"] is None:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend: no display needed
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as error:
        print("matplotlib not available - skipping confusion matrix image:", error)
        return False

    cm = np.array(metrics["confusionMatrix"])
    names = metrics["classes"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Annotate each cell with its count, in a readable contrast color.
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(CONFUSION_PNG_PATH, dpi=120)
    plt.close(fig)
    return True


def write_reports(metrics):
    """Write evaluation_report.txt + .json, and a PNG if matplotlib is present."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report_text(metrics))
    print("\nSaved evaluation report to", REPORT_TXT_PATH)

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved evaluation report to", REPORT_JSON_PATH)

    if save_confusion_matrix_png(metrics):
        print("Saved confusion matrix image to", CONFUSION_PNG_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1) Load both sources and clean them with the same rules.
    print("Loading starter data from", DATA_PATH)
    base = clean(load_base_data())

    print("Loading feedback from", FEEDBACK_PATH)
    feedback = clean(load_feedback_data())

    n_base = len(base)
    n_feedback = len(feedback)

    # 2) Combine, then de-duplicate ACROSS both sources too, so a correction
    #    that merely confirms an existing example doesn't get counted twice.
    combined = pd.concat([base, feedback], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    # 3) Report the dataset makeup.
    print()
    print("=== Dataset summary ===")
    print(f"Base examples (starter):     {n_base}")
    print(f"Feedback examples used:      {n_feedback}")
    print(f"Final combined dataset size: {len(combined)}")
    print("Class distribution:")
    distribution = combined["label"].value_counts()
    for label in VALID_LABELS:
        print(f"  {label:<7}: {int(distribution.get(label, 0))}")
    print()

    if combined.empty:
        raise SystemExit("No usable training rows after cleaning. Aborting.")

    texts = combined["text"].tolist()
    labels = combined["label"].tolist()

    # 4) Build a deterministic label <-> index mapping (sorted).
    classes = sorted(set(labels))
    label_to_idx = {label: i for i, label in enumerate(classes)}
    num_classes = len(classes)
    print("Classes:", label_to_idx)

    # 5) Encode all text into embeddings with the SAME model as train_nn.py.
    print("Loading embedding model:", EMBEDDING_MODEL_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Encoding text into embeddings...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    X_all = torch.tensor(embeddings, dtype=torch.float32)
    y_all = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)

    input_size = X_all.shape[1]  # 384 for all-MiniLM-L6-v2
    print("Embedding size:", input_size)
    if input_size != EMBEDDING_SIZE:
        print(f"  (note: expected {EMBEDDING_SIZE}, got {input_size})")

    # 6) Decide whether we have enough data for a held-out test set.
    use_split = len(combined) >= MIN_FOR_SPLIT

    if use_split:
        print(f"\n{len(combined)} examples (>= {MIN_FOR_SPLIT}) -> using an 80/20 train/test split.")
        # Stratify when every class has at least 2 samples, so each split keeps
        # all three labels; otherwise fall back to a plain random split.
        label_counts = combined["label"].value_counts()
        stratify = y_all if label_counts.min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    else:
        print(f"\n{len(combined)} examples (< {MIN_FOR_SPLIT}) -> training on all data (no test split).")
        X_train, y_train = X_all, y_all
        X_test, y_test = None, None

    # 7) Build the model, loss, and optimizer, then train.
    model = PriorityNet(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training...")
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS} - loss {loss.item():.4f}")

    # 8) Evaluate. We collect everything into a `metrics` dict so the same
    #    numbers can be printed AND written to the report files below.
    target_names = [c for c in classes]  # e.g. ["High", "Low", "Medium"]
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseExamples": n_base,
        "feedbackExamplesUsed": n_feedback,
        "finalDatasetSize": len(combined),
        "classDistribution": {
            label: int(distribution.get(label, 0)) for label in VALID_LABELS
        },
        "classes": target_names,
        "usedSplit": use_split,
        "trainAccuracy": None,
        "testAccuracy": None,
        "confusionMatrix": None,        # list-of-lists (rows = true, cols = pred)
        "classificationReport": None,   # dict keyed by class + averages
        "classificationReportText": None,
    }

    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).argmax(dim=1)
    train_acc = accuracy_score(y_train.numpy(), train_preds.numpy())
    metrics["trainAccuracy"] = round(float(train_acc), 4)
    print(f"\nTrain accuracy: {train_acc * 100:.1f}%")

    if use_split:
        labels_idx = list(range(num_classes))

        with torch.no_grad():
            test_preds = model(X_test).argmax(dim=1)
        test_acc = accuracy_score(y_test.numpy(), test_preds.numpy())
        metrics["testAccuracy"] = round(float(test_acc), 4)
        print(f"Test accuracy:  {test_acc * 100:.1f}%")

        cm = confusion_matrix(y_test.numpy(), test_preds.numpy(), labels=labels_idx)
        metrics["confusionMatrix"] = cm.tolist()
        print("\nConfusion matrix (rows = true, cols = predicted):")
        print(format_confusion_matrix(cm, target_names))

        report_text = classification_report(
            y_test.numpy(),
            test_preds.numpy(),
            labels=labels_idx,
            target_names=target_names,
            zero_division=0,
        )
        report_dict = classification_report(
            y_test.numpy(),
            test_preds.numpy(),
            labels=labels_idx,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )
        metrics["classificationReportText"] = report_text
        metrics["classificationReport"] = report_dict
        print("\nClassification report:")
        print(report_text)

    # 8b) Write the persistent evaluation reports (txt + json + optional png).
    write_reports(metrics)

    # 9) Save the model + label map to the SAME files the backend reads.
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_size": HIDDEN_SIZE,
            "dropout": DROPOUT,
            "num_classes": num_classes,
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
        MODEL_PATH,
    )

    idx_to_label = {str(i): label for label, i in label_to_idx.items()}
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, indent=2)

    print("\nSaved model to", MODEL_PATH)
    print("Saved label map to", LABEL_MAP_PATH)
    print("Done! Restart the backend to use it: py -m uvicorn main:app --reload")


if __name__ == "__main__":
    main()
