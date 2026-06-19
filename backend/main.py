"""
Gmail Priority Classifier - FastAPI backend
------------------------------------------------------------------------------
This backend scores an email's text into High / Medium / Low priority.

It has TWO scoring engines:
  1. Neural network (preferred): sentence-transformers embeddings + a small
     PyTorch classifier trained by train_nn.py.
  2. Rule-based keyword scoring (fallback): the original keyword matching. Used
     automatically if the model isn't trained yet, the files are missing, or the
     ML libraries aren't installed.

Everything runs LOCALLY. No email text is ever sent to an external API.

Run it with:
    cd backend
    pip install -r requirements.txt
    py train_nn.py                       # train the model (one time / when data changes)
    py -m uvicorn main:app --reload      # start the server at http://127.0.0.1:8000
------------------------------------------------------------------------------
"""

import os
import csv
import json
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Optional ML imports.
# ---------------------------------------------------------------------------
# We import torch / sentence-transformers inside a try block so the backend
# still runs (in rule-based mode) even if those heavy libraries aren't
# installed. This keeps the fallback path always available.
try:
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer

    TORCH_AVAILABLE = True
except Exception as _import_error:  # pragma: no cover - depends on environment
    TORCH_AVAILABLE = False
    print("[backend] ML libraries not available, rule-based only:", _import_error)


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "priority_model.pt")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Where user corrections are appended. Stays local, next to this file, so no
# email text ever leaves the machine. Created on first correction if missing.
FEEDBACK_PATH = "feedback.csv"
FEEDBACK_HEADER = ["timestamp", "text", "predictedLabel", "correctedLabel", "confidence"]


app = FastAPI(title="Gmail Priority Classifier")

# CORS so the extension's content script on https://mail.google.com can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# RULE-BASED SCORING (fallback) - kept intact from before
# ===========================================================================
# Strong signals. (+3 each)
HIGH_KEYWORDS = [
    "urgent", "action needed", "action required", "payment failed",
    "subscription canceled", "subscription has been canceled",
    "subscription has been cancelled", "account suspended", "security alert",
    "password reset", "verification code", "interview", "offer", "deadline",
    "final notice", "due today",
]

# Moderate signals. (+1 each)
MEDIUM_KEYWORDS = [
    "meeting", "reminder", "application received", "invoice", "receipt",
    "billing", "trial", "free trial", "upgrade", "upgrade to paid",
    "appointment", "recall reminder", "confirmation",
]

# Promotional / low-value signals. (-1 each)
LOW_KEYWORDS = [
    "newsletter", "digest", "privacy policy", "promotion", "sale",
    "unsubscribe", "alumni spotlight", "performance report",
]


def rule_based_score(text: str) -> dict:
    """The original keyword scoring. Always available, no ML needed."""
    normalized = " ".join(text.split()).lower()

    score = 0
    matched_high, matched_medium, matched_low = [], [], []

    for keyword in HIGH_KEYWORDS:
        if keyword in normalized:
            score += 3
            matched_high.append(keyword)
    for keyword in MEDIUM_KEYWORDS:
        if keyword in normalized:
            score += 1
            matched_medium.append(keyword)
    for keyword in LOW_KEYWORDS:
        if keyword in normalized:
            score -= 1
            matched_low.append(keyword)

    if score >= 3:
        label = "High"
    elif score >= 1:
        label = "Medium"
    else:
        label = "Low"

    return {
        "label": label,
        "score": score,
        "matchedHigh": matched_high,
        "matchedMedium": matched_medium,
        "matchedLow": matched_low,
        "source": "rule-based",
    }


# ===========================================================================
# NEURAL NETWORK SCORING (preferred)
# ===========================================================================
# Global state filled in by load_neural_model() at startup.
MODEL_READY = False
nn_model = None
embedder = None
label_map = None  # e.g. {"0": "High", "1": "Low", "2": "Medium"}


if TORCH_AVAILABLE:
    # Architecture MUST match train_nn.py: Linear -> ReLU -> Dropout -> Linear.
    class PriorityNet(nn.Module):
        def __init__(self, input_size, num_classes, hidden_size=128, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes),
            )

        def forward(self, x):
            return self.net(x)


def load_neural_model():
    """Try to load the trained model + embedder. Sets MODEL_READY accordingly.

    Any failure (missing files, libraries, or load error) leaves MODEL_READY
    False, so /score transparently uses the rule-based fallback.
    """
    global MODEL_READY, nn_model, embedder, label_map

    if not TORCH_AVAILABLE:
        print("[backend] torch/sentence-transformers missing -> rule-based mode")
        return

    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH)):
        print("[backend] model not trained yet (run train_nn.py) -> rule-based mode")
        return

    try:
        # Label map: index -> label.
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            loaded_label_map = json.load(f)

        # Rebuild the network from the saved checkpoint and load its weights.
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model = PriorityNet(
            checkpoint["input_size"],
            checkpoint["num_classes"],
            checkpoint.get("hidden_size", 128),
            checkpoint.get("dropout", 0.3),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # Load the same embedding model that was used during training.
        emb_name = checkpoint.get("embedding_model", EMBEDDING_MODEL_NAME)
        print("[backend] loading embedding model:", emb_name)
        loaded_embedder = SentenceTransformer(emb_name)

        # Commit to globals only after everything succeeded.
        nn_model = model
        embedder = loaded_embedder
        label_map = loaded_label_map
        MODEL_READY = True
        print("[backend] neural network loaded -> neural-network mode")
    except Exception as error:
        MODEL_READY = False
        print("[backend] failed to load neural model, using rule-based:", error)


def neural_score(text: str) -> dict:
    """Score text with the neural network. Returns label + confidence."""
    # Turn the text into an embedding (shape: 1 x embedding_size).
    embedding = embedder.encode([text], convert_to_numpy=True)
    x = torch.tensor(embedding, dtype=torch.float32)

    # Run the network and convert logits to probabilities with softmax.
    with torch.no_grad():
        logits = nn_model(x)
        probabilities = torch.softmax(logits, dim=1)[0]

    # The predicted class is the highest probability; the score is that
    # probability (the model's confidence in its choice).
    confidence, predicted_index = torch.max(probabilities, dim=0)
    label = label_map[str(predicted_index.item())]

    return {
        "label": label,
        "score": round(confidence.item(), 4),
        "matchedHigh": [],
        "matchedMedium": [],
        "matchedLow": [],
        "source": "neural-network",
    }


# Try to load the model as soon as the module is imported (i.e. on server start).
load_neural_model()


# ===========================================================================
# API
# ===========================================================================
class ScoreRequest(BaseModel):
    """Body of POST /score -> {"text": "email row text here"}"""

    text: str


class FeedbackRequest(BaseModel):
    """Body of POST /feedback.

    Sent when the user corrects a badge in Gmail. We record the original
    prediction alongside the corrected label so the data can later be used to
    retrain the model.
    """

    text: str
    predictedLabel: str
    correctedLabel: str
    confidence: float


@app.get("/health")
def health():
    """Simple liveness check."""
    return {"status": "ok"}


@app.get("/model-status")
def model_status():
    """Report which scoring engine is active so you can confirm setup."""
    return {
        "mode": "neural-network" if MODEL_READY else "rule-based",
        "modelLoaded": MODEL_READY,
        "embeddingModel": EMBEDDING_MODEL_NAME if MODEL_READY else None,
        "torchAvailable": TORCH_AVAILABLE,
    }


@app.post("/score")
def score(request: ScoreRequest):
    """Score one email row's text.

    Uses the neural network when available; otherwise falls back to the
    rule-based keyword scoring. If neural scoring throws at runtime, we still
    fall back so the endpoint never fails.
    """
    if MODEL_READY:
        try:
            return neural_score(request.text)
        except Exception as error:
            print("[backend] neural scoring failed, falling back:", error)

    return rule_based_score(request.text)


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """Record a user's label correction to feedback.csv (local only).

    Creates feedback.csv with a header row the first time it's called, then
    appends one row per correction:
        timestamp,text,predictedLabel,correctedLabel,confidence
    """
    file_exists = os.path.exists(FEEDBACK_PATH)

    # newline="" is the documented way to let the csv module manage line
    # endings so rows aren't double-spaced on Windows.
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(FEEDBACK_HEADER)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            request.text,
            request.predictedLabel,
            request.correctedLabel,
            request.confidence,
        ])

    return {"status": "ok", "recorded": True}
