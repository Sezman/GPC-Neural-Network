"""
Gmail Priority Classifier - neural network trainer
------------------------------------------------------------------------------
This script trains a small neural network that classifies an email's text into
High / Medium / Low priority.

How it works (beginner-friendly overview):
  1. Read labeled examples from email_training_data.csv (text, label).
  2. Turn each email's text into a numeric vector ("embedding") using a
     pre-trained sentence-transformers model. Similar sentences get similar
     vectors, which is why this works far better than raw keyword matching.
  3. Train a tiny feed-forward neural network (PyTorch) to map an embedding to
     one of the 3 priority classes.
  4. Save the trained model and the label mapping into the model/ folder so the
     backend (main.py) can load and use them.

Run it with:
    cd backend
    py train_nn.py
------------------------------------------------------------------------------
"""

import os
import json

import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Configuration (kept at the top so it's easy to tweak)
# ---------------------------------------------------------------------------
DATA_PATH = "email_training_data.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "priority_model.pt")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# A small, fast, widely-used embedding model (384-dimensional output).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

HIDDEN_SIZE = 128   # neurons in the hidden layer
DROPOUT = 0.3       # randomly zero 30% of neurons during training (reduces overfitting)
EPOCHS = 300        # how many times we loop over the data
LEARNING_RATE = 1e-3


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
# IMPORTANT: this class must match the one in main.py so the saved weights load
# correctly. Architecture: Linear -> ReLU -> Dropout -> Linear -> logits.
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


def main():
    # 1) Load the labeled training data.
    print("Loading training data from", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    print(f"Loaded {len(texts)} examples.")

    # 2) Build a label <-> index mapping. Sorted so it's deterministic.
    #    e.g. ["High", "Low", "Medium"] -> High=0, Low=1, Medium=2
    classes = sorted(set(labels))
    label_to_idx = {label: i for i, label in enumerate(classes)}
    num_classes = len(classes)
    print("Classes:", label_to_idx)

    # Target tensor (the correct class index for each example).
    y = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)

    # 3) Convert text into embeddings using sentence-transformers.
    print("Loading embedding model:", EMBEDDING_MODEL_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Encoding text into embeddings...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    X = torch.tensor(embeddings, dtype=torch.float32)

    input_size = X.shape[1]  # embedding dimension (384 for all-MiniLM-L6-v2)
    print("Embedding size:", input_size)

    # 4) Create the model, loss function, and optimizer.
    model = PriorityNet(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()  # standard loss for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5) Training loop. The dataset is tiny, so we train on it all at once.
    print("Training...")
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()       # reset gradients
        logits = model(X)           # forward pass
        loss = criterion(logits, y) # how wrong are we?
        loss.backward()             # backpropagation
        optimizer.step()            # update weights

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS} - loss {loss.item():.4f}")

    # 6) Report training accuracy.
    model.eval()
    with torch.no_grad():
        predictions = model(X).argmax(dim=1)
    accuracy = accuracy_score(y.numpy(), predictions.numpy())
    print(f"Training accuracy: {accuracy * 100:.1f}%")

    # 7) Save the model + everything main.py needs to rebuild and use it.
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

    # Save index -> label, e.g. {"0": "High", "1": "Low", "2": "Medium"}
    idx_to_label = {str(i): label for label, i in label_to_idx.items()}
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, indent=2)

    print("Saved model to", MODEL_PATH)
    print("Saved label map to", LABEL_MAP_PATH)
    print("Done! Start the backend with: py -m uvicorn main:app --reload")


if __name__ == "__main__":
    main()
