# Gmail Priority Classifier

A beginner-friendly **Chrome Extension (Manifest V3)** that runs on Gmail and
adds a small **priority badge** — **High**, **Medium**, or **Low** — to each
visible email row in your inbox.

Scoring runs in a small **local FastAPI backend** (`backend/`). The extension
sends each email's text to the backend and uses the response.

The backend now uses a **neural network classifier** (sentence-transformers
embeddings + a small PyTorch model) as its primary scoring engine. It keeps the
original **rule-based keyword scoring** as a fallback, used automatically when
the model hasn't been trained yet, the model files are missing, or the ML
libraries aren't installed.

There are **two layers of fallback**, so you always get badges:

1. Backend **neural network** → if unavailable, backend **rule-based** scoring.
2. If the **backend itself is offline**, the extension falls back to the same
   rules running in JavaScript.

Everything runs **locally** — no email text is ever sent to an external API.

---

## What it does

- Runs only on `https://mail.google.com/*`.
- Detects the email rows currently visible in your Gmail list.
- Reads each row's visible text (sender, subject, snippet).
- Scores the email using a small keyword rulebook.
- Adds a colored badge at the start of each row:
  - **High** — strong/red (urgent, deadline, action required, security, password)
  - **Medium** — orange (interview, meeting, invoice, payment, offer, application)
  - **Low** — subtle/gray (nothing notable matched)
- Uses a `MutationObserver` so badges keep getting added as Gmail dynamically
  loads more emails (scrolling, switching labels, new mail).
- Avoids adding duplicate badges to the same row.
- Shows a small **floating control panel** (bottom-left) with two pill toggles.

### How scoring works

**Primary: neural network.** The backend turns the email text into an embedding
and runs a small trained classifier that outputs High / Medium / Low plus a
confidence score (e.g. `0.91`). Because it learns from examples, it can get
cases that confuse pure keywords right — e.g. *"Special offer: save big this
weekend"* is **Low** (promotional) even though "offer" is a High keyword.

**Fallback: rule-based keywords.** When the model isn't available, scoring falls
back to weighted keywords. Each **High** phrase adds **3 points**, each
**Medium** adds **1**, each **Low** subtracts **1**, and the total maps to:

| Score      | Label  |
| ---------- | ------ |
| 3 or more  | High   |
| 1 or 2     | Medium |
| 0 or less  | Low    |

The keyword lists live in `backend/main.py` (and are mirrored in `content.js`
for the offline-extension fallback) — they are easy to customize.

### Control panel (two pill toggles)

A small floating panel titled **Gmail Priority** appears in the corner of Gmail
with two modern pill-style switches:

- **Badges** — ON shows the priority badges and keeps scoring rows; OFF hides
  all badges (no badge UI appears). Rows are still scored internally even when
  this is OFF, so **Hide Low** keeps working.
- **Hide Low** — ON hides every row classified **Low**; OFF shows all rows
  again. If an email's classification changes once Gmail finishes loading its
  text (e.g. Low → High), its visibility updates automatically.

Both toggle states are saved in `localStorage`, so they persist after you
refresh Gmail.

### Correcting a label (feedback)

Every badge is clickable. Click one and a small menu opens offering **High**,
**Medium**, and **Low** (the current label is check-marked). Pick the correct
one and:

- The badge updates **immediately**.
- The correction is `POST`ed to the backend's **`/feedback`** endpoint, which
  appends it to `backend/feedback.csv` for later retraining.
- **Hide Low** reacts on the spot: correct a row to **Low** while Hide Low is ON
  and it disappears; correct a hidden Low row to **High/Medium** and it returns.

The correction includes the model's original prediction and confidence, so
`feedback.csv` captures exactly where the model was wrong. If the backend is
offline the badge still updates — only the CSV write is skipped (with a console
warning). No correction data ever leaves your machine.

To turn those corrections into a better model, run
`py retrain_with_feedback.py` in `backend/` and restart the server. See
**Retraining from feedback** in the [top-level README](../README.md).

---

## Project structure

```
EmailClassifier/
  gmail-priority-extension/
    manifest.json   # Manifest V3 config (host permission for the backend + content script)
    content.js      # Detect rows, score them, add badges, handle corrections
    styles.css      # Badge styling + correction menu + pill toggle panel
    README.md       # This file

  backend/
    main.py                  # FastAPI app: /score, /feedback, /health, /model-status
    requirements.txt
    train_nn.py              # Trains the neural network from the starter CSV
    retrain_with_feedback.py # Retrains on the starter CSV + feedback.csv
    email_training_data.csv  # Starter labeled examples (text,label)
    feedback.csv             # User label corrections (created on first feedback)
    model/
      priority_model.pt      # Saved PyTorch model (created by the trainers)
      label_map.json         # Index -> label mapping (created by the trainers)
    reports/                 # Evaluation reports written by retrain_with_feedback.py
      evaluation_report.txt  # Human-readable metrics summary
      evaluation_report.json # Same metrics, machine-readable
      confusion_matrix.png   # Confusion matrix image (only if matplotlib installed)
```

---

## How to run the backend

The backend holds the scoring logic. Run it first so the extension can call it.

### 1. Install dependencies

```
cd backend
pip install -r requirements.txt
```

> On Windows, if `pip` / `python` point to the Microsoft Store stub, use the
> `py` launcher instead: `py -m pip install -r requirements.txt`.
>
> The first install is large (PyTorch + sentence-transformers). The first run
> also downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB) once.

### 2. Train the neural network

```
py train_nn.py
```

This reads `email_training_data.csv`, builds embeddings, trains the classifier,
prints the **training accuracy**, and saves `model/priority_model.pt` and
`model/label_map.json`. Re-run it any time you add more labeled examples.

> You can skip this step — the backend still works using rule-based scoring
> until a model is trained.

### 3. Start the server

```
py -m uvicorn main:app --reload
```

This starts the server at **http://127.0.0.1:8000**. Quick checks:

- Health: http://127.0.0.1:8000/health → `{"status":"ok"}`
- **Model status: http://127.0.0.1:8000/model-status** → tells you whether the
  backend is using the `neural-network` or the `rule-based` fallback.
- Interactive docs: http://127.0.0.1:8000/docs to try `POST /score`.

The `--reload` flag auto-restarts the server when you edit `main.py`.

### The model improves as you add data

The project ships with a **small starter dataset** (~66 examples), so the model
is only as good as those samples. To make it smarter, add more **real, labeled**
rows to `email_training_data.csv` (`text,label` with labels High/Medium/Low) and
re-run `py train_nn.py`. More varied, realistic examples → better predictions.

### Endpoints

`POST /score` accepts:

```json
{ "text": "email row text here" }
```

and returns (neural-network mode):

```json
{
  "label": "High",
  "score": 0.91,
  "matchedHigh": [],
  "matchedMedium": [],
  "matchedLow": [],
  "source": "neural-network"
}
```

In rule-based mode, `score` is an integer point total, the `matched*` arrays
list the keywords that fired, and `source` is `"rule-based"`.

`POST /feedback` records a user's label correction. It accepts:

```json
{
  "text": "email row text",
  "predictedLabel": "Low",
  "correctedLabel": "High",
  "confidence": 0.82
}
```

and appends a row to `backend/feedback.csv` (created with a header the first
time), returning `{ "status": "ok", "recorded": true }`. The CSV columns are
`timestamp,text,predictedLabel,correctedLabel,confidence`.

`GET /health` returns `{ "status": "ok" }`.

`GET /model-status` returns e.g.
`{ "mode": "neural-network", "modelLoaded": true, "embeddingModel": "all-MiniLM-L6-v2", "torchAvailable": true }`.

---

## How to load the extension in Chrome

1. Open Chrome and go to `chrome://extensions`.
2. Turn on **Developer mode** (toggle in the top-right corner).
3. Click **Load unpacked**.
4. Select the `gmail-priority-extension/` folder.
5. Open or refresh **https://mail.google.com**.
6. Open DevTools (F12) → **Console** and you should see:
   `[Gmail Priority Classifier] content script loaded.`
7. Badges should appear at the start of your visible email rows.

> Tip: after editing any file, return to `chrome://extensions` and click the
> **reload** (↻) icon on the extension card, then refresh Gmail.

---

## How scoring + the backend fit together

1. `content.js` extracts each row's normalized text.
2. It `POST`s that text to `http://127.0.0.1:8000/score`.
3. The backend returns the label, score, and matched keywords.
4. The extension paints the badge and applies the **Hide Low** filter.

To avoid spamming the backend, the extension **caches results by normalized
text** and never sends the same text twice (and rows already scored are
skipped). So a busy inbox only generates a handful of requests.

### What happens if the backend is offline

The extension is designed to keep working without the backend:

- Each `POST /score` is wrapped in a `try/catch`.
- If the request fails (server not running, error, etc.), the extension
  **falls back to the identical rule-based scoring in JavaScript**
  (`scoreEmail()` in `content.js`).
- It logs a clear warning in the DevTools Console:
  `[GPC] Backend unavailable, falling back to local scoring.`
- Each badge's log line shows its `source`: `backend` or `local-fallback`, so
  you can tell at a glance which path produced a result.

In other words: backend running → scores come from the neural network (or the
backend's rule-based fallback); backend down → scores come from the bundled JS
rules. Either way you get badges.

> Note: the **rule-based keyword lists** are duplicated in `backend/main.py` and
> `content.js` so the JavaScript offline fallback matches the backend's
> rule-based fallback. If you change those keywords, update **both** files. (The
> neural network is trained separately from `email_training_data.csv`.)

---

## ⚠️ Important warning: Gmail's DOM is unstable

Gmail does **not** offer a public, documented HTML structure for the inbox.
Google can change class names and layout at any time, and that **will**
eventually break selectors used by this extension.

If badges stop appearing after a Gmail update, here is where to adjust:

- **Email row selector** — in `content.js`, the function `getEmailRows()` uses
  `document.querySelectorAll("tr.zA")`. The class `zA` is Gmail's current row
  class. If it changes:
  1. Open Gmail, press F12 to open DevTools.
  2. Use the element inspector to click on a single email row.
  3. Find the element that wraps the whole row (sender + subject + snippet).
  4. Update the selector in `getEmailRows()` to match the new element/class.

- **Badge position** — `addBadgeToRow()` inserts the badge at the start of the
  row via `row.insertBefore(badge, row.firstChild)`. If the badge lands in an
  awkward spot after a Gmail change, adjust where it's inserted.

- **Text extraction** — we use `row.innerText`. If Gmail restructures rows, you
  may want to target a more specific element (e.g. the subject) instead.

Because the `MutationObserver` watches the whole `document.body`, the extension
is fairly resilient to Gmail re-rendering — but the **row selector** is the most
likely thing to need maintenance.

---

## What it is NOT (yet)

- No access to your real email data via an API — it only reads text already
  rendered on the page (the DOM).
- It does not change Gmail's real labels or move/sort your mail.
- The model is trained on a tiny starter dataset, so predictions improve only as
  you add more labeled examples.

---

## Already done

- ✅ **FastAPI backend** — scoring moved into a local Python service.
- ✅ **ML classifier** — sentence-transformer embeddings + a PyTorch neural
  network, with rule-based keyword scoring kept as a fallback.
- ✅ **Local & private** — everything runs on `127.0.0.1`; no email text leaves
  your machine.
- ✅ **Feedback / corrections** — click a badge to correct its label; the fix is
  applied instantly and logged to `backend/feedback.csv` for retraining.
- ✅ **Retrain from feedback** — `retrain_with_feedback.py` folds `feedback.csv`
  corrections into the starter data and retrains, closing the learning loop.

## Future improvement ideas

- **Bigger / real dataset** — train on many more labeled emails for accuracy.
- **Gmail API / OAuth** — read structured email data securely instead of
  scraping the DOM, which removes the fragile-selector problem.
- **Real Gmail labels** — apply actual Gmail labels (e.g. "High Priority")
  instead of visual-only badges.
- **In-browser inference** — run the model on-device (e.g. ONNX Runtime Web) so
  the backend isn't needed at all.
