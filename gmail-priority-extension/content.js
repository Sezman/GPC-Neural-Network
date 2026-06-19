/*
 * Gmail Priority Classifier - content.js
 * ----------------------------------------------------------------------------
 * This script runs inside the Gmail web page. It looks at the email rows that
 * are currently visible in the inbox list, scores each one with a simple
 * rule-based function, and adds a small "priority" badge (High / Medium / Low)
 * to the start of the row.
 *
 * This is an MVP. There is NO machine learning here yet - just keyword rules.
 * See README.md for future improvement ideas.
 * ----------------------------------------------------------------------------
 */

// 1) Let us know the extension actually loaded (open DevTools -> Console to see).
console.log("[Gmail Priority Classifier] content script loaded.");

/*
 * 1b) Panel state (two toggles).
 *
 * STATE MANAGEMENT
 * There are exactly two pieces of state, one per pill toggle:
 *   - badgesEnabled : are the priority badges visible?
 *   - hideLow       : are rows classified "Low" hidden?
 *
 * LOCALSTORAGE
 * Both are persisted in localStorage so they survive a Gmail refresh. We read
 * them once at startup (below) and write them back inside the toggle handlers.
 * Defaults: badges ON, hide-low OFF.
 *
 * HOW SHOW/HIDE WORKS
 * We never hand-edit each row. Instead applyState() reflects the two booleans
 * onto two CSS classes on <body> (gpc-disabled, gpc-hide-low) and styles.css
 * does the rest. This applies instantly to every row at once.
 *
 * IMPORTANT: scoring is NOT tied to badge visibility. Rows are always scored
 * and tagged with data-gpc-label, even when badges are OFF, so "Hide Low" keeps
 * working independently of whether badges are shown.
 */
const STORAGE_ENABLED = "gpc-enabled";
const STORAGE_HIDE_LOW = "gpc-hide-low";

// Read persisted state. Badges default ON (only "false" disables); hide-low
// defaults OFF (only "true" enables).
let badgesEnabled = localStorage.getItem(STORAGE_ENABLED) !== "false";
let hideLow = localStorage.getItem(STORAGE_HIDE_LOW) === "true";

/*
 * Reflect the current state onto <body> as CSS classes:
 *   - gpc-disabled  -> hides all badge elements
 *   - gpc-hide-low  -> hides rows tagged data-gpc-label="Low"
 * Both are independent: hide-low works whether or not badges are visible.
 */
function applyState() {
  document.body.classList.toggle("gpc-disabled", !badgesEnabled);
  document.body.classList.toggle("gpc-hide-low", hideLow);
}

/*
 * 2) Weighted keyword groups.
 *
 * Each phrase below signals a different level of importance. The scoring
 * function adds (or subtracts) points based on which phrases appear in the
 * row text. These lists are the easiest thing to customize - just add or
 * remove phrases. Multi-word phrases are matched as a whole.
 */

// Strong signals of something important / time-sensitive. (+3 each)
// Note: we include spelling variants ("canceled" vs "cancelled") and longer
// real-world phrasings ("subscription has been canceled") because includes()
// only matches the EXACT substring it is given.
const HIGH_KEYWORDS = [
  "urgent",
  "action needed",
  "action required",
  "payment failed",
  "subscription canceled",
  "subscription has been canceled",
  "subscription has been cancelled",
  "account suspended",
  "security alert",
  "password reset",
  "verification code",
  "interview",
  "offer",
  "deadline",
  "final notice",
  "due today",
];

// Moderately relevant, often informational but worth noticing. (+1 each)
const MEDIUM_KEYWORDS = [
  "meeting",
  "reminder",
  "application received",
  "invoice",
  "receipt",
  "billing",
  "trial",
  "free trial",
  "upgrade",
  "upgrade to paid",
  "appointment",
  "recall reminder",
  "confirmation",
];

// Low-value / promotional noise that should pull the score DOWN. (-1 each)
const LOW_KEYWORDS = [
  "newsletter",
  "digest",
  "privacy policy",
  "promotion",
  "sale",
  "unsubscribe",
  "alumni spotlight",
  "performance report",
];

/*
 * 3) Score the email text and return both the number and the label.
 *
 * Weighted scoring:
 *   - Each HIGH phrase found adds 3 points.
 *   - Each MEDIUM phrase found adds 1 point.
 *   - Each LOW phrase found subtracts 1 point.
 *
 * Classification:
 *   - score >= 3 -> High
 *   - score >= 1 -> Medium
 *   - otherwise  -> Low
 *
 * We return the score AND the matched keywords so the caller can log exactly
 * which phrases triggered the result - this makes debugging easy.
 *
 * @param {string} normalizedText - already lowercased + whitespace-normalized
 * @returns {{ score: number, label: string, matched: object }}
 */
function scoreEmail(normalizedText) {
  let score = 0;

  // Track which phrases matched, grouped by level, for debugging.
  const matched = { high: [], medium: [], low: [] };

  // Strong signals push the score up the most. (+3 each)
  for (const keyword of HIGH_KEYWORDS) {
    if (normalizedText.includes(keyword)) {
      score += 3;
      matched.high.push(keyword);
    }
  }

  // Moderate signals give a small boost. (+1 each)
  for (const keyword of MEDIUM_KEYWORDS) {
    if (normalizedText.includes(keyword)) {
      score += 1;
      matched.medium.push(keyword);
    }
  }

  // Promotional / low-value signals pull the score down. (-1 each)
  for (const keyword of LOW_KEYWORDS) {
    if (normalizedText.includes(keyword)) {
      score -= 1;
      matched.low.push(keyword);
    }
  }

  // Map the numeric score to one of three labels.
  let label;
  if (score >= 3) {
    label = "High";
  } else if (score >= 1) {
    label = "Medium";
  } else {
    label = "Low";
  }

  return { score, label, matched };
}

/*
 * 3b) Extract clean, normalized text from a row.
 *
 * Two important details that fix earlier bugs:
 *   - We SKIP our own badge element so its label text ("High"/"Low"/...) never
 *     pollutes the text we score on later passes.
 *   - We collapse all runs of whitespace (newlines, tabs, non-breaking spaces)
 *     into single spaces, then lowercase. Gmail rows contain odd whitespace
 *     between cells, and without this, phrase matching can silently fail.
 *
 * @param {Element} row - one email row element
 * @returns {string} normalized, lowercased visible text
 */
function getRowText(row) {
  let raw = "";

  // Build the text from each child cell, skipping any badge we added.
  for (const child of row.children) {
    if (child.classList && child.classList.contains("gpc-badge")) {
      continue;
    }
    // Prefer innerText (visible text only). But when "Hide Low" has hidden a row
    // with display:none, innerText returns "" - so fall back to textContent.
    // This keeps hidden rows re-scorable, so if their classification changes
    // (e.g. Low -> High once Gmail finishes loading), their visibility updates.
    raw += " " + (child.innerText || child.textContent || "");
  }

  // Normalize: collapse whitespace, trim, lowercase.
  return raw.replace(/\s+/g, " ").trim().toLowerCase();
}

/*
 * 3c) Backend-backed scoring (with caching + offline fallback).
 *
 * Scoring now lives in a local FastAPI backend. The extension POSTs the row
 * text and uses the response. To stay fast and avoid spamming the server:
 *   - scoreCache: maps normalized text -> result, so identical emails are only
 *     ever scored once.
 *   - inFlight: maps normalized text -> in-progress Promise, so if several rows
 *     share the same text we only fire ONE request.
 *
 * If the backend is offline or returns an error, we transparently fall back to
 * the local scoreEmail() rules and log a clear warning.
 */
const BACKEND_URL = "http://127.0.0.1:8000/score";

const scoreCache = new Map(); // text -> { score, label, matched, source }
const inFlight = new Map(); // text -> Promise

/*
 * Ask the backend to score the text. Always resolves (never rejects): on any
 * failure it returns the local rule-based result instead.
 *
 * @param {string} text - normalized email text
 * @returns {Promise<{score, label, matched, source}>}
 */
async function requestScore(text) {
  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });

    if (!response.ok) {
      throw new Error("Backend responded with HTTP " + response.status);
    }

    const data = await response.json();

    // Reshape the backend's matchedHigh/Medium/Low into our internal shape.
    return {
      score: data.score,
      label: data.label,
      matched: {
        high: data.matchedHigh || [],
        medium: data.matchedMedium || [],
        low: data.matchedLow || [],
      },
      source: "backend",
    };
  } catch (error) {
    // FALLBACK: backend offline or errored -> use the built-in JS rules.
    console.warn(
      "[GPC] Backend unavailable, falling back to local scoring.",
      error
    );
    const local = scoreEmail(text);
    return {
      score: local.score,
      label: local.label,
      matched: local.matched,
      source: "local-fallback",
    };
  }
}

/*
 * Get a score for the text, using the cache / in-flight dedup so we never send
 * the same text twice.
 *
 * @param {string} text - normalized email text
 * @returns {Promise<{score, label, matched, source}>}
 */
function scoreText(text) {
  // Already scored this exact text before? Reuse it instantly.
  if (scoreCache.has(text)) {
    return Promise.resolve(scoreCache.get(text));
  }

  // A request for this exact text is already in progress? Reuse that Promise.
  if (inFlight.has(text)) {
    return inFlight.get(text);
  }

  // Otherwise start a new request, remember it, and cache the result.
  const promise = requestScore(text).then((result) => {
    scoreCache.set(text, result);
    inFlight.delete(text);
    return result;
  });

  inFlight.set(text, promise);
  return promise;
}

/*
 * 4) Find the email rows that are currently in the DOM.
 *
 * WARNING: Gmail's DOM is NOT a stable, documented API. Google can change
 * class names and structure at any time, which may break this selector.
 *
 * In Gmail's inbox list, each email row is currently a <tr> with the class
 * "zA". If badges stop appearing after a Gmail update, this is the FIRST place
 * to check: open DevTools, inspect an email row, and update the selector below
 * to match the new row element.
 *
 * @returns {Element[]} an array of row elements
 */
function getEmailRows() {
  return Array.from(document.querySelectorAll("tr.zA"));
}

/*
 * 5) Add (or update) the badge on a single row.
 *
 * IMPORTANT BUG FIX:
 * Gmail often inserts an EMPTY row skeleton first and fills in the subject
 * text a moment later. The previous version locked each row with a
 * "processed" flag on the very first pass, so rows scored while empty stayed
 * stuck on "Low" forever.
 *
 * The fix: we re-score a row whenever its TEXT changes (tracked via
 * row.dataset.gpcText), and we only touch the DOM when the label actually
 * changes. This means:
 *   - A row that loaded empty (Low) gets upgraded to High/Medium once its text
 *     arrives.
 *   - We never re-send the same text to the backend (gpcText guard + cache).
 *   - We never add a second badge (we reuse/replace the existing one).
 *   - The MutationObserver does not loop forever, because once a row's text and
 *     label are settled, re-running makes no request and no DOM change.
 *
 * @param {Element} row - one email row element
 */
function addBadgeToRow(row) {
  // NOTE: we score every row regardless of the Badges toggle. When badges are
  // OFF, the badge element is simply hidden by the gpc-disabled CSS class - but
  // the row is still tagged with its label so "Hide Low" keeps working.

  // Get clean, normalized, lowercased text (excludes our own badge).
  const text = getRowText(row);

  // Skip totally empty rows (skeletons not yet filled in). We'll catch them on
  // a later observer pass once Gmail populates the text.
  if (text === "") {
    return;
  }

  // Have we already scored this exact text for this row? If so, there is
  // nothing new to do - this is what stops us from re-sending the same row to
  // the backend on every MutationObserver tick.
  if (row.dataset.gpcText === text) {
    return;
  }
  // Remember what we're scoring so duplicate ticks are ignored.
  row.dataset.gpcText = text;

  // Score asynchronously (backend, with cache + local fallback), then paint.
  scoreText(text).then((result) => {
    // Gmail may have recycled this row to a different email while we waited.
    // Only apply the result if the row still holds the text we scored.
    if (row.dataset.gpcText !== text) {
      return;
    }
    applyBadge(row, text, result);
  });
}

/*
 * 5b) Paint (or update) the badge on a row from a scoring result.
 *
 * Kept separate from addBadgeToRow so the async flow stays readable. This is
 * where the row gets its label tag (for Hide Low) and its badge element.
 *
 * @param {Element} row    - the email row
 * @param {string}  text   - the normalized text that was scored (for logging)
 * @param {object}  result - { score, label, matched, source }
 */
function applyBadge(row, text, result) {
  const { score, label, matched, source } = result;

  // Tag the row with its label so the "Hide Low" CSS rule can target Low rows.
  row.dataset.gpcLabel = label;

  // Strong, detailed logging so every classification can be verified. "source"
  // shows whether the backend or the local fallback produced this result.
  console.log(
    "[GPC] %c" + label + " (score " + score + ", " + source + ")",
    "font-weight:bold",
    {
      text: text,
      matchedHigh: matched.high,
      matchedMedium: matched.medium,
      matchedLow: matched.low,
      score: score,
      label: label,
      source: source,
    }
  );

  // Is there already a badge on this row?
  const existing = row.querySelector(":scope > .gpc-badge");

  // If a badge exists and already shows the correct label, do nothing.
  if (existing && existing.dataset.label === label) {
    return;
  }

  // If a badge exists but the label changed (e.g. Low -> High once text
  // loaded), update it in place instead of adding a duplicate.
  if (existing) {
    existing.textContent = label;
    existing.dataset.label = label;
    // Remember the model's own prediction + confidence for /feedback. These
    // are the "before" values reported if the user later corrects the label.
    existing.dataset.predicted = label;
    existing.dataset.confidence = String(score);
    existing.className = "gpc-badge gpc-" + label.toLowerCase();
    return;
  }

  // No badge yet: build one and insert it at the START of the row.
  const badge = document.createElement("span");
  badge.textContent = label;
  badge.dataset.label = label; // used for the duplicate/rescore checks above
  // The model's prediction + its confidence, sent verbatim as predictedLabel
  // and confidence when the user corrects this badge.
  badge.dataset.predicted = label;
  badge.dataset.confidence = String(score);
  badge.className = "gpc-badge gpc-" + label.toLowerCase();
  // Clicking the badge opens a small menu to correct the label.
  badge.addEventListener("click", (event) => onBadgeClick(event, row, badge));
  row.insertBefore(badge, row.firstChild);
}

/*
 * 5c) Feedback / correction flow.
 *
 * Clicking a badge opens a tiny menu with High / Medium / Low. Picking one:
 *   1. POSTs the correction to the backend's /feedback endpoint, which logs it
 *      to feedback.csv for later retraining. (Stays entirely local.)
 *   2. Immediately repaints the badge with the corrected label.
 *   3. Re-tags the row (data-gpc-label) so "Hide Low" reacts instantly: a row
 *      corrected to Low while Hide Low is ON disappears; one corrected away
 *      from Low reappears. The CSS rule does the actual show/hide.
 */
const FEEDBACK_URL = "http://127.0.0.1:8000/feedback";
const CORRECTION_LABELS = ["High", "Medium", "Low"];

// Only one correction menu is open at a time; track it so we can close it.
let openMenu = null;

function closeMenu() {
  if (openMenu) {
    openMenu.remove();
    openMenu = null;
    document.removeEventListener("click", closeMenu, true);
  }
}

/*
 * Open the correction menu next to the clicked badge. We stop propagation so
 * the click that opens the menu doesn't immediately trigger the outside-click
 * handler that closes it, and so Gmail doesn't open the email.
 */
function onBadgeClick(event, row, badge) {
  event.preventDefault();
  event.stopPropagation();

  // A second click on the same badge toggles the menu shut.
  const wasOpen = openMenu !== null;
  closeMenu();
  if (wasOpen) {
    return;
  }

  const menu = document.createElement("div");
  menu.className = "gpc-menu";

  for (const choice of CORRECTION_LABELS) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "gpc-menu-item gpc-menu-" + choice.toLowerCase();
    item.textContent = choice;
    // Mark the current label so the user can see what it's set to.
    if (choice === badge.dataset.label) {
      item.classList.add("gpc-menu-current");
    }
    item.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      closeMenu();
      correctLabel(row, badge, choice);
    });
    menu.appendChild(item);
  }

  // Position the menu just below the badge using fixed coordinates.
  const rect = badge.getBoundingClientRect();
  menu.style.top = rect.bottom + 4 + "px";
  menu.style.left = rect.left + "px";

  document.body.appendChild(menu);
  openMenu = menu;

  // Close when clicking anywhere else. Capture phase so it fires before other
  // handlers; the opening click already stopped propagating so it won't self-close.
  document.addEventListener("click", closeMenu, true);
}

/*
 * Apply a user correction: send it to the backend, then update the UI. The
 * predicted label + confidence come from the badge's dataset (the model's
 * original output). The visual update happens regardless of whether the POST
 * succeeds, so the user always sees their correction take effect.
 */
function correctLabel(row, badge, newLabel) {
  const text = row.dataset.gpcText || getRowText(row);
  const predictedLabel = badge.dataset.predicted || badge.dataset.label;
  const confidence = Number(badge.dataset.confidence) || 0;

  // No change? Nothing to record.
  if (newLabel === badge.dataset.label) {
    return;
  }

  // 1) Send the correction to the local backend (fire and continue).
  fetch(FEEDBACK_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: text,
      predictedLabel: predictedLabel,
      correctedLabel: newLabel,
      confidence: confidence,
    }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Backend responded with HTTP " + response.status);
      }
      console.log("[GPC] feedback recorded:", predictedLabel, "->", newLabel);
    })
    .catch((error) => {
      console.warn("[GPC] feedback could not be saved (backend offline?).", error);
    });

  // 2) Update the badge + row immediately. Re-tagging data-gpc-label lets the
  // Hide Low CSS rule hide/unhide this row on the spot.
  badge.textContent = newLabel;
  badge.dataset.label = newLabel;
  badge.className = "gpc-badge gpc-" + newLabel.toLowerCase();
  row.dataset.gpcLabel = newLabel;

  // 3) Keep the cache in sync so a later re-score of identical text doesn't
  // overwrite the user's correction.
  const cached = scoreCache.get(text);
  if (cached) {
    cached.label = newLabel;
  }
}

/*
 * 6) Process every visible row once.
 *
 * Called on initial load and whenever Gmail adds more rows.
 */
function processVisibleRows() {
  const rows = getEmailRows();
  for (const row of rows) {
    addBadgeToRow(row);
  }
}

/*
 * 7) Watch for Gmail loading more emails.
 *
 * Gmail is a single-page app: it adds and removes rows as you scroll, switch
 * labels, or receive new mail - WITHOUT reloading the page. A MutationObserver
 * lets us react to those DOM changes and badge the new rows too.
 *
 * We observe the whole document body for simplicity. For an MVP this is fine;
 * a more advanced version could observe a narrower container for performance.
 */
const observer = new MutationObserver(() => {
  // Re-add the panel if Gmail ever wipes it during a re-render. buildPanel()
  // is guarded, so this never creates a duplicate.
  buildPanel();
  processVisibleRows();
});

observer.observe(document.body, {
  childList: true, // watch for added/removed elements
  subtree: true, // ...anywhere inside the body
});

/*
 * 8a) Create one pill toggle (label + sliding switch).
 *
 * Plain DOM, no frameworks. The visual ON/OFF state is just a "gpc-on" class
 * on the switch element; styles.css slides the knob and recolors the pill.
 *
 * @param {string} labelText - text shown beside the switch
 * @param {boolean} isOn     - initial state
 * @param {(on:boolean)=>void} onToggle - called with the new state on click
 * @returns {HTMLElement} the row element to append to the panel
 */
function createToggle(labelText, isOn, onToggle) {
  const rowEl = document.createElement("div");
  rowEl.className = "gpc-toggle-row";

  // Text label beside the switch.
  const label = document.createElement("span");
  label.className = "gpc-toggle-label";
  label.textContent = labelText;

  // The switch itself is a button (keyboard-accessible) with a knob inside.
  const sw = document.createElement("button");
  sw.type = "button";
  sw.className = "gpc-switch";
  sw.setAttribute("role", "switch");

  const knob = document.createElement("span");
  knob.className = "gpc-knob";
  sw.appendChild(knob);

  // Paint the current state onto the switch.
  const render = (on) => {
    sw.classList.toggle("gpc-on", on);
    sw.setAttribute("aria-checked", on ? "true" : "false");
  };
  render(isOn);

  // TOGGLE EVENT HANDLER: flip the visual state, then notify the caller so it
  // can update state, persist to localStorage, and apply the change.
  sw.addEventListener("click", () => {
    const next = !sw.classList.contains("gpc-on");
    render(next);
    onToggle(next);
  });

  rowEl.appendChild(label);
  rowEl.appendChild(sw);
  return rowEl;
}

/*
 * 8b) Build the floating control panel (two pill toggles).
 *
 * Guarded so it is only ever created once - this prevents duplicate panels or
 * toggles when Gmail dynamically re-renders the page.
 */
function buildPanel() {
  if (document.getElementById("gpc-panel")) {
    return; // already built - avoids duplicates
  }

  const panel = document.createElement("div");
  panel.id = "gpc-panel";

  // Title
  const title = document.createElement("div");
  title.id = "gpc-title";
  title.textContent = "Gmail Priority";
  panel.appendChild(title);

  // Toggle 1: Badges (show/hide badges). Rows are still scored when OFF.
  panel.appendChild(
    createToggle("Badges", badgesEnabled, (on) => {
      badgesEnabled = on;
      // Persist so the choice survives a refresh.
      localStorage.setItem(STORAGE_ENABLED, on ? "true" : "false");
      applyState();
    })
  );

  // Toggle 2: Hide Low (hide/show rows classified Low).
  panel.appendChild(
    createToggle("Hide Low", hideLow, (on) => {
      hideLow = on;
      // Persist so the choice survives a refresh.
      localStorage.setItem(STORAGE_HIDE_LOW, on ? "true" : "false");
      applyState();
    })
  );

  document.body.appendChild(panel);
}

// 9) Initialize: apply saved state, build the panel, and score visible rows.
applyState();
buildPanel();
processVisibleRows();
