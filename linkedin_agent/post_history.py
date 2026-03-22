"""
post_history.py — Article Deduplication Memory
────────────────────────────────────────────────
Prevents the agent from posting about the same news story twice.

HOW IT WORKS:
─────────────
After each successful run, the agent saves the article it posted about to
`post_history.json` (stored in the output/ directory). On the next run,
the agent calls filter_seen_articles() to remove any articles that are too
similar to previously seen ones BEFORE spending Claude API credits on them.

TWO-LEVEL DEDUPLICATION:
──────────────────────────
1. EXACT match  — Same URL means the identical article. Always filtered out.

2. NEAR-DUPLICATE — Title word overlap (Jaccard similarity) above the
   configured threshold. Catches the same story from different news sources.
   Example: "OpenAI releases GPT-5" vs "OpenAI's GPT-5 is here" would score
   high similarity and be correctly filtered even with different URLs.

   Formula: Jaccard(A, B) = |A ∩ B| / |A ∪ B|
   Where A and B are the sets of meaningful (non-stopword) words in each title.

STORAGE FORMAT:
───────────────
post_history.json is a JSON array of article records:
[
  {
    "url": "https://...",
    "title": "...",
    "source": "...",
    "used_at": "2026-03-22T14:46:00"   ← ISO timestamp of when used
  },
  ...
]

CONFIGURATION:
──────────────
SIMILARITY_THRESHOLD : float (default 0.5)
    Jaccard similarity score above which two articles are considered
    the same story. 0.5 means 50% word overlap.
    → Raise to 0.7 for stricter matching (fewer false positives)
    → Lower to 0.3 for looser matching (catches more near-duplicates)

HISTORY_MAX_ENTRIES : int (default 100)
    Maximum number of articles stored in history. Older entries are
    rotated out once this limit is reached, so history doesn't grow forever.
"""

import json
import re
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
# Jaccard similarity threshold for near-duplicate detection.
# 0.5 = articles sharing 50%+ of meaningful title words are considered the same.
SIMILARITY_THRESHOLD = 0.5

# Max history entries to keep on disk.
HISTORY_MAX_ENTRIES = 100

# ── Storage path ────────────────────────────────────────────────────────────────
# Stored alongside the generated posts for easy inspection.
# Path is resolved relative to this file, not the CWD, for reliability.
_BASE_DIR = Path(__file__).parent
HISTORY_FILE = _BASE_DIR / "output" / "post_history.json"

# ── English stop words to ignore during similarity comparison ──────────────────
# These common words add noise to similarity scores without adding signal.
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "its", "it", "this", "that",
    "these", "those", "from", "by", "as", "into", "about", "over", "new",
    "ai", "says", "says", "report", "reports", "new", "how", "why", "what",
}


def _title_to_word_set(title: str) -> set[str]:
    """
    Convert an article title into a set of meaningful lowercase words.

    Removes stop words, punctuation, and single-character tokens so the
    Jaccard similarity focuses on the substantive content of the title.

    Parameters
    ──────────
    title : str  — Raw article title string

    Returns
    ───────
    set[str]  — Set of meaningful words for similarity comparison.
    """
    # Lowercase and remove all non-alphanumeric characters (except spaces)
    cleaned = re.sub(r"[^a-z0-9\s]", "", title.lower())
    words = cleaned.split()
    # Filter out stop words and single-character tokens
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Compute the Jaccard similarity coefficient between two word sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns 0.0 if both sets are empty (avoids division by zero).

    Parameters
    ──────────
    set_a : set  — Word set for the first title
    set_b : set  — Word set for the second title

    Returns
    ───────
    float  — Similarity score between 0.0 (no overlap) and 1.0 (identical).
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def load_history() -> list[dict]:
    """
    Load the article history from disk.

    Returns an empty list if the file doesn't exist yet (first run).
    Gracefully handles corrupted JSON by returning an empty list.

    Returns
    ───────
    list[dict]  — List of previously used article records.
    """
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # If history file is corrupted, start fresh rather than crashing
        return []


def save_to_history(article: dict) -> None:
    """
    Append a newly used article to the history file on disk.

    Adds the current timestamp as `used_at` so you can see when each
    article was posted about. Rotates out oldest entries if HISTORY_MAX_ENTRIES
    is exceeded, keeping the file from growing unboundedly.

    Parameters
    ──────────
    article : dict  — The article dict selected by the agent for posting.
                      Must contain at minimum: 'url', 'title', 'source'.
    """
    history = load_history()

    # Build a compact record — we don't need the full description in history
    record = {
        "url":     article.get("url", ""),
        "title":   article.get("title", ""),
        "source":  article.get("source", ""),
        "used_at": datetime.now().isoformat(timespec="seconds"),
    }
    history.append(record)

    # Rotate: keep only the most recent HISTORY_MAX_ENTRIES records
    if len(history) > HISTORY_MAX_ENTRIES:
        history = history[-HISTORY_MAX_ENTRIES:]

    # Ensure the output/ directory exists before writing
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def is_duplicate(article: dict, history: list[dict]) -> tuple[bool, str]:
    """
    Check whether an article is too similar to any previously used article.

    Applies two checks in order:
      1. Exact URL match — definitively the same article.
      2. Jaccard title similarity — same story from a different source.

    Parameters
    ──────────
    article : dict       — Candidate article from the news fetcher.
    history : list[dict] — Previously used articles from load_history().

    Returns
    ───────
    tuple[bool, str]
        (True, reason_string)  if duplicate detected
        (False, "")            if not a duplicate
    """
    candidate_url = article.get("url", "")
    candidate_words = _title_to_word_set(article.get("title", ""))

    for seen in history:
        # ── Check 1: Exact URL ────────────────────────────────────────────────
        if candidate_url and seen.get("url") == candidate_url:
            return True, f"exact URL match (seen on {seen.get('used_at', '?')})"

        # ── Check 2: Title similarity ─────────────────────────────────────────
        seen_words = _title_to_word_set(seen.get("title", ""))
        similarity = _jaccard_similarity(candidate_words, seen_words)
        if similarity >= SIMILARITY_THRESHOLD:
            return True, (
                f"similar to '{seen['title'][:50]}…' "
                f"(similarity={similarity:.0%}, seen on {seen.get('used_at', '?')})"
            )

    return False, ""


def filter_seen_articles(articles: list[dict], history: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split a list of articles into new (unseen) and duplicate (already used) sets.

    The agent calls this after fetching news and before running PM reflections,
    so Claude API credits are never spent on articles we'd discard anyway.

    Parameters
    ──────────
    articles : list[dict]  — All articles returned by the news fetcher
    history  : list[dict]  — Previously used articles from load_history()

    Returns
    ───────
    tuple[list[dict], list[dict]]
        (fresh_articles, skipped_articles)
        fresh_articles   — Articles the agent has NOT posted about before
        skipped_articles — Articles filtered out as duplicates (with reason attached)
    """
    fresh = []
    skipped = []

    for article in articles:
        duplicate, reason = is_duplicate(article, history)
        if duplicate:
            # Attach the reason so agent.py can display it in the terminal
            article["_skip_reason"] = reason
            skipped.append(article)
        else:
            fresh.append(article)

    return fresh, skipped
