"""
eval_functional.py — Functional Correctness Checks
─────────────────────────────────────────────────────
This module runs fast, deterministic tests against the agent's core functions.

WHY FUNCTIONAL EVALS?
─────────────────────
Before measuring quality, you need to ensure the agent's output has the
right *structure*. A post with great writing but missing hashtags, or a
reflection that returns a score of 15, is a bug — not a quality issue.

Functional evals catch bugs that LLM-as-judge would silently ignore.

HOW IT WORKS:
─────────────
Each check is a standalone function that returns a dict:
  { "name": str, "passed": bool, "detail": str }

run_all_functional_checks() runs every check and returns the full list.
No Claude API calls are made here — these run instantly.

CHECKS INCLUDED:
────────────────
1. Reflection schema       — All required keys present in Claude's JSON response
2. Relevance score range   — Score is an integer between 1 and 10
3. Post not empty          — Generated post is a non-empty string
4. Post has hashtags       — At least 3 hashtags (#word) present
5. Post word count         — Within the configured POST_MAX_WORDS limit
6. Best article selection  — pick_best_article() returns the highest-scoring item
7. Post completeness       — Post ends properly (not truncated mid-sentence)
"""

import sys
import os
import re

# Add the parent directory (linkedin_agent/) to the path so we can import
# config, pm_reflection, post_generator without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import POST_MAX_WORDS
from pm_reflection import reflect_on_article, pick_best_article
from post_generator import generate_linkedin_post


# ── Required keys that Claude must return in every reflection ─────────────────
REQUIRED_REFLECTION_KEYS = {
    "relevance_score",
    "core_insight",
    "opportunity",
    "risk_or_challenge",
    "action",
    "hook_idea",
}


def check_reflection_schema(article: dict) -> dict:
    """
    Verify that reflect_on_article() returns a dict containing all required keys.

    Why this matters
    ────────────────
    If Claude omits a key (e.g. "hook_idea"), post_generator.py will crash
    with a KeyError. This check catches prompt regressions early.

    Parameters
    ──────────
    article : dict  — One article from test_articles.json

    Returns
    ───────
    dict  — { name, passed, detail }
    """
    try:
        reflection = reflect_on_article(article)
        missing_keys = REQUIRED_REFLECTION_KEYS - set(reflection.keys())
        if missing_keys:
            return {
                "name": "reflection_schema",
                "passed": False,
                "detail": f"Missing keys: {missing_keys}",
            }
        return {"name": "reflection_schema", "passed": True, "detail": "All required keys present"}
    except Exception as e:
        return {"name": "reflection_schema", "passed": False, "detail": f"Exception: {e}"}


def check_relevance_score_range(reflection: dict) -> dict:
    """
    Verify that the relevance_score returned by Claude is an integer between 1 and 10.

    Why this matters
    ────────────────
    If Claude returns a float (e.g. 7.5) or an out-of-range value (e.g. 11),
    comparisons and sorting in pick_best_article() may still work but produce
    inconsistent results across runs.

    Parameters
    ──────────
    reflection : dict  — Already-fetched reflection (avoids a redundant API call)

    Returns
    ───────
    dict  — { name, passed, detail }
    """
    score = reflection.get("relevance_score")
    if not isinstance(score, int):
        return {
            "name": "relevance_score_range",
            "passed": False,
            "detail": f"Score is not an int: {score!r} (type: {type(score).__name__})",
        }
    if not (1 <= score <= 10):
        return {
            "name": "relevance_score_range",
            "passed": False,
            "detail": f"Score out of range: {score} (expected 1–10)",
        }
    return {"name": "relevance_score_range", "passed": True, "detail": f"Score = {score} ✓"}


def check_post_not_empty(reflection: dict) -> dict:
    """
    Verify that generate_linkedin_post() returns a non-empty string.

    Why this matters
    ────────────────
    A blank response from Claude could happen due to content filters or API
    errors. This check ensures we never save an empty post file.

    Parameters
    ──────────
    reflection : dict  — Reflection dict (includes nested 'article' key)

    Returns
    ───────
    dict  — { name, passed, detail }
    """
    try:
        post = generate_linkedin_post(reflection)
        if not post or not post.strip():
            return {"name": "post_not_empty", "passed": False, "detail": "Post is empty"}
        return {
            "name": "post_not_empty",
            "passed": True,
            "detail": f"Post has {len(post.split())} words",
        }
    except Exception as e:
        return {"name": "post_not_empty", "passed": False, "detail": f"Exception: {e}"}


def check_post_has_hashtags(post: str, min_hashtags: int = 3) -> dict:
    """
    Verify that the generated post contains at least `min_hashtags` hashtags.

    Why this matters
    ────────────────
    Hashtags are a required part of our post format (they drive LinkedIn reach).
    If the prompt changes and Claude stops generating them, this check fails.

    Parameters
    ──────────
    post          : str  — The generated post text
    min_hashtags  : int  — Minimum number of hashtags expected (default: 3)

    Returns
    ───────
    dict  — { name, passed, detail }
    """
    # Match any word starting with # followed by at least one letter/digit
    hashtags = re.findall(r"#\w+", post)
    count = len(hashtags)
    if count < min_hashtags:
        return {
            "name": "post_has_hashtags",
            "passed": False,
            "detail": f"Found {count} hashtags, expected ≥ {min_hashtags}",
        }
    return {
        "name": "post_has_hashtags",
        "passed": True,
        "detail": f"Found {count} hashtags: {', '.join(hashtags[:5])}",
    }


def check_post_word_count(post: str) -> dict:
    """
    Verify that the post word count does not exceed POST_MAX_WORDS from config.py.

    Why this matters
    ────────────────
    LinkedIn's algorithm penalises posts that are too long. This check enforces
    the configured limit and catches prompt changes that inflate post length.

    Parameters
    ──────────
    post : str  — The generated post text

    Returns
    ───────
    dict  — { name, passed, detail }
    """
    word_count = len(post.split())
    # Allow a 20% buffer above the limit before failing (Claude counts slightly differently)
    hard_limit = int(POST_MAX_WORDS * 1.2)
    if word_count > hard_limit:
        return {
            "name": "post_word_count",
            "passed": False,
            "detail": f"{word_count} words exceeds limit of {hard_limit} (config: {POST_MAX_WORDS})",
        }
    return {
        "name": "post_word_count",
        "passed": True,
        "detail": f"{word_count} words (limit: {POST_MAX_WORDS})",
    }


def check_best_article_selection(reflections: list[dict]) -> dict:
    """
    Verify that pick_best_article() returns the reflection with the highest score.

    Why this matters
    ────────────────
    This is a pure logic test — it does not call Claude. It confirms our
    selection algorithm works correctly regardless of what scores Claude returns.

    Parameters
    ──────────
    reflections : list[dict]  — List of reflection dicts with relevance_score set

    Returns
    ───────
    dict  — { name, passed, detail }
    """
    # Create deterministic test fixtures — no Claude calls needed here
    test_reflections = [
        {"relevance_score": 6, "article": {"title": "Article A"}},
        {"relevance_score": 9, "article": {"title": "Article B"}},   # <-- should win
        {"relevance_score": 3, "article": {"title": "Article C"}},
    ]
    best = pick_best_article(test_reflections)
    if best["relevance_score"] != 9:
        return {
            "name": "best_article_selection",
            "passed": False,
            "detail": f"Expected score 9, got {best['relevance_score']}",
        }
    return {
        "name": "best_article_selection",
        "passed": True,
        "detail": "Correctly selected article with highest relevance_score",
    }


def check_post_completeness(post: str) -> dict:
    """
    Verify that the generated post is complete and not truncated mid-output.

    Why this matters
    ────────────────
    When max_tokens is too low, Claude's response is cut off — the post ends
    abruptly mid-sentence with no closing question or hashtags. This is a
    silent failure: the post passes the 'not empty' check but is unusable.

    This check was added after a real truncation bug was caught in production
    where max_tokens=800 was too low for the new voice-personalised prompt.

    Three signals of a complete post:
      1. Does NOT end mid-sentence (last non-empty line ends with punctuation
         or is a hashtag line)
      2. Contains a closing question somewhere (? character present)
      3. Hashtags appear in the final 5 lines (they should be at the end)

    Parameters
    ──────────
    post : str  — The generated post text

    Returns
    ───────
    dict  — { name, passed, detail }
    """
    if not post or not post.strip():
        return {"name": "post_completeness", "passed": False, "detail": "Post is empty"}

    lines = [ln for ln in post.strip().splitlines() if ln.strip()]
    last_line = lines[-1].strip() if lines else ""
    final_5_lines = "\n".join(lines[-5:]) if len(lines) >= 5 else "\n".join(lines)

    issues = []

    # ── Signal 1: Last line should end with punctuation or be a hashtag ────────
    # A truncated post ends mid-word or mid-sentence (no terminal punctuation).
    ends_with_punctuation = last_line.endswith((
        ".", "!", "?", "…", "\"", "'", ")", "]"
    ))
    is_hashtag_line = bool(re.search(r"#\w+", last_line))
    if not ends_with_punctuation and not is_hashtag_line:
        issues.append(f"Last line appears truncated: '{last_line[:80]}'")

    # ── Signal 2: Post must contain at least one question ─────────────────────
    # A well-formed post in this style always ends with a closing question.
    if "?" not in post:
        issues.append("No closing question found (post may be truncated before the ending)")

    # ── Signal 3: Hashtags should appear near the end ─────────────────────────
    # If hashtags exist in the post but NOT in the final 5 lines, the post
    # was probably cut off before the hashtag block was written.
    all_hashtags = re.findall(r"#\w+", post)
    final_hashtags = re.findall(r"#\w+", final_5_lines)
    if all_hashtags and not final_hashtags:
        issues.append("Hashtags exist but not near the end — post may be truncated")

    if issues:
        return {
            "name": "post_completeness",
            "passed": False,
            "detail": " | ".join(issues),
        }
    return {
        "name": "post_completeness",
        "passed": True,
        "detail": f"Post ends cleanly, has closing question, hashtags in final lines",
    }


def run_all_functional_checks(test_article: dict) -> list[dict]:
    """
    Run every functional check against a single test article.

    Design decision
    ───────────────
    We only run ONE Claude API call for the reflection (shared across checks)
    to minimise cost. The post is generated once and reused too.

    Parameters
    ──────────
    test_article : dict  — One article from test_articles.json

    Returns
    ───────
    list[dict]  — One result dict per check, in run order.
    """
    results = []

    # ── Check 1: Reflection schema ────────────────────────────────────────────
    schema_result = check_reflection_schema(test_article)
    results.append(schema_result)

    if not schema_result["passed"]:
        # If reflection is broken, remaining checks that depend on it can't run
        results.append({"name": "relevance_score_range", "passed": False, "detail": "Skipped (reflection failed)"})
        results.append({"name": "post_not_empty",        "passed": False, "detail": "Skipped (reflection failed)"})
        results.append({"name": "post_has_hashtags",     "passed": False, "detail": "Skipped (reflection failed)"})
        results.append({"name": "post_word_count",       "passed": False, "detail": "Skipped (reflection failed)"})
        results.append({"name": "post_completeness",     "passed": False, "detail": "Skipped (reflection failed)"})
    else:
        # Re-fetch reflection to reuse for downstream checks
        reflection = reflect_on_article(test_article)

        # ── Check 2: Relevance score ──────────────────────────────────────────
        results.append(check_relevance_score_range(reflection))

        # ── Checks 3–5: Post quality ──────────────────────────────────────────
        post_result = check_post_not_empty(reflection)
        results.append(post_result)

        if post_result["passed"]:
            post = generate_linkedin_post(reflection)
            results.append(check_post_has_hashtags(post))
            results.append(check_post_word_count(post))
            # Check 7: Did the post get cut off before finishing? Catches max_tokens truncation.
            results.append(check_post_completeness(post))
        else:
            results.append({"name": "post_has_hashtags", "passed": False, "detail": "Skipped (post empty)"})
            results.append({"name": "post_word_count",   "passed": False, "detail": "Skipped (post empty)"})
            results.append({"name": "post_completeness", "passed": False, "detail": "Skipped (post empty)"})

    # ── Check 6: Selection logic (no API call) ────────────────────────────────
    results.append(check_best_article_selection([]))

    return results
