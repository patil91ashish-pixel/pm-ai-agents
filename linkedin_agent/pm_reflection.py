"""
pm_reflection.py — AI Article → PM Insight (via Claude)
──────────────────────────────────────────────────────────
This module is the "thinking" layer of the agent.

For each AI news article, it asks Claude to wear the hat of a senior Product
Manager and analyse the story through that lens. Claude responds with a
structured JSON object containing a relevance score and several insight fields.

The best-scoring article is then passed to post_generator.py to become the
LinkedIn post.

METRICS INTEGRATION:
─────────────────────
reflect_on_article() accepts an optional MetricsTracker argument.
When provided, it records tokens consumed, cost, and latency for each
Claude call so the caller can display a full run summary.
"""

import json
import time
from typing import Optional

import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, POST_TARGET_AUDIENCE
from metrics import MetricsTracker


def reflect_on_article(article: dict, tracker: Optional[MetricsTracker] = None) -> dict:
    """
    Ask Claude to analyse one AI news article and score its relevance for PMs.

    Prompt strategy
    ───────────────
    We give Claude a specific persona ("senior PM and AI strategy advisor") and
    ask it to return a strict JSON object. Requesting JSON directly (instead of
    free-form prose) makes parsing reliable and keeps downstream code simple.

    Claude is instructed to respond ONLY with JSON — no preamble, no explanation.
    However, some model versions wrap the JSON in a markdown code fence (```json …```)
    so we defensively strip that before parsing.

    Parameters
    ──────────
    article : dict
        A single article dict from fetch_ai_news(), containing at minimum
        'title', 'source', and 'description'.
    tracker : Optional[MetricsTracker]
        If provided, token counts, cost, and latency for this Claude call
        are recorded in the tracker. Pass None to skip metrics (default).

    Returns
    ───────
    dict  — The parsed JSON fields from Claude, plus the original article
            nested under the key "article". Fields returned by Claude:

        • relevance_score  (int  1–10) : How useful is this story for PMs?
        • core_insight     (str)       : One-sentence PM takeaway
        • opportunity      (str)       : What PMs can gain from this development
        • risk_or_challenge(str)       : What PMs need to watch out for
        • action           (str)       : One concrete step a PM can take today
        • hook_idea        (str)       : Suggested LinkedIn opener for this story
    """

    # Initialise the Anthropic client with our API key from .env
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build the prompt — we embed article fields directly into the string.
    # The JSON schema in the prompt acts as a strict output contract for Claude.
    prompt = f"""You are a senior Product Manager and AI strategy advisor.

Read the following AI news article and provide a structured reflection on how it impacts Product Managers.

Article Title: {article['title']}
Source: {article['source']}
Summary: {article['description']}

Please provide your reflection in the following JSON format (respond ONLY with valid JSON, no extra text):
{{
  "relevance_score": <integer 1-10, how relevant this is to PMs>,
  "core_insight": "<one sentence summarizing the key PM insight>",
  "opportunity": "<specific opportunity this creates for Product Managers>",
  "risk_or_challenge": "<a risk or challenge PMs need to watch out for>",
  "action": "<one concrete action a PM can take today based on this news>",
  "hook_idea": "<a compelling, attention-grabbing opening line for a LinkedIn post about this>"
}}"""

    # Call the Claude API — max_tokens=600 is enough for the JSON object.
    # We use a single-turn conversation (one "user" message, no system prompt).
    # time.perf_counter() gives sub-millisecond precision for latency measurement.
    _start = time.perf_counter()
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    _latency = time.perf_counter() - _start

    # Record token usage, cost, and latency if a tracker was provided.
    # The step label uses the first 40 chars of the article title for readability.
    if tracker is not None:
        tracker.record(
            step_name=f"reflect: {article['title'][:40]}",
            message=message,
            latency_sec=_latency,
        )

    # Extract the raw text from the first (and only) content block
    raw = message.content[0].text.strip()

    # ── JSON parsing with code-fence handling ─────────────────────────────────
    # Claude sometimes wraps JSON in markdown fences like:
    #   ```json
    #   { … }
    #   ```
    # We strip the fence so json.loads() can parse the clean object.
    if raw.startswith("```"):
        raw = raw.split("```")[1]       # grab content between first pair of fences
        if raw.startswith("json"):
            raw = raw[4:]               # remove the "json" language tag after the open fence
    reflection = json.loads(raw.strip())

    # Attach the original article dict so post_generator.py has the source details
    # without needing to pass it separately.
    reflection["article"] = article
    return reflection


def pick_best_article(reflections: list[dict]) -> dict:
    """
    Select the most PM-relevant article from a list of reflections.

    Strategy
    ────────
    Each reflection contains a 'relevance_score' (1–10) assigned by Claude.
    We simply return the reflection with the highest score.

    If a reflection is malformed and missing the key, its score defaults to 0
    so it naturally loses the comparison.

    Parameters
    ──────────
    reflections : list[dict]
        All reflection dicts returned by reflect_on_article(), one per article.

    Returns
    ───────
    dict  — The single reflection with the highest relevance_score.
    """
    return max(reflections, key=lambda r: r.get("relevance_score", 0))

