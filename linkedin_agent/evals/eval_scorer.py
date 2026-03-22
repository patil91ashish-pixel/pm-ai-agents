"""
eval_scorer.py — LLM-as-Judge Quality Scoring
────────────────────────────────────────────────
This module scores the quality of generated LinkedIn posts using Claude itself
as the evaluator — a technique called "LLM-as-Judge".

WHY LLM-AS-JUDGE?
─────────────────
Quality is subjective and hard to check with rules alone. Questions like
"Is this hook compelling?" or "Is this actionable?" require language understanding.
Claude can evaluate these dimensions reliably if given a clear rubric.

KEY DESIGN PRINCIPLE — Separate judge from generator:
─────────────────────────────────────────────────────
We use the SAME Claude model as both generator and judge, but with a completely
different prompt and persona. This is acceptable for relative comparisons
(e.g. "did my prompt change improve quality?") but be aware of self-preference
bias — Claude may favour its own writing style.

SCORING DIMENSIONS (each 1–10):
────────────────────────────────
  pm_relevance    — How useful is this insight for Product Managers specifically?
  hook_strength   — Does the first line make you stop scrolling?
  actionability   — Are the 3 bullet takeaways concrete and doable?
  tone            — Professional + conversational, not robotic or generic?
  hashtag_quality — Relevant hashtags in appropriate quantity (5–7)?

Returns
───────
  A dict with all 5 dimension scores, an overall average, and brief justifications.

INTERPRETING SCORES:
────────────────────
  8–10 : Excellent — publish as-is
  6–7  : Good — minor polish needed
  4–5  : Mediocre — prompt or model change recommended
  1–3  : Poor — significant regression detected
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


# ── Scoring rubric injected into every judge prompt ───────────────────────────
# Each dimension has a 1-10 scale with clear anchors so Claude applies the
# rubric consistently across different posts and runs.
SCORING_RUBRIC = """
Score each dimension from 1 to 10 using these anchors:
  10 = Exceptional, publish immediately
   8 = Strong, minor improvements possible
   6 = Acceptable, but noticeably average
   4 = Below average, needs significant work
   2 = Poor, fails the dimension almost entirely
   1 = Complete failure (empty, broken, or irrelevant)
"""


def score_post(post: str, article: dict) -> dict:
    """
    Ask Claude to evaluate a generated LinkedIn post across 5 quality dimensions.

    How it works
    ────────────
    1. Build a judge prompt that includes the post, the source article (context),
       and the scoring rubric.
    2. Ask Claude to respond ONLY in JSON — one score + justification per dimension.
    3. Parse the JSON and compute an `overall` average score.
    4. Return the full score dict for storage and display.

    Parameters
    ──────────
    post    : str  — The generated LinkedIn post text to be evaluated.
    article : dict — The source article, used to verify relevance in scoring.

    Returns
    ───────
    dict with keys:
        pm_relevance       (int)  : 1–10 score
        hook_strength      (int)  : 1–10 score
        actionability      (int)  : 1–10 score
        tone               (int)  : 1–10 score
        hashtag_quality    (int)  : 1–10 score
        overall            (float): Average of all 5 dimensions
        justifications     (dict) : One-sentence reason per dimension
        error              (str)  : Only present if scoring failed
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # The judge prompt gives Claude a clear persona (impartial evaluator),
    # the full context (source article + post), and a strict JSON schema to follow.
    judge_prompt = f"""You are an impartial LinkedIn content quality evaluator.

Your job is to score the following LinkedIn post written for Product Managers.

── SOURCE ARTICLE ──
Title: {article['title']}
Summary: {article['description']}

── LINKEDIN POST TO EVALUATE ──
{post}

── SCORING RUBRIC ──
{SCORING_RUBRIC}

── DIMENSIONS TO SCORE ──
1. pm_relevance    — Does this post provide genuinely useful insight for Product Managers?
                     (NOT generic AI commentary — specific to PM workflows, decisions, and challenges)
2. hook_strength   — Does the opening line stop a scrolling reader? Is it specific and intriguing?
3. actionability   — Are the bullet-point takeaways concrete actions a PM can take THIS WEEK?
4. tone            — Is the writing style professional yet conversational? Does it feel human?
5. hashtag_quality — Are 5–7 relevant hashtags included? Are they appropriate for LinkedIn reach?

Respond ONLY with valid JSON in this exact format (no extra text, no code fences):
{{
  "pm_relevance":    {{"score": <int 1-10>, "justification": "<one sentence>"}},
  "hook_strength":   {{"score": <int 1-10>, "justification": "<one sentence>"}},
  "actionability":   {{"score": <int 1-10>, "justification": "<one sentence>"}},
  "tone":            {{"score": <int 1-10>, "justification": "<one sentence>"}},
  "hashtag_quality": {{"score": <int 1-10>, "justification": "<one sentence>"}}
}}"""

    try:
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": judge_prompt}],
        )

        raw = message.content[0].text.strip()

        # Defensively strip any markdown code fences Claude may have added
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        scores_raw = json.loads(raw.strip())

        # Flatten into a clean dict with top-level scores and justifications
        scores = {}
        justifications = {}
        for dimension, data in scores_raw.items():
            scores[dimension] = data["score"]
            justifications[dimension] = data["justification"]

        # Compute overall average rounded to 2 decimal places
        scores["overall"] = round(sum(scores.values()) / len(scores_raw), 2)
        scores["justifications"] = justifications
        return scores

    except Exception as e:
        # Return an error dict instead of crashing — the runner handles this gracefully
        return {
            "pm_relevance": 0,
            "hook_strength": 0,
            "actionability": 0,
            "tone": 0,
            "hashtag_quality": 0,
            "overall": 0.0,
            "justifications": {},
            "error": str(e),
        }


def score_multiple_posts(posts_and_articles: list[dict]) -> list[dict]:
    """
    Score a batch of posts sequentially and return all results.

    Design decision
    ───────────────
    We call Claude once per post rather than batching. This keeps each
    judge call focused and avoids inter-post bias (Claude rating relative
    to others in the batch instead of against the rubric).

    Parameters
    ──────────
    posts_and_articles : list[dict]
        Each dict must have keys:
            article_id  (str)  : e.g. "article_001"
            article     (dict) : The source article
            post        (str)  : The generated LinkedIn post

    Returns
    ───────
    list[dict]  — Each dict has: article_id, article_title, and all score fields
                  from score_post().
    """
    results = []
    for item in posts_and_articles:
        scores = score_post(item["post"], item["article"])
        results.append({
            "article_id": item["article_id"],
            "article_title": item["article"]["title"],
            **scores,  # spread all score fields (pm_relevance, hook_strength, etc.)
        })
    return results
