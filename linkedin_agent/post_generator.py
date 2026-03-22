"""
post_generator.py — LinkedIn Post Writer (via Claude)
───────────────────────────────────────────────────────
This module is the final creative step in the agent pipeline.

It receives the highest-scoring article along with the PM reflection produced
by pm_reflection.py, then instructs Claude to write a ready-to-publish
LinkedIn post tailored for Product Managers.

The prompt is carefully engineered with explicit formatting rules so the output
requires zero editing before copying into LinkedIn.

METRICS INTEGRATION:
─────────────────────
generate_linkedin_post() accepts an optional MetricsTracker argument.
When provided, it records token counts, cost, and latency for the
post-generation Claude call.
"""

import time
from typing import Optional

import anthropic
from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    POST_TARGET_AUDIENCE,
    POST_STYLE,
    POST_MAX_WORDS,
    AUTHOR_STYLE,           # Option B: explicit style rules from MY_STYLE.md
    AUTHOR_EXAMPLE_POSTS,   # Option A: few-shot real posts from my_posts.md
)
from metrics import MetricsTracker


def generate_linkedin_post(reflection: dict, tracker: Optional[MetricsTracker] = None) -> str:
    """
    Generate a polished LinkedIn post from a PM reflection.

    Input format
    ────────────
    The `reflection` dict is the object returned by pm_reflection.reflect_on_article().
    It contains Claude's analysis fields AND the original article nested under
    the key "article". Both are used to build the prompt.

    Prompt engineering decisions
    ─────────────────────────────
    • Persona      : "Expert LinkedIn content creator for Product Managers" — this
                     frames the writing style as PM-focused, not generic tech.
    • Context dump : We give Claude both the raw article AND the structured PM
                     analysis so it has two complementary information sources.
    • Requirements : 9 explicit post rules prevent generic, low-quality output.
                     - Rule 1  : Forces use of the hook idea from pm_reflection.
                     - Rules 3 : Bullet-point format improves LinkedIn scannability.
                     - Rule 4  : Closing question drives comment engagement.
                     - Rule 8  : Bans clichés ("game-changer", "fast-paced world").
                     - Rule 9  : First-person voice makes the post feel personal.
    • Output only  : "Write ONLY the post text" ensures no preamble or notes from Claude.

    Parameters
    ──────────
    reflection : dict
        Must contain the keys produced by reflect_on_article():
            article, core_insight, opportunity, risk_or_challenge, action, hook_idea
    tracker : Optional[MetricsTracker]
        If provided, token usage, cost, and latency for this Claude call are
        recorded in the tracker. Pass None to skip metrics (default).

    Returns
    ───────
    str  — The final LinkedIn post text, ready to copy-paste. Typically 150–300 words.
    """

    # Initialise the Anthropic client using the key loaded from .env
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Unpack the nested article dict (title, source, description, url, …)
    article = reflection["article"]

    # Build the full prompt. It has 5 clearly-labeled sections:
    #   1. AUTHOR IDENTITY  : Tells Claude whose voice it is writing in
    #   2. STYLE RULES      : Explicit rules from MY_STYLE.md (Option B)
    #   3. EXAMPLE POSTS    : Real posts by the author as writing reference (Option A)
    #   4. CONTENT          : The article and PM analysis to write about
    #   5. POST REQUIREMENTS: Structural requirements for this specific post
    #
    # The examples section is only included if AUTHOR_EXAMPLE_POSTS is non-empty.
    # This keeps the prompt clean if the file is missing.
    examples_section = ""
    if AUTHOR_EXAMPLE_POSTS.strip():
        examples_section = f"""
── MY EXAMPLE POSTS (mirror this voice exactly) ──
{AUTHOR_EXAMPLE_POSTS}
"""

    prompt = f"""You are ghostwriting a LinkedIn post for a Product Manager who writes in a very specific, recognisable voice.

Your job is to write AS THIS PERSON — not as a generic AI assistant.
Match their voice, structure, tone, and sentence patterns as closely as possible.

── AUTHOR'S STYLE PROFILE ──
{AUTHOR_STYLE}
{examples_section}
── TODAY'S AI NEWS STORY ──
Title: {article['title']}
Source: {article['source']}
Summary: {article['description']}

── PM ANALYSIS (use these insights in the post) ──
Core Insight: {reflection['core_insight']}
Opportunity for PMs: {reflection['opportunity']}
Risk / Challenge: {reflection['risk_or_challenge']}
Actionable Step: {reflection['action']}
Hook Idea to build from: {reflection['hook_idea']}

── POST REQUIREMENTS ──
1. Open with a bold headline in the author's format: "Topic | What happened / what I found"
2. Tell a story arc: context → what happened → honest reflection
3. Use emoji + bold section headers to organise (e.g. 🚀 The Problem, ✅ What Worked, 💡 Honest Take)
4. Use --> arrows for key callout moments or standout observations
5. Share specific, concrete details — numbers, tool names, exact limitations found
6. Be honest about what this does NOT do or what the limitations are
7. Close with a specific italicised question the author is genuinely curious about
8. End with 5–8 relevant hashtags on a separate final line
9. Keep total post under {POST_MAX_WORDS} words
10. Write ONLY the post. No preamble, no commentary, no headers outside the post itself."""

    # max_tokens=1500: Previously 800, which was sufficient for the old short prompt.
    # The new voice-personalised prompt (~4K input tokens) produces longer, richer posts
    # with section headers, --> callouts, a closing question AND hashtags.
    # 1500 gives Claude enough room to write a complete post without truncation.
    # Rule of thumb: 1 token ≈ 0.75 words, so 1500 tokens ≈ ~1100 words of headroom.
    # time.perf_counter() gives sub-millisecond latency precision.
    _start = time.perf_counter()
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    _latency = time.perf_counter() - _start

    # Record token usage, cost, and latency if a tracker was provided.
    if tracker is not None:
        tracker.record(
            step_name="generate_post",
            message=message,
            latency_sec=_latency,
        )

    # Return the generated post text, stripping any leading/trailing whitespace
    return message.content[0].text.strip()

