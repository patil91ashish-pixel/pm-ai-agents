"""
config.py — Central Configuration
───────────────────────────────────
This file is the single place to control all agent behaviour.
It loads your secrets from the .env file and exposes named constants
that every other module imports. Change values here to customise the agent
without touching the logic files.

VOICE PERSONALISATION:
──────────────────────
Two files in the project root drive the agent's writing voice:
  • MY_STYLE.md   — Explicit style rules (Option B)
  • my_posts.md   — Real example posts / few-shot examples (Option A)
Both are loaded here at startup and stored as AUTHOR_STYLE and
AUTHOR_EXAMPLE_POSTS. post_generator.py injects both into the prompt.
"""

import os
from pathlib import Path
from dotenv import load_dotenv  # reads key=value pairs from a .env file into environment variables

# ── Load .env file ───────────────────────────────────────────────────────────
# python-dotenv looks for a file called ".env" in the current directory and
# loads its contents into os.environ so we can read them with os.getenv().
load_dotenv()

# ── API Keys (loaded from .env) ──────────────────────────────────────────────
# These are kept out of source code for security reasons.
# Never hard-code API keys directly here — use the .env file instead.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")   # Required: Claude LLM access
NEWS_API_KEY = os.getenv("NEWS_API_KEY")              # Optional: live AI news headlines

# ── Claude Model ─────────────────────────────────────────────────────────────
# The Anthropic model ID to use for both PM reflection and post generation.
# "claude-sonnet-4-5" offers a strong balance of quality and speed.
# You can swap this for "claude-opus-4-5" for higher quality (slower & pricier)
# or "claude-haiku-3-5" for faster, cheaper responses.
CLAUDE_MODEL = "claude-sonnet-4-5"

# ── News Fetching Settings ───────────────────────────────────────────────────
# Controls what kind of articles are fetched from NewsAPI.

NEWS_QUERY = "artificial intelligence"   # The search keyword sent to NewsAPI.
                                         # Change this to focus on sub-topics,
                                         # e.g. "generative AI", "AI regulation", "LLM".

NEWS_LANGUAGE = "en"                     # Filter results to English articles only.

NEWS_MAX_ARTICLES = 8                    # How many articles to download and evaluate.
                                         # More = better selection, but more Claude API calls.

NEWS_HOURS_BACK = 48                     # Only fetch articles published within this window.
                                         # 48 = last two days. Increase for a wider net.

# ── LinkedIn Post Settings ───────────────────────────────────────────────────
# These values are injected directly into the Claude prompts to shape the post.

POST_TARGET_AUDIENCE = "Product Managers"  # Who the post is written for.
                                            # Claude will frame insights around their world.

POST_STYLE = "professional yet conversational, insightful, and inspiring"
                                            # Tone descriptor passed to Claude.
                                            # Edit freely to match your personal voice.

POST_MAX_WORDS = 300                        # Target upper word limit for the post.
                                            # LinkedIn performs best with 150–300 word posts.

# ── Output Settings ──────────────────────────────────────────────────────────
# Where generated posts are saved on disk.
OUTPUT_DIR = "output"                       # Relative path; created automatically if missing.

# ── Voice Personalisation ──────────────────────────────────────
# Load the author's style profile and example posts from markdown files.
# These are read once at startup (cheap) and exported for use in post_generator.py.
#
# MY_STYLE.md   — Writing style rules (Option B: explicit style profile)
# my_posts.md   — Real example posts   (Option A: few-shot examples)
#
# Using Path(__file__).parent ensures the files are found relative to config.py
# regardless of which directory the user runs the agent from.

_BASE_DIR = Path(__file__).parent  # linkedin_agent/ project root


def _load_file(filename: str, fallback: str) -> str:
    """
    Load a personalisation file from the project root.

    This helper makes the loading fault-tolerant: if the file is missing
    (e.g. on a fresh clone), it returns a safe fallback string instead
    of raising a FileNotFoundError and crashing the agent.

    Parameters
    ──────────
    filename : str  — Name of the file to load (e.g. 'MY_STYLE.md')
    fallback : str  — String to return if the file doesn't exist

    Returns
    ───────
    str  — File contents, or the fallback string.
    """
    filepath = _BASE_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return fallback


# Option B: Style rules profile
# Falls back to a generic style description if MY_STYLE.md is missing.
AUTHOR_STYLE = _load_file(
    "MY_STYLE.md",
    fallback="professional yet conversational, insightful, first-person",
)

# Option A: Few-shot example posts
# Falls back to empty string (no examples) if my_posts.md is missing.
AUTHOR_EXAMPLE_POSTS = _load_file(
    "my_posts.md",
    fallback="",
)
