"""
news_fetcher.py — AI News Retrieval
──────────────────────────────────────
Responsible for pulling the latest AI-related news from the internet.

Primary source : NewsAPI (https://newsapi.org) — a free-tier news aggregator
                 covering thousands of sources worldwide.
Fallback       : A small hand-curated list of evergreen AI stories used when
                 the NewsAPI key is missing or the network call fails.

Each article is returned as a plain Python dict so other modules don't depend
on any particular HTTP or data library.
"""

import requests
from datetime import datetime, timedelta
from config import NEWS_API_KEY, NEWS_QUERY, NEWS_LANGUAGE, NEWS_MAX_ARTICLES, NEWS_HOURS_BACK


def fetch_ai_news() -> list[dict]:
    """
    Fetch the latest AI-related news articles from NewsAPI.

    How it works
    ────────────
    1. Calculate the earliest publish date we care about (now − NEWS_HOURS_BACK).
    2. Call the NewsAPI /everything endpoint with our AI keyword and date filter.
    3. Normalise each raw article into a clean dict (title, description, url, …).
    4. Skip any article that is missing a title or description (unusable for AI).
    5. If the request fails for any reason, fall back to hand-curated articles.

    Returns
    ───────
    list[dict]  — Each dict has keys:
        • title        (str) : Headline of the article
        • description  (str) : Short summary / lead paragraph
        • content      (str) : Full body text (may be truncated by NewsAPI)
        • url          (str) : Link to the original article
        • source       (str) : Publication name, e.g. "TechCrunch"
        • published_at (str) : ISO-8601 publish timestamp, e.g. "2024-03-19T10:00:00Z"
    """

    # Calculate the oldest acceptable publish date, formatted for NewsAPI
    cutoff_time = datetime.utcnow() - timedelta(hours=NEWS_HOURS_BACK)
    from_date = cutoff_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # NewsAPI endpoint — /everything searches across all sources and blogs
    url = "https://newsapi.org/v2/everything"

    # Build the query parameters dict; NewsAPI ignores unknown keys safely
    params = {
        "q": NEWS_QUERY,              # Keyword filter (e.g. "artificial intelligence")
        "language": NEWS_LANGUAGE,    # Only return articles in this language
        "from": from_date,            # Exclude articles older than our cutoff
        "sortBy": "relevancy",        # "relevancy" ranks matches by keyword fit;
                                      # alternatives: "publishedAt", "popularity"
        "pageSize": NEWS_MAX_ARTICLES, # Maximum number of results to return
        "apiKey": NEWS_API_KEY,        # Authentication token from .env
    }

    try:
        # Make the HTTP GET request with a 10-second timeout to avoid hanging
        response = requests.get(url, params=params, timeout=10)

        # Raise an exception for HTTP error codes (4xx client, 5xx server errors)
        response.raise_for_status()

        # Parse the JSON body into a Python dict
        data = response.json()

        articles = []
        for a in data.get("articles", []):
            # Guard: skip articles where title or description is absent.
            # Without these fields, Claude cannot generate a meaningful reflection.
            if not a.get("title") or not a.get("description"):
                continue

            # Normalise the raw NewsAPI article shape into our standard dict.
            # Using .get() with defaults prevents KeyError if NewsAPI changes its schema.
            articles.append({
                "title": a["title"],
                "description": a.get("description", ""),
                "content": a.get("content", ""),       # NewsAPI truncates content to ~200 chars on free tier
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "Unknown"),  # source is a nested object
                "published_at": a.get("publishedAt", ""),
            })

        return articles

    except requests.exceptions.RequestException as e:
        # Network errors, DNS failures, timeouts, bad HTTP status codes all land here.
        # Instead of crashing, we fall back to sample articles so the agent can still run.
        print(f"[WARNING] NewsAPI request failed: {e}. Using fallback articles.")
        return _fallback_articles()


def _fallback_articles() -> list[dict]:
    """
    Return a small set of hand-written AI news articles.

    Purpose
    ───────
    Used when the NEWS_API_KEY is not configured or the network request fails.
    These articles let you test the PM reflection and post-generation pipeline
    locally without needing a live API key.

    All articles follow the same dict shape as fetch_ai_news() so downstream
    code doesn't need to handle a different format.
    """
    return [
        {
            "title": "OpenAI launches GPT-5 with enhanced reasoning capabilities",
            "description": "The latest model shows dramatic improvements in multi-step reasoning, coding, and long-form analysis, setting a new benchmark for enterprise AI tools.",
            "content": "",
            "url": "https://openai.com/blog",
            "source": "OpenAI Blog",
            "published_at": datetime.utcnow().isoformat(),  # mark as now so it passes any date filter
        },
        {
            "title": "Google DeepMind's AI co-scientist accelerates drug discovery by 10x",
            "description": "A new AI agent from DeepMind can autonomously design and test molecular compounds, cutting pharmaceutical R&D timelines from years to months.",
            "content": "",
            "url": "https://deepmind.google/blog",
            "source": "DeepMind Blog",
            "published_at": datetime.utcnow().isoformat(),
        },
        {
            "title": "Anthropic releases Claude 3 with advanced code generation and agentic workflows",
            "description": "Claude's latest version features improved tool use, allowing it to autonomously complete multi-step software engineering tasks end-to-end.",
            "content": "",
            "url": "https://anthropic.com/news",
            "source": "Anthropic",
            "published_at": datetime.utcnow().isoformat(),
        },
    ]

