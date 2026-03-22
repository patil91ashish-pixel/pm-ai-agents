"""
agent.py — Main Orchestrator
──────────────────────────────
This is the entry point for the LinkedIn Post Generator Agent.

It ties together all three pipeline stages in order:
  1. news_fetcher.py  → fetch the latest AI headlines from NewsAPI
  2. pm_reflection.py → ask Claude to score each article's PM relevance
  3. post_generator.py → ask Claude to write a LinkedIn post from the best article

The agent also handles:
  • Config validation (are API keys set?)
  • A rich terminal UI with progress spinners and colour-coded output
  • Saving the final post to the output/ folder as a timestamped .txt file

Run this file directly:
    python3 agent.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# rich — a library for beautiful terminal formatting.
# Console : handles all print output with markup support (colours, bold, etc.)
# Panel   : draws a bordered box around content
# Progress: animated spinner/progress bar
# Rule    : draws a horizontal dividing line
# Text    : wraps raw strings so they can be passed into Panels safely
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.text import Text
from rich import print as rprint

# Import settings and all pipeline modules
from config import ANTHROPIC_API_KEY, NEWS_API_KEY, OUTPUT_DIR, CLAUDE_MODEL
from news_fetcher import fetch_ai_news                          # Step 1
from pm_reflection import reflect_on_article, pick_best_article # Step 2
from post_generator import generate_linkedin_post               # Step 3
from metrics import MetricsTracker                              # Observability
from post_history import load_history, filter_seen_articles, save_to_history  # Memory

# Global Rich console — used throughout the file instead of plain print()
# so we get colour, styling, and emoji support in the terminal.
console = Console()


def validate_config():
    """
    Check that required API keys are present in the environment before running.

    ANTHROPIC_API_KEY  — Required. Without it, all Claude calls will fail immediately.
    NEWS_API_KEY       — Optional. If missing, the agent uses fallback articles and
                         warns the user rather than crashing.

    Behaviour
    ─────────
    • If ANTHROPIC_API_KEY is missing → print a helpful error panel and exit with code 1.
    • If NEWS_API_KEY is missing      → print a yellow warning and continue.
    """
    missing = []

    # Check if the key is absent OR still set to the placeholder from .env.example
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
        missing.append("ANTHROPIC_API_KEY")
    if not NEWS_API_KEY or NEWS_API_KEY == "your_newsapi_key_here":
        missing.append("NEWS_API_KEY (optional – fallback articles will be used)")

    if "ANTHROPIC_API_KEY" in missing:
        # Print a styled error panel with setup instructions and exit immediately.
        # sys.exit(1) signals failure to any shell scripts wrapping this agent.
        console.print(
            Panel(
                "[bold red]Missing required API key: ANTHROPIC_API_KEY[/bold red]\n\n"
                "1. Copy [cyan].env.example[/cyan] → [cyan].env[/cyan]\n"
                "2. Add your Anthropic API key from [link=https://console.anthropic.com]console.anthropic.com[/link]",
                title="⚠️  Configuration Error",
                border_style="red",
            )
        )
        sys.exit(1)

    if "NEWS_API_KEY (optional – fallback articles will be used)" in missing:
        # Non-fatal: just warn and continue — fallback articles will be used instead.
        console.print(
            "[yellow]⚠️  NEWS_API_KEY not set — using fallback articles for demo.[/yellow]\n"
        )


def save_post(post: str, article_title: str) -> str:
    """
    Save the generated LinkedIn post to a timestamped text file in the output/ directory.

    File naming convention
    ──────────────────────
    post_YYYY-MM-DD_HH-MM_<sanitised_article_title>.txt
    Example: post_2024-03-19_14-30_OpenAI_launches_GPT_5_with_enha.txt

    The article title is truncated to 40 characters and non-alphanumeric characters
    are replaced with underscores to produce a safe filename on all operating systems.

    Parameters
    ──────────
    post          : str  — The full generated LinkedIn post text.
    article_title : str  — Title of the source article (used in the filename).

    Returns
    ───────
    str  — Absolute path to the saved file, printed by run_agent() after saving.
    """
    # Create the output directory if it doesn't exist yet (exist_ok avoids errors on repeat runs)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    # Build a filesystem-safe filename component from the article title
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    safe_title = "".join(c if c.isalnum() else "_" for c in article_title[:40])
    filename = output_path / f"post_{date_str}_{safe_title}.txt"

    # Write the file with UTF-8 encoding to preserve emojis and special characters
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Source article: {article_title}\n")
        f.write("=" * 60 + "\n\n")  # visual separator between metadata and post body
        f.write(post)

    return str(filename)


def run_agent():
    """
    Main pipeline function — runs all four steps of the agent in sequence.

    Steps
    ─────
    1. Validate API keys            → validate_config()
    2. Fetch AI news                → news_fetcher.fetch_ai_news()
    3. Reflect on each article      → pm_reflection.reflect_on_article()  (one Claude call per article)
    4. Pick the best article        → pm_reflection.pick_best_article()
    5. Generate the LinkedIn post   → post_generator.generate_linkedin_post()
    6. Display and save the post    → console.print() + save_post()

    Each step is wrapped in a Rich Progress spinner so the user sees live feedback
    while waiting for network/Claude API calls to complete.
    """

    # ── Welcome banner ──────────────────────────────────────────────────────
    # Panel.fit() auto-sizes the box to the content width
    console.print(
        Panel.fit(
            "[bold cyan]🤖 LinkedIn Post Generator Agent[/bold cyan]\n"
            "[dim]AI News → PM Insight → LinkedIn Post[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Fail fast if keys are missing — no point hitting the network otherwise
    validate_config()

    # Initialise the MetricsTracker for this run.
    # It will accumulate token counts, costs, and latency for every Claude call.
    tracker = MetricsTracker(model=CLAUDE_MODEL)

    # ── Step 1: Fetch News ──────────────────────────────────────────────────
    console.print(Rule("[bold]Step 1 · Fetching AI News[/bold]", style="cyan"))

    # Progress context manager shows an animated spinner while the NewsAPI call runs.
    # transient=True removes the spinner line from the terminal once the block exits.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning latest AI headlines...", total=None)
        articles = fetch_ai_news()    # actual network call; may use fallback if it fails
        progress.update(task, completed=True)

    # Print a numbered list of fetched article titles for transparency
    console.print(f"  [green]✓[/green] Found [bold]{len(articles)}[/bold] AI news articles\n")
    for i, a in enumerate(articles, 1):
        console.print(f"  [dim]{i}.[/dim] {a['title']} [dim]({a['source']})[/dim]")
    console.print()

    # ── Step 1b: Deduplication ────────────────────────────────────────────────
    # Load the post history and filter out any articles that are too similar
    # to ones we've already posted about. This happens BEFORE Claude API calls
    # so we don't waste tokens reflecting on articles we'd discard anyway.
    history = load_history()
    articles, skipped = filter_seen_articles(articles, history)

    if skipped:
        console.print(f"  [yellow]⚠️  Skipped {len(skipped)} already-seen article(s):[/yellow]")
        for a in skipped:
            console.print(f"  [dim]  ✗ {a['title'][:70]} — {a.get('_skip_reason', 'duplicate')}[/dim]")
        console.print()

    # If ALL articles were filtered, there's nothing new to post about today.
    # Exit cleanly (code 0) — this isn't an error, just a "nothing new" state.
    if not articles:
        console.print(
            "[yellow]All fetched articles have been posted about recently. "
            "Try again later or increase NEWS_HOURS_BACK in config.py.[/yellow]"
        )
        sys.exit(0)

    console.print(f"  [green]✓[/green] [bold]{len(articles)}[/bold] fresh article(s) ready for analysis\n")

    # ── Step 2: Reflect on each article ────────────────────────────────────
    # For each article we make one Claude API call, so this is the most time-consuming step.
    # We show a per-article spinner with the article index to give progress feedback.
    console.print(Rule("[bold]Step 2 · Analyzing PM Relevance[/bold]", style="cyan"))
    reflections = []  # accumulate successful reflections here

    for i, article in enumerate(articles, 1):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            t = progress.add_task(
                f"Reflecting on article {i}/{len(articles)}: {article['title'][:55]}...",
                total=None,
            )
            try:
                # Ask Claude to analyse this article and return a scored reflection dict
                reflection = reflect_on_article(article, tracker=tracker)
                reflections.append(reflection)
                progress.update(t, completed=True)

                # Show the PM relevance score inline so you can see which articles rank highest
                score = reflection.get("relevance_score", "?")
                console.print(
                    f"  [green]✓[/green] [{i}/{len(articles)}] Score: [bold]{score}/10[/bold] — {article['title'][:60]}"
                )
            except Exception as e:
                # If Claude fails for one article (API error, JSON parse error, etc.)
                # we skip it rather than aborting the entire run.
                progress.update(t, completed=True)
                console.print(f"  [red]✗[/red] [{i}/{len(articles)}] Skipped ({e})")
    console.print()

    # If EVERY article failed to reflect, there's nothing to generate a post from.
    if not reflections:
        console.print("[red]No articles could be analyzed. Please check your API keys.[/red]")
        sys.exit(1)

    # ── Step 3: Pick best article ───────────────────────────────────────────
    # pick_best_article() simply finds the reflection with the highest relevance_score.
    console.print(Rule("[bold]Step 3 · Selecting Best Story[/bold]", style="cyan"))
    best = pick_best_article(reflections)

    # Display the selected article and its PM insights to the user
    console.print(f"  [green]✓[/green] Selected: [bold]{best['article']['title']}[/bold]")
    console.print(
        f"  [dim]Relevance Score: {best['relevance_score']}/10 | "
        f"Source: {best['article']['source']}[/dim]\n"
    )
    # Show the four insight fields so the user can see Claude's reasoning
    console.print(f"  [cyan]Core PM Insight:[/cyan] {best['core_insight']}")
    console.print(f"  [green]Opportunity:[/green] {best['opportunity']}")
    console.print(f"  [yellow]Risk:[/yellow] {best['risk_or_challenge']}")
    console.print(f"  [magenta]Action:[/magenta] {best['action']}")
    console.print()

    # ── Step 4: Generate LinkedIn Post ─────────────────────────────────────
    console.print(Rule("[bold]Step 4 · Writing LinkedIn Post[/bold]", style="cyan"))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Claude is crafting your LinkedIn post...", total=None)
        # One final Claude call that takes the full reflection and writes the post
        post = generate_linkedin_post(best, tracker=tracker)
        progress.update(t, completed=True)

    # Report word count so the user knows if it's within LinkedIn's sweet spot (150–300 words)
    console.print(f"  [green]✓[/green] Post generated ({len(post.split())} words)\n")

    # ── Display Final Post ──────────────────────────────────────────────────
    console.print(Rule("[bold magenta]✨  Your LinkedIn Post[/bold magenta]", style="magenta"))
    console.print()
    # Text() wraps the post string so Rich doesn't interpret hashtags as markup tags
    console.print(
        Panel(
            Text(post),
            border_style="magenta",
            padding=(1, 2),   # (top/bottom, left/right) padding inside the panel
        )
    )
    console.print()

    # ── Save to file & update history ──────────────────────────────────────────
    # Persist the post to disk so you don't lose it after closing the terminal.
    # Also record the used article in post_history.json so future runs skip it.
    saved_path = save_post(post, best["article"]["title"])
    console.print(f"  [green]💾 Saved to:[/green] [cyan]{saved_path}[/cyan]")

    # Save the article to history AFTER a successful post save.
    # This ensures history is only updated when the full run was successful.
    save_to_history(best["article"])
    console.print(f"  [green]🧠 Article added to history:[/green] [dim]future runs will skip this story[/dim]\n")

    # ── Print metrics summary ───────────────────────────────────────
    # Printed after everything else so it doesn’t interrupt the post display.
    tracker.print_summary()

    console.print(Rule(style="dim"))
    console.print(
        "[dim]  Copy the post above and paste it directly into LinkedIn. Happy posting! 🚀[/dim]\n"
    )


# ── Entry point ─────────────────────────────────────────────────────────────
# This block only runs when the file is executed directly (python3 agent.py).
# It does NOT run when this module is imported by another script or test.
if __name__ == "__main__":
    run_agent()
