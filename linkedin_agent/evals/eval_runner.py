"""
eval_runner.py — Main Eval Orchestrator
─────────────────────────────────────────
This is the entry point for the AI eval suite. Run it with:

    python3 evals/eval_runner.py

WHAT IT DOES (in order):
─────────────────────────
  1. Load the fixed test dataset from test_articles.json
  2. Run FUNCTIONAL EVALS on one representative article (fast, no cost)
     → Checks schema, score range, hashtags, word count, selection logic
  3. Run QUALITY EVALS (LLM-as-judge) on ALL test articles
     → For each article: generate a post, then score it with Claude
  4. Save results to evals/results/run_YYYY-MM-DD_HH-MM.json
  5. Load the PREVIOUS run result and show a REGRESSION DIFF
     → Green = improved, Red = regressed, Dim = unchanged

WHY THIS STRUCTURE?
───────────────────
Separating functional (fast, free) from quality (slow, costs API credits)
lets you run functional checks on every code change without burning budget,
and run full quality evals only when you change prompts or models.

UNDERSTANDING THE REGRESSION DIFF:
───────────────────────────────────
The diff compares average dimension scores between this run and the last:
  • A drop of ≥ 0.5 points is flagged as a REGRESSION (red)
  • A gain of ≥ 0.5 points is flagged as an IMPROVEMENT (green)
  • Changes < 0.5 are considered noise and shown in dim text

This threshold prevents you from chasing random LLM variance.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Add parent directory to path so we can import agent modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ANTHROPIC_API_KEY
from pm_reflection import reflect_on_article
from post_generator import generate_linkedin_post
from evals.eval_functional import run_all_functional_checks
from evals.eval_scorer import score_multiple_posts

# ── Paths ──────────────────────────────────────────────────────────────────────
EVALS_DIR = Path(__file__).parent                          # evals/ directory
TEST_ARTICLES_PATH = EVALS_DIR / "test_articles.json"      # fixed test dataset
RESULTS_DIR = EVALS_DIR / "results"                        # where run JSONs are stored

# ── Constants ─────────────────────────────────────────────────────────────────
# Minimum score change (absolute) to be flagged as a regression or improvement
REGRESSION_THRESHOLD = 0.5

console = Console()


def load_test_articles() -> list[dict]:
    """
    Load the fixed test article dataset from JSON.

    Why fixed?
    ──────────
    Using hand-curated, static articles ensures eval results are comparable
    across runs. If we used live news, scores could change due to topic
    difficulty rather than agent quality.

    Returns
    ───────
    list[dict]  — List of article dicts with id, title, description, source, url.
    """
    with open(TEST_ARTICLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: dict) -> str:
    """
    Save the current eval run results to a timestamped JSON file in evals/results/.

    Why save to disk?
    ─────────────────
    Persisting results lets the runner compare the current run against previous
    ones (regression detection). Without storage, every run is isolated and
    provides no trend data.

    Parameters
    ──────────
    results : dict  — Full results dict including functional checks and quality scores.

    Returns
    ───────
    str  — Absolute path to the saved file.
    """
    RESULTS_DIR.mkdir(exist_ok=True)  # create results/ folder if it doesn't exist
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filepath = RESULTS_DIR / f"run_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return str(filepath)


def load_previous_run() -> dict | None:
    """
    Find and load the most recent previous eval run result from evals/results/.

    How it works
    ────────────
    Scans results/ for all run_*.json files, sorts them alphabetically
    (ISO timestamps sort correctly as strings), and returns the second-to-last
    (the one before the current run).

    Returns
    ───────
    dict | None  — The previous run's results dict, or None if no prior run exists.
    """
    if not RESULTS_DIR.exists():
        return None
    run_files = sorted(RESULTS_DIR.glob("run_*.json"))

    # Need at least 2 files: the one we just saved (latest) and one before it
    if len(run_files) < 2:
        return None

    # Second-to-last file is the previous run (latest was just saved this session)
    prev_file = run_files[-2]
    with open(prev_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_average_scores(quality_results: list[dict]) -> dict:
    """
    Compute the average score for each quality dimension across all test articles.

    Parameters
    ──────────
    quality_results : list[dict]  — List of per-article score dicts from eval_scorer.

    Returns
    ───────
    dict  — { "pm_relevance": float, "hook_strength": float, … , "overall": float }
    """
    dimensions = ["pm_relevance", "hook_strength", "actionability", "tone", "hashtag_quality", "overall"]
    averages = {}
    for dim in dimensions:
        scores = [r[dim] for r in quality_results if isinstance(r.get(dim), (int, float))]
        averages[dim] = round(sum(scores) / len(scores), 2) if scores else 0.0
    return averages


def print_functional_results(checks: list[dict]):
    """
    Render functional eval results as a rich table with PASS/FAIL indicators.

    Parameters
    ──────────
    checks : list[dict]  — Results from eval_functional.run_all_functional_checks().
    """
    table = Table(title="Functional Checks", border_style="cyan", show_header=True)
    table.add_column("Check", style="bold", width=28)
    table.add_column("Result", width=8)
    table.add_column("Detail", style="dim")

    for check in checks:
        status = "[bold green]PASS ✓[/bold green]" if check["passed"] else "[bold red]FAIL ✗[/bold red]"
        table.add_row(check["name"], status, check["detail"])

    console.print(table)


def print_quality_results(quality_results: list[dict]):
    """
    Render LLM-as-judge quality scores as a rich table, one row per test article.

    Parameters
    ──────────
    quality_results : list[dict]  — Scored results from eval_scorer.score_multiple_posts().
    """
    table = Table(title="Quality Scores (LLM-as-Judge)", border_style="magenta", show_header=True)
    table.add_column("Article", style="bold", width=35, no_wrap=True)
    table.add_column("PM Rel", justify="center", width=7)
    table.add_column("Hook", justify="center", width=6)
    table.add_column("Action", justify="center", width=7)
    table.add_column("Tone", justify="center", width=6)
    table.add_column("Hashtag", justify="center", width=8)
    table.add_column("Overall", justify="center", style="bold", width=8)

    for result in quality_results:
        # Truncate long article titles for table readability
        title_short = result["article_title"][:33] + "…" if len(result["article_title"]) > 34 else result["article_title"]

        # Colour-code the overall score: green ≥ 7, yellow ≥ 5, red < 5
        overall = result.get("overall", 0)
        if overall >= 7:
            overall_str = f"[green]{overall}[/green]"
        elif overall >= 5:
            overall_str = f"[yellow]{overall}[/yellow]"
        else:
            overall_str = f"[red]{overall}[/red]"

        table.add_row(
            title_short,
            str(result.get("pm_relevance", "—")),
            str(result.get("hook_strength", "—")),
            str(result.get("actionability", "—")),
            str(result.get("tone", "—")),
            str(result.get("hashtag_quality", "—")),
            overall_str,
        )

    console.print(table)


def print_regression_diff(current_avgs: dict, previous_avgs: dict):
    """
    Compare current run averages to the previous run and highlight changes.

    Colour coding
    ─────────────
    Green  (↑) : score improved by ≥ REGRESSION_THRESHOLD
    Red    (↓) : score regressed by ≥ REGRESSION_THRESHOLD
    Dim    (→) : change below threshold (likely noise)

    Parameters
    ──────────
    current_avgs  : dict  — Average scores for this run
    previous_avgs : dict  — Average scores for the previous run
    """
    table = Table(title="Regression Diff vs Previous Run", border_style="yellow", show_header=True)
    table.add_column("Dimension", style="bold", width=20)
    table.add_column("Previous", justify="center", width=10)
    table.add_column("Current", justify="center", width=10)
    table.add_column("Change", justify="center", width=12)

    for dim in ["pm_relevance", "hook_strength", "actionability", "tone", "hashtag_quality", "overall"]:
        prev = previous_avgs.get(dim, 0)
        curr = current_avgs.get(dim, 0)
        delta = round(curr - prev, 2)

        if delta >= REGRESSION_THRESHOLD:
            change_str = f"[bold green]↑ +{delta}[/bold green]"
        elif delta <= -REGRESSION_THRESHOLD:
            change_str = f"[bold red]↓ {delta}[/bold red]"
        else:
            change_str = f"[dim]→ {delta:+.2f}[/dim]"

        table.add_row(dim, str(prev), str(curr), change_str)

    console.print(table)


def run_evals():
    """
    Main entry point — orchestrates all eval stages in sequence.

    Stages
    ──────
    1. Load test articles
    2. Functional evals (1 article, no cost, fast)
    3. Quality evals (all articles, 1 Claude call per article for post, 1 for score)
    4. Save results to JSON
    5. Load previous run & show regression diff
    """
    # ── Banner ─────────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🧪 LinkedIn Agent — AI Eval Suite[/bold cyan]\n"
            "[dim]Functional Checks → Quality Scoring → Regression Diff[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Validate that the Anthropic key is set before running any Claude calls
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
        console.print("[bold red]Error: ANTHROPIC_API_KEY not set in .env[/bold red]")
        sys.exit(1)

    # ── Stage 1: Load test articles ────────────────────────────────────────────
    console.print(Rule("[bold]Stage 1 · Loading Test Dataset[/bold]", style="cyan"))
    articles = load_test_articles()
    console.print(f"  [green]✓[/green] Loaded [bold]{len(articles)}[/bold] test articles from test_articles.json\n")

    # ── Stage 2: Functional Evals ─────────────────────────────────────────────
    # Use the first article as the representative test case (keeps API cost minimal)
    console.print(Rule("[bold]Stage 2 · Functional Checks[/bold]", style="cyan"))
    console.print(f"  [dim]Running against:[/dim] {articles[0]['title'][:70]}\n")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console, transient=True) as progress:
        t = progress.add_task("Running functional checks...", total=None)
        functional_results = run_all_functional_checks(articles[0])
        progress.update(t, completed=True)

    print_functional_results(functional_results)
    passed = sum(1 for c in functional_results if c["passed"])
    total = len(functional_results)
    status_colour = "green" if passed == total else "yellow" if passed > total // 2 else "red"
    console.print(f"\n  [{status_colour}]{passed}/{total} checks passed[/{status_colour}]\n")

    # ── Stage 3: Quality Evals (LLM-as-judge) ─────────────────────────────────
    console.print(Rule("[bold]Stage 3 · Quality Scoring (LLM-as-Judge)[/bold]", style="cyan"))
    console.print("  [dim]Generating posts and scoring each article…[/dim]\n")

    posts_and_articles = []
    for i, article in enumerate(articles, 1):
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console, transient=True) as progress:
            t = progress.add_task(
                f"  [{i}/{len(articles)}] Generating + scoring: {article['title'][:50]}…",
                total=None
            )
            try:
                # Generate a reflection then a post for this article
                reflection = reflect_on_article(article)
                post = generate_linkedin_post(reflection)
                posts_and_articles.append({
                    "article_id": article["id"],
                    "article": article,
                    "post": post,
                })
                progress.update(t, completed=True)
                console.print(f"  [green]✓[/green] [{i}/{len(articles)}] {article['title'][:60]}")
            except Exception as e:
                progress.update(t, completed=True)
                console.print(f"  [red]✗[/red] [{i}/{len(articles)}] Skipped: {e}")

    console.print()

    # Score all generated posts with Claude as judge
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console, transient=True) as progress:
        t = progress.add_task("Claude is evaluating all posts…", total=None)
        quality_results = score_multiple_posts(posts_and_articles)
        progress.update(t, completed=True)

    console.print()
    print_quality_results(quality_results)

    # Compute per-dimension averages for regression comparison
    current_avgs = compute_average_scores(quality_results)
    console.print(
        f"\n  [bold]Average overall score:[/bold] "
        f"[{'green' if current_avgs['overall'] >= 7 else 'yellow'}]{current_avgs['overall']}/10[/]\n"
    )

    # ── Stage 4: Save results ──────────────────────────────────────────────────
    console.print(Rule("[bold]Stage 4 · Saving Results[/bold]", style="cyan"))
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "functional": functional_results,
        "quality": quality_results,
        "averages": current_avgs,
    }
    saved_path = save_results(run_data)
    console.print(f"  [green]💾 Results saved to:[/green] [cyan]{saved_path}[/cyan]\n")

    # ── Stage 5: Regression diff ───────────────────────────────────────────────
    console.print(Rule("[bold]Stage 5 · Regression Diff[/bold]", style="cyan"))
    prev_run = load_previous_run()
    if prev_run:
        prev_avgs = prev_run.get("averages", {})
        print_regression_diff(current_avgs, prev_avgs)
        console.print(
            f"\n  [dim]Threshold: changes ≥ {REGRESSION_THRESHOLD} pts flagged as regression/improvement[/dim]\n"
        )
    else:
        console.print("  [dim]No previous run found — run again to see regression diff.[/dim]\n")

    console.print(Rule(style="dim"))
    console.print("[dim]  Eval complete. Review results above or open the saved JSON for full details.[/dim]\n")


if __name__ == "__main__":
    run_evals()
