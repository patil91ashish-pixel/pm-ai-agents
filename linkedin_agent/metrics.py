"""
metrics.py — Token, Cost & Latency Tracker
────────────────────────────────────────────
Provides a lightweight MetricsTracker class that records usage data from
every Claude API call and summarises it at the end of a run.

WHY TRACK THESE METRICS?
─────────────────────────
  • Token consumption : Reveals which pipeline step is most "expensive"
                        and signals when a prompt has ballooned unexpectedly.
  • Cost (USD)        : Makes Claude API usage tangible. One full agent run
                        costs roughly $0.05–$0.30 depending on article count.
  • Latency (seconds) : Shows where wall-clock time is spent — helps decide
                        when to parallelise or switch models.

HOW IT WORKS:
─────────────
  1. Create a MetricsTracker instance at the start of a run.
  2. After each Claude API call, call tracker.record(step_name, message, latency).
     The Anthropic SDK returns token counts in message.usage automatically.
  3. At the end of the run, call tracker.print_summary() for a rich table
     and tracker.to_dict() to save metrics to a results JSON file.

PRICING (claude-sonnet-4-5 as of 2025):
─────────────────────────────────────────
  Input tokens  : $3.00 per million tokens
  Output tokens : $15.00 per million tokens

  These rates are defined in PRICING below. Update them if Anthropic
  changes their pricing at: https://www.anthropic.com/pricing
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.rule import Rule

# ── Pricing constants (per million tokens) ────────────────────────────────────
# Source: https://www.anthropic.com/pricing
# Update these if Anthropic adjusts rates.
PRICING = {
    "claude-sonnet-4-5": {
        "input_per_million": 3.00,    # USD per 1M input tokens
        "output_per_million": 15.00,  # USD per 1M output tokens
    },
    "claude-haiku-3-5": {
        "input_per_million": 0.80,
        "output_per_million": 4.00,
    },
    "claude-opus-4-5": {
        "input_per_million": 15.00,
        "output_per_million": 75.00,
    },
}

# Fallback rates used when the model isn't in PRICING (avoids KeyError)
DEFAULT_PRICING = {"input_per_million": 3.00, "output_per_million": 15.00}

console = Console()


@dataclass
class CallRecord:
    """
    Stores usage data for a single Claude API call.

    Attributes
    ──────────
    step_name     : str   — Human-readable label, e.g. "reflect: GPT-5 article"
    model         : str   — Claude model ID used for this call
    input_tokens  : int   — Number of tokens in the prompt (input)
    output_tokens : int   — Number of tokens in Claude's response (output)
    cost_usd      : float — Computed dollar cost for this call
    latency_sec   : float — Wall-clock seconds from request start to response
    """
    step_name: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_sec: float


class MetricsTracker:
    """
    Accumulates and reports token, cost, and latency data across a full agent run.

    Usage pattern
    ─────────────
    tracker = MetricsTracker(model="claude-sonnet-4-5")

    start = time.perf_counter()
    message = client.messages.create(...)
    tracker.record("reflect: article_1", message, time.perf_counter() - start)

    tracker.print_summary()   # prints rich table to terminal
    metrics_dict = tracker.to_dict()  # returns data for JSON storage

    Design decisions
    ─────────────────
    • MetricsTracker is passed as an optional parameter into each pipeline
      function. If None is passed, functions work exactly as before — no
      breaking changes.
    • Cost is computed immediately on record() so it's always consistent
      with the token counts stored alongside it.
    """

    def __init__(self, model: str):
        """
        Initialise the tracker for a specific Claude model.

        Parameters
        ──────────
        model : str  — Claude model ID (must match a key in PRICING, or
                       DEFAULT_PRICING is used as fallback).
        """
        self.model = model
        self.records: list[CallRecord] = []  # ordered list of all recorded calls
        self._pricing = PRICING.get(model, DEFAULT_PRICING)

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Compute the USD cost for a given number of input and output tokens.

        Parameters
        ──────────
        input_tokens  : int  — Prompt tokens consumed
        output_tokens : int  — Response tokens generated

        Returns
        ───────
        float  — Cost in USD, rounded to 6 decimal places for precision.
        """
        input_cost  = (input_tokens  / 1_000_000) * self._pricing["input_per_million"]
        output_cost = (output_tokens / 1_000_000) * self._pricing["output_per_million"]
        return round(input_cost + output_cost, 6)

    def record(self, step_name: str, message, latency_sec: float):
        """
        Record usage data from a completed Claude API call.

        Parameters
        ──────────
        step_name   : str    — Label for this call, e.g. "reflect: EU AI Act"
        message     : anthropic.types.Message
                              The Anthropic SDK response object. Token counts
                              are read from message.usage.input_tokens and
                              message.usage.output_tokens.
        latency_sec : float  — Elapsed seconds (use time.perf_counter() for precision).
        """
        input_tokens  = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost_usd      = self._compute_cost(input_tokens, output_tokens)

        self.records.append(CallRecord(
            step_name=step_name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_sec=round(latency_sec, 2),
        ))

    def total_input_tokens(self) -> int:
        """Return the sum of input tokens across all recorded calls."""
        return sum(r.input_tokens for r in self.records)

    def total_output_tokens(self) -> int:
        """Return the sum of output tokens across all recorded calls."""
        return sum(r.output_tokens for r in self.records)

    def total_cost_usd(self) -> float:
        """Return the total cost in USD across all recorded calls."""
        return round(sum(r.cost_usd for r in self.records), 6)

    def total_latency_sec(self) -> float:
        """Return the cumulative latency (sum, not wall-clock) across all calls."""
        return round(sum(r.latency_sec for r in self.records), 2)

    def print_summary(self):
        """
        Print a formatted rich table showing per-call and totals metrics.

        Columns: Step | Input Tokens | Output Tokens | Cost (USD) | Latency (s)
        A totals row is appended at the bottom in bold.
        """
        console.print()
        console.print(Rule("[bold yellow]📊 Run Metrics[/bold yellow]", style="yellow"))

        table = Table(border_style="yellow", show_header=True, show_footer=False)
        table.add_column("Step",           style="bold", width=38, no_wrap=True)
        table.add_column("Input Tok",      justify="right", width=11)
        table.add_column("Output Tok",     justify="right", width=11)
        table.add_column("Cost (USD)",     justify="right", width=11)
        table.add_column("Latency (s)",    justify="right", width=11)

        # One row per recorded Claude call
        for r in self.records:
            table.add_row(
                r.step_name[:36] + "…" if len(r.step_name) > 37 else r.step_name,
                f"{r.input_tokens:,}",
                f"{r.output_tokens:,}",
                f"${r.cost_usd:.5f}",
                f"{r.latency_sec:.2f}s",
            )

        # Totals row — visually separated with a rule style
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{self.total_input_tokens():,}[/bold]",
            f"[bold]{self.total_output_tokens():,}[/bold]",
            f"[bold green]${self.total_cost_usd():.5f}[/bold green]",
            f"[bold]{self.total_latency_sec():.2f}s[/bold]",
        )

        console.print(table)
        console.print()

    def to_dict(self) -> dict:
        """
        Serialise all metrics data to a plain dict suitable for JSON storage.

        Returns
        ───────
        dict with keys:
            model          : str    — Claude model used
            total_input_tokens  : int
            total_output_tokens : int
            total_cost_usd      : float
            total_latency_sec   : float
            calls               : list[dict] — one dict per recorded call
        """
        return {
            "model": self.model,
            "total_input_tokens":  self.total_input_tokens(),
            "total_output_tokens": self.total_output_tokens(),
            "total_cost_usd":      self.total_cost_usd(),
            "total_latency_sec":   self.total_latency_sec(),
            "calls": [
                {
                    "step":          r.step_name,
                    "input_tokens":  r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_usd":      r.cost_usd,
                    "latency_sec":   r.latency_sec,
                }
                for r in self.records
            ],
        }
