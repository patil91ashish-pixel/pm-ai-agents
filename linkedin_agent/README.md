# 🤖 LinkedIn Post Generator Agent

> **Part of [pm-ai-agents](../README.md) — a PM building AI agents in public.**

A fully autonomous agent that monitors AI news, scores articles by PM relevance, and generates LinkedIn posts written **in your personal voice**.

Built by a Product Manager learning to code agents — not a polished product, but a documented learning journey.

---

## ✨ What It Does

```
AI News (NewsAPI)
    ↓
PM Relevance Scoring (Claude)    ← scores each article 1–10 for PM relevance
    ↓
Deduplication Memory             ← skips articles you've already posted about
    ↓
LinkedIn Post Generation (Claude) ← writes in YOUR voice using your style profile
    ↓
Save to output/ + History
```

At the end of every run, a metrics table shows token usage, cost, and latency per step.

---

## 🗂️ Project Structure

```
linkedin_agent/
├── agent.py              # Main runner — ties the pipeline together
├── config.py             # All settings in one place
├── news_fetcher.py       # Fetches AI headlines from NewsAPI (+ fallback)
├── pm_reflection.py      # Claude scores each article for PM relevance
├── post_generator.py     # Claude writes the LinkedIn post
├── post_history.py       # Deduplication memory (avoids repeating stories)
├── metrics.py            # Token, cost, and latency tracking
│
├── MY_STYLE.example.md   # ← Copy to MY_STYLE.md and fill in YOUR style rules
├── my_posts.example.md   # ← Copy to my_posts.md and paste YOUR real posts
│
├── samples/              # Before/after: generic vs personalised post quality
│   ├── README.md
│   ├── before_generic_prompt.txt
│   └── after_personalised_prompt.txt
│
├── evals/
│   ├── eval_runner.py        # Main eval orchestrator (run this)
│   ├── eval_functional.py    # 7 structural checks (schema, completeness, etc.)
│   ├── eval_scorer.py        # LLM-as-judge quality scoring (5 dimensions)
│   └── test_articles.json    # 8 fixed test articles for reproducible evals
│
├── .env.example          # Template — copy to .env and add your keys
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/patil91ashish-pixel/pm-ai-agents.git
cd pm-ai-agents/linkedin_agent
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and add:
```
ANTHROPIC_API_KEY=your_key_here       # Required — get from console.anthropic.com
NEWS_API_KEY=your_key_here            # Optional — get from newsapi.org (free tier works)
```

### 3. Set up YOUR voice (the most important step)

```bash
cp MY_STYLE.example.md MY_STYLE.md
cp my_posts.example.md my_posts.md
```

- Edit `MY_STYLE.md` — describe how you write (tone, structure, what to avoid)
- Edit `my_posts.md` — paste 2–3 of your best real LinkedIn posts as examples

**This is what makes the output sound like you and not a generic AI.**
See `samples/` for a real before/after comparison of the difference this makes.

### 4. Run the agent

```bash
python3 agent.py
```

---

## 🧪 Running Evals

```bash
python3 evals/eval_runner.py
```

This runs:
- **7 functional checks** — schema validation, completeness, hashtags, word count, etc.
- **LLM-as-judge quality scoring** — 5 dimensions (PM relevance, hook, actionability, tone, hashtag quality)
- **Regression diff** — compares against the previous eval run

Latest result: **7/7 checks PASS, avg quality 8.07/10**

---

## 📊 Sample Metrics Output

```
 📊 Run Metrics
┌──────────────────────────┬────────────┬────────────┬───────────┬───────────┐
│ Step                     │ Input Tok  │ Output Tok │ Cost(USD) │ Latency   │
├──────────────────────────┼────────────┼────────────┼───────────┼───────────┤
│ reflect: OpenAI GPT-5…   │ 258        │ 247        │ $0.00448  │ 8.91s     │
│ reflect: EU AI Act…      │ 282        │ 246        │ $0.00448  │ 7.30s     │
│ generate_post            │ 4,134      │ 1,244      │ $0.03106  │ 26.38s    │
├──────────────────────────┼────────────┼────────────┼───────────┼───────────┤
│ TOTAL                    │ 6,013      │ 2,756      │ $0.05855  │ 71.60s    │
└──────────────────────────┴────────────┴────────────┴───────────┴───────────┘
```

Real cost: ~$0.05–0.07 per full run with Claude Sonnet.

---

## ⚙️ Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CLAUDE_MODEL` | `claude-sonnet-4-5` | Swap for haiku (cheaper) or opus (better) |
| `NEWS_MAX_ARTICLES` | `8` | How many articles to evaluate per run |
| `NEWS_HOURS_BACK` | `48` | How far back to search for news |
| `POST_MAX_WORDS` | `300` | Target post length |
| `SIMILARITY_THRESHOLD` | `0.5` | Jaccard threshold for dedup (in post_history.py) |

---

## 🧠 What I Learned Building This

- `max_tokens=800` truncated posts silently after I added voice personalisation (prompt grew from ~550 to ~4K tokens). Fixed to 1500. The eval suite now catches this with a `post_completeness` check.
- Deduplication with Jaccard title similarity (threshold 0.5) correctly filters the same story from different sources without over-filtering.
- LLM-as-judge evals are powerful but the rubric must match your actual output format — my hashtag scores jumped from 1/10 → 7.75/10 just from fixing the max_tokens bug.

---

## 📝 License

MIT
