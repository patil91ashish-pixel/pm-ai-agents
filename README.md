# 🤖 PM AI Agents

> A Product Manager learning to build AI agents — one real problem at a time.

I'm Ashish, a PM at foundit. This repo is my public build log.

The goal isn't to show polished finished products. It's to document what I'm learning, what broke, what surprised me, and what I'd do differently — through working code.

Every agent here solves a workflow problem I actually have.

---

## 🗂️ Agents

| Agent | What it does | Status |
|-------|-------------|--------|
| [linkedin-agent](./linkedin_agent/) | Reads AI news → reflects on PM relevance → generates LinkedIn posts in your voice | ✅ Live |

---

## 🧱 How Each Agent Is Built

Each agent follows the same design principles:

- **Modular** — Each step is a separate Python module, independently testable
- **Evaluated** — Every agent includes a functional eval suite + LLM-as-judge quality scoring
- **Observable** — Token usage, cost, and latency tracked per API call
- **Personalised** — Voice profile system so output sounds like you, not a generic AI

---

## 📖 Part of My "Learning with AI" Series

I write about what I'm building on LinkedIn. If you find this useful, follow along:

**[linkedin.com/in/ashish-patil-pm](https://www.linkedin.com/in/ashish-patil-pm)**

---

## 🚀 Get Started

Each agent has its own `README.md` with setup instructions. Start with the LinkedIn agent:

```bash
cd linkedin_agent
cp .env.example .env
# Add your API keys to .env
pip install -r requirements.txt
python3 agent.py
```

---

## 📄 License

MIT — fork it, build on it, make it your own.
