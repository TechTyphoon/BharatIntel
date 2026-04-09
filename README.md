# BharatIntel — AI-Powered Daily Intelligence Briefing System

An automated, end-to-end pipeline that discovers, ingests, filters, ranks, summarises, and publishes a professional daily news briefing as a PDF — without manual intervention.

Includes a **React web dashboard** with a settings panel where evaluators can plug in their own API keys and generate briefings on-demand.

Built for the **Jyot India Foundation** AI Generalist Technical Assignment.

> **[📄 View Sample Briefing PDF](output/sample/briefing_2026-04-09.pdf)**

---

## System Architecture

BharatIntel follows an **agent-based modular architecture** where each stage of the pipeline is an independent, testable agent with a single responsibility:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Collector   │───▶│   Curator   │───▶│ Summarizer  │───▶│  Publisher   │
│  (11 sources)│    │ (dedup+rank)│    │ (LLM write) │    │ (PDF/HTML)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      RSS              Semantic           Section             ReportLab
      API (GNews)      Dedup              Writers             PDF gen
      Web Scraper      Multi-dim          Headlines           HTML/Jinja2
                       LLM Ranking        Executive           JSON archive
                       Diversity           Summary
                       Reranking
```

**Pipeline Orchestrator** wires the four agents sequentially with per-stage error isolation, timing, and diagnostics. **APScheduler** provides cron-based daily execution.

### Module Breakdown

| Module | Purpose |
|--------|---------|
| `core/` | Shared foundations — logging, exceptions, retry, unified LLM client |
| `agents/collector/` | Parallel news fetching from RSS, APIs, web scrapers |
| `agents/curator/` | Semantic deduplication via sentence-transformers |
| `agents/ranker/` | Multi-dimensional LLM scoring + heuristic signals + diversity reranking |
| `agents/summarizer/` | LLM-powered section narratives, headlines, executive summary |
| `agents/publisher/` | PDF (ReportLab), HTML (Jinja2), JSON archival |
| `pipeline/` | Orchestrator + APScheduler cron |
| `config/` | Settings, sources, and LLM prompts (YAML) |

---

## News Categories

The system covers the **six required categories**:

1. **Geopolitics & International Affairs** — Diplomatic events, conflicts, sanctions, UN developments
2. **Technology & Artificial Intelligence** — AI/ML breakthroughs, tech policy, cybersecurity
3. **Indian National Politics & Governance** — Parliamentary developments, Supreme Court rulings, policy shifts
4. **Science & New Discoveries** — Peer-reviewed breakthroughs, space, medical discoveries
5. **Civilisation, Culture & Society** — Heritage, cultural events, interfaith dialogue, India's civilisational standing
6. **Editor's Picks** — LLM autonomously surfaces exceptional stories that cross category boundaries

---

## Setup Instructions

### Prerequisites

- Python 3.10+ (tested on 3.12)
- API keys for at least one LLM provider and GNews

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd BharatIntel

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **API Keys** — Copy the template and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your keys:
#   GNEWS_API_KEY=<your-gnews-key>      (free at gnews.io)
#   XAI_API_KEY=<your-xai-key>          (for Grok ranking)
#   GEMINI_API_KEY=<your-gemini-key>    (for Gemini summarization)
```

2. **LLM Models** — Set in `.env`:
   - `LLM_RANK_MODEL` — Fast model for article ranking (default: `gemini/gemini-2.5-flash`)
   - `LLM_SUMMARY_MODEL` — Quality model for summarization (default: `gemini/gemini-2.5-flash`)

3. **Sources** — Edit `config/sources.yaml` to add/remove news sources
4. **Prompts** — All LLM prompts are in `config/prompts.yaml` (versioned, documented)

### Running

```bash
# Single run (fetch → rank → summarize → PDF)
python main.py

# Scheduled daily run (cron: 6 AM daily, configurable in settings.yaml)
python main.py --schedule
```

Output is written to `output/`:
- `briefing_YYYY-MM-DD.pdf` — The daily briefing
- `briefing_YYYY-MM-DD.html` — HTML version
- `briefing_YYYY-MM-DD.json` — Machine-readable archive

### Web Dashboard (Optional)

```bash
# Terminal 1: Start backend API
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173` to access the dashboard. Click the ⚙ gear icon to configure API keys, then click **Generate Brief** to run the pipeline from the browser.

---

## Design Decisions

### Why litellm for LLM integration?
litellm provides a unified interface across 100+ LLM providers (OpenAI, Anthropic, Google Gemini, xAI Grok, open-source via Ollama). This means switching models requires changing one environment variable — no code changes. We use a two-tier strategy: a fast/cheap model for ranking (Grok) and a quality model for summarization (Gemini).

### Why agent-based architecture?
Each agent (Collector, Curator/Ranker, Summarizer, Publisher) is independently testable, configurable, and replaceable. The pipeline orchestrator simply wires them together. This makes the system robust to partial failures — if one source fails, others still contribute.

### Why multi-dimensional ranking?
A single relevance score conflates importance, novelty, and timeliness. Our ranker scores articles across 4 independent LLM dimensions + 3 heuristic signals, then applies a weighted composite with diversity-aware reranking to prevent category domination.

### Why ReportLab for PDF?
ReportLab is a pure-Python PDF generator with no system-level dependencies (unlike WeasyPrint which needs GTK/Cairo). This makes deployment and CI/CD simpler. The system supports both backends via a config switch.

### Source selection rationale
Sources span Reuters, BBC, Al Jazeera (global), The Hindu, NDTV (India), Nature, Science Daily (science), GNews API (aggregation), and Hacker News (tech community) — chosen for authority, breadth, and India-awareness.

---

## Challenges Encountered

1. **Rate limiting** — Batched LLM calls with configurable batch sizes to stay within provider rate limits.
2. **Category balance** — Without diversity reranking, popular categories dominated. The diversity penalty ensures fair representation.
3. **Token budget management** — The summarizer tracks cumulative token usage and gracefully truncates sections if exhausted.
4. **Lazy imports** — Optional dependencies (ReportLab, WeasyPrint) are lazily imported so the system doesn't crash at startup.

---

## Key Dependencies

`litellm` (unified LLM API), `httpx` (async HTTP), `feedparser` (RSS), `sentence-transformers` (semantic dedup), `reportlab` (PDF), `structlog` (logging), `apscheduler` (scheduling), `jinja2` (HTML). Full list in `requirements.txt`.

---

## Project Structure

```
BharatIntel/
├── main.py                    # CLI entry point
├── config/                    # settings.yaml, sources.yaml, prompts.yaml
├── core/                      # LLM client, logger, retry, exceptions
├── agents/
│   ├── collector/             # RSS, API, web scraper sources
│   ├── curator/               # Semantic dedup (sentence-transformers)
│   ├── ranker/                # Multi-dim LLM scoring + composite
│   ├── summarizer/            # Section/headline/executive writers
│   └── publisher/             # PDF (ReportLab), HTML, JSON
├── pipeline/                  # Orchestrator + scheduler
├── api/server.py              # FastAPI REST API + settings
├── frontend/                  # React + Vite dashboard
└── output/sample/             # Sample briefing PDF
```
