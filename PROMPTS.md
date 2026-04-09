# BharatIntel — LLM Prompt Documentation

This document lists all LLM prompts used in the BharatIntel pipeline, their purpose, design rationale, and the iterative decisions behind each.

All prompts are stored in `config/prompts.yaml` and referenced by key — **no prompts are hardcoded in Python files**.

---

## 1. Multi-Dimensional Article Scoring (`rank_multidim_v1`)

**Used by:** `agents/ranker/llm_scorer.py` → `RankerAgent`

**Purpose:** Score each article across four independent quality dimensions for ranking.

### System Prompt
```
You are an expert news editor scoring articles for a daily intelligence briefing.
Score each article across FOUR independent dimensions.
Be consistent: similar articles should get similar scores.
Score independently — a niche breakthrough can be high-novelty but low-impact.
```

### User Prompt (template)
```
Score each article across four dimensions for inclusion in a daily news briefing.

Articles:
{articles_block}

Dimensions (each scored 1-10):
- relevance:  How relevant to informed readers of a daily briefing
- impact:     How many people affected, how significantly
- novelty:    New/surprising information vs. routine/expected
- timeliness: Breaking/developing now vs. old/stale

Also assign each article a primary category.
Use "editors_picks" ONLY for stories of exceptional, cross-cutting significance
that do not fit neatly into the other five categories.
```

### Design Rationale

- **Why four dimensions instead of one score?** A single "relevance" score conflates importance, novelty, and freshness. An article about a routine GDP report is high-relevance but low-novelty. A niche scientific discovery is high-novelty but low-impact. Separating dimensions lets the composite ranker weight them independently.
- **Why scored 1-10?** Integer scales reduce LLM hallucination compared to floating-point. 1-10 is familiar to humans and LLMs alike.
- **Why "Score independently"?** Without this instruction, LLMs tend to correlate scores — a "breaking" article gets high scores across all dimensions. The independence instruction reduces this bias.
- **Why batch processing?** Scoring articles individually costs ~25 LLM calls. Batching (10 articles per call) reduces this to 2-3 calls with comparable quality, saving 80%+ on API costs.
- **Category assignment in scoring:** Having the LLM assign categories during scoring (rather than separately) saves an extra LLM call and leverages the context the model already has about each article's content.
- **Editor's Picks:** The prompt explicitly instructs the LLM to use `editors_picks` only for exceptional cross-cutting stories, preventing overuse of this catch-all category.

### Evolution
- **v0 (rank_article_v1):** Single article, single score. Expensive and inconsistent.
- **v1 (rank_batch_v1):** Batch scoring with single score. Cost-efficient but lost nuance.
- **v2 (rank_multidim_v1, current):** Multi-dimensional batch scoring. Best balance of cost, nuance, and consistency.

---

## 2. Section Summary Writer (`summarize_section_v2`)

**Used by:** `agents/summarizer/section_writer.py` → `SectionWriter`

**Purpose:** Generate a narrative summary and key takeaways for a single category section.

### System Prompt
```
You are a senior news editor writing a daily intelligence briefing.
Your writing is factual, concise, and informative.
You never speculate or editorialize. You synthesize multiple sources into a cohesive narrative.
Write in third-person, present tense.
You also distill key takeaways — short, actionable bullet points a busy reader can scan.
```

### User Prompt (template)
```
Write a briefing summary and key takeaways for the "{category}" section.

Source articles:
{articles_block}

Instructions:
SUMMARY:
- Synthesize the key developments into a coherent 3-5 sentence paragraph.
- Cover the most significant facts and developments across the articles.
- Do NOT list articles one by one. Weave them into a narrative.
- If articles are unrelated, cover each briefly but maintain flow.
- Do NOT use bullet points in the summary. Write prose only.
- Do NOT include source attributions in the summary text.

KEY TAKEAWAYS:
- Extract 2-4 key takeaways from the articles.
- Each takeaway should be a single, specific, factual statement (15-30 words).
- Focus on concrete facts, numbers, or decisions — not vague observations.
- Takeaways should be scannable by a busy reader.
```

### Design Rationale

- **"Synthesize, do not list"**: Early prompts produced listicle-style summaries ("Article 1 reports X. Article 2 reports Y."). The explicit narrative instruction forces genuine synthesis — connecting themes, noting patterns, and writing prose that reads like a real briefing.
- **"Third-person, present tense"**: Matches the convention of intelligence briefings and professional newsletters. Avoids the casual tone LLMs default to.
- **"Do NOT include source attributions"**: Sources are shown separately in the PDF. Embedding "according to Reuters" in the summary wastes precious words and clutters the narrative.
- **Key takeaways (15-30 words each)**: Designed for the target audience — a senior figure who may scan takeaways before reading the full summary. Word count limits prevent vague statements like "tensions remain high."
- **"Concrete facts, numbers, or decisions"**: Without this constraint, LLMs produce generic takeaways. This forces specificity: "NVIDIA data center revenue up 154% YoY" rather than "Tech stocks had a good day."

### Quality Gates
The section writer enforces a minimum summary length (80 chars, configurable) and validates that takeaways are non-empty strings. Failures trigger a three-tier fallback:
1. Retry with the same prompt
2. Fall back to `summarize_section_v1` (simpler prompt without takeaways)
3. Generate a minimal section from article titles

---

## 3. Headline Writer (`write_headlines_v1`)

**Used by:** `agents/summarizer/headline_writer.py` → `HeadlineWriter`

**Purpose:** Generate the top N headlines for the briefing masthead.

### System Prompt
```
You are a senior news editor crafting the top headlines for a daily intelligence briefing.
Headlines must be punchy, accurate, and informative.
Each headline gets a single-sentence context line.
```

### User Prompt (template)
```
Generate the top {count} headlines for today's briefing from the following articles.

Articles (ranked by importance):
{articles_block}

Instructions:
- Pick the {count} most significant stories.
- Rewrite each headline to be concise and impactful (max 15 words).
- Write one sentence of context for each (the "oneliner").
- Preserve the original article URL and source name exactly.
- Order by importance (most important first).
```

### Design Rationale

- **"Rewrite each headline"**: Original headlines from news sources are often clickbait, overly long, or lack context. Rewriting ensures the briefing has a consistent, professional voice.
- **"Max 15 words"**: Forces brevity. Original headlines from wire services can be 30+ words. A 15-word limit ensures scannability.
- **"Preserve URL and source name exactly"**: Critical for attribution integrity. The LLM must not hallucinate URLs or misattribute sources.
- **"Order by importance"**: The LLM sees articles pre-sorted by rank score, but has freedom to reorder based on narrative judgment (e.g., grouping related stories).

---

## 4. Executive Summary (`executive_summary_v1`)

**Used by:** `agents/summarizer/executive_writer.py` → `ExecutiveWriter`

**Purpose:** Generate the top-level overview that opens the briefing — a bird's-eye view of the day.

### System Prompt
```
You are a senior intelligence analyst writing the executive overview for a daily news briefing.
Your overview provides a birds-eye view of the day's most important developments.
You write with authority, clarity, and brevity. No filler, no speculation.
You identify the 3-5 stories that matter most and explain why they matter together.
```

### User Prompt (template)
```
Write the executive summary for today's daily intelligence briefing.

The briefing contains {section_count} thematic sections and {headline_count} top headlines.

Section summaries:
{sections_block}

Top headlines:
{headlines_block}

Instructions:
OVERVIEW:
- Write a 2-3 paragraph overview of the day's most significant developments.
- Start with the single biggest story of the day.
- Draw connections between storylines where they exist.
- Give the reader a clear sense of "what happened today and why it matters."
- Write in third-person, present tense. No editorializing.

KEY DEVELOPMENTS:
- List 3-5 key developments across ALL categories.
- Each development: one specific sentence (15-25 words).

TONE:
- Assess the overall tone of today's news in 1-2 words
  (e.g. "cautiously optimistic", "volatile", "mixed", "stable", "escalatory").
```

### Design Rationale

- **"Senior intelligence analyst" persona**: This is the most important piece of text in the briefing. The persona shift from "news editor" (sections) to "intelligence analyst" (executive) produces more strategic, connective writing.
- **"Start with the single biggest story"**: Prevents the LLM from opening with a generic "Today was a busy day in global affairs" preamble. Forces it to lead with substance.
- **"Draw connections between storylines"**: The unique value of an executive summary over individual sections. If geopolitical tensions are affecting markets, this is where that link is made explicit.
- **Tone assessment**: A one-word tone descriptor (displayed as a badge in the PDF) gives the reader an instant emotional read of the day's news before they dive into details.
- **Input includes both sections AND headlines**: The executive writer receives the already-generated section summaries and headlines, so it synthesizes a meta-narrative rather than re-reading raw articles. This is more token-efficient and produces better cross-section connections.

---

## 5. Legacy Prompts (retained for fallback)

### `rank_article_v1` — Single Article Scoring
Used as a reference; replaced by `rank_multidim_v1` for production. Retained because the section writer's three-tier fallback can invoke `summarize_section_v1` if v2 fails.

### `rank_batch_v1` — Batch Single-Score Ranking
Intermediate evolution step. Retained for backward compatibility.

### `summarize_section_v1` — Summary-Only Section Writer
Simpler prompt without key takeaways. Used as fallback tier 2 if `summarize_section_v2` fails.

---

## Prompt Design Principles

1. **Structured JSON responses**: Every prompt requests JSON output with a defined schema. This enables programmatic parsing and validation — no regex extraction needed.
2. **Explicit constraints**: Each instruction tells the LLM what NOT to do (no bullet points in summaries, no source attributions, no editorializing). LLMs follow negative constraints more reliably than positive ones.
3. **Versioned naming**: Prompts are named `_v1`, `_v2` etc. This allows A/B testing and graceful rollback without code changes — just update the reference in the agent.
4. **Centralized storage**: All prompts live in `config/prompts.yaml`. No prompt text exists in Python code. This makes prompt iteration fast (edit YAML, re-run) and audit-friendly.
5. **Audience awareness**: The target reader is "a senior scholarly figure whose work spans international diplomacy, Bharatiya Darshana, legal discourse, and civilizational thought." This informed the intelligence-analyst persona, the emphasis on strategic connections, and the inclusion of the Civilisation category.
