"""
BharatIntel — Local LLM Post-Processor

Enhances pipeline output by replacing snippet-based fallback summaries
with locally-generated LLM prose using a small HuggingFace model,
then regenerates PDF/HTML via the publisher.

Usage:
    python scripts/enhance_briefing.py [output/briefing_YYYY-MM-DD.json]

Requires: transformers, accelerate, torch
"""

from __future__ import annotations

import json
import time
import sys
import os
from datetime import date
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


def load_model():
    """Load SmolLM2-360M-Instruct for local generation."""
    from transformers import pipeline as hf_pipeline

    print(f"[*] Loading local LLM ({MODEL_NAME})...")
    t = time.time()
    gen = hf_pipeline(
        "text-generation",
        model=MODEL_NAME,
        device="cpu",
    )
    print(f"[*] Model loaded in {time.time()-t:.1f}s")
    return gen


def generate(gen, prompt: str, max_tokens: int = 250) -> str:
    """Generate text from a prompt using the local model."""
    messages = [{"role": "user", "content": prompt}]
    output = gen(messages, max_new_tokens=max_tokens, temperature=0.4, do_sample=True)
    return output[0]["generated_text"][-1]["content"].strip()


# ── Section summary prompt ───────────────────────────────────────────
def build_section_prompt(category: str, titles: list[str]) -> str:
    """Build a summarization prompt for a section using article titles."""
    bullet_list = "\n".join(f"- {t}" for t in titles[:8])
    return (
        f"You are a senior news analyst. Write a concise 3-5 sentence summary "
        f"for the \"{category}\" section of today's intelligence briefing.\n\n"
        f"Today's articles in this section:\n{bullet_list}\n\n"
        f"Synthesize the key themes and developments into a coherent paragraph. "
        f"Do NOT list articles. Write analytical prose only."
    )


# ── Key takeaways prompt ────────────────────────────────────────────
def build_takeaway_prompt(summary: str) -> str:
    return (
        f"Based on this summary, list exactly 3 key takeaways. "
        f"Each must be one concise sentence starting with a dash (-).\n\n"
        f"Summary: {summary}\n\n"
        f"Takeaways:"
    )


# ── Headline oneliner prompt ────────────────────────────────────────
def build_oneliner_prompt(title: str) -> str:
    return f"Write a single-sentence summary (max 20 words) for this headline: {title}"


# ── Executive summary prompt ────────────────────────────────────────
def build_executive_prompt(section_summaries: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"- {cat}: {summ[:200]}" for cat, summ in section_summaries)
    return (
        f"You are writing the executive summary for today's daily intelligence briefing. "
        f"Based on these section summaries, write a 3-4 sentence overview.\n\n"
        f"Sections:\n{lines}\n\n"
        f"Write a concise executive summary paragraph. Be factual and direct."
    )


def enhance_briefing(input_path: str):
    """Enhance a pipeline-generated briefing with local LLM summaries."""
    with open(input_path, "r") as f:
        briefing = json.load(f)

    date_str = briefing.get("date", "unknown")
    sections = briefing.get("sections", [])
    headlines = briefing.get("headlines", [])

    print(f"[*] Loaded briefing: {input_path}")
    print(f"    Date: {date_str}")
    print(f"    Sections: {len(sections)}, Headlines: {len(headlines)}")

    gen = load_model()
    total_t = time.time()

    # ── 1. Enhance section summaries ─────────────────────────────────
    for i, section in enumerate(sections):
        category = section.get("category", "Unknown")
        titles = section.get("article_titles", [])

        print(f"\n[{i+1}/{len(sections)}] Section: {category} ({len(titles)} articles)...")
        t = time.time()
        summary = generate(gen, build_section_prompt(category, titles), max_tokens=200)
        print(f"    Summary ({time.time()-t:.1f}s): {summary[:120]}...")
        section["summary"] = summary

        # Key takeaways
        t = time.time()
        takeaway_text = generate(gen, build_takeaway_prompt(summary), max_tokens=150)
        takeaways = [
            line.strip().lstrip("- ").strip()
            for line in takeaway_text.strip().split("\n")
            if line.strip().startswith("-")
        ]
        if len(takeaways) >= 2:
            section["key_takeaways"] = takeaways[:3]
            print(f"    Takeaways ({time.time()-t:.1f}s): {len(section['key_takeaways'])} items")

    # ── 2. Enhance headline oneliners ────────────────────────────────
    print(f"\n[*] Enhancing {len(headlines)} headline oneliners...")
    for j, h in enumerate(headlines):
        t = time.time()
        h["oneliner"] = generate(gen, build_oneliner_prompt(h["title"]), max_tokens=40)
        print(f"    [{j+1}] ({time.time()-t:.1f}s) {h['oneliner'][:80]}")

    # ── 3. Generate executive summary ────────────────────────────────
    print(f"\n[*] Generating executive summary...")
    section_summaries = [(s["category"], s["summary"]) for s in sections]
    t = time.time()
    exec_overview = generate(gen, build_executive_prompt(section_summaries), max_tokens=200)
    print(f"    ({time.time()-t:.1f}s) {exec_overview[:120]}...")

    briefing["executive_summary"] = {
        "overview": exec_overview,
        "key_developments": [h["title"] for h in headlines[:5]],
        "tone": "analytical",
    }

    briefing["token_usage"] = {
        "model": f"{MODEL_NAME} (local CPU)",
        "note": "Enhanced with local LLM post-processing",
    }

    # ── 4. Save enhanced JSON ────────────────────────────────────────
    with open(input_path, "w") as f:
        json.dump(briefing, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Enhanced JSON saved ({time.time()-total_t:.1f}s total)")

    # ── 5. Regenerate PDF and HTML ───────────────────────────────────
    print("\n[*] Regenerating PDF and HTML...")
    try:
        from agents.publisher.agent import PublisherAgent
        from agents.summarizer.models import (
            Briefing as BriefingModel,
            BriefingSection,
            ExecutiveSummary,
            Headline as HeadlineModel,
        )

        b = BriefingModel(
            date=briefing["date"],
            executive_summary=ExecutiveSummary(**briefing["executive_summary"]),
            headlines=[HeadlineModel(**h) for h in briefing["headlines"]],
            sections=[BriefingSection(**s) for s in briefing["sections"]],
            token_usage=briefing.get("token_usage", {}),
        )

        agent = PublisherAgent()
        result = agent.publish(b)
        print(f"[✓] PDF: {result.pdf_path}")
        print(f"[✓] HTML: {result.html_path}")
    except Exception as exc:
        print(f"[!] PDF/HTML regeneration failed: {exc}")
        print("    JSON briefing was still saved successfully.")

    return briefing


if __name__ == "__main__":
    today = date.today().isoformat()
    default_input = f"output/briefing_{today}.json"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    elif os.path.exists(default_input):
        input_file = default_input
    else:
        output_dir = Path("output")
        jsons = sorted(output_dir.glob("briefing_*.json"))
        if jsons:
            input_file = str(jsons[-1])
        else:
            print("No briefing JSON found in output/")
            sys.exit(1)

    enhance_briefing(input_file)
