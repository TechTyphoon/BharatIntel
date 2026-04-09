"""
BharatIntel — FastAPI Server

Exposes the news briefing pipeline over HTTP:
  GET  /api/get-latest        → latest briefing JSON
  POST /api/generate-brief    → trigger pipeline run
  GET  /api/download-pdf      → download latest PDF
  GET  /api/status            → pipeline run status
  GET  /api/settings          → get current API key config
  POST /api/settings          → update API keys
  POST /api/settings/validate → test if a key works
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import date
from glob import glob
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="BharatIntel API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = PROJECT_ROOT / "output"
ENV_PATH = PROJECT_ROOT / ".env"

# Simple in-memory state for pipeline status
_pipeline_state = {"running": False, "last_error": None, "last_run": None}

# Keys we allow users to configure via the UI
ALLOWED_KEYS = {
    "GEMINI_API_KEY": "Google Gemini",
    "XAI_API_KEY": "xAI (Grok)",
    "OPENROUTER_API_KEY": "OpenRouter",
    "GROQ_API_KEY": "Groq",
    "OPENAI_API_KEY": "OpenAI",
    "ANTHROPIC_API_KEY": "Anthropic",
}


def _find_latest_briefing(ext: str = ".json") -> Path | None:
    """Find the most recent briefing file by date in filename."""
    pattern = str(OUTPUT_DIR / f"briefing_*{ext}")
    files = sorted(glob(pattern))
    return Path(files[-1]) if files else None


def _mask_key(key: str) -> str:
    """Show first 8 and last 4 chars of an API key, mask the rest."""
    if len(key) <= 12:
        return "****"
    return key[:8] + "****" + key[-4:]


def _read_env_keys() -> dict[str, str]:
    """Read current API keys from environment (masked for display)."""
    result = {}
    for key_name, label in ALLOWED_KEYS.items():
        val = os.environ.get(key_name, "")
        result[key_name] = {
            "label": label,
            "is_set": bool(val),
            "masked_value": _mask_key(val) if val else "",
        }
    return result


def _update_env_file(updates: dict[str, str]) -> None:
    """Update .env file with new key values, preserving other content."""
    lines = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text().splitlines()

    # Track which keys we've updated
    updated_keys = set()

    new_lines = []
    for line in lines:
        # Check if this line sets one of our keys
        matched = False
        for key_name in updates:
            if re.match(rf"^{re.escape(key_name)}\s*=", line):
                new_lines.append(f"{key_name}={updates[key_name]}")
                updated_keys.add(key_name)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    # Append any keys not already in the file
    for key_name, value in updates.items():
        if key_name not in updated_keys:
            new_lines.append(f"{key_name}={value}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n")

    # Also update current process environment
    for key_name, value in updates.items():
        os.environ[key_name] = value


# ── Pydantic models ─────────────────────────────────────────────────

class SettingsUpdate(BaseModel):
    keys: dict[str, str]  # { "GEMINI_API_KEY": "AIza...", ... }


class ValidateKeyRequest(BaseModel):
    provider: str  # e.g. "GEMINI_API_KEY"
    key: str


# ── Endpoints ─────────────────────────────────────────────────────────


@app.get("/api/get-latest")
async def get_latest():
    """Return the latest briefing JSON."""
    path = _find_latest_briefing(".json")
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="No briefing found. Run the pipeline first.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.post("/api/generate-brief")
async def generate_brief():
    """Trigger a full pipeline run (collect → curate → summarize → publish)."""
    if _pipeline_state["running"]:
        raise HTTPException(status_code=409, detail="Pipeline is already running.")

    _pipeline_state["running"] = True
    _pipeline_state["last_error"] = None

    try:
        from core.logger import setup_logging

        setup_logging()

        from pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        result = await orch.run()

        _pipeline_state["last_run"] = date.today().isoformat()

        # Check if LLM errors occurred (rate limits, auth failures)
        details = result.to_dict()
        llm_warning = _detect_llm_issues(details)

        if not result.success:
            _pipeline_state["last_error"] = result.error
            msg = result.error
            if llm_warning:
                msg += f" | {llm_warning}"
            return JSONResponse(
                status_code=207,
                content={
                    "status": "partial",
                    "message": msg,
                    "details": details,
                },
            )

        msg = f"Briefing generated for {date.today().isoformat()}"
        if llm_warning:
            msg += f" | {llm_warning}"

        return {
            "status": "success",
            "message": msg,
            "details": details,
        }
    except Exception as exc:
        _pipeline_state["last_error"] = str(exc)
        err_str = str(exc).lower()
        if "rate" in err_str or "limit" in err_str or "429" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=500,
                detail="API keys are rate-limited. Please wait a few minutes or add your own API key via the ⚙ Settings button.",
            )
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _pipeline_state["running"] = False


def _detect_llm_issues(details: dict) -> str:
    """Scan pipeline result for LLM-related warnings."""
    detail_str = json.dumps(details).lower()
    issues = []
    if "fallback" in detail_str or "all_models_failed" in detail_str:
        issues.append("Some sections used fallback content due to API limits")
    if "rate_limit" in detail_str or "429" in detail_str or "quota" in detail_str:
        issues.append("API keys hit rate limits")
    if issues:
        return ". ".join(issues) + ". Click ⚙ Settings to add your own API key for best results."
    return ""


@app.get("/api/download-pdf")
async def download_pdf():
    """Download the latest generated PDF."""
    path = _find_latest_briefing(".pdf")
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="No PDF found. Run the pipeline first.")

    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
    )


@app.get("/api/status")
async def status():
    """Return current pipeline status."""
    latest = _find_latest_briefing(".json")
    return {
        "pipeline_running": _pipeline_state["running"],
        "last_run": _pipeline_state["last_run"],
        "last_error": _pipeline_state["last_error"],
        "latest_briefing": latest.name if latest else None,
    }


# ── Settings / API-Key Management ────────────────────────────────────


@app.get("/api/settings")
async def get_settings():
    """Return current API key configuration (values masked)."""
    return {"keys": _read_env_keys()}


@app.post("/api/settings")
async def update_settings(body: SettingsUpdate):
    """Persist updated API keys to .env and process environment."""
    # Validate: only accept known key names
    for key_name in body.keys:
        if key_name not in ALLOWED_KEYS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown key: {key_name}. Allowed: {list(ALLOWED_KEYS.keys())}",
            )

    _update_env_file(body.keys)

    # Auto-select the best LLM model based on newly available keys
    from core.llm_client import auto_select_model
    best_model = auto_select_model()
    os.environ["LLM_SUMMARY_MODEL"] = best_model
    os.environ["LLM_RANK_MODEL"] = best_model

    return {"status": "saved", "keys": _read_env_keys()}


@app.post("/api/settings/validate")
async def validate_key(body: ValidateKeyRequest):
    """Quick-test an API key by making a minimal LLM call."""
    if body.provider not in ALLOWED_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {body.provider}")

    try:
        import litellm

        model_map = {
            "GEMINI_API_KEY": ("gemini/gemini-2.5-flash", "GEMINI_API_KEY"),
            "XAI_API_KEY": ("xai/grok-3-mini", "XAI_API_KEY"),
            "OPENROUTER_API_KEY": ("openrouter/google/gemma-3-4b-it:free", "OPENROUTER_API_KEY"),
            "GROQ_API_KEY": ("groq/llama-3.1-8b-instant", "GROQ_API_KEY"),
            "OPENAI_API_KEY": ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": ("anthropic/claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
        }

        model, env_var = model_map.get(body.provider, (None, None))
        if not model:
            return {"valid": False, "error": "No test model configured for this provider"}

        # Temporarily set the key for testing
        old_val = os.environ.get(env_var, "")
        os.environ[env_var] = body.key

        try:
            resp = await asyncio.wait_for(
                litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": "Say OK"}],
                    max_tokens=5,
                    temperature=0,
                ),
                timeout=15,
            )
            return {"valid": True, "message": f"{ALLOWED_KEYS[body.provider]} key is working!"}
        except asyncio.TimeoutError:
            return {"valid": False, "error": "Request timed out — try again"}
        except Exception as e:
            err_str = str(e).lower()
            label = ALLOWED_KEYS[body.provider]

            # Rate limit = key is valid, just quota exhausted
            if "rate" in err_str and "limit" in err_str or "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                return {"valid": True, "message": f"{label} key is valid! (quota currently exhausted — will work when quota resets)"}

            # Auth errors = key is invalid
            if "401" in err_str or "403" in err_str or "invalid" in err_str or "unauthorized" in err_str or "authentication" in err_str:
                return {"valid": False, "error": f"Invalid {label} key — check and try again"}

            # Payment / billing errors
            if "402" in err_str or "billing" in err_str or "payment" in err_str:
                return {"valid": False, "error": f"{label} key has no credits — add billing"}

            # Generic — extract a short message
            short = str(e).split("\n")[0][:120]
            return {"valid": False, "error": short}
        finally:
            # Restore original value
            if old_val:
                os.environ[env_var] = old_val
            else:
                os.environ.pop(env_var, None)

    except ImportError:
        return {"valid": False, "error": "litellm not installed"}
