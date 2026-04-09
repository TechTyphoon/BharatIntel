const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

export async function getLatestBriefing() {
  const res = await fetch(`${API_BASE}/get-latest`);
  if (!res.ok) {
    if (res.status === 404) return null;
    throw new Error(`Failed to fetch briefing: ${res.status}`);
  }
  return res.json();
}

export async function generateBrief() {
  const res = await fetch(`${API_BASE}/generate-brief`, { method: "POST" });
  if (res.status === 409) {
    throw new Error("Pipeline is already running.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Pipeline failed: ${res.status}`);
  }

  // Pipeline started — poll status until completion
  while (true) {
    await new Promise((r) => setTimeout(r, 3000));
    const st = await getStatus();
    if (!st.pipeline_running) {
      if (st.last_error) {
        throw new Error(st.last_error);
      }
      const result = st.last_result || {};
      if (result.status === "partial") {
        return { ...result, _partial: true };
      }
      return result.status ? result : { status: "success", message: "Briefing generated successfully." };
    }
  }
}

export async function getStatus() {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error("Failed to fetch status");
  return res.json();
}

export function getPdfUrl() {
  return `${API_BASE}/download-pdf`;
}

// ── Settings / API Key Management ──────────────────────────────────

export async function getSettings() {
  const res = await fetch(`${API_BASE}/settings`);
  if (!res.ok) throw new Error("Failed to fetch settings");
  return res.json();
}

export async function saveSettings(keys) {
  const res = await fetch(`${API_BASE}/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ keys }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Failed to save settings");
  }
  return res.json();
}

export async function validateKey(provider, key) {
  const res = await fetch(`${API_BASE}/settings/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, key }),
  });
  if (!res.ok) throw new Error("Validation request failed");
  return res.json();
}
