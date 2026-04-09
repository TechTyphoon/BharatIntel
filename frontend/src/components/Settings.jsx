import { useState, useEffect } from "react";
import { getSettings, saveSettings, validateKey } from "../api";

const PROVIDER_INFO = {
  GEMINI_API_KEY: {
    label: "Google Gemini",
    icon: "✦",
    placeholder: "AIzaSy...",
    helpUrl: "https://aistudio.google.com/apikey",
  },
  XAI_API_KEY: {
    label: "xAI (Grok)",
    icon: "𝕏",
    placeholder: "xai-...",
    helpUrl: "https://console.x.ai/",
  },
  OPENROUTER_API_KEY: {
    label: "OpenRouter",
    icon: "⚡",
    placeholder: "sk-or-v1-...",
    helpUrl: "https://openrouter.ai/keys",
  },
  GROQ_API_KEY: {
    label: "Groq",
    icon: "🔥",
    placeholder: "gsk_...",
    helpUrl: "https://console.groq.com/keys",
  },
  OPENAI_API_KEY: {
    label: "OpenAI",
    icon: "◐",
    placeholder: "sk-...",
    helpUrl: "https://platform.openai.com/api-keys",
  },
  ANTHROPIC_API_KEY: {
    label: "Anthropic",
    icon: "◈",
    placeholder: "sk-ant-...",
    helpUrl: "https://console.anthropic.com/settings/keys",
  },
};

export default function Settings({ open, onClose }) {
  const [keys, setKeys] = useState({});
  const [serverKeys, setServerKeys] = useState({});
  const [saving, setSaving] = useState(false);
  const [validating, setValidating] = useState({});
  const [validationResults, setValidationResults] = useState({});
  const [message, setMessage] = useState(null);

  useEffect(() => {
    if (open) {
      loadSettings();
    }
  }, [open]);

  async function loadSettings() {
    try {
      const data = await getSettings();
      setServerKeys(data.keys || {});
      setKeys({});
      setValidationResults({});
      setMessage(null);
    } catch {
      setMessage({ type: "error", text: "Failed to load settings" });
    }
  }

  function handleKeyChange(provider, value) {
    setKeys((prev) => ({ ...prev, [provider]: value }));
    // Clear validation when user edits
    setValidationResults((prev) => {
      const next = { ...prev };
      delete next[provider];
      return next;
    });
  }

  async function handleValidate(provider) {
    const value = keys[provider];
    if (!value) return;

    setValidating((prev) => ({ ...prev, [provider]: true }));
    try {
      const result = await validateKey(provider, value);
      setValidationResults((prev) => ({ ...prev, [provider]: result }));
    } catch {
      setValidationResults((prev) => ({
        ...prev,
        [provider]: { valid: false, error: "Network error" },
      }));
    } finally {
      setValidating((prev) => ({ ...prev, [provider]: false }));
    }
  }

  async function handleSave() {
    // Only save keys that have been entered
    const toSave = {};
    for (const [k, v] of Object.entries(keys)) {
      if (v && v.trim()) toSave[k] = v.trim();
    }

    if (Object.keys(toSave).length === 0) {
      setMessage({ type: "error", text: "No keys to save" });
      return;
    }

    setSaving(true);
    setMessage(null);
    try {
      const result = await saveSettings(toSave);
      setServerKeys(result.keys || {});
      setKeys({});
      setMessage({ type: "success", text: "API keys saved successfully!" });
    } catch (err) {
      setMessage({ type: "error", text: err.message });
    } finally {
      setSaving(false);
    }
  }

  if (!open) return null;

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>API Key Configuration</h2>
          <button className="settings-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <p className="settings-description">
          Enter your own API keys to power the AI briefing pipeline. At least
          one key is needed. Keys are stored locally on the server.
        </p>

        {message && (
          <div className={`settings-message settings-message-${message.type}`}>
            {message.text}
          </div>
        )}

        <div className="settings-keys">
          {Object.entries(PROVIDER_INFO).map(([provider, info]) => {
            const server = serverKeys[provider] || {};
            const validation = validationResults[provider];
            const isValidating = validating[provider];

            return (
              <div key={provider} className="key-row">
                <div className="key-label">
                  <span className="key-icon">{info.icon}</span>
                  <span className="key-name">{info.label}</span>
                  {server.is_set && !keys[provider] && (
                    <span className="key-status key-status-set" title="Key is configured">
                      ✓ Set
                    </span>
                  )}
                  <a
                    className="key-help"
                    href={info.helpUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Get key →
                  </a>
                </div>
                <div className="key-input-row">
                  <input
                    type="password"
                    className="key-input"
                    placeholder={
                      server.is_set
                        ? server.masked_value
                        : info.placeholder
                    }
                    value={keys[provider] || ""}
                    onChange={(e) => handleKeyChange(provider, e.target.value)}
                  />
                  <button
                    className="btn btn-small btn-validate"
                    disabled={!keys[provider] || isValidating}
                    onClick={() => handleValidate(provider)}
                  >
                    {isValidating ? "…" : "Test"}
                  </button>
                </div>
                {validation && (
                  <div
                    className={`key-validation ${
                      validation.valid
                        ? "key-validation-ok"
                        : "key-validation-fail"
                    }`}
                  >
                    {validation.valid
                      ? `✓ ${validation.message}`
                      : `✗ ${validation.error}`}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="settings-actions">
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? "Saving…" : "Save Keys"}
          </button>
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
