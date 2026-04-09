export default function StatusBar({ status, error, onOpenSettings }) {
  if (!status && !error) return null;

  const hasSettingsHint = (text) =>
    text && (text.includes("Settings") || text.includes("API key") || text.includes("rate-limit"));

  return (
    <div className={`status-bar ${error ? "status-error" : "status-info"}`}>
      {error ? (
        <p>
          ⚠️ {error}
          {hasSettingsHint(error) && onOpenSettings && (
            <button className="status-settings-link" onClick={onOpenSettings}>
              Open Settings
            </button>
          )}
        </p>
      ) : (
        <p>
          {status}
          {hasSettingsHint(status) && onOpenSettings && (
            <button className="status-settings-link" onClick={onOpenSettings}>
              Open Settings
            </button>
          )}
        </p>
      )}
    </div>
  );
}
