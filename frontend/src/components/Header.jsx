export default function Header({ briefingDate, onGenerate, onDownload, onSettings, loading }) {
  return (
    <header className="header">
      <div className="header-left">
        <h1 className="logo">
          <span className="logo-bharat">Bharat</span>
          <span className="logo-intel">Intel</span>
        </h1>
        <p className="tagline">AI-Powered Daily Intelligence Briefing</p>
      </div>
      <div className="header-right">
        {briefingDate && (
          <span className="briefing-date">{formatDate(briefingDate)}</span>
        )}
        <button
          className="btn btn-primary"
          onClick={onGenerate}
          disabled={loading}
        >
          {loading ? "Generating…" : "Generate Brief"}
        </button>
        {briefingDate && (
          <button className="btn btn-secondary" onClick={onDownload}>
            Download PDF
          </button>
        )}
        <button
          className="btn-settings-gear"
          onClick={onSettings}
          title="API Key Settings"
        >
          ⚙
        </button>
      </div>
    </header>
  );
}

function formatDate(dateStr) {
  try {
    const d = new Date(dateStr + "T00:00:00");
    return d.toLocaleDateString("en-IN", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  } catch {
    return dateStr;
  }
}
