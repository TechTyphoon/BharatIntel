const CATEGORY_LABELS = {
  geopolitics: "Geopolitics",
  technology: "Technology",
  indian_politics: "Indian Politics",
  science: "Science",
  civilisation: "Civilisation",
  editors_picks: "Editor's Picks",
};

const CATEGORY_ICONS = {
  geopolitics: "🌍",
  technology: "💻",
  indian_politics: "🏛️",
  science: "🔬",
  civilisation: "🏛",
  editors_picks: "⭐",
};

export default function Section({ section }) {
  const key = section.category.toLowerCase().replace(/\s+/g, "_");
  const label = CATEGORY_LABELS[key] || section.category;
  const icon = CATEGORY_ICONS[key] || "📰";

  return (
    <section className="briefing-section">
      <div className="section-header">
        <span className="section-icon">{icon}</span>
        <h2 className="section-label">{label}</h2>
        <span className="article-count">
          {section.article_count} article{section.article_count !== 1 ? "s" : ""}
        </span>
      </div>

      <p className="section-summary">{section.summary}</p>

      {section.key_takeaways?.length > 0 && (
        <div className="takeaways">
          <h4>Key Takeaways</h4>
          <ul>
            {section.key_takeaways.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </div>
      )}

      {section.article_titles?.length > 0 && (
        <div className="source-articles">
          <h4>Sources</h4>
          <ul className="source-list">
            {section.article_titles.map((title, i) => (
              <li key={i}>
                {section.article_urls?.[i] ? (
                  <a
                    href={section.article_urls[i]}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {decodeHtml(title)}
                  </a>
                ) : (
                  <span>{decodeHtml(title)}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function decodeHtml(text) {
  const el = document.createElement("textarea");
  el.innerHTML = text;
  return el.value;
}
