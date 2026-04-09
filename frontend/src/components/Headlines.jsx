export default function Headlines({ headlines }) {
  if (!headlines?.length) return null;

  return (
    <section className="headlines">
      <h2 className="section-title">Top Headlines</h2>
      <div className="headline-list">
        {headlines.map((h, i) => (
          <a
            key={i}
            href={h.url}
            target="_blank"
            rel="noopener noreferrer"
            className="headline-card"
          >
            <span className="headline-number">{i + 1}</span>
            <div className="headline-content">
              <h3 className="headline-title">{h.title}</h3>
              <p className="headline-oneliner">{h.oneliner}</p>
              <span className="headline-source">{h.source}</span>
            </div>
          </a>
        ))}
      </div>
    </section>
  );
}
