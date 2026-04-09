export default function Executive({ data }) {
  if (!data) return null;

  return (
    <section className="executive">
      <h2 className="section-title">Executive Summary</h2>
      <p className="executive-overview">{data.overview}</p>
      {data.key_developments?.length > 0 && (
        <div className="key-developments">
          <h3>Key Developments</h3>
          <ul>
            {data.key_developments.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </div>
      )}
      {data.tone && (
        <span className="tone-badge">Tone: {data.tone}</span>
      )}
    </section>
  );
}
