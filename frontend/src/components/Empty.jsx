export default function Empty() {
  return (
    <div className="empty-state">
      <div className="empty-icon">📰</div>
      <h2>No Briefing Available</h2>
      <p>
        Click <strong>Generate Brief</strong> to run the AI pipeline and create
        today's intelligence briefing.
      </p>
    </div>
  );
}
