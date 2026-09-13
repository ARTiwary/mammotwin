export default function QualityStatus({ quality }) {
  if (!quality) return null;

  const flagLabels = {
    is_blank: "Image appears blank",
    is_low_contrast: "Low contrast",
    breast_area_too_small: "Breast area too small",
  };

  const activeFlags = Object.entries(flagLabels).filter(([key]) => quality[key]);

  return (
    <div className={`panel ${quality.passed ? "ok" : "attention"}`}>
      <div className="panel-header">
        <h2>Quality check</h2>
        <span className={`badge ${quality.passed ? "ok" : "attention"}`}>
          <span className="dot" />
          {quality.passed ? "Passed" : "Flagged"}
        </span>
      </div>

      {quality.passed ? (
        <p className="muted">
          No quality issues detected. This image was judged suitable for automated analysis.
        </p>
      ) : (
        <>
          <p className="muted">
            This image was flagged before analysis. Results below should be interpreted with
            extra caution.
          </p>
          <div className="flag-list">
            {activeFlags.map(([key]) => (
              <span key={key} className="badge attention">{flagLabels[key]}</span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
