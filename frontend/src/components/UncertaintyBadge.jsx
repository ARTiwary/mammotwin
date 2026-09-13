export default function UncertaintyBadge({ uncertainty }) {
  if (!uncertainty) return null;

  return (
    <div className={`panel ${uncertainty.needs_review ? "attention" : "ok"}`}>
      <div className="panel-header">
        <h2>Uncertainty</h2>
        <span className={`badge ${uncertainty.needs_review ? "attention" : "ok"}`}>
          <span className="dot" />
          {uncertainty.needs_review ? "Needs expert review" : "Confident"}
        </span>
      </div>

      <div className="readout-grid">
        <div className="readout">
          <div className="readout-label">Confidence</div>
          <div className="readout-value">{(uncertainty.confidence * 100).toFixed(1)}%</div>
        </div>
        <div className="readout">
          <div className="readout-label">Mean P(malignant)</div>
          <div className="readout-value small">{(uncertainty.mean_prob_malignant * 100).toFixed(1)}%</div>
        </div>
        <div className="readout">
          <div className="readout-label">Std. dev. across passes</div>
          <div className="readout-value small">±{(uncertainty.std_prob_malignant * 100).toFixed(1)}%</div>
        </div>
        <div className="readout">
          <div className="readout-label">MC-Dropout passes</div>
          <div className="readout-value small">{uncertainty.n_passes}</div>
        </div>
      </div>

      <p className="disclaimer-line">
        Estimated via Monte Carlo Dropout ({uncertainty.n_passes} stochastic forward passes).
        A low-confidence flag means the model itself is unsure — it should never be read as
        reassurance when the flag is absent.
      </p>
    </div>
  );
}
