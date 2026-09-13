export default function ClassificationResult({ classification }) {
  if (!classification) return null;

  const isMalignant = classification.predicted_label === "malignant";
  const pct = (classification.probability_malignant * 100).toFixed(1);

  return (
    <div className={`panel ${isMalignant ? "attention" : "ok"}`}>
      <div className="panel-header">
        <h2>Classification</h2>
        <span className={`badge ${isMalignant ? "attention" : "ok"}`}>
          <span className="dot" />
          {classification.predicted_label}
        </span>
      </div>

      <div className="readout-grid">
        <div className="readout">
          <div className="readout-label">P(malignant)</div>
          <div className="readout-value">{pct}%</div>
        </div>
        <div className="readout">
          <div className="readout-label">Operating threshold</div>
          <div className="readout-value small">{classification.operating_threshold.toFixed(3)}</div>
        </div>
        {classification.target_sensitivity != null && (
          <div className="readout">
            <div className="readout-label">Tuned for sensitivity</div>
            <div className="readout-value small">{(classification.target_sensitivity * 100).toFixed(0)}%</div>
          </div>
        )}
        <div className="readout">
          <div className="readout-label">Model</div>
          <div className="readout-value small">{classification.model_name.replace(/_/g, " ")}</div>
        </div>
      </div>

      <p className="disclaimer-line">
        This is a research model output, not a diagnosis. The threshold above was tuned on
        validation data for high sensitivity, which trades away specificity — expect a meaningful
        rate of benign cases flagged as malignant.
      </p>
    </div>
  );
}
