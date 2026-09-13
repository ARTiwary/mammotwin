import { useState } from "react";
import UploadPanel from "./UploadPanel.jsx";

export default function LongitudinalCompare({ onCompare, loading, error, result }) {
  const [priorFile, setPriorFile] = useState(null);
  const [currentFile, setCurrentFile] = useState(null);

  const canCompare = priorFile && currentFile && !loading;

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Longitudinal comparison (simulated)</h2>
      </div>

      <p className="muted">
        This project does not have genuine prior/current exam pairs for the same patient.
        Uploading two images here demonstrates the comparison mechanism only — it is never a
        real change-detection result.
      </p>

      <div className="dual-upload" style={{ marginTop: 14 }}>
        <UploadPanel label={priorFile ? `Prior: ${priorFile.name}` : "Prior exam"} onFileSelected={setPriorFile} disabled={loading} />
        <UploadPanel label={currentFile ? `Current: ${currentFile.name}` : "Current exam"} onFileSelected={setCurrentFile} disabled={loading} />
      </div>

      <button
        className="btn"
        style={{ marginTop: 14 }}
        disabled={!canCompare}
        onClick={() => onCompare(priorFile, currentFile)}
      >
        {loading ? "Comparing…" : "Compare"}
      </button>

      {error && <p className="error-text" style={{ marginTop: 12 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 18 }}>
          <span className="badge attention"><span className="dot" />Simulated result — not a real comparison</span>

          <div className="score-change-row">
            <span>P(malignant) prior: <strong>{(result.prior_probability_malignant * 100).toFixed(1)}%</strong></span>
            <span className="arrow">→</span>
            <span>current: <strong>{(result.current_probability_malignant * 100).toFixed(1)}%</strong></span>
            <span className="arrow">
              ({result.score_change >= 0 ? "+" : ""}{(result.score_change * 100).toFixed(1)} pts)
            </span>
          </div>

          {result.overlay_png_base64 && (
            <div className="image-viewer">
              <img src={`data:image/png;base64,${result.overlay_png_base64}`} alt="Naive pixel-difference visualization" />
            </div>
          )}

          <p className="disclaimer-line">{result.message}</p>
        </div>
      )}
    </div>
  );
}
