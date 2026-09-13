import { useState } from "react";
import UploadPanel from "../components/UploadPanel.jsx";
import QualityStatus from "../components/QualityStatus.jsx";
import LocalizationView from "../components/LocalizationView.jsx";
import ClassificationResult from "../components/ClassificationResult.jsx";
import UncertaintyBadge from "../components/UncertaintyBadge.jsx";
import GradCamView from "../components/GradCamView.jsx";
import SegmentationView from "../components/SegmentationView.jsx";
import LongitudinalCompare from "../components/LongitudinalCompare.jsx";
import { predictUpload, compareLongitudinal } from "../api/client.js";

const STAGES = [
  { key: "quality", label: "Quality check" },
  { key: "localization", label: "Localization" },
  { key: "classification", label: "Classification" },
  { key: "explainability", label: "Explainability" },
  { key: "segmentation", label: "Segmentation" },
  { key: "uncertainty", label: "Uncertainty" },
];

export default function Dashboard() {
  const [mode, setMode] = useState("single"); // "single" | "longitudinal"

  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const [longLoading, setLongLoading] = useState(false);
  const [longError, setLongError] = useState(null);
  const [longResult, setLongResult] = useState(null);

  async function handleFileSelected(selected) {
    setFile(selected);
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const response = await predictUpload(selected);
      setResult(response);
    } catch (e) {
      setError(e.message || "Something went wrong while analyzing this image.");
    } finally {
      setLoading(false);
    }
  }

  async function handleCompare(priorFile, currentFile) {
    setLongError(null);
    setLongResult(null);
    setLongLoading(true);
    try {
      const response = await compareLongitudinal(priorFile, currentFile);
      setLongResult(response);
    } catch (e) {
      setLongError(e.message || "Something went wrong while comparing these images.");
    } finally {
      setLongLoading(false);
    }
  }

  function stageStatus(key) {
    if (!result) return "";
    if (key === "quality" && !result.quality.passed) return "attention";
    if (key === "uncertainty" && result.uncertainty.needs_review) return "attention";
    if (key === "classification" && result.classification.predicted_label === "malignant") return "attention";
    return "done";
  }

  function stageNote(key) {
    if (!result) return "Waiting for an upload";
    const r = result[key];
    if (key === "localization") return r.available ? `${r.boxes.length} region(s)` : "Not available";
    if (key === "segmentation") return r.available ? "Mask predicted" : "Not available";
    if (key === "quality") return r.passed ? "Passed" : "Flagged";
    if (key === "classification") return r.predicted_label;
    if (key === "uncertainty") return r.needs_review ? "Needs review" : "Confident";
    if (key === "explainability") return "Heatmap ready";
    return "";
  }

  return (
    <div className="app-shell">
      <div className="disclaimer-band">
        <strong>MammoTwin</strong> is an academic research prototype. Outputs are not a medical
        diagnosis and must not inform clinical decisions.
      </div>

      <header className="app-header">
        <h1>MammoTwin</h1>
        <p>
          Explainable, uncertainty-aware mammogram lesion analysis — a research dashboard, not a
          diagnostic tool.
        </p>
        <div style={{ marginTop: 14, display: "flex", gap: 8 }}>
          <button className={`btn ${mode === "single" ? "" : "secondary"}`} onClick={() => setMode("single")}>
            Single image
          </button>
          <button className={`btn ${mode === "longitudinal" ? "" : "secondary"}`} onClick={() => setMode("longitudinal")}>
            Longitudinal (simulated)
          </button>
        </div>
      </header>

      {mode === "single" ? (
        <div className="app-body">
          <aside className="sidebar">
            <UploadPanel
              label={file ? file.name : "Upload mammogram"}
              onFileSelected={handleFileSelected}
              disabled={loading}
            />
            {loading && <p className="muted" style={{ marginTop: 10 }}>Running the full pipeline…</p>}
            {error && <p className="error-text" style={{ marginTop: 10 }}>{error}</p>}

            <div className="stage-tracker">
              {STAGES.map((s) => (
                <div className={`stage-item ${stageStatus(s.key)}`} key={s.key}>
                  <div>
                    <div className="stage-label">{s.label}</div>
                    <div className="stage-note">{stageNote(s.key)}</div>
                  </div>
                </div>
              ))}
            </div>
          </aside>

          <main className="results-column">
            {!result && !loading && (
              <div className="panel">
                <p className="panel-empty">
                  Upload a mammogram to run it through quality checking, localization,
                  classification, explainability, segmentation, and uncertainty estimation.
                </p>
              </div>
            )}

            {result && (
              <>
                <QualityStatus quality={result.quality} />
                <LocalizationView localization={result.localization} />
                <ClassificationResult classification={result.classification} />
                <GradCamView explainability={result.explainability} />
                <SegmentationView segmentation={result.segmentation} />
                <UncertaintyBadge uncertainty={result.uncertainty} />
              </>
            )}
          </main>
        </div>
      ) : (
        <main className="results-column" style={{ maxWidth: 760 }}>
          <LongitudinalCompare
            onCompare={handleCompare}
            loading={longLoading}
            error={longError}
            result={longResult}
          />
        </main>
      )}
    </div>
  );
}
