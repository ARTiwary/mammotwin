export default function LocalizationView({ localization }) {
  if (!localization) return null;

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Localization</h2>
        {localization.available && localization.boxes.length > 0 && (
          <span className="panel-meta">{localization.boxes.length} region(s) detected</span>
        )}
      </div>

      {!localization.available ? (
        <p className="panel-empty">{localization.message || "No detector model is available."}</p>
      ) : (
        <>
          <div className="image-viewer">
            <img src={`data:image/png;base64,${localization.overlay_png_base64}`} alt="Detected regions" />
          </div>
          {localization.boxes.length === 0 ? (
            <p className="muted" style={{ marginTop: 10 }}>{localization.message}</p>
          ) : (
            <div className="readout-grid">
              {localization.boxes.map((box, i) => (
                <div className="readout" key={i}>
                  <div className="readout-label">Region {i + 1} score</div>
                  <div className="readout-value small">{(box.score * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
