export default function GradCamView({ explainability }) {
  if (!explainability) return null;

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Explainability</h2>
        <span className="panel-meta">Grad-CAM · {explainability.target_class}-class evidence</span>
      </div>

      <div className="image-viewer">
        <img
          src={`data:image/png;base64,${explainability.overlay_png_base64}`}
          alt="Grad-CAM heatmap overlay"
        />
      </div>

      <p className="disclaimer-line">{explainability.disclaimer}</p>
    </div>
  );
}
