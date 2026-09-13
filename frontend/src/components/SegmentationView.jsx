export default function SegmentationView({ segmentation }) {
  if (!segmentation) return null;

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Segmentation</h2>
      </div>

      {!segmentation.available ? (
        <p className="panel-empty">{segmentation.message || "No segmentation model is available."}</p>
      ) : (
        <>
          <div className="image-viewer">
            <img
              src={`data:image/png;base64,${segmentation.overlay_png_base64}`}
              alt="Predicted lesion mask overlay"
            />
          </div>
          <p className="disclaimer-line">
            Predicted mask probability overlay{segmentation.used_localization_box
              ? ", cropped around the detected region above"
              : ""}. Segmentation quality on this project's holdout data is weak (test Dice ≈ 0.16)
            — treat this as a coarse, exploratory indicator only.
          </p>
        </>
      )}
    </div>
  );
}
