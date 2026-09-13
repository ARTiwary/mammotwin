import { useRef, useState } from "react";

/**
 * A single-file drop zone. Calls onFileSelected(file) with a plain File
 * object -- the caller (Dashboard) owns what happens with it (preview,
 * upload, etc.) so this component stays a dumb input.
 */
export default function UploadPanel({ onFileSelected, disabled, label = "Upload mammogram" }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(fileList) {
    if (fileList && fileList[0]) onFileSelected(fileList[0]);
  }

  return (
    <div
      className={`upload-dropzone${dragging ? " dragging" : ""}`}
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      role="button"
      tabIndex={0}
    >
      <div className="upload-title">{label}</div>
      <div className="upload-hint">Drag a file here, or click to browse — PNG, JPG, or DICOM</div>
      <input
        ref={inputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.tif,.tiff,.dcm,.dicom"
        disabled={disabled}
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
