export default function ConfirmDialog({
  open,
  title,
  description,
  confirmText = "Delete",
  cancelText = "Cancel",
  onConfirm,
  onCancel
}) {
  if (!open) return null;

  return (
    <div className="dialog-overlay">
      <div className="card dialog-card">
        <h3 className="dialog-title">{title}</h3>
        <p className="muted dialog-description">{description}</p>

        <div className="dialog-actions">
          <button className="secondary" onClick={onCancel}>
            {cancelText}
          </button>
          <button onClick={onConfirm} className="btn-danger">
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
