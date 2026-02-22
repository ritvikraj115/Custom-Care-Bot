import { useState } from "react";
import axios from "../api/axios";
import { toast } from "react-toastify";
import ConfirmDialog from "./ConfirmDialog";

export default function BotCard({ bot, onDelete }) {
  const [open, setOpen] = useState(false);

  const deleteBot = async () => {
    try {
      await axios.delete(`/bots/${bot._id}`);
      toast.success("Assistant deleted");
      onDelete(bot._id);
    } catch {
      toast.error("Failed to delete assistant");
    } finally {
      setOpen(false);
    }
  };

  return (
    <>
      <div className="card bot-card">
        <h3>{bot.name}</h3>
        <p className="muted">{bot.purpose}</p>

        <div className="bot-card-actions">
          <a href={`/bots/${bot._id}`} className="bot-card-manage">Manage</a>
          <button
            className="secondary bot-card-delete"
            onClick={() => setOpen(true)}
          >
            Delete
          </button>
        </div>
      </div>

      <ConfirmDialog
        open={open}
        title={`Delete "${bot.name}"?`}
        description="This assistant and its documents will be permanently removed. This action cannot be undone."
        confirmText="Delete assistant"
        onConfirm={deleteBot}
        onCancel={() => setOpen(false)}
      />
    </>
  );
}
