import { useEffect, useState } from "react";
import axios from "../api/axios";
import BotCard from "../components/BotCard";

export default function Dashboard() {
  const [bots, setBots] = useState([]);

  useEffect(() => {
    axios.get("/bots").then(res => setBots(res.data));
  }, []);

  const handleDelete = botId => {
    setBots(prev => prev.filter(b => b._id !== botId));
  };

  const hasBots = bots.length > 0;

  return (
    <div className="container dashboard-page">
      <div className="page-header dashboard-header">
        <h2 className="page-title">AI Assistants</h2>
        <p className="page-subtitle">
          Manage and configure AI assistants for your organization.
          Each assistant operates independently with its own knowledge base.
        </p>
      </div>

      <div className="page-actions">
        <h3 className="section-title">
          {hasBots ? "Your assistants" : "Get started"}
        </h3>
        <a href="/create-bot">
          <button>Create assistant</button>
        </a>
      </div>

      {hasBots ? (
        <div className="grid">
          {bots.map(bot => (
            <BotCard key={bot._id} bot={bot} onDelete={handleDelete} />
          ))}
        </div>
      ) : (
        <div className="card empty-state">
          <h3>No assistants yet</h3>
          <p className="muted">
            Create your first AI assistant to start uploading documents
            and answering questions using your organization's knowledge.
          </p>
          <a href="/create-bot">
            <button>Create your first assistant</button>
          </a>
        </div>
      )}
    </div>
  );
}
