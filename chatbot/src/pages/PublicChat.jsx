import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import axios from "../api/axios";
import ChatWindow from "../components/ChatWindow";

export default function PublicChat() {
  const { botId } = useParams();
  const [sessionId, setSessionId] = useState(null);
  const [starterQuestions, setStarterQuestions] = useState([]);

  useEffect(() => {
    axios.post(`/chat/session/${botId}`).then(res => {
      setSessionId(res.data.sessionId);
      setStarterQuestions(
        Array.isArray(res.data?.starterQuestions)
          ? res.data.starterQuestions
          : []
      );
    });
  }, [botId]);

  if (!sessionId) {
    return (
      <div className="auth-shell">
        <div className="card auth-layout">
          <h3 className="section-title">Preparing assistant chat</h3>
          <p className="muted">
            Setting up your session. This usually takes a few seconds.
          </p>
        </div>
      </div>
    );
  }

  return <ChatWindow sessionId={sessionId} starterQuestions={starterQuestions} />;
}
