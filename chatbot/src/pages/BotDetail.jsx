import { useParams } from "react-router-dom";
import { useState, useEffect, useRef, useCallback } from "react";
import axios from "../api/axios";

const JOB_TERMINAL_STATES = new Set(["completed", "failed"]);

export default function BotDetail() {
  const { botId } = useParams();

  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [escalations, setEscalations] = useState([]);
  const [loadingEscalations, setLoadingEscalations] = useState(false);
  const [answerDrafts, setAnswerDrafts] = useState({});
  const [resolving, setResolving] = useState({});
  const [analytics, setAnalytics] = useState(null);
  const [loadingAnalytics, setLoadingAnalytics] = useState(false);
  const [ingestionJob, setIngestionJob] = useState(null);
  const pollRef = useRef(null);

  const baseUrl = window.location.origin;

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const loadDocuments = useCallback(async () => {
    const res = await axios.get(`/documents/bot/${botId}`);
    setDocuments(res.data);
  }, [botId]);

  const loadEscalations = useCallback(async () => {
    setLoadingEscalations(true);
    try {
      const res = await axios.get(`/escalations/bot/${botId}`);
      setEscalations(res.data);
    } catch (err) {
      console.error("Failed to load escalations:", err);
    } finally {
      setLoadingEscalations(false);
    }
  }, [botId]);

  const loadAnalytics = useCallback(async () => {
    setLoadingAnalytics(true);
    try {
      const res = await axios.get(`/bots/${botId}/analytics`);
      setAnalytics(res.data);
    } catch (err) {
      console.error("Failed to load analytics:", err);
    } finally {
      setLoadingAnalytics(false);
    }
  }, [botId]);

  const pollJob = useCallback(async jobId => {
    try {
      const res = await axios.get(`/documents/jobs/${jobId}`);
      const job = res.data;
      setIngestionJob(job);

      if (JOB_TERMINAL_STATES.has((job.status || "").toLowerCase())) {
        stopPolling();
        await loadDocuments();
        await loadAnalytics();
      }
    } catch (err) {
      console.error("Failed to poll ingestion job:", err);
      stopPolling();
    }
  }, [loadAnalytics, loadDocuments, stopPolling]);

  useEffect(() => {
    loadDocuments();
    loadEscalations();
    loadAnalytics();

    return () => stopPolling();
  }, [loadAnalytics, loadDocuments, loadEscalations, stopPolling]);

  const handleFileChange = e => {
    setSelectedFiles(Array.from(e.target.files));
  };

  const handleSubmit = async e => {
    e.preventDefault();
    if (!selectedFiles.length) return;

    setUploading(true);

    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append("files", file);
    });
    formData.append("botId", botId);

    try {
      const res = await axios.post("/documents/upload", formData);
      setSelectedFiles([]);

      if (res.data?.status === "QUEUED" && res.data?.jobId) {
        setIngestionJob({
          id: res.data.jobId,
          status: "queued",
          rebuildMode: res.data.rebuildMode,
          progress: {
            phase: "queued",
            current: 0,
            total: 1,
            percent: 0,
            message: "Queued for processing"
          }
        });

        stopPolling();
        await pollJob(res.data.jobId);
        pollRef.current = setInterval(() => {
          pollJob(res.data.jobId);
        }, 1800);
      } else {
        setIngestionJob({
          status: (res.data?.status || "NO_CHANGES").toLowerCase(),
          progress: {
            phase: "done",
            current: 1,
            total: 1,
            percent: 100,
            message: res.data?.message || "No new documents to ingest"
          }
        });
      }

      await loadDocuments();
    } catch (err) {
      console.error("Document upload failed:", err);
      setIngestionJob({
        status: "failed",
        progress: {
          phase: "failed",
          current: 0,
          total: 1,
          percent: 0,
          message: "Upload failed"
        },
        error: err.response?.data || err.message
      });
    } finally {
      setUploading(false);
    }
  };

  const handleResolve = async escalationId => {
    const answer = (answerDrafts[escalationId] || "").trim();
    if (!answer) return;

    setResolving(prev => ({ ...prev, [escalationId]: true }));
    try {
      await axios.post("/escalations/resolve", {
        escalationId,
        answer
      });

      setAnswerDrafts(prev => ({ ...prev, [escalationId]: "" }));
      setEscalations(prev => prev.filter(item => item._id !== escalationId));
      await loadAnalytics();
    } catch (err) {
      console.error("Failed to resolve escalation:", err);
    } finally {
      setResolving(prev => ({ ...prev, [escalationId]: false }));
    }
  };

  const summary = analytics?.summary || {};
  const rates = analytics?.rates || {};
  const unresolved = analytics?.hotspots?.unresolvedQuestions || [];
  const negativeHotspots = analytics?.hotspots?.negativeHotspots || [];

  return (
    <div className="container">
      <div className="page-header">
        <h2 className="page-title">Assistant Configuration</h2>
        <p className="page-subtitle">
          Manage knowledge files, monitor quality, and resolve escalations.
        </p>
      </div>

      <div className="detail-layout">
        <div className="card detail-layout-full analytics-card">
          <h3 className="section-title">Knowledge Gap Analytics</h3>
          {loadingAnalytics ? (
            <p className="muted">Loading analytics...</p>
          ) : (
            <>
              <div className="analytics-grid">
                <div className="analytics-item">
                  <span className="analytics-label">Containment</span>
                  <span className="analytics-value">{Number(rates.containmentRate || 0).toFixed(2)}%</span>
                </div>
                <div className="analytics-item">
                  <span className="analytics-label">Escalation</span>
                  <span className="analytics-value">{Number(rates.escalationRate || 0).toFixed(2)}%</span>
                </div>
                <div className="analytics-item">
                  <span className="analytics-label">Negative Feedback</span>
                  <span className="analytics-value">{Number(rates.negativeFeedbackRate || 0).toFixed(2)}%</span>
                </div>
                <div className="analytics-item">
                  <span className="analytics-label">Likely No-Docs</span>
                  <span className="analytics-value">{Number(rates.likelyNoDocsRate || 0).toFixed(2)}%</span>
                </div>
                <div className="analytics-item">
                  <span className="analytics-label">Open Escalations</span>
                  <span className="analytics-value">{Number(summary.openEscalations || 0)}</span>
                </div>
                <div className="analytics-item">
                  <span className="analytics-label">Total Sessions</span>
                  <span className="analytics-value">{Number(summary.sessions || 0)}</span>
                </div>
              </div>

              <div className="analytics-hotspots">
                <div className="analytics-panel">
                  <h4>Top unresolved questions</h4>
                  {unresolved.length === 0 ? (
                    <p className="muted">No unresolved hotspots.</p>
                  ) : (
                    <ul>
                      {unresolved.map((item, idx) => (
                        <li key={`${item.question}-${idx}`}>
                          <span>{item.question}</span>
                          <strong>{item.count}</strong>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <div className="analytics-panel">
                  <h4>Most negative topics</h4>
                  {negativeHotspots.length === 0 ? (
                    <p className="muted">No negative hotspots yet.</p>
                  ) : (
                    <ul>
                      {negativeHotspots.map((item, idx) => (
                        <li key={`${item.question}-${idx}`}>
                          <span>{item.question}</span>
                          <strong>{item.count}</strong>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </>
          )}
        </div>

        <div className="card">
          <h3 className="section-title">Knowledge Files</h3>
          <p className="muted">
            Files available to this assistant for retrieval.
          </p>

          {documents.length === 0 ? (
            <p className="muted">No documents uploaded yet.</p>
          ) : (
            <ul className="file-list">
              {documents.map(doc => (
                <li key={doc._id}>{doc.fileName}</li>
              ))}
            </ul>
          )}
        </div>

        <div className="card section-stack update-card">
          <div className="update-header-row">
            <div>
              <h3 className="section-title">Update Knowledge</h3>
              <p className="muted">
                Uploads are deduplicated and processed asynchronously.
              </p>
            </div>
            <div className="update-badge">
              {documents.length} indexed
            </div>
          </div>

          <form onSubmit={handleSubmit} className="upload-form">
            <input
              id="knowledge-upload-input"
              className="file-input-hidden"
              type="file"
              multiple
              accept="application/pdf"
              onChange={handleFileChange}
            />

            <label htmlFor="knowledge-upload-input" className="upload-dropzone">
              <span className="upload-dropzone-title">
                Drop PDF files here or click to browse
              </span>
              <span className="upload-dropzone-subtitle">
                Supports multiple files. PDF format only.
              </span>
            </label>

            {selectedFiles.length > 0 ? (
              <div className="selected-files">
                {selectedFiles.map(file => (
                  <span
                    key={`${file.name}-${file.size}`}
                    className="file-chip"
                    title={file.name}
                  >
                    {file.name}
                  </span>
                ))}
              </div>
            ) : (
              <p className="form-hint upload-empty-hint">
                No files selected yet.
              </p>
            )}

            <div className="upload-actions">
              <button disabled={uploading || selectedFiles.length === 0}>
                {uploading ? "Submitting..." : "Upload & Queue"}
              </button>
              <span className="upload-meta">
                {selectedFiles.length > 0
                  ? `${selectedFiles.length} file(s) ready`
                  : "Select one or more PDF files to continue"}
              </span>
            </div>
          </form>

          {ingestionJob && (
            <div className="job-status-card">
              <div className="job-status-head">
                <strong>Ingestion Job</strong>
                <span className={`job-pill ${String(ingestionJob.status || "").toLowerCase()}`}>
                  {String(ingestionJob.status || "unknown").toUpperCase()}
                </span>
              </div>
              <p className="muted job-status-message">
                {ingestionJob?.progress?.message || "Waiting for updates"}
              </p>
              <div className="job-progress-track">
                <div
                  className="job-progress-fill"
                  style={{ width: `${Number(ingestionJob?.progress?.percent || 0)}%` }}
                />
              </div>
              <p className="job-progress-meta">
                {Number(ingestionJob?.progress?.percent || 0)}% · {ingestionJob.rebuildMode || "full"} mode
              </p>
            </div>
          )}

          <div className="share-link-group">
            <label className="form-label">Public Chat Link</label>
            <input readOnly value={`${baseUrl}/chat/${botId}`} />
          </div>
        </div>

        <div className="card detail-layout-full">
          <h3 className="section-title">Escalated Questions</h3>

          {loadingEscalations ? (
            <p className="muted">Loading escalations...</p>
          ) : escalations.length === 0 ? (
            <p className="muted">No open escalations for this bot.</p>
          ) : (
            <div className="escalation-list">
              {escalations.map(item => (
                <div key={item._id} className="escalation-item">
                  <p className="escalation-meta">
                    {item.createdAt
                      ? new Date(item.createdAt).toLocaleString()
                      : "New escalation"}
                  </p>

                  <p className="escalation-question">{item.question}</p>

                  <textarea
                    rows={3}
                    placeholder="Write the answer that should be used in future."
                    value={answerDrafts[item._id] || ""}
                    onChange={e =>
                      setAnswerDrafts(prev => ({
                        ...prev,
                        [item._id]: e.target.value
                      }))
                    }
                  />

                  <button
                    disabled={
                      resolving[item._id] ||
                      !(answerDrafts[item._id] || "").trim()
                    }
                    onClick={() => handleResolve(item._id)}
                  >
                    {resolving[item._id] ? "Saving..." : "Answer & Resolve"}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
