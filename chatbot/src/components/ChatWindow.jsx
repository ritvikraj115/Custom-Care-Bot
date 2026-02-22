import { useState, useEffect, useRef, useCallback } from "react";
import axios from "../api/axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "../styles/chat.css";

const SUGGEST_CACHE_TTL_MS = 45 * 1000;
const SUGGEST_CACHE_LIMIT = 120;

export default function ChatWindow({ sessionId, starterQuestions = [] }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [suggestionIndex, setSuggestionIndex] = useState(-1);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const bottomRef = useRef(null);
  const analysisTimersRef = useRef({});
  const suggestTimerRef = useRef(null);
  const suggestRequestRef = useRef(0);
  const suggestionCacheRef = useRef(new Map());
  const lastSuggestKeyRef = useRef("");

  const parseIntentLabel = trace => {
    if (!Array.isArray(trace)) return null;
    const intentStep = trace.find(
      step =>
        step &&
        step.step === "IntentClassifier" &&
        typeof step.detail === "string"
    );
    if (!intentStep) return null;
    const match = intentStep.detail.match(/(?:^|;)intent=([a-z_]+)/i);
    return match ? match[1].toLowerCase() : null;
  };

  const normalizeReferences = refs => {
    if (!Array.isArray(refs)) return [];
    return refs
      .map(ref => ({
        type: (ref?.type || "").toString(),
        title: (ref?.title || "").toString(),
        url: (ref?.url || ref?.source_url || "").toString(),
        pdf: (ref?.pdf || "").toString(),
        chunkRef: (ref?.chunk_ref || "").toString(),
        note: (ref?.note || "").toString(),
        score: Number(ref?.score || 0)
      }))
      .filter(ref =>
        ref.url || ref.title || ref.pdf || ref.chunkRef || ref.note
      );
  };

  const buildAnalysisTimeline = trace => {
    const stepsTrace = Array.isArray(trace) ? trace : [];
    const getStep = name =>
      stepsTrace.find(step => step && step.step === name);
    const hasStep = name => !!getStep(name);
    const sumMs = names => {
      const total = names.reduce((acc, name) => {
        const ms = getStep(name)?.ms;
        return acc + (typeof ms === "number" ? ms : 0);
      }, 0);
      return total > 0 ? Math.round(total) : null;
    };
    const statusText = (status, detail) => {
      const prefix =
        status === "done"
          ? "Done"
          : status === "skipped"
            ? "Skipped"
            : "Pending";
      return detail ? `${prefix}: ${detail}` : prefix;
    };

    const intentLabel = parseIntentLabel(stepsTrace);
    const intentReadable = intentLabel
      ? intentLabel.replace(/_/g, " ")
      : null;

    const followup =
      getStep("SemanticMemoryLookup")?.detail === "followup";
    const toolStep = getStep("ToolRetrieval");
    const toolDetail = toolStep?.detail;
    const toolMessage =
      toolDetail === "results"
        ? "Tool results found"
        : toolDetail === "empty"
          ? "No tool results found"
          : toolStep
            ? "Tool lookup completed"
            : "Tool lookup not used";

    const usedPrimary = hasStep("PrimaryRetrieval");
    const usedSecondary = hasStep("SecondaryRetrieval");
    const usedRetrieval = usedPrimary || usedSecondary;
    const retrievalDetail = usedSecondary
      ? "Secondary retrieval (fresh chunks)"
      : usedPrimary
        ? "Primary retrieval"
        : followup
          ? "Resolved from conversation"
          : toolStep || intentLabel === "latest_social_updates"
            ? "Skipped (tool-first path)"
            : "Skipped (not required)";

    const analyzerRan = hasStep("AnalyzerNode");
    const humanHandoff = hasStep("HumanInLoopNode");
    const ownerOverride =
      getStep("OwnerResolutionNode")?.detail === "override";

    const answerDetail = humanHandoff
      ? "Escalated to support"
      : ownerOverride
        ? "Owner answer applied"
        : analyzerRan
          ? "Response composed and quality-checked"
          : "Response composed";

    return [
      {
        step: "Intent detection",
        ms: sumMs(["IntentClassifier"]),
        detail: statusText(
          hasStep("IntentClassifier") ? "done" : "pending",
          intentReadable ? `Detected: ${intentReadable}` : "Classifying request"
        )
      },
      {
        step: "Memory & feedback scan",
        ms: sumMs(["CheckFeedbackState", "SemanticMemoryLookup"]),
        detail: statusText(
          hasStep("CheckFeedbackState") || hasStep("SemanticMemoryLookup")
            ? "done"
            : "pending",
          followup
            ? "Used conversation context"
            : "Checked prior feedback and context"
        )
      },
      {
        step: "Docs/website retrieval",
        ms: sumMs(["PrimaryRetrieval", "SecondaryRetrieval"]),
        detail: statusText(
          usedRetrieval ? "done" : "skipped",
          retrievalDetail
        )
      },
      {
        step: "Tool lookup (social/web)",
        ms: sumMs(["ToolRetrieval"]),
        detail: statusText(toolStep ? "done" : "skipped", toolMessage)
      },
      {
        step: "Answer drafting",
        ms: sumMs([
          "AnalyzerNode",
          "OwnerResolutionNode",
          "FinalizeResponse",
          "HumanInLoopNode"
        ]),
        detail: statusText(
          hasStep("FinalizeResponse") || humanHandoff ? "done" : "pending",
          answerDetail
        )
      }
    ];
  };

  const finalizeAnalysisTimeline = trace => {
    if (!Array.isArray(trace) || trace.length === 0) {
      return [
        {
          step: "Intent detection",
          ms: 0,
          detail: "Done: Request classified"
        },
        {
          step: "Memory & feedback scan",
          ms: 0,
          detail: "Done: Context checked"
        },
        {
          step: "Docs/website retrieval",
          ms: 0,
          detail: "Done: Sources gathered"
        },
        {
          step: "Tool lookup (social/web)",
          ms: 0,
          detail: "Skipped: Not required"
        },
        {
          step: "Answer drafting",
          ms: 0,
          detail: "Done: Response composed"
        }
      ];
    }
    return buildAnalysisTimeline(trace);
  };

  const statusFromDetail = detail => {
    const text = (detail || "").toLowerCase();
    if (text.startsWith("pending")) return "pending";
    if (text.startsWith("queued")) return "queued";
    if (text.startsWith("running")) return "running";
    if (text.startsWith("done")) return "done";
    if (text.startsWith("estimated done")) return "done";
    if (text.startsWith("skipped")) return "skipped";
    return "";
  };

  const pendingAnalysis = () => ([
    {
      step: "Intent detection",
      ms: 0,
      detail: "Pending: Waiting to classify request"
    },
    {
      step: "Memory & feedback scan",
      ms: 0,
      detail: "Pending: Checking prior context"
    },
    {
      step: "Docs/website retrieval",
      ms: 0,
      detail: "Pending: Searching relevant sources"
    },
    {
      step: "Tool lookup (social/web)",
      ms: 0,
      detail: "Pending: Verifying live sources"
    },
    {
      step: "Answer drafting",
      ms: 0,
      detail: "Pending: Composing response"
    }
  ]);

  const updateAnalysis = (index, updater) => {
    setMessages(prev => {
      const copy = [...prev];
      const msg = copy[index];
      if (!msg) return prev;
      const current = Array.isArray(msg.analysis) ? msg.analysis : [];
      const next = typeof updater === "function" ? updater(current) : updater;
      copy[index] = {
        ...msg,
        analysis: next
      };
      return copy;
    });
  };

  const clearAnalysisTimers = index => {
    const timers = analysisTimersRef.current[index];
    if (Array.isArray(timers)) {
      timers.forEach(timerId => clearTimeout(timerId));
    }
    delete analysisTimersRef.current[index];
  };

  const updateStepDetail = (analysis, stepName, detail) =>
    analysis.map(item =>
      item.step === stepName
        ? { ...item, detail }
        : item
    );

  const isLikelySocialQuery = query => {
    const q = (query || "").toLowerCase();
    const latestPatterns = [
      "latest",
      "recent",
      "today",
      "current",
      "news",
      "update",
      "updates"
    ];
    const socialPatterns = [
      "linkedin",
      "facebook",
      "instagram",
      "insta",
      "social media",
      "post",
      "posts"
    ];
    return latestPatterns.some(p => q.includes(p)) &&
      socialPatterns.some(p => q.includes(p));
  };

  const startAnalysisProgress = (index, query) => {
    clearAnalysisTimers(index);
    updateAnalysis(index, pendingAnalysis());

    const likelySocial = isLikelySocialQuery(query);
    const timers = [];

    const schedule = (delay, fn) => {
      timers.push(setTimeout(fn, delay));
    };

    schedule(150, () => {
      updateAnalysis(index, current =>
        updateStepDetail(
          current,
          "Intent detection",
          "Running: Classifying request"
        )
      );
    });

    schedule(450, () => {
      updateAnalysis(index, current =>
        updateStepDetail(
          current,
          "Intent detection",
          "Estimated done: Intent categorized"
        )
      );
    });

    schedule(500, () => {
      updateAnalysis(index, current =>
        updateStepDetail(
          current,
          "Memory & feedback scan",
          "Running: Checking prior context"
        )
      );
    });

    schedule(900, () => {
      updateAnalysis(index, current =>
        updateStepDetail(
          current,
          "Memory & feedback scan",
          "Estimated done: Context checked"
        )
      );
    });

    schedule(950, () => {
      updateAnalysis(index, current =>
        updateStepDetail(
          current,
          "Docs/website retrieval",
          "Running: Searching relevant sources"
        )
      );
    });

    schedule(1400, () => {
      updateAnalysis(index, current =>
        updateStepDetail(
          current,
          "Docs/website retrieval",
          "Estimated done: Sources retrieved"
        )
      );
    });

    schedule(1450, () => {
      updateAnalysis(index, current =>
        updateStepDetail(
          current,
          "Tool lookup (social/web)",
          likelySocial
            ? "Running: Scanning social/web updates"
            : "Queued: Only if docs are thin"
        )
      );
    });

    schedule(1800, () => {
      updateAnalysis(index, current =>
        updateStepDetail(
          current,
          "Answer drafting",
          "Running: Composing response"
        )
      );
    });

    analysisTimersRef.current[index] = timers;
  };

  const applySuggestion = value => {
    const text = (value || "").trim();
    if (!text) return;
    setInput(`${text} `);
    setSuggestions([]);
    setSuggestionIndex(-1);
    setIsSuggesting(false);
  };

  const handleInputKeyDown = e => {
    if (e.key === "ArrowDown" && suggestions.length) {
      e.preventDefault();
      setSuggestionIndex(prev => {
        const next = prev + 1;
        if (next >= suggestions.length) return 0;
        return next;
      });
      return;
    }

    if (e.key === "ArrowUp" && suggestions.length) {
      e.preventDefault();
      setSuggestionIndex(prev => {
        if (prev <= 0) return suggestions.length - 1;
        return prev - 1;
      });
      return;
    }

    if (e.key === "Tab" && suggestions.length) {
      e.preventDefault();
      const idx = suggestionIndex >= 0 ? suggestionIndex : 0;
      const selected = suggestions[idx];
      if (selected?.text) applySuggestion(selected.text);
      return;
    }

    if (e.key === "Enter") {
      if (suggestions.length && suggestionIndex >= 0) {
        e.preventDefault();
        const selected = suggestions[suggestionIndex];
        if (selected?.text) applySuggestion(selected.text);
        return;
      }
      send();
    }
  };

  const querySignature = useCallback(raw => (
    (raw || "")
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim()
  ), []);

  const shouldTriggerSuggestion = useCallback(raw => {
    const value = raw || "";
    const signature = querySignature(value);
    if (!signature || signature.length < 4) return false;

    const boundaryEnded = /[\s.,!?;:]$/.test(value);
    const parts = signature.split(" ");
    const last = parts[parts.length - 1] || "";

    if (boundaryEnded) return true;
    return last.length >= 3;
  }, [querySignature]);

  const readSuggestionCache = useCallback(key => {
    const entry = suggestionCacheRef.current.get(key);
    if (!entry) return null;
    if (Date.now() - entry.ts > SUGGEST_CACHE_TTL_MS) {
      suggestionCacheRef.current.delete(key);
      return null;
    }
    return entry.items;
  }, []);

  const writeSuggestionCache = useCallback((key, items) => {
    const cache = suggestionCacheRef.current;
    cache.set(key, { ts: Date.now(), items });
    if (cache.size > SUGGEST_CACHE_LIMIT) {
      const oldest = cache.keys().next().value;
      if (oldest) cache.delete(oldest);
    }
  }, []);

  const renderAssistantReply = async (text, index) => {
    const normalized = typeof text === "string" ? text.trim() : "";
    if (!normalized) return;

    const words = normalized.split(/\s+/);

    // Long answers are rendered instantly to avoid perceived lag.
    if (words.length > 40) {
      setMessages(prev => {
        const copy = [...prev];
        if (!copy[index]) return prev;
        copy[index] = {
          ...copy[index],
          content: normalized
        };
        return copy;
      });
      return;
    }

    const chunkSize = words.length > 20 ? 4 : 3;
    const delayMs = words.length > 20 ? 6 : 10;

    for (let i = 0; i < words.length; i += chunkSize) {
      const chunk = words.slice(i, i + chunkSize).join(" ");
      await new Promise(r => setTimeout(r, delayMs));
      setMessages(prev => {
        const copy = [...prev];
        if (!copy[index]) return prev;
        const prior = copy[index].content ? `${copy[index].content} ` : "";
        copy[index] = {
          ...copy[index],
          content: `${prior}${chunk}`.trim()
        };
        return copy;
      });
    }
  };

  // ----------------------------
  // Send message (normal)
  // ----------------------------
  const send = async (overrideText = null) => {
    const draft =
      typeof overrideText === "string"
        ? overrideText
        : input;
    if (!draft.trim()) return;

    const userText = draft.trim();
    setInput("");
    setSuggestions([]);
    setSuggestionIndex(-1);
    setIsSuggesting(false);

    setMessages(prev => [...prev, { role: "user", content: userText }]);

    const assistantIndex = messages.length + 1;
    setMessages(prev => [
      ...prev,
      {
        role: "assistant",
        content: "",
        experienceId: null,
        feedback: null,
        retryAvailable: false,
        escalated: false,
        meta: null,
        analysis: pendingAnalysis()
      }
    ]);
    startAnalysisProgress(assistantIndex, userText);

    try {
      const res = await axios.post("/chat/message", {
        sessionId,
        message: userText
      });

      const reply =
        typeof res.data.reply === "string"
          ? res.data.reply
          : "Sorry, something went wrong.";

      clearAnalysisTimers(assistantIndex);
      setMessages(prev => {
        const copy = [...prev];
        copy[assistantIndex] = {
          ...copy[assistantIndex],
          experienceId: res.data.experienceId,
          retryAvailable: res.data.retryAvailable || false,
          escalated: res.data.escalationSuggested || false,
          meta: {
            confidence: res.data.confidence,
            sourceType: res.data.sourceType,
            references: normalizeReferences(res.data.references),
            referencesCount: Array.isArray(res.data.references)
              ? res.data.references.length
              : 0
          },
          analysis: finalizeAnalysisTimeline(res.data.analysis)
        };
        return copy;
      });
      await renderAssistantReply(reply, assistantIndex);
    } catch (err) {
      console.error(err);
      clearAnalysisTimers(assistantIndex);
      setMessages(prev => {
        const copy = [...prev];
        copy[assistantIndex] = {
          role: "assistant",
          content: "Something went wrong. Please try again.",
          feedback: null,
          meta: null,
          analysis: []
        };
        return copy;
      });
    }
  };

  // ----------------------------
  // Retry with secondary chunks
  // ----------------------------
  const retryAnswer = async index => {
    const msg = messages[index];
    if (!msg?.experienceId || !msg.retryAvailable) return;

    setMessages(prev => {
      const copy = [...prev];
      copy[index] = {
        ...copy[index],
        retryAvailable: false
      };
      return copy;
    });

    const assistantIndex = messages.length;
    setMessages(prev => [
      ...prev,
      {
        role: "assistant",
        content: "",
        experienceId: null,
        feedback: null,
        retryAvailable: false,
        escalated: false,
        meta: null,
        analysis: pendingAnalysis()
      }
    ]);
    const lastUser = [...messages]
      .slice(0, index)
      .reverse()
      .find(item => item?.role === "user");
    startAnalysisProgress(assistantIndex, lastUser?.content || "");

    try {
      const res = await axios.post("/chat/retry", {
        sessionId,
        experienceId: msg.experienceId
      });

      clearAnalysisTimers(assistantIndex);
      setMessages(prev => {
        const copy = [...prev];
        copy[assistantIndex] = {
          ...copy[assistantIndex],
          experienceId: res.data.experienceId,
          retryAvailable: res.data.retryAvailable || false,
          escalated: res.data.escalationSuggested || false,
          meta: {
            confidence: res.data.confidence,
            sourceType: res.data.sourceType,
            references: normalizeReferences(res.data.references),
            referencesCount: Array.isArray(res.data.references)
              ? res.data.references.length
              : 0
          },
          analysis: finalizeAnalysisTimeline(res.data.analysis)
        };
        return copy;
      });
      await renderAssistantReply(res.data.reply, assistantIndex);
    } catch (err) {
      console.error("Retry failed:", err);
      clearAnalysisTimers(assistantIndex);
    }
  };

  // ----------------------------
  // Submit feedback
  // ----------------------------
  const submitFeedback = async (index, value) => {
    const msg = messages[index];
    if (!msg?.experienceId || msg.feedback) return;

    try {
      const res = await axios.post("/chat/feedback", {
        experienceId: msg.experienceId,
        feedback: value
      });

      setMessages(prev => {
        const copy = [...prev];
        copy[index] = {
          ...copy[index],
          feedback: value,
          retryAvailable: res.data.retryAllowed || false,
          escalated: res.data.escalationSuggested || false
        };
        return copy;
      });
    } catch (err) {
      console.error("Feedback failed:", err);
    }
  };

  useEffect(() => {
    if (suggestTimerRef.current) {
      clearTimeout(suggestTimerRef.current);
      suggestTimerRef.current = null;
    }

    const rawInput = input || "";
    const query = querySignature(rawInput);

    if (!sessionId || !shouldTriggerSuggestion(rawInput)) {
      setSuggestions([]);
      setSuggestionIndex(-1);
      setIsSuggesting(false);
      return;
    }

    const cached = readSuggestionCache(query);
    if (cached) {
      setSuggestions(cached);
      setSuggestionIndex(-1);
    }

    if (query === lastSuggestKeyRef.current) {
      setIsSuggesting(false);
      return;
    }

    const boundaryEnded = /[\s.,!?;:]$/.test(rawInput);
    const delayMs = boundaryEnded ? 120 : 360;

    const requestId = suggestRequestRef.current + 1;
    suggestRequestRef.current = requestId;

    suggestTimerRef.current = setTimeout(async () => {
      setIsSuggesting(true);
      try {
        const res = await axios.get("/chat/autocomplete", {
          params: {
            sessionId,
            q: query,
            limit: 3,
            futureWords: 3
          }
        });

        if (suggestRequestRef.current !== requestId) return;

        const itemsRaw = Array.isArray(res.data?.suggestions)
          ? res.data.suggestions
          : [];

        const items = itemsRaw.filter(item =>
          typeof item?.text === "string" &&
          item.text.trim() &&
          item.text.trim().toLowerCase() !== query.toLowerCase()
        );

        setSuggestions(items);
        setSuggestionIndex(-1);
        writeSuggestionCache(query, items);
        lastSuggestKeyRef.current = query;
      } catch (err) {
        if (suggestRequestRef.current !== requestId) return;
        // Keep previous suggestions on transient backend failures.
      } finally {
        if (suggestRequestRef.current === requestId) {
          setIsSuggesting(false);
        }
      }
    }, delayMs);

    return () => {
      if (suggestTimerRef.current) {
        clearTimeout(suggestTimerRef.current);
        suggestTimerRef.current = null;
      }
    };
  }, [
    input,
    sessionId,
    querySignature,
    readSuggestionCache,
    shouldTriggerSuggestion,
    writeSuggestionCache
  ]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => () => {
    if (suggestTimerRef.current) {
      clearTimeout(suggestTimerRef.current);
      suggestTimerRef.current = null;
    }
    Object.values(analysisTimersRef.current).forEach(timers => {
      if (Array.isArray(timers)) {
        timers.forEach(timerId => clearTimeout(timerId));
      }
    });
    analysisTimersRef.current = {};
  }, []);

  const starterItems = Array.isArray(starterQuestions)
    ? starterQuestions
      .map(item => {
        if (typeof item === "string") {
          return { text: item, count: 0 };
        }
        return {
          text: (item?.text || "").toString(),
          count: Number(item?.count || 0)
        };
      })
      .filter(item => item.text.trim())
      .slice(0, 5)
    : [];
  const maxStarterCount = starterItems.reduce(
    (maxVal, item) => Math.max(maxVal, Number(item?.count || 0)),
    0
  );

  return (
    <div className="gemini-root">
      <div className="gemini-chat">
        {messages.length === 0 && !input.trim() && starterItems.length > 0 && (
          <div className="starter-panel fade-in">
            <div className="starter-head">
              <div className="starter-head-copy">
                <span className="starter-kicker">Start Here</span>
                <span className="starter-title">Top Questions</span>
              </div>
              <span className="starter-subtitle">
                Ask one instantly or type your own
              </span>
            </div>
            <div className="starter-list">
              {starterItems.map((item, idx) => {
                const rawCount = Number(item?.count || 0);
                const popularity = maxStarterCount > 0
                  ? Math.max(12, Math.round((rawCount / maxStarterCount) * 100))
                  : 0;

                return (
                  <button
                    key={`${item.text}-${idx}`}
                    type="button"
                    className="starter-chip"
                    onClick={() => send(item.text)}
                  >
                    <span className="starter-rank">{idx + 1}</span>
                    <span className="starter-chip-main">
                      <span className="starter-chip-text">{item.text}</span>
                      <span className="starter-chip-meta">Quick start</span>
                    </span>
                    {rawCount > 0 && (
                      <span className="starter-chip-side">
                        <span className="starter-chip-count">{rawCount} asks</span>
                        <span className="starter-chip-bar">
                          <span
                            className="starter-chip-bar-fill"
                            style={{ width: `${popularity}%` }}
                          />
                        </span>
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
            <div className="starter-foot">
              These update as users interact with this bot.
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`gemini-msg ${m.role}`}>
            <div className="gemini-msg-inner fade-in">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {m.content}
              </ReactMarkdown>

              {m.role === "assistant" && m.meta && (
                <div className="meta-row">
                  <div className="meta-card">
                    <span className="meta-label">Confidence</span>
                    <span className="meta-value">
                      {Number(m.meta.confidence || 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="meta-card">
                    <span className="meta-label">Source</span>
                    <span className="meta-value">
                      {m.meta.sourceType || "docs"}
                    </span>
                  </div>
                  <div className="meta-card">
                    <span className="meta-label">Refs</span>
                    <span className="meta-value">
                      {m.meta.referencesCount || 0}
                    </span>
                  </div>
                </div>
              )}

              {m.role === "assistant" &&
                Array.isArray(m.meta?.references) &&
                m.meta.references.length > 0 && (
                <div className="reference-block">
                  <div className="reference-title">Sources</div>
                  <div className="reference-list">
                    {m.meta.references.map((ref, idx) => {
                      const label = ref.title || ref.pdf || ref.chunkRef || ref.note || `Source ${idx + 1}`;
                      const sub = ref.pdf && ref.chunkRef
                        ? `${ref.pdf} · ${ref.chunkRef}`
                        : ref.chunkRef || ref.pdf || ref.type;
                      const scoreText = Number(ref.score) > 0
                        ? `score ${Number(ref.score).toFixed(2)}`
                        : "";

                      return (
                        <div key={`${label}-${idx}`} className="reference-item">
                          <div className="reference-main">
                            {ref.url ? (
                              <a
                                href={ref.url}
                                target="_blank"
                                rel="noreferrer"
                                className="reference-link"
                              >
                                {label}
                              </a>
                            ) : (
                              <span className="reference-label">{label}</span>
                            )}
                            {(sub || scoreText) && (
                              <span className="reference-meta">
                                {[sub, scoreText].filter(Boolean).join(" · ")}
                              </span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {m.role === "assistant" &&
                Array.isArray(m.analysis) &&
                m.analysis.length > 0 && (
                <div className="analysis-trace">
                  {m.analysis.map((t, idx) => (
                    <div
                      key={idx}
                      className={`analysis-item ${statusFromDetail(t.detail)}`}
                    >
                      <span className="analysis-step">{t.step}</span>
                      {typeof t.ms === "number" && (
                        <span className="analysis-ms">{t.ms}ms</span>
                      )}
                      {t.detail && (
                        <span className="analysis-detail">{t.detail}</span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Feedback + Retry UX */}
              {m.role === "assistant" && m.experienceId && (
                <div className="feedback-bar">
                  <button
                    className={`feedback-btn ${
                      m.feedback === "positive" ? "active positive" : ""
                    }`}
                    disabled={!!m.feedback}
                    onClick={() => submitFeedback(i, "positive")}
                  >
                    Helpful
                  </button>

                  <button
                    className={`feedback-btn ${
                      m.feedback === "negative" ? "active negative" : ""
                    }`}
                    disabled={!!m.feedback}
                    onClick={() => submitFeedback(i, "negative")}
                  >
                    Not helpful
                  </button>

                  {m.retryAvailable && (
                    <button
                      className="retry-btn"
                      onClick={() => retryAnswer(i)}
                    >
                      Try another answer
                    </button>
                  )}

                  {m.escalated && (
                    <div className="escalation-msg">
                      This question has been forwarded to support.
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="gemini-input-wrap">
        <div className="gemini-input">
          <div className="gemini-input-field">
            <input
              value={input}
              onChange={e => {
                setInput(e.target.value);
                setSuggestionIndex(-1);
              }}
              placeholder="Ask Gemini"
              onKeyDown={handleInputKeyDown}
            />
            {(suggestions.length > 0 || isSuggesting) &&
              shouldTriggerSuggestion(input) && (
              <div className="autocomplete-popover">
                <div className="autocomplete-head">
                  <span className="autocomplete-title">Smart Suggestions</span>
                  {isSuggesting && (
                    <span className="autocomplete-loading">Updating...</span>
                  )}
                </div>
                {suggestions.map((item, idx) => (
                  <button
                    key={`${item.text}-${idx}`}
                    type="button"
                    className={`autocomplete-item ${
                      idx === suggestionIndex ? "active" : ""
                    }`}
                    onMouseDown={e => e.preventDefault()}
                    onClick={() => applySuggestion(item.text)}
                  >
                    <span className="autocomplete-text">{item.text}</span>
                    <span className="autocomplete-meta">
                      {Math.round(Number(item.confidence || 0) * 100)}%
                      {" \u00b7 "}
                      +{Number(item.future_words || 1)}w
                    </span>
                  </button>
                ))}
                <div className="autocomplete-foot">
                  Press Tab to insert
                </div>
              </div>
            )}
          </div>
          <button
            className="gemini-send-btn"
            onClick={send}
            disabled={!input.trim()}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
