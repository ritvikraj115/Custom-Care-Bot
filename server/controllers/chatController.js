import Bot from "../models/Bot.js";
import ChatSession from "../models/ChatSession.js";
import ChatMessage from "../models/ChatMessage.js";
import Experience from "../models/Experience.js";
import mongoose from "mongoose";
import crypto from "crypto";
import axios from "axios";
import Escalation from "../models/Escalation.js";
import { docServiceUrl } from "../config/serviceUrls.js";

const RAG_TIMEOUT_MS = Number.parseInt(
  process.env.RAG_TIMEOUT_MS || "",
  10
);
const RAG_TIMEOUT_SAFE_MS = Number.isFinite(RAG_TIMEOUT_MS)
  ? RAG_TIMEOUT_MS
  : 90000;
const DEBUG_RAG_PAYLOAD = (process.env.DEBUG_RAG_PAYLOAD || "").trim() === "1";

const SESSION_CACHE_TTL_MS = 5 * 60 * 1000;
const SESSION_CACHE_MAX = 2000;
const SUGGEST_CACHE_TTL_MS = 30 * 1000;
const SUGGEST_CACHE_MAX = 4000;
const TOP_QUESTIONS_CACHE_TTL_MS = 90 * 1000;
const TOP_QUESTIONS_CACHE_MAX = 1200;
const CONVERSATION_CACHE_TTL_MS = 2 * 60 * 1000;
const CONVERSATION_CACHE_MAX = 3000;
const ANSWER_CACHE_TTL_MS = 5 * 60 * 1000;
const ANSWER_CACHE_MAX = 5000;
const CONVERSATION_LIMIT = 12;

const SESSION_CACHE = new Map();
const SUGGEST_CACHE = new Map();
const TOP_QUESTIONS_CACHE = new Map();
const CONVERSATION_CACHE = new Map();
const ANSWER_CACHE = new Map();

const cacheGet = (map, key, ttlMs) => {
  const entry = map.get(key);
  if (!entry) return null;
  if (Date.now() - entry.ts > ttlMs) {
    map.delete(key);
    return null;
  }
  return entry.value;
};

const cacheSet = (map, key, value, maxSize) => {
  map.set(key, { ts: Date.now(), value });
  if (map.size > maxSize) {
    const oldestKey = map.keys().next().value;
    if (oldestKey) map.delete(oldestKey);
  }
};

const normalizeSuggestQuery = value =>
  (value || "").toString().toLowerCase().replace(/\s+/g, " ").trim();
const normalizeQuestionKey = value =>
  (value || "").toString().toLowerCase().replace(/\s+/g, " ").trim();


/* ==================================================
   Utilities
================================================== */

const computeAvgSimilarity = chunks => {
  if (!chunks || !chunks.length) return 0;
  const sum = chunks.reduce((a, c) => a + (c.score || 0), 0);
  return sum / chunks.length;
};

const normalizeNumber = value =>
  typeof value === "number" && !Number.isNaN(value) ? value : 0;

const chunkIdSet = chunks =>
  new Set((chunks || []).map(c => c.chunk_ref || c.chunkId || "unknown"));

const chunkSetsEqual = (a, b) => {
  if (a.size !== b.size) return false;
  for (const item of a) {
    if (!b.has(item)) return false;
  }
  return true;
};

const includesAny = (text, patterns) =>
  patterns.some(p => text.includes(p));

const extractIntentLabelFromTrace = trace => {
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

const resolveCanonicalQuestion = (conversation, currentQuestion, intentLabel) => {
  if (intentLabel !== "dissatisfied_retry") {
    return currentQuestion;
  }
  if (!Array.isArray(conversation) || conversation.length === 0) {
    return currentQuestion;
  }

  const current = (currentQuestion || "").trim().toLowerCase();
  const userMessages = conversation
    .filter(item => item?.role === "user" && typeof item.content === "string")
    .map(item => item.content.trim())
    .filter(Boolean);

  if (userMessages.length <= 1) {
    return currentQuestion;
  }

  for (let i = userMessages.length - 1; i >= 0; i -= 1) {
    const candidate = userMessages[i];
    const isLast = i === userMessages.length - 1;
    if (isLast && candidate.toLowerCase() === current) {
      continue;
    }
    return candidate;
  }

  return currentQuestion;
};

const shouldUseGraphIntentRouting = (question, conversation = []) => {
  const q = (question || "").toLowerCase();
  const hasConversation = Array.isArray(conversation) && conversation.length > 0;

  const dissatisfactionPatterns = [
    "not satisfied",
    "different answer",
    "another answer",
    "try again",
    "regenerate",
    "not helpful",
    "wrong answer",
    "not correct"
  ];

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
    "post"
  ];

  const websitePatterns = [
    "website",
    "web site",
    "webpage",
    "web page",
    "homepage",
    "home page",
    "pricing page",
    "contact page"
  ];

  const dissatisfied =
    hasConversation && includesAny(q, dissatisfactionPatterns);
  const latestSocial =
    includesAny(q, latestPatterns) && includesAny(q, socialPatterns);
  const websiteIntent = includesAny(q, websitePatterns);

  return dissatisfied || latestSocial || websiteIntent;
};

const buildConversation = async (sessionId, limit = 12) => {
  const items = await ChatMessage.find({ sessionId })
    .sort({ createdAt: -1 })
    .limit(limit)
    .lean();

  return items
    .reverse()
    .map(m => ({ role: m.role, content: m.content }));
};

const getConversationCached = async (sessionId, limit = CONVERSATION_LIMIT) => {
  const key = sessionId.toString();
  const cached = cacheGet(CONVERSATION_CACHE, key, CONVERSATION_CACHE_TTL_MS);
  if (Array.isArray(cached)) {
    return cached.slice(-limit);
  }

  const fresh = await buildConversation(sessionId, limit);
  cacheSet(CONVERSATION_CACHE, key, fresh, CONVERSATION_CACHE_MAX);
  return fresh;
};

const appendConversationCache = (
  sessionId,
  role,
  content,
  maxKeep = CONVERSATION_LIMIT * 3
) => {
  const key = sessionId.toString();
  const current = cacheGet(CONVERSATION_CACHE, key, CONVERSATION_CACHE_TTL_MS) || [];
  const updated = [
    ...current,
    { role, content: (content || "").toString() }
  ].slice(-maxKeep);
  cacheSet(CONVERSATION_CACHE, key, updated, CONVERSATION_CACHE_MAX);
};

const DYNAMIC_PATTERNS = [
  "latest",
  "recent",
  "today",
  "current",
  "news",
  "update",
  "updates"
];

const RETRY_PATTERNS = [
  "not satisfied",
  "different answer",
  "another answer",
  "try again",
  "regenerate",
  "not helpful",
  "wrong answer",
  "not correct"
];

const FEEDBACK_BLOCK_THRESHOLD = -2;
const RETRY_SCORE_FLOOR = -1;
const ESCALATION_SCORE_FLOOR = 0;
const SECONDARY_MIN_SIMILARITY = 0.15;
const OWNER_RESOLVED_SCORE = 5;

const isRetryPrompt = question =>
  includesAny((question || "").toLowerCase(), RETRY_PATTERNS);

const normalizeObjectIdString = value => {
  if (!value) return null;
  return value.toString();
};

const buildSemanticGroupFilter = (experience, semanticIdOverride = null) => {
  const semanticId = semanticIdOverride || experience?.semanticId;
  if (semanticId) {
    return {
      botId: experience.botId,
      semanticId
    };
  }
  return { _id: experience._id };
};

const ensureSemanticIdForExperience = async (experience) => {
  if (!experience) return null;
  if (experience.semanticId) {
    return normalizeObjectIdString(experience.semanticId);
  }
  const assigned = new mongoose.Types.ObjectId();
  experience.semanticId = assigned;
  await experience.save();
  return assigned.toString();
};

const getGroupChunkRefs = async (experience, semanticIdOverride = null) => {
  if (!experience) return [];
  const filter = buildSemanticGroupFilter(experience, semanticIdOverride);
  const groupRows = await Experience.find(filter, { retrievedChunks: 1 }).lean();
  const refs = new Set();
  for (const row of groupRows) {
    for (const chunk of row?.retrievedChunks || []) {
      const id = (chunk?.chunkId || "").toString().trim();
      if (id) refs.add(id);
    }
  }
  return Array.from(refs);
};

const syncSemanticFeedbackToVector = async ({
  experience,
  semanticId,
  feedbackScore,
  negativeCount,
  delta
}) => {
  try {
    await axios.post(
      docServiceUrl("/experience/update-feedback"),
      {
        semantic_id: semanticId || null,
        bot_id: experience.botId.toString(),
        experience_id: experience._id.toString(),
        delta,
        feedback_score: feedbackScore,
        negative_count: negativeCount
      }
    );
  } catch (vecErr) {
    console.error(
      "?????? Vector feedback update failed:",
      vecErr.response?.data || vecErr.message
    );
  }
};

const applySemanticFeedbackDelta = async ({
  experience,
  delta = 0,
  reason = ""
}) => {
  const semanticId = await ensureSemanticIdForExperience(experience);
  const groupFilter = buildSemanticGroupFilter(experience, semanticId);
  const seed = await Experience.findOne(groupFilter).sort({ createdAt: -1 }).lean();
  const currentScore = normalizeNumber(seed?.feedbackScore);
  const currentNegative = normalizeNumber(seed?.negativeCount);

  const feedbackScore = currentScore + delta;
  const negativeCount = currentNegative + (delta < 0 ? 1 : 0);

  await Experience.updateMany(groupFilter, {
    feedbackScore,
    negativeCount,
    feedbackAt: new Date(),
    feedbackReason: reason || ""
  });

  if (delta !== 0) {
    await syncSemanticFeedbackToVector({
      experience,
      semanticId,
      feedbackScore,
      negativeCount,
      delta
    });
  }

  return {
    semanticId,
    previousScore: currentScore,
    previousNegativeCount: currentNegative,
    feedbackScore,
    negativeCount
  };
};

const upsertSemanticExperience = async ({
  session,
  semanticId,
  question,
  answer,
  retrievedChunks,
  avgChunkSimilarity,
  feedbackScore,
  negativeCount,
  retrievalVariant,
  preferredExperienceId = null
}) => {
  let experience = null;
  if (preferredExperienceId) {
    experience = await Experience.findById(preferredExperienceId);
  }

  if (!experience && semanticId) {
    experience = await Experience.findOne({
      botId: session.botId,
      semanticId,
      retrievalVariant,
      resolvedByOwner: false
    }).sort({ createdAt: -1 });
  }

  if (!experience && semanticId) {
    experience = await Experience.findOne({
      botId: session.botId,
      semanticId,
      resolvedByOwner: false
    }).sort({ createdAt: -1 });
  }

  const payload = {
    botId: session.botId,
    clientId: session.tenantId,
    sessionId: session._id,
    semanticId,
    question,
    answer,
    retrievedChunks: (retrievedChunks || []).map(c => ({
      chunkId: c.chunk_ref || "unknown",
      similarity: c.score ?? 0
    })),
    avgChunkSimilarity,
    feedbackScore,
    negativeCount,
    status: "active",
    retrievalVariant
  };

  if (experience && !experience.resolvedByOwner) {
    experience.question = payload.question;
    experience.answer = payload.answer;
    experience.retrievedChunks = payload.retrievedChunks;
    experience.avgChunkSimilarity = payload.avgChunkSimilarity;
    experience.feedbackScore = payload.feedbackScore;
    experience.negativeCount = payload.negativeCount;
    experience.feedbackAt = new Date();
    experience.status = payload.status;
    experience.retrievalVariant = payload.retrievalVariant;
    experience.sessionId = payload.sessionId;
    await experience.save();
    return experience;
  }

  return Experience.create(payload);
};

const ensureEscalation = async ({ experience, reason }) => {
  if (!experience || experience.resolvedByOwner) return false;

  try {
    const existing = await Escalation.findOne({
      status: "open",
      experienceIds: experience._id
    });

    if (existing) return true;

    await Experience.updateOne(
      { _id: experience._id },
      { status: "escalated" }
    );

    const cleanReason =
      typeof reason === "string" ? reason.trim() : "";

    await Escalation.create({
      botId: experience.botId,
      question: experience.question,
      experienceIds: [experience._id],
      ...(cleanReason ? { reason: cleanReason } : {})
    });

    return true;
  } catch (err) {
    console.error("Escalation create failed:", err.message);
    return false;
  }
};

const getBotLinkPayload = async (botId, clientId) => {
  try {
    const bot = await Bot.findOne({
      _id: botId,
      tenantId: clientId
    }).lean();

    const socialLinks = {};
    if (bot?.facebookUrl) {
      socialLinks.facebook = bot.facebookUrl;
    }
    if (bot?.instagramUrl) {
      socialLinks.instagram = bot.instagramUrl;
    }

    return {
      websiteUrl: bot?.websiteUrl || "",
      socialLinks
    };
  } catch (err) {
    console.error("Failed to load bot links:", err.message);
    return { websiteUrl: "", socialLinks: {} };
  }
};

// index experience into vector DB (async-safe)
const indexExperienceVector = async ({
  experienceId,
  question,
  session,
  avgChunkSimilarity,
  feedbackScore = 0,
  negativeCount = 0,
  semanticId = null,
  ownerAnswer = false
}) => {
  try {
    await axios.post(docServiceUrl("/experience/index"), {
      experience_id: experienceId,
      text: question,
      bot_id: session.botId.toString(),
      client_id: session.tenantId.toString(),
      semantic_id: semanticId,
      feedback_score: feedbackScore,
      negative_count: negativeCount,
      owner_answer: ownerAnswer,
      avg_chunk_similarity: avgChunkSimilarity
    });
  } catch (err) {
    console.error("?????? Experience indexing failed:", err.message);
  }
};

// search experience memory
const searchExperience = async (session, question) => {
  try {
    const res = await axios.post(
      docServiceUrl("/experience/search"),
      {
        query: question,
        bot_id: session.botId.toString()
      }
    );
    return res.data;
  } catch (err) {
    console.error("⚠️ Experience search failed:", err.message);
    return null;
  }
};

const recordAutocompleteQuestion = async (session, question) => {
  try {
    await axios.post(
      docServiceUrl("/autocomplete/record-question"),
      {
        query: question,
        bot_id: session.botId.toString(),
        client_id: session.tenantId.toString()
      }
    );

    const prefix = `${session.botId.toString()}|${session.tenantId.toString()}|`;
    for (const key of TOP_QUESTIONS_CACHE.keys()) {
      if (key.startsWith(prefix)) {
        TOP_QUESTIONS_CACHE.delete(key);
      }
    }
  } catch (err) {
    console.error(
      "Autocomplete record failed:",
      err.response?.data || err.message
    );
  }
};

const fetchTopStarterQuestions = async (botId, clientId, limit = 5) => {
  const safeLimit = Math.max(1, Math.min(Number(limit) || 5, 5));
  const cacheKey = `${botId}|${clientId}|${safeLimit}`;
  const cached = cacheGet(
    TOP_QUESTIONS_CACHE,
    cacheKey,
    TOP_QUESTIONS_CACHE_TTL_MS
  );
  if (cached) {
    return cached;
  }

  try {
    const response = await axios.post(
      docServiceUrl("/autocomplete/top-questions"),
      {
        bot_id: botId,
        client_id: clientId,
        limit: safeLimit
      },
      { timeout: 4500 }
    );

    const itemsRaw = Array.isArray(response.data?.questions)
      ? response.data.questions
      : [];
    const items = itemsRaw
      .map(item => ({
        text: (item?.text || "").toString().trim(),
        count: Number(item?.count || 0)
      }))
      .filter(item => item.text);

    cacheSet(TOP_QUESTIONS_CACHE, cacheKey, items, TOP_QUESTIONS_CACHE_MAX);
    return items;
  } catch (err) {
    console.error(
      "Autocomplete top questions failed:",
      err.response?.data || err.message
    );
    return [];
  }
};

/* ==================================================
   Create anonymous chat session
================================================== */

export const createSession = async (req, res) => {
  try {
    const { botId } = req.params;
    if (!botId) {
      return res.status(400).json({ message: "Bot ID required" });
    }

    const bot = await Bot.findById(botId);
    if (!bot || !bot.publicAccess || !bot.isActive) {
      return res.status(404).json({ message: "Bot not available" });
    }

    const ip =
      req.headers["x-forwarded-for"]?.split(",")[0] ||
      req.ip ||
      "unknown";

    const ipHash = crypto
      .createHash("sha256")
      .update(ip)
      .digest("hex");

    const session = await ChatSession.create({
      botId: bot._id,
      tenantId: bot.tenantId,
      ipHash,
      userAgent: req.headers["user-agent"] || "unknown",
      lastActiveAt: new Date()
    });

    cacheSet(
      SESSION_CACHE,
      session._id.toString(),
      {
        botId: session.botId.toString(),
        tenantId: session.tenantId.toString()
      },
      SESSION_CACHE_MAX
    );

    const starterQuestions = await fetchTopStarterQuestions(
      session.botId.toString(),
      session.tenantId.toString(),
      5
    );

    res.json({
      sessionId: session._id,
      starterQuestions
    });
  } catch (err) {
    console.error("Create session error:", err);
    res.status(500).json({ message: "Failed to create session" });
  }
};

export const getAutocompleteSuggestions = async (req, res) => {
  try {
    const { sessionId, q } = req.query;
    if (!sessionId) {
      return res.status(400).json({ message: "sessionId is required" });
    }

    const query = normalizeSuggestQuery(q);
    if (!query || query.length < 4) {
      return res.json({ suggestions: [], status: null });
    }

    const sessionKey = sessionId.toString();
    let sessionMeta = cacheGet(SESSION_CACHE, sessionKey, SESSION_CACHE_TTL_MS);
    if (!sessionMeta) {
      const session = await ChatSession.findById(sessionId).lean();
      if (!session) {
        return res.status(404).json({ message: "Invalid session" });
      }
      sessionMeta = {
        botId: session.botId.toString(),
        tenantId: session.tenantId.toString()
      };
      cacheSet(SESSION_CACHE, sessionKey, sessionMeta, SESSION_CACHE_MAX);
    }

    const limitRaw = Number.parseInt((req.query.limit || "3").toString(), 10);
    const futureRaw = Number.parseInt((req.query.futureWords || "3").toString(), 10);
    const maxSuggestions = Number.isFinite(limitRaw)
      ? Math.max(1, Math.min(limitRaw, 3))
      : 3;
    const maxFutureWords = Number.isFinite(futureRaw)
      ? Math.max(1, Math.min(futureRaw, 3))
      : 3;

    const suggestCacheKey = [
      sessionKey,
      sessionMeta.botId,
      sessionMeta.tenantId,
      query,
      maxSuggestions,
      maxFutureWords
    ].join("|");

    const cached = cacheGet(SUGGEST_CACHE, suggestCacheKey, SUGGEST_CACHE_TTL_MS);
    if (cached) {
      return res.json(cached);
    }

    const response = await axios.post(
      docServiceUrl("/autocomplete/suggest"),
      {
        query,
        bot_id: sessionMeta.botId,
        client_id: sessionMeta.tenantId,
        max_suggestions: maxSuggestions,
        max_future_words: maxFutureWords
      },
      { timeout: 4500 }
    );

    const payload = response.data || { suggestions: [], status: null };
    cacheSet(SUGGEST_CACHE, suggestCacheKey, payload, SUGGEST_CACHE_MAX);
    return res.json(payload);
  } catch (err) {
    console.error(
      "Autocomplete suggest failed:",
      err.response?.data || err.message
    );

    const sessionKey = (req.query.sessionId || "").toString();
    const query = normalizeSuggestQuery(req.query.q || "");
    if (sessionKey && query) {
      // Return the newest cached item for this session as resilience fallback.
      for (const [key, entry] of Array.from(SUGGEST_CACHE.entries()).reverse()) {
        if (!key.startsWith(`${sessionKey}|`)) continue;
        if (Date.now() - entry.ts > SUGGEST_CACHE_TTL_MS) continue;
        if (typeof entry?.value === "object" && entry.value) {
          return res.json(entry.value);
        }
      }
    }

    return res.json({ suggestions: [], status: null });
  }
};

/* ==================================================
   NON-STREAMING CHAT
================================================== */
export const sendMessage = async (req, res) => {
  try {
    const { sessionId, message } = req.body;
    if (!sessionId || !message?.trim()) {
      return res.status(400).json({ message: "Invalid message" });
    }

    const session = await ChatSession.findById(sessionId);
    if (!session) {
      return res.status(404).json({ message: "Invalid session" });
    }

    const question = message.trim();
    const normalizedQuestion = normalizeQuestionKey(question);
    const retryPromptRequest = isRetryPrompt(normalizedQuestion);
    const answerCacheKey = `${session.botId.toString()}|${session.tenantId.toString()}|${normalizedQuestion}`;

    // 1) Save USER message
    await ChatMessage.create({
      sessionId,
      role: "user",
      content: question
    });
    appendConversationCache(sessionId, "user", question);
    recordAutocompleteQuestion(session, question);

    let conversation = await getConversationCached(sessionId, CONVERSATION_LIMIT);
    if (
      conversation.length === 0 ||
      conversation[conversation.length - 1]?.role !== "user" ||
      (conversation[conversation.length - 1]?.content || "").trim() !== question
    ) {
      conversation = [...conversation, { role: "user", content: question }]
        .slice(-CONVERSATION_LIMIT);
    }
    const forceGraphIntentRouting = shouldUseGraphIntentRouting(
      question,
      conversation
    );
    const bypassAnswerCache = (
      forceGraphIntentRouting ||
      includesAny(normalizedQuestion, DYNAMIC_PATTERNS) ||
      includesAny(normalizedQuestion, RETRY_PATTERNS)
    );

    if (!bypassAnswerCache) {
      const cachedPayload = cacheGet(ANSWER_CACHE, answerCacheKey, ANSWER_CACHE_TTL_MS);
      if (cachedPayload) {
        await ChatMessage.create({
          sessionId,
          role: "assistant",
          content: cachedPayload.reply
        });
        appendConversationCache(sessionId, "assistant", cachedPayload.reply);

        session.lastActiveAt = new Date();
        await session.save();

        return res.json({
          ...cachedPayload,
          reused: true
        });
      }
    }

    /* ===============================
       EXPERIENCE SEARCH (SEMANTIC)
    =============================== */

    const experienceHit = await searchExperience(session, question);
    const hitFeedbackScore = normalizeNumber(experienceHit?.feedback_score);
    const hitNegativeCount = normalizeNumber(experienceHit?.negative_count);
    const hitOwnerAnswer = !!experienceHit?.owner_answer;
    const shouldAutoForward = hitFeedbackScore < FEEDBACK_BLOCK_THRESHOLD;
    let semanticId = experienceHit?.semantic_id || null;

    let semanticExperience = null;

    if (experienceHit?.experience_id) {
      semanticExperience = await Experience.findById(
        experienceHit.experience_id
      );

      if (!semanticExperience && semanticId) {
        if (hitOwnerAnswer) {
          semanticExperience = await Experience.findOne({
            botId: session.botId,
            semanticId,
            resolvedByOwner: true
          }).sort({ createdAt: -1 });
        }

        if (!semanticExperience) {
          semanticExperience = await Experience.findOne({
            botId: session.botId,
            semanticId
          }).sort({ createdAt: -1 });
        }
      }
    }

    if (retryPromptRequest && !semanticExperience && semanticId) {
      semanticExperience = await Experience.findOne({
        botId: session.botId,
        semanticId
      }).sort({ createdAt: -1 });
    }

    if (semanticExperience && !retryPromptRequest) {
      const isOwnerAnswer =
        semanticExperience.resolvedByOwner || hitOwnerAnswer;
      const feedbackScore = Number.isFinite(
        semanticExperience.feedbackScore
      )
        ? semanticExperience.feedbackScore
        : hitFeedbackScore;

      if (!semanticExperience.semanticId) {
        const assignedSemanticId =
          semanticId || new mongoose.Types.ObjectId();
        semanticExperience.semanticId = assignedSemanticId;
        await semanticExperience.save();

        semanticId = assignedSemanticId.toString();

        indexExperienceVector({
          experienceId: semanticExperience._id,
          question: semanticExperience.question,
          session,
          avgChunkSimilarity: semanticExperience.avgChunkSimilarity,
          feedbackScore,
          negativeCount: normalizeNumber(
            semanticExperience.negativeCount
          ),
          semanticId,
          ownerAnswer: isOwnerAnswer
        });
      } else {
        semanticId = semanticExperience.semanticId.toString();
      }

      if (feedbackScore < FEEDBACK_BLOCK_THRESHOLD) {
        const reply = "We are working on this";

        if (shouldAutoForward) {
          await ensureEscalation({
            experience: semanticExperience,
            reason: "auto-forward"
          });
        }

        await ChatMessage.create({
          sessionId,
          role: "assistant",
          content: reply
        });
        appendConversationCache(sessionId, "assistant", reply);

        session.lastActiveAt = new Date();
        await session.save();

        return res.json({
          reply,
          experienceId: semanticExperience._id,
          reused: true,
          retryAvailable: false,
          retrievalVariant: semanticExperience.retrievalVariant,
          confidence: normalizeNumber(semanticExperience.avgChunkSimilarity),
          references: [],
          sourceType: semanticExperience.retrievalVariant === "owner"
            ? "human"
            : "docs",
          analysis: [
            { step: "CheckFeedbackState", ms: 0, detail: "feedback_block" },
            { step: "HumanInLoopNode", ms: 0, detail: "handoff" }
          ],
          escalationSuggested: shouldAutoForward
        });
      }

      if (shouldAutoForward) {
        await ensureEscalation({
          experience: semanticExperience,
          reason: "auto-forward"
        });
      }

      await ChatMessage.create({
        sessionId,
        role: "assistant",
        content: semanticExperience.answer
      });
      appendConversationCache(sessionId, "assistant", semanticExperience.answer);

      session.lastActiveAt = new Date();
      await session.save();

      return res.json({
        reply: semanticExperience.answer,
        experienceId: semanticExperience._id,
        reused: true,
          retryAvailable: !isOwnerAnswer && feedbackScore >= RETRY_SCORE_FLOOR,
        retrievalVariant: semanticExperience.retrievalVariant,
        confidence: normalizeNumber(semanticExperience.avgChunkSimilarity),
        references: [],
        sourceType: semanticExperience.retrievalVariant === "owner"
          ? "human"
          : "docs",
        analysis: [
          { step: "CheckFeedbackState", ms: 0, detail: "memory_hit" },
          { step: "FinalizeResponse", ms: 0, detail: "reuse" }
        ],
        escalationSuggested: shouldAutoForward
      });
    }

    if (
      experienceHit &&
      hitFeedbackScore < FEEDBACK_BLOCK_THRESHOLD &&
      !retryPromptRequest
    ) {
      const reply = "We are working on this";

      if (shouldAutoForward && experienceHit?.experience_id) {
        const exp = await Experience.findById(experienceHit.experience_id);
        if (exp) {
          await ensureEscalation({ experience: exp, reason: "auto-forward" });
        }
      }

      await ChatMessage.create({
        sessionId,
        role: "assistant",
        content: reply
      });
      appendConversationCache(sessionId, "assistant", reply);

      session.lastActiveAt = new Date();
      await session.save();

      return res.json({
        reply,
        experienceId: experienceHit.experience_id || null,
        reused: true,
        retryAvailable: false,
        confidence: 0,
        references: [],
        sourceType: "human",
        analysis: [
          { step: "CheckFeedbackState", ms: 0, detail: "feedback_block" },
          { step: "HumanInLoopNode", ms: 0, detail: "handoff" }
        ],
        escalationSuggested: shouldAutoForward
      });
    }

    /* ===============================
       RAG CALL (PRIMARY)
    =============================== */

    let retrievalVariant = "primary";
    let retryPromptFeedback = null;
    let excludeChunkRefs = [];

    if (retryPromptRequest && semanticExperience && !semanticExperience.resolvedByOwner) {
      retryPromptFeedback = await applySemanticFeedbackDelta({
        experience: semanticExperience,
        delta: -1,
        reason: "prompt_retry_secondary"
      });
      semanticId = retryPromptFeedback.semanticId;
      retrievalVariant = "secondary";
      excludeChunkRefs = await getGroupChunkRefs(
        semanticExperience,
        semanticId
      );

      if (retryPromptFeedback.feedbackScore < 1) {
        await ensureEscalation({
          experience: semanticExperience,
          reason: "prompt_retry_low_score"
        });
      }

      if (retryPromptFeedback.feedbackScore < FEEDBACK_BLOCK_THRESHOLD) {
        await ensureEscalation({
          experience: semanticExperience,
          reason: "prompt_retry_feedback_block"
        });
        const blockedReply = "We are working on this";
        await ChatMessage.create({
          sessionId,
          role: "assistant",
          content: blockedReply
        });
        appendConversationCache(sessionId, "assistant", blockedReply);

        session.lastActiveAt = new Date();
        await session.save();

        return res.json({
          reply: blockedReply,
          experienceId: semanticExperience._id,
          reused: true,
          retryAvailable: false,
          retrievalVariant: semanticExperience.retrievalVariant || "primary",
          confidence: 0,
          references: [],
          sourceType: "human",
          analysis: [
            { step: "CheckFeedbackState", ms: 0, detail: "feedback_block" },
            { step: "HumanInLoopNode", ms: 0, detail: "handoff" }
          ],
          escalationSuggested: true
        });
      }
    }

    let reply = "Sorry, I couldn't retrieve an answer right now.";
    let retrievedChunks = [];

    const botLinks = await getBotLinkPayload(
      session.botId,
      session.tenantId
    );

    const ragPayload = {
      query: question,
      bot_id: session.botId.toString(),
      client_id: session.tenantId.toString(),
      retrieval_variant: retrievalVariant,
      top_k: 3,
      conversation
    };
    if (retrievalVariant === "secondary" && excludeChunkRefs.length > 0) {
      ragPayload.exclude_chunk_refs = excludeChunkRefs;
    }

    if (botLinks.websiteUrl) {
      ragPayload.website_url = botLinks.websiteUrl;
    }
    if (Object.keys(botLinks.socialLinks).length) {
      ragPayload.social_links = botLinks.socialLinks;
    }

    const ragRes = await axios.post(
      docServiceUrl("/answer"),
      ragPayload,
      { timeout: RAG_TIMEOUT_SAFE_MS }
    );

    if (DEBUG_RAG_PAYLOAD) {
      console.log("RAG response meta:", {
        hasAnswer: !!ragRes?.data?.answer,
        chunkCount: Array.isArray(ragRes?.data?.chunks)
          ? ragRes.data.chunks.length
          : 0,
        confidence: ragRes?.data?.confidence ?? null,
        sourceType: ragRes?.data?.source_type || "docs",
        noDocs: !!ragRes?.data?.no_docs
      });
    }

    if (ragRes?.data?.answer?.trim()) {
      reply = ragRes.data.answer.trim();
    }

    if (Array.isArray(ragRes?.data?.chunks)) {
      retrievedChunks = ragRes.data.chunks;
    }

    const confidence = normalizeNumber(ragRes?.data?.confidence);
    const references = Array.isArray(ragRes?.data?.references)
      ? ragRes.data.references
      : [];
    const sourceType = ragRes?.data?.source_type || "docs";
    const analysis = Array.isArray(ragRes?.data?.trace)
      ? ragRes.data.trace
      : [];
    const noDocs = !!ragRes?.data?.no_docs;
    const graphEscalationRequired =
      sourceType === "human" || noDocs;
    const intentLabel = extractIntentLabelFromTrace(analysis);
    const canonicalQuestion = resolveCanonicalQuestion(
      conversation,
      question,
      intentLabel
    );

    // Save ASSISTANT message
    await ChatMessage.create({
      sessionId,
      role: "assistant",
      content: reply
    });
    appendConversationCache(sessionId, "assistant", reply);

    const avgChunkSimilarity =
      computeAvgSimilarity(retrievedChunks);

    const assignedSemanticId = semanticId || new mongoose.Types.ObjectId();

    const seedFeedbackScore = retryPromptFeedback
      ? retryPromptFeedback.feedbackScore
      : experienceHit
        ? hitFeedbackScore
        : 0;
    const seedNegativeCount = retryPromptFeedback
      ? retryPromptFeedback.negativeCount
      : experienceHit
        ? hitNegativeCount
        : 0;

    const canonicalQuestionFinal = retryPromptRequest && semanticExperience
      ? semanticExperience.question
      : canonicalQuestion;

    const experience = await upsertSemanticExperience({
      session,
      semanticId: assignedSemanticId,
      question: canonicalQuestionFinal,
      answer: reply,
      retrievedChunks,
      avgChunkSimilarity,
      feedbackScore: seedFeedbackScore,
      negativeCount: seedNegativeCount,
      retrievalVariant
    });

    if (graphEscalationRequired) {
      await ensureEscalation({
        experience,
        reason: noDocs
          ? "no-docs-threshold"
          : "graph-human-handoff"
      });
    }

    // index experience asynchronously
    indexExperienceVector({
      experienceId: experience._id,
      question: canonicalQuestionFinal,
      session,
      avgChunkSimilarity,
      feedbackScore: seedFeedbackScore,
      negativeCount: seedNegativeCount,
      semanticId: assignedSemanticId.toString(),
      ownerAnswer: false
    });

    session.lastActiveAt = new Date();
    await session.save();

    /* ===============================
       RESPONSE TO FRONTEND
    =============================== */

    const responsePayload = {
      reply,
      experienceId: experience._id,
      reused: false,
      retryAvailable: !experience.resolvedByOwner &&
        seedFeedbackScore >= RETRY_SCORE_FLOOR,
      retrievalVariant,
      confidence,
      references,
      sourceType,
      analysis,
      escalationSuggested: graphEscalationRequired
    };

    if (!bypassAnswerCache) {
      cacheSet(ANSWER_CACHE, answerCacheKey, responsePayload, ANSWER_CACHE_MAX);
    }

    return res.json(responsePayload);
  } catch (err) {
    console.error("Send message error:", err);
    return res.status(500).json({
      reply: "Something went wrong. Please try again."
    });
  }
};

/* ==================================================
   STREAMING (FULLY PRESERVED)
================================================== */

export const streamMessage = async (req, res) => {
  const { sessionId, message } = req.query;

  if (!sessionId || !message?.trim()) {
    return res.status(400).end();
  }

  const session = await ChatSession.findById(sessionId);
  if (!session) {
    return res.status(404).end();
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const question = message.trim();

  await ChatMessage.create({
    sessionId,
    role: "user",
    content: question
  });
  appendConversationCache(sessionId, "user", question);
  recordAutocompleteQuestion(session, question);

  let conversation = await getConversationCached(sessionId, CONVERSATION_LIMIT);
  if (
    conversation.length === 0 ||
    conversation[conversation.length - 1]?.role !== "user" ||
    (conversation[conversation.length - 1]?.content || "").trim() !== question
  ) {
    conversation = [...conversation, { role: "user", content: question }]
      .slice(-CONVERSATION_LIMIT);
  }
  const forceGraphIntentRouting = shouldUseGraphIntentRouting(
    question,
    conversation
  );

  let fullAnswer = "";
    let retrievedChunks = [];

    try {
    const experienceHit = await searchExperience(session, question);
    const hitFeedbackScore = normalizeNumber(experienceHit?.feedback_score);
    const hitNegativeCount = normalizeNumber(experienceHit?.negative_count);
    const hitOwnerAnswer = !!experienceHit?.owner_answer;
    const shouldAutoForward = hitFeedbackScore < -2;
    let semanticId = experienceHit?.semantic_id || null;

    let semanticExperience = null;

    if (experienceHit?.experience_id) {
      semanticExperience = await Experience.findById(
        experienceHit.experience_id
      );

      if (!semanticExperience && semanticId) {
        if (hitOwnerAnswer) {
          semanticExperience = await Experience.findOne({
            botId: session.botId,
            semanticId,
            resolvedByOwner: true
          }).sort({ createdAt: -1 });
        }

        if (!semanticExperience) {
          semanticExperience = await Experience.findOne({
            botId: session.botId,
            semanticId
          }).sort({ createdAt: -1 });
        }
      }
    }

    if (semanticExperience) {
      const isOwnerAnswer =
        semanticExperience.resolvedByOwner || hitOwnerAnswer;
      const feedbackScore = Number.isFinite(
        semanticExperience.feedbackScore
      )
        ? semanticExperience.feedbackScore
        : hitFeedbackScore;

      if (!semanticExperience.semanticId) {
        const assignedSemanticId =
          semanticId || new mongoose.Types.ObjectId();
        semanticExperience.semanticId = assignedSemanticId;
        await semanticExperience.save();

        semanticId = assignedSemanticId.toString();

        indexExperienceVector({
          experienceId: semanticExperience._id,
          question: semanticExperience.question,
          session,
          avgChunkSimilarity: semanticExperience.avgChunkSimilarity,
          feedbackScore,
          negativeCount: normalizeNumber(
            semanticExperience.negativeCount
          ),
          semanticId,
          ownerAnswer: isOwnerAnswer
        });
      }

      const answerText =
        feedbackScore < -2
          ? "We are working on this"
          : semanticExperience.answer;

      if (feedbackScore < -2 && shouldAutoForward) {
        await ensureEscalation({
          experience: semanticExperience,
          reason: "auto-forward"
        });
      }

      if (feedbackScore >= -2 && shouldAutoForward) {
        await ensureEscalation({
          experience: semanticExperience,
          reason: "auto-forward"
        });
      }

      for (const token of answerText.split(" ")) {
        fullAnswer += token + " ";
        res.write(`data: ${token} `);
        await new Promise(r => setTimeout(r, 10));
      }

      res.write("data: \n\n");
      res.end();

      await ChatMessage.create({
        sessionId,
        role: "assistant",
        content: fullAnswer.trim()
      });
      appendConversationCache(sessionId, "assistant", fullAnswer.trim());

      session.lastActiveAt = new Date();
      await session.save();
      return;
    }

    if (experienceHit && hitFeedbackScore < -2) {
      const answerText = "We are working on this";

      if (shouldAutoForward && experienceHit?.experience_id) {
        const exp = await Experience.findById(experienceHit.experience_id);
        if (exp) {
          await ensureEscalation({ experience: exp, reason: "auto-forward" });
        }
      }

      for (const token of answerText.split(" ")) {
        fullAnswer += token + " ";
        res.write(`data: ${token} `);
        await new Promise(r => setTimeout(r, 10));
      }

      res.write("data: \n\n");
      res.end();

      await ChatMessage.create({
        sessionId,
        role: "assistant",
        content: fullAnswer.trim()
      });
      appendConversationCache(sessionId, "assistant", fullAnswer.trim());

      session.lastActiveAt = new Date();
      await session.save();
      return;
    }

    const botLinks = await getBotLinkPayload(
      session.botId,
      session.tenantId
    );

    const ragPayload = {
      query: question,
      bot_id: session.botId.toString(),
      client_id: session.tenantId.toString(),
      top_k: 3,
      conversation
    };

    if (botLinks.websiteUrl) {
      ragPayload.website_url = botLinks.websiteUrl;
    }
    if (Object.keys(botLinks.socialLinks).length) {
      ragPayload.social_links = botLinks.socialLinks;
    }

    const ragRes = await axios.post(
      docServiceUrl("/answer"),
      ragPayload,
      { timeout: RAG_TIMEOUT_SAFE_MS }
    );

    if (DEBUG_RAG_PAYLOAD) {
      console.log("RAG response meta:", {
        hasAnswer: !!ragRes?.data?.answer,
        chunkCount: Array.isArray(ragRes?.data?.chunks)
          ? ragRes.data.chunks.length
          : 0,
        confidence: ragRes?.data?.confidence ?? null,
        sourceType: ragRes?.data?.source_type || "docs",
        noDocs: !!ragRes?.data?.no_docs
      });
    }

    const answer =
      typeof ragRes?.data?.answer === "string" && ragRes.data.answer.trim()
        ? ragRes.data.answer.trim()
        : "Sorry, I couldn't retrieve an answer right now.";
    const sourceType = ragRes?.data?.source_type || "docs";
    const noDocs = !!ragRes?.data?.no_docs;
    const graphEscalationRequired =
      sourceType === "human" || noDocs;
    const analysis = Array.isArray(ragRes?.data?.trace)
      ? ragRes.data.trace
      : [];
    const intentLabel = extractIntentLabelFromTrace(analysis);
    const canonicalQuestion = resolveCanonicalQuestion(
      conversation,
      question,
      intentLabel
    );

    if (Array.isArray(ragRes?.data?.chunks)) {
      retrievedChunks = ragRes.data.chunks;
    }

    for (const token of answer.split(" ")) {
      fullAnswer += token + " ";
      res.write(`data: ${token} `);
      await new Promise(r => setTimeout(r, 10));
    }

    res.write("data: \n\n");
    res.end();

    await ChatMessage.create({
      sessionId,
      role: "assistant",
      content: fullAnswer.trim()
    });
    appendConversationCache(sessionId, "assistant", fullAnswer.trim());

    const avgChunkSimilarity =
      computeAvgSimilarity(retrievedChunks);

    const assignedSemanticId =
      semanticId || new mongoose.Types.ObjectId();

    const seedFeedbackScore =
      experienceHit ? hitFeedbackScore : 0;
    const seedNegativeCount =
      experienceHit ? hitNegativeCount : 0;

    const experience = await Experience.create({
      botId: session.botId,
      clientId: session.tenantId,
      sessionId: session._id,
      semanticId: assignedSemanticId,
      question: canonicalQuestion,
      answer: fullAnswer.trim(),
      retrievedChunks: retrievedChunks.map(c => ({
        chunkId: c.chunk_ref || "unknown",
        similarity: c.score ?? 0
      })),
      avgChunkSimilarity,
      feedbackScore: seedFeedbackScore,
      negativeCount: seedNegativeCount,
      status: "active",
      retrievalVariant: "primary"
    });

    if (graphEscalationRequired) {
      await ensureEscalation({
        experience,
        reason: noDocs
          ? "no-docs-threshold"
          : "graph-human-handoff"
      });
    }

    indexExperienceVector({
      experienceId: experience._id,
      question: canonicalQuestion,
      session,
      avgChunkSimilarity,
      feedbackScore: seedFeedbackScore,
      negativeCount: seedNegativeCount,
      semanticId: assignedSemanticId.toString(),
      ownerAnswer: false
    });

    session.lastActiveAt = new Date();
    await session.save();
  } catch (err) {
    console.error("Streaming error:", err.response?.data || err.message);
    res.end();
  }
};

/* ==================================================
   History
================================================== */

export const getHistory = async (req, res) => {
  try {
    const { sessionId } = req.params;
    if (!sessionId) {
      return res.status(400).json({ message: "Session ID required" });
    }

    const messages = await ChatMessage.find({ sessionId })
      .sort({ createdAt: 1 })
      .lean();

    res.json(messages);
  } catch (err) {
    console.error("Get history error:", err);
    res.status(500).json({ message: "Failed to fetch history" });
  }
};

/* ==================================================
   Feedback
================================================== */

export const submitFeedback = async (req, res) => {
  try {
    const { experienceId, feedback, reason } = req.body;

    if (!experienceId || !feedback) {
      return res.status(400).json({
        message: "experienceId and feedback are required"
      });
    }

    if (!['positive', 'negative', 'neutral'].includes(feedback)) {
      return res.status(400).json({
        message: "Invalid feedback value"
      });
    }

    const experience = await Experience.findById(experienceId);
    if (!experience) {
      return res.status(404).json({
        message: "Experience not found"
      });
    }

    const delta =
      feedback === "positive" ? 1 : feedback === "negative" ? -1 : 0;

    let feedbackScore = normalizeNumber(experience.feedbackScore);
    let negativeCount = normalizeNumber(experience.negativeCount);

    if (delta !== 0) {
      const updated = await applySemanticFeedbackDelta({
        experience,
        delta,
        reason: reason || ""
      });
      feedbackScore = updated.feedbackScore;
      negativeCount = updated.negativeCount;
    } else {
      const semanticId = await ensureSemanticIdForExperience(experience);
      const groupFilter = buildSemanticGroupFilter(experience, semanticId);
      await Experience.updateMany(groupFilter, {
        feedbackAt: new Date(),
        feedbackReason: reason || ""
      });

      const seed = await Experience.findOne(groupFilter)
        .sort({ createdAt: -1 })
        .lean();
      feedbackScore = normalizeNumber(seed?.feedbackScore);
      negativeCount = normalizeNumber(seed?.negativeCount);
    }

    const retryAllowed =
      feedback === "negative" &&
      !experience.resolvedByOwner &&
      feedbackScore >= RETRY_SCORE_FLOOR;

    const escalationSuggested =
      feedback === "negative" &&
      !retryAllowed &&
      !experience.resolvedByOwner;

    if (escalationSuggested) {
      await ensureEscalation({ experience, reason });
    }

    return res.json({
      success: true,
      feedback,
      feedbackScore,
      negativeCount,
      retryAllowed,
      escalationSuggested
    });

  } catch (err) {
    console.error("Feedback error:", err);
    return res.status(500).json({
      message: "Failed to submit feedback"
    });
  }
};

export const retryWithSecondary = async (req, res) => {
  try {
    const { sessionId, experienceId } = req.body;

    if (!sessionId || !experienceId) {
      return res.status(400).json({
        message: "sessionId and experienceId are required"
      });
    }

    const session = await ChatSession.findById(sessionId);
    if (!session) {
      return res.status(404).json({ message: "Invalid session" });
    }

    const previousExperience = await Experience.findById(experienceId);
    if (!previousExperience) {
      return res.status(404).json({
        message: "Experience not found"
      });
    }

    if (previousExperience.resolvedByOwner) {
      return res.status(400).json({
        message: "Owner-resolved answer cannot be retried",
        escalationSuggested: true
      });
    }

    const question = previousExperience.question;
    const semanticId = await ensureSemanticIdForExperience(previousExperience);

    // Save retry user message (UX continuity)
    await ChatMessage.create({
      sessionId,
      role: "user",
      content: question
    });
    appendConversationCache(sessionId, "user", question);

    let conversation = await getConversationCached(sessionId, CONVERSATION_LIMIT);
    if (
      conversation.length === 0 ||
      conversation[conversation.length - 1]?.role !== "user" ||
      (conversation[conversation.length - 1]?.content || "").trim() !== question
    ) {
      conversation = [...conversation, { role: "user", content: question }]
        .slice(-CONVERSATION_LIMIT);
    }

    const feedbackUpdate = await applySemanticFeedbackDelta({
      experience: previousExperience,
      delta: -1,
      reason: "secondary_retry_requested"
    });
    const updatedScore = feedbackUpdate.feedbackScore;
    const updatedNegative = feedbackUpdate.negativeCount;

    if (updatedScore < 1) {
      await ensureEscalation({
        experience: previousExperience,
        reason: "retry_low_score"
      });
    }

    if (updatedScore < FEEDBACK_BLOCK_THRESHOLD) {
      await ensureEscalation({
        experience: previousExperience,
        reason: "retry_feedback_block"
      });

      const blockedReply = "We are working on this";
      await ChatMessage.create({
        sessionId,
        role: "assistant",
        content: blockedReply
      });
      appendConversationCache(sessionId, "assistant", blockedReply);

      session.lastActiveAt = new Date();
      await session.save();

      return res.json({
        reply: blockedReply,
        experienceId: previousExperience._id,
        retry: false,
        retryAvailable: false,
        escalationSuggested: true
      });
    }

    let reply = "Sorry, I couldn't retrieve an answer right now.";
    let retrievedChunks = [];
    const excludeChunkRefs = await getGroupChunkRefs(
      previousExperience,
      semanticId
    );

    const botLinks = await getBotLinkPayload(
      session.botId,
      session.tenantId
    );

    const ragPayload = {
      query: question,
      bot_id: session.botId.toString(),
      client_id: session.tenantId.toString(),
      retrieval_variant: "secondary",
      top_k: 3,
      exclude_chunk_refs: excludeChunkRefs,
      conversation
    };

    if (botLinks.websiteUrl) {
      ragPayload.website_url = botLinks.websiteUrl;
    }
    if (Object.keys(botLinks.socialLinks).length) {
      ragPayload.social_links = botLinks.socialLinks;
    }

    const ragRes = await axios.post(
      docServiceUrl("/answer"),
      ragPayload,
      { timeout: RAG_TIMEOUT_SAFE_MS }
    );

    if (ragRes?.data?.answer?.trim()) {
      reply = ragRes.data.answer.trim();
    }

    if (Array.isArray(ragRes?.data?.chunks)) {
      retrievedChunks = ragRes.data.chunks;
    }

    const confidence = normalizeNumber(ragRes?.data?.confidence);
    const references = Array.isArray(ragRes?.data?.references)
      ? ragRes.data.references
      : [];
    const sourceType = ragRes?.data?.source_type || "docs";
    const analysis = Array.isArray(ragRes?.data?.trace)
      ? ragRes.data.trace
      : [];

    const noDocs = !!ragRes?.data?.no_docs;
    const avgChunkSimilarity =
      computeAvgSimilarity(retrievedChunks);

    const previousChunkSet = chunkIdSet(
      previousExperience.retrievedChunks?.map(c => ({
        chunk_ref: c.chunkId
      }))
    );
    const newChunkSet = chunkIdSet(retrievedChunks);
    const sameChunks = chunkSetsEqual(previousChunkSet, newChunkSet);

    const secondarySatisfied =
      retrievedChunks.length > 0 &&
      avgChunkSimilarity >= SECONDARY_MIN_SIMILARITY &&
      !sameChunks;

    if (noDocs) {
      const noDocsReply = ragRes?.data?.answer || "No relevant docs found.";

      await ChatMessage.create({
        sessionId,
        role: "assistant",
        content: noDocsReply
      });
      appendConversationCache(sessionId, "assistant", noDocsReply);

      session.lastActiveAt = new Date();
      await session.save();

      return res.json({
        reply: noDocsReply,
        experienceId: previousExperience._id,
        retry: false,
        retryAvailable: updatedScore >= RETRY_SCORE_FLOOR,
        escalationSuggested: false,
        confidence,
        references,
        sourceType,
        analysis
      });
    }

    if (!secondarySatisfied) {
      const fallbackReply = previousExperience.answer;

      await ChatMessage.create({
        sessionId,
        role: "assistant",
        content: fallbackReply
      });
      appendConversationCache(sessionId, "assistant", fallbackReply);

      session.lastActiveAt = new Date();
      await session.save();

      return res.json({
        reply: fallbackReply,
        experienceId: previousExperience._id,
        retry: false,
        retryAvailable: updatedScore >= RETRY_SCORE_FLOOR,
        escalationSuggested: false,
        confidence,
        references,
        sourceType,
        analysis
      });
    }

    await ChatMessage.create({
      sessionId,
      role: "assistant",
      content: reply
    });
    appendConversationCache(sessionId, "assistant", reply);

    const retryExperience = await upsertSemanticExperience({
      session,
      semanticId,
      question,
      answer: reply,
      retrievedChunks,
      avgChunkSimilarity,
      feedbackScore: updatedScore,
      negativeCount: updatedNegative,
      retrievalVariant: "secondary",
      preferredExperienceId: previousExperience._id
    });

    indexExperienceVector({
      experienceId: retryExperience._id,
      question,
      session,
      avgChunkSimilarity,
      feedbackScore: updatedScore,
      negativeCount: updatedNegative,
      semanticId: semanticId.toString(),
      ownerAnswer: false
    });

    session.lastActiveAt = new Date();
    await session.save();

    return res.json({
      reply,
      experienceId: retryExperience._id,
      retry: true,
      retryAvailable: updatedScore >= RETRY_SCORE_FLOOR,
      escalationNext: updatedScore < ESCALATION_SCORE_FLOOR,
      escalationSuggested: false,
      confidence,
      references,
      sourceType,
      analysis
    });

  } catch (err) {
    console.error("Retry error:", err);
    return res.status(500).json({
      message: "Retry failed"
    });
  }
};


export const getEscalationsForBot = async (req, res) => {
  try {
    const { botId } = req.params;
    const status =
      typeof req.query.status === "string" ? req.query.status : "open";

    if (!botId) {
      return res.status(400).json({ message: "botId is required" });
    }

    if (status && !["open", "resolved"].includes(status)) {
      return res.status(400).json({ message: "Invalid status value" });
    }

    const bot = await Bot.findOne({
      _id: botId,
      tenantId: req.clientId
    }).lean();

    if (!bot) {
      return res.status(404).json({ message: "Bot not found" });
    }

    const query = { botId };
    if (status) query.status = status;

    const escalations = await Escalation.find(query)
      .sort({ createdAt: -1 })
      .lean();

    return res.json(escalations);
  } catch (err) {
    console.error("Get escalations error:", err);
    return res.status(500).json({ message: "Failed to fetch escalations" });
  }
};


export const resolveEscalation = async (req, res) => {
  try {
    const { escalationId, answer } = req.body;

    if (!escalationId || !answer?.trim()) {
      return res.status(400).json({
        message: "escalationId and answer are required"
      });
    }

    const escalation = await Escalation.findById(escalationId);
    if (!escalation) {
      return res.status(404).json({ message: "Escalation not found" });
    }

    const bot = await Bot.findById(escalation.botId).lean();
    if (!bot || bot.tenantId.toString() !== req.clientId) {
      return res.status(403).json({ message: "Not authorized" });
    }

    const seedExperience = await Experience.findOne({
      _id: { $in: escalation.experienceIds }
    }).lean();

    const clientId = seedExperience?.clientId || bot?.tenantId;

    if (!clientId) {
      return res.status(500).json({
        message: "Unable to resolve client context"
      });
    }

    let semanticId = null;

    try {
      const searchRes = await axios.post(
        docServiceUrl("/experience/search"),
        {
          query: escalation.question,
          bot_id: escalation.botId.toString()
        }
      );

      if (searchRes?.data?.semantic_id) {
        semanticId = searchRes.data.semantic_id;
      }
    } catch (err) {
      console.error("?????? Experience search failed:", err.message);
    }

    if (!semanticId && seedExperience?.semanticId) {
      semanticId = seedExperience.semanticId;
    }

    if (!semanticId) {
      semanticId = new mongoose.Types.ObjectId();
    }

    await Experience.deleteMany({
      botId: escalation.botId,
      semanticId
    });

    const ownerExperience = await Experience.create({
      botId: escalation.botId,
      clientId,
      semanticId,
      question: escalation.question,
      answer: answer.trim(),
      retrievedChunks: [],
      avgChunkSimilarity: 0,
      feedbackScore: OWNER_RESOLVED_SCORE,
      negativeCount: 0,
      resolvedByOwner: true,
      status: "active",
      retrievalVariant: "owner"
    });

    indexExperienceVector({
      experienceId: ownerExperience._id,
      question: escalation.question,
      session: { botId: escalation.botId, tenantId: clientId },
      avgChunkSimilarity: 0,
      feedbackScore: OWNER_RESOLVED_SCORE,
      negativeCount: 0,
      semanticId: ownerExperience.semanticId.toString(),
      ownerAnswer: true
    });

    escalation.status = "resolved";
    escalation.resolvedAt = new Date();
    escalation.resolvedExperienceId = ownerExperience._id;
    await escalation.save();

    return res.json({
      success: true,
      experienceId: ownerExperience._id
    });
  } catch (err) {
    console.error("Resolve escalation error:", err);
    return res.status(500).json({
      message: "Failed to resolve escalation"
    });
  }
};
