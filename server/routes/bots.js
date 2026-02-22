import express from "express";
import axios from "axios";
import fs from "fs";
import Bot from "../models/Bot.js";
import Document from "../models/Document.js";
import Experience from "../models/Experience.js";
import Escalation from "../models/Escalation.js";
import ChatSession from "../models/ChatSession.js";
import ChatMessage from "../models/ChatMessage.js";
import { protect } from "../middleware/authMiddleware.js";
import { refreshBotSocialIndex } from "../services/socialRefreshService.js";
import { docServiceUrl } from "../config/serviceUrls.js";

const router = express.Router();

const toPercent = (num, den) => {
  if (!den) return 0;
  return Number(((num / den) * 100).toFixed(2));
};

// CREATE BOT
router.post("/", protect, async (req, res) => {
  try {
    const {
      botName,
      botPurpose,
      description,
      websiteUrl,
      facebookUrl,
      instagramUrl
    } = req.body;

    const bot = await Bot.create({
      name: botName,
      description,
      websiteUrl,
      facebookUrl,
      instagramUrl,
      purpose: botPurpose,
      tenantId: req.clientId
    });

    if (bot.facebookUrl || bot.instagramUrl) {
      refreshBotSocialIndex(bot).catch(() => {});
    }

    res.json(bot);
  } catch (err) {
    console.error(err);
    res.status(400).json({ message: "Failed to create bot" });
  }
});

// LIST BOTS
router.get("/", protect, async (req, res) => {
  const bots = await Bot.find({ tenantId: req.clientId });
  res.json(bots);
});

// BOT ANALYTICS
router.get("/:id/analytics", protect, async (req, res) => {
  try {
    const bot = await Bot.findOne({
      _id: req.params.id,
      tenantId: req.clientId
    }).lean();

    if (!bot) {
      return res.status(404).json({ message: "Bot not found" });
    }

    const sessions = await ChatSession.find({ botId: bot._id })
      .select("_id")
      .lean();
    const sessionIds = sessions.map(row => row._id);

    const messageAgg = sessionIds.length
      ? await ChatMessage.aggregate([
        { $match: { sessionId: { $in: sessionIds } } },
        { $group: { _id: "$role", count: { $sum: 1 } } }
      ])
      : [];

    const messageCountByRole = {
      user: 0,
      assistant: 0
    };
    for (const row of messageAgg) {
      if (row?._id === "user" || row?._id === "assistant") {
        messageCountByRole[row._id] = Number(row.count || 0);
      }
    }

    const [
      totalExperiences,
      escalatedExperiences,
      negativeExperiences,
      secondaryExperiences,
      ownerExperiences,
      likelyNoDocsExperiences,
      openEscalations,
      resolvedEscalations,
      unresolvedQuestions,
      negativeHotspots
    ] = await Promise.all([
      Experience.countDocuments({ botId: bot._id }),
      Experience.countDocuments({ botId: bot._id, status: "escalated" }),
      Experience.countDocuments({ botId: bot._id, feedbackScore: { $lt: 0 } }),
      Experience.countDocuments({ botId: bot._id, retrievalVariant: "secondary" }),
      Experience.countDocuments({ botId: bot._id, retrievalVariant: "owner" }),
      Experience.countDocuments({
        botId: bot._id,
        retrievalVariant: { $ne: "owner" },
        avgChunkSimilarity: { $lt: 0.08 }
      }),
      Escalation.countDocuments({ botId: bot._id, status: "open" }),
      Escalation.countDocuments({ botId: bot._id, status: "resolved" }),
      Escalation.aggregate([
        { $match: { botId: bot._id, status: "open" } },
        {
          $group: {
            _id: "$question",
            count: { $sum: 1 },
            lastCreatedAt: { $max: "$createdAt" }
          }
        },
        { $sort: { count: -1, lastCreatedAt: -1 } },
        { $limit: 5 },
        {
          $project: {
            _id: 0,
            question: "$_id",
            count: 1,
            lastCreatedAt: 1
          }
        }
      ]),
      Experience.aggregate([
        { $match: { botId: bot._id, feedbackScore: { $lt: 0 } } },
        {
          $group: {
            _id: "$question",
            count: { $sum: 1 },
            worstScore: { $min: "$feedbackScore" }
          }
        },
        { $sort: { count: -1, worstScore: 1 } },
        { $limit: 5 },
        {
          $project: {
            _id: 0,
            question: "$_id",
            count: 1,
            worstScore: 1
          }
        }
      ])
    ]);

    const totalSessions = sessionIds.length;
    const totalMessages =
      Number(messageCountByRole.user || 0) + Number(messageCountByRole.assistant || 0);

    const containmentCount = Math.max(0, totalExperiences - escalatedExperiences);

    return res.json({
      summary: {
        sessions: totalSessions,
        messages: totalMessages,
        userMessages: Number(messageCountByRole.user || 0),
        assistantMessages: Number(messageCountByRole.assistant || 0),
        experiences: totalExperiences,
        escalatedExperiences,
        openEscalations,
        resolvedEscalations,
        ownerResolvedExperiences: ownerExperiences
      },
      rates: {
        containmentRate: toPercent(containmentCount, totalExperiences),
        escalationRate: toPercent(escalatedExperiences, totalExperiences),
        negativeFeedbackRate: toPercent(negativeExperiences, totalExperiences),
        secondaryRetryRate: toPercent(secondaryExperiences, totalExperiences),
        likelyNoDocsRate: toPercent(likelyNoDocsExperiences, totalExperiences)
      },
      hotspots: {
        unresolvedQuestions,
        negativeHotspots
      }
    });
  } catch (err) {
    console.error("Bot analytics failed:", err);
    return res.status(500).json({ message: "Failed to load bot analytics" });
  }
});

// DELETE BOT
router.delete("/:id", protect, async (req, res) => {
  const bot = await Bot.findOne({
    _id: req.params.id,
    tenantId: req.clientId
  });

  if (!bot) {
    return res.status(404).json({ message: "Bot not found" });
  }

  try {
    await axios.post(docServiceUrl("/bot/delete"), {
      bot_id: bot._id.toString(),
      client_id: req.clientId.toString()
    });
  } catch (err) {
    console.error("Vector cleanup failed:", err.response?.data || err.message);
    return res.status(502).json({
      message: "Vector cleanup failed. Bot not deleted."
    });
  }

  const docs = await Document.find({
    botId: bot._id,
    clientId: req.clientId
  }).lean();

  for (const doc of docs) {
    if (doc?.filePath) {
      try {
        fs.unlinkSync(doc.filePath);
      } catch (err) {
        console.error("Failed to delete file:", doc.filePath, err.message);
      }
    }
  }

  await Document.deleteMany({
    botId: bot._id,
    clientId: req.clientId
  });

  await Escalation.deleteMany({ botId: bot._id });
  await Experience.deleteMany({ botId: bot._id });

  const sessions = await ChatSession.find({ botId: bot._id })
    .select("_id")
    .lean();
  const sessionIds = sessions.map(s => s._id);
  if (sessionIds.length) {
    await ChatMessage.deleteMany({
      sessionId: { $in: sessionIds }
    });
  }
  await ChatSession.deleteMany({ botId: bot._id });

  await bot.deleteOne();
  res.json({ message: "Bot deleted successfully" });
});

export default router;
