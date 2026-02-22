import express from "express";
import {
  createSession,
  getAutocompleteSuggestions,
  sendMessage,
  getHistory,
  streamMessage,
  submitFeedback,
  retryWithSecondary
} from "../controllers/chatController.js";

const router = express.Router();

router.post("/session/:botId", createSession);
router.get("/autocomplete", getAutocompleteSuggestions);
router.post("/message", sendMessage);
router.get("/history/:sessionId", getHistory);

// 🔥 STREAMING ROUTE
router.get("/stream", streamMessage);

router.post("/feedback", submitFeedback);
router.post("/retry", retryWithSecondary);
// Backward compatibility for older clients
router.post("/chat/retry", retryWithSecondary);


export default router;

