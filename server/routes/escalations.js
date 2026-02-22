import express from "express";
import {
  getEscalationsForBot,
  resolveEscalation
} from "../controllers/chatController.js";
import { protect } from "../middleware/authMiddleware.js";

const router = express.Router();

router.get("/bot/:botId", protect, getEscalationsForBot);
router.post("/resolve", protect, resolveEscalation);

export default router;
