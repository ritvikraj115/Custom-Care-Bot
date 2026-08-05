import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import chatRoutes from "./routes/chat.js";
import escalationRoutes from "./routes/escalations.js";

import authRoutes from "./routes/auth.js";
import botRoutes from "./routes/bots.js";
import documentRoutes from "./routes/documents.js";
import { startSocialRefreshScheduler } from "./services/socialRefreshService.js";

dotenv.config();

const app = express();

// middleware
const allowedOrigins = (process.env.CORS_ORIGIN || "")
  .split(",")
  .map(origin => origin.trim())
  .filter(Boolean);

app.use(cors(allowedOrigins.length ? { origin: allowedOrigins } : undefined));
app.use(express.json());
app.disable("etag");

app.get("/api/health", (req, res) => {
  res.json({ status: "ok", service: "custom-care-backend", timestamp: Date.now() });
});

// routes
app.use("/api/auth", authRoutes);
app.use("/api/bots", botRoutes);
app.use("/api/documents", documentRoutes);
app.use("/api/chat", chatRoutes);
app.use("/api/escalations", escalationRoutes);

// db + server
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => {
    console.log("MongoDB connected");
    app.listen(process.env.PORT || 5000, () => {
      console.log("Server running");
      startSocialRefreshScheduler();
    });
  })
  .catch(err => console.error(err));

