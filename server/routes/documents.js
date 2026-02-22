import express from "express";
import multer from "multer";
import fs from "fs";
import crypto from "crypto";

import Document from "../models/Document.js";
import Bot from "../models/Bot.js";
import { protect } from "../middleware/authMiddleware.js";
import {
  enqueueIngestionJob,
  getIngestionJob
} from "../services/ingestionJobService.js";

const upload = multer({ dest: "uploads/" });
const router = express.Router();

const hashFile = filePath =>
  new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = fs.createReadStream(filePath);

    stream.on("data", chunk => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(hash.digest("hex")));
  });

const safeUnlink = async filePath => {
  if (!filePath) return;
  try {
    await fs.promises.unlink(filePath);
  } catch (err) {
    if (err?.code !== "ENOENT") {
      console.error("Failed to delete temp upload:", filePath, err.message);
    }
  }
};

/**
 * Get documents for a bot
 */
router.get("/bot/:botId", protect, async (req, res) => {
  const docs = await Document.find({
    botId: req.params.botId,
    clientId: req.clientId
  }).sort("createdAt");

  res.json(docs);
});

/**
 * Upload documents and enqueue ingestion
 */
router.post(
  "/upload",
  protect,
  upload.array("files"),
  async (req, res) => {
    try {
      const botIdRaw = (req.body.botId || "").toString();
      if (!botIdRaw) {
        for (const file of req.files || []) {
          await safeUnlink(file.path);
        }
        return res.status(400).json({ message: "botId is required" });
      }

      const bot = await Bot.findOne({
        _id: botIdRaw,
        tenantId: req.clientId
      }).lean();

      if (!bot) {
        for (const file of req.files || []) {
          await safeUnlink(file.path);
        }
        return res.status(404).json({ message: "Bot not found" });
      }

      const files = Array.isArray(req.files) ? req.files : [];
      if (!files.length) {
        return res.status(400).json({ message: "No files provided" });
      }

      const botId = bot._id.toString();
      const oldDocs = await Document.find({
        botId,
        clientId: req.clientId
      }).lean();

      const existingHashes = new Set(
        oldDocs.map(doc => doc.contentHash).filter(Boolean)
      );

      for (const doc of oldDocs) {
        if (doc?.contentHash || !doc?.filePath) continue;
        try {
          const backfilledHash = await hashFile(doc.filePath);
          existingHashes.add(backfilledHash);
          await Document.updateOne(
            { _id: doc._id },
            { contentHash: backfilledHash }
          );
        } catch (err) {
          console.error(
            "Failed to backfill document hash:",
            doc._id?.toString(),
            err.message
          );
        }
      }

      const newDocs = [];
      const skippedDuplicates = [];

      for (const file of files) {
        const contentHash = await hashFile(file.path);

        if (existingHashes.has(contentHash)) {
          skippedDuplicates.push(file.originalname);
          await safeUnlink(file.path);
          continue;
        }

        existingHashes.add(contentHash);

        const created = await Document.create({
          clientId: req.clientId,
          botId,
          fileName: file.originalname,
          filePath: file.path,
          fileSize: file.size,
          contentHash
        });

        newDocs.push(created);
      }

      if (!newDocs.length) {
        return res.json({
          status: "NO_CHANGES",
          message: "All uploaded files already exist",
          skippedDuplicates,
          documents: []
        });
      }

      const hasExistingDocs = oldDocs.length > 0;
      const rebuildMode = hasExistingDocs ? "incremental" : "full";
      const filesToProcess = rebuildMode === "incremental"
        ? newDocs
        : [...oldDocs, ...newDocs];

      const ingestionJob = enqueueIngestionJob({
        clientId: req.clientId.toString(),
        botId,
        rebuildMode,
        includeWebsite: rebuildMode === "full" && !!bot.websiteUrl,
        websiteUrl: bot.websiteUrl || "",
        files: filesToProcess.map(doc => ({
          filePath: doc.filePath,
          fileName: doc.fileName
        })),
        stats: {
          uploadedCount: files.length,
          dedupedCount: skippedDuplicates.length,
          indexedCount: newDocs.length
        }
      });

      return res.status(202).json({
        status: "QUEUED",
        rebuildMode,
        jobId: ingestionJob.id,
        skippedDuplicates,
        documents: newDocs
      });
    } catch (err) {
      console.error("Document upload failed:", err);
      return res.status(500).json({
        message: "Failed to queue document ingestion"
      });
    }
  }
);

/**
 * Get ingestion job status
 */
router.get("/jobs/:jobId", protect, async (req, res) => {
  const job = getIngestionJob(req.params.jobId);
  if (!job) {
    return res.status(404).json({ message: "Job not found" });
  }

  if (job.clientId !== req.clientId.toString()) {
    return res.status(403).json({ message: "Not authorized" });
  }

  return res.json({
    id: job.id,
    status: job.status,
    botId: job.botId,
    rebuildMode: job.rebuildMode,
    stats: job.stats,
    progress: job.progress,
    error: job.error,
    result: job.result,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    startedAt: job.startedAt,
    completedAt: job.completedAt
  });
});

export default router;
