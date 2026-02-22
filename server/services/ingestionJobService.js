import axios from "axios";
import fs from "fs";
import FormData from "form-data";
import crypto from "crypto";
import { docServiceUrl } from "../config/serviceUrls.js";

const JOB_TTL_MS = 6 * 60 * 60 * 1000;
const JOB_MAX_ENTRIES = 2000;

const JOBS = new Map();
const QUEUE = [];
let WORKER_ACTIVE = false;

const cleanupJobs = () => {
  const now = Date.now();
  for (const [id, job] of JOBS.entries()) {
    if (!job?.updatedAt) continue;
    if (now - job.updatedAt > JOB_TTL_MS) {
      JOBS.delete(id);
    }
  }

  if (JOBS.size <= JOB_MAX_ENTRIES) return;
  const ordered = Array.from(JOBS.values())
    .sort((a, b) => (a.updatedAt || 0) - (b.updatedAt || 0));
  const overBy = JOBS.size - JOB_MAX_ENTRIES;
  for (let i = 0; i < overBy; i += 1) {
    const old = ordered[i];
    if (old?.id) JOBS.delete(old.id);
  }
};

const patchJob = (jobId, patch) => {
  const current = JOBS.get(jobId);
  if (!current) return null;
  const next = {
    ...current,
    ...patch,
    updatedAt: Date.now()
  };
  JOBS.set(jobId, next);
  cleanupJobs();
  return next;
};

const buildProgress = (phase, current, total, message) => {
  const safeTotal = Math.max(1, Number(total) || 1);
  const safeCurrent = Math.max(0, Math.min(Number(current) || 0, safeTotal));
  const percent = Math.round((safeCurrent / safeTotal) * 100);
  return {
    phase,
    current: safeCurrent,
    total: safeTotal,
    percent,
    message
  };
};

const processJob = async jobId => {
  const job = JOBS.get(jobId);
  if (!job) return;

  patchJob(jobId, {
    status: "running",
    progress: buildProgress("prepare", 0, job.files.length + 1, "Preparing ingestion payload"),
    startedAt: Date.now()
  });

  try {
    const form = new FormData();
    form.append("client_id", job.clientId);
    form.append("bot_id", job.botId);
    form.append("rebuild_mode", job.rebuildMode || "full");

    if (job.includeWebsite && job.websiteUrl) {
      form.append("website_url", job.websiteUrl);
    }

    let processed = 0;
    const total = job.files.length + 1;
    for (const row of job.files) {
      if (!row?.filePath || !row?.fileName) continue;
      form.append(
        "files",
        fs.createReadStream(row.filePath),
        row.fileName
      );
      processed += 1;
      patchJob(jobId, {
        progress: buildProgress(
          "prepare",
          processed,
          total,
          `Queued ${processed}/${job.files.length} file(s)`
        )
      });
    }

    patchJob(jobId, {
      progress: buildProgress("index", total - 1, total, "Running indexing pipeline")
    });

    const response = await axios.post(
      docServiceUrl("/process"),
      form,
      {
        headers: form.getHeaders(),
        timeout: 0
      }
    );

    patchJob(jobId, {
      status: "completed",
      completedAt: Date.now(),
      progress: buildProgress("done", total, total, "Ingestion completed"),
      result: response.data || {}
    });
  } catch (err) {
    patchJob(jobId, {
      status: "failed",
      completedAt: Date.now(),
      error: err.response?.data || err.message || "Ingestion failed",
      progress: buildProgress("failed", 0, 1, "Ingestion failed")
    });
  }
};

const runWorker = async () => {
  if (WORKER_ACTIVE) return;
  WORKER_ACTIVE = true;

  try {
    while (QUEUE.length > 0) {
      const nextJobId = QUEUE.shift();
      if (!nextJobId) continue;
      await processJob(nextJobId);
    }
  } finally {
    WORKER_ACTIVE = false;
  }
};

export const enqueueIngestionJob = payload => {
  const id = crypto.randomUUID();
  const now = Date.now();

  const job = {
    id,
    status: "queued",
    clientId: payload.clientId,
    botId: payload.botId,
    rebuildMode: payload.rebuildMode || "full",
    includeWebsite: !!payload.includeWebsite,
    websiteUrl: payload.websiteUrl || "",
    files: Array.isArray(payload.files) ? payload.files : [],
    stats: payload.stats || {},
    result: null,
    error: null,
    progress: buildProgress("queued", 0, 1, "Queued for processing"),
    createdAt: now,
    updatedAt: now,
    startedAt: null,
    completedAt: null
  };

  JOBS.set(id, job);
  QUEUE.push(id);
  cleanupJobs();
  runWorker();
  return job;
};

export const getIngestionJob = jobId => {
  const job = JOBS.get(jobId);
  if (!job) return null;
  return { ...job };
};
