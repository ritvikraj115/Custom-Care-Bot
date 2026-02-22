import axios from "axios";
import Bot from "../models/Bot.js";
import { docServiceUrl } from "../config/serviceUrls.js";

const SOCIAL_REFRESH_INTERVAL_MS = Math.max(
  15 * 60 * 1000,
  Number.parseInt(process.env.SOCIAL_REFRESH_INTERVAL_MS || "", 10) ||
    (6 * 60 * 60 * 1000)
);

const SOCIAL_REFRESH_BATCH_LIMIT = Math.max(
  1,
  Number.parseInt(process.env.SOCIAL_REFRESH_BATCH_LIMIT || "", 10) || 50
);

let intervalRef = null;
let cycleRunning = false;

const hasAnySocialLink = bot =>
  !!(bot?.facebookUrl || bot?.instagramUrl);

const buildPayload = bot => ({
  bot_id: bot._id.toString(),
  website_url: bot.websiteUrl || "",
  social_links: {
    ...(bot.facebookUrl ? { facebook: bot.facebookUrl } : {}),
    ...(bot.instagramUrl ? { instagram: bot.instagramUrl } : {})
  },
  query_hints: [
    bot.name || "",
    bot.description || "",
    bot.purpose || ""
  ].filter(Boolean),
  max_results_per_platform: 2
});

export const refreshBotSocialIndex = async bot => {
  if (!bot || !hasAnySocialLink(bot)) return false;
  try {
    await axios.post(
      docServiceUrl("/social/refresh"),
      buildPayload(bot),
      { timeout: 30000 }
    );
    return true;
  } catch (err) {
    console.error(
      "Social pre-index refresh failed:",
      bot._id?.toString(),
      err.response?.data || err.message
    );
    return false;
  }
};

export const runSocialRefreshCycle = async () => {
  if (cycleRunning) return;
  cycleRunning = true;

  try {
    const bots = await Bot.find({
      isActive: true
    })
      .select("_id name description purpose websiteUrl facebookUrl instagramUrl")
      .sort({ updatedAt: -1 })
      .limit(SOCIAL_REFRESH_BATCH_LIMIT)
      .lean();

    let refreshed = 0;
    for (const bot of bots) {
      if (!hasAnySocialLink(bot)) continue;
      const ok = await refreshBotSocialIndex(bot);
      if (ok) refreshed += 1;
    }

    if (refreshed > 0) {
      console.log(`Social pre-index refresh cycle completed (${refreshed} bot(s))`);
    }
  } catch (err) {
    console.error("Social refresh cycle failed:", err.message);
  } finally {
    cycleRunning = false;
  }
};

export const startSocialRefreshScheduler = () => {
  if (intervalRef) return;
  runSocialRefreshCycle();
  intervalRef = setInterval(runSocialRefreshCycle, SOCIAL_REFRESH_INTERVAL_MS);
};
