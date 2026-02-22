import axios from "axios";
import { docServiceUrl } from "../config/serviceUrls.js";

/**
 * Push experience embedding to vector DB (async-safe)
 */
export const indexExperience = async ({
  experienceId,
  question,
  botId,
  clientId,
  feedbackScore = 0,
  negativeCount = 0,
  semanticId = null,
  ownerAnswer = false,
  avgChunkSimilarity
}) => {
  try {
    await axios.post(docServiceUrl("/experience/index"), {
      experience_id: experienceId,
      text: question,
      bot_id: botId.toString(),
      client_id: clientId.toString(),
      semantic_id: semanticId,
      feedback_score: feedbackScore,
      negative_count: negativeCount,
      owner_answer: ownerAnswer,
      avg_chunk_similarity: avgChunkSimilarity
    });
  } catch (err) {
    console.error("Experience indexing failed:", err.message);
  }
};
