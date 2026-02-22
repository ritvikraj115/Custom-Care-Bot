import mongoose from "mongoose";

const chatSessionSchema = new mongoose.Schema({
  botId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Bot",
    required: true
  },
  tenantId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true
  },
  source: {
    type: String,
    enum: ["public", "embed"],
    default: "public"
  },
  ipHash: String,
  userAgent: String,
  lastActiveAt: Date
}, { timestamps: true });

export default mongoose.model("ChatSession", chatSessionSchema);
