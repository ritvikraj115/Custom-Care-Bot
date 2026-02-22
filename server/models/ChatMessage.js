import mongoose from "mongoose";

const chatMessageSchema = new mongoose.Schema({
  sessionId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "ChatSession",
    required: true
  },
  role: {
    type: String,
    enum: ["user", "assistant"],
    required: true
  },
  content: {
    type: String,
    required: true
  }
}, { timestamps: true });

chatMessageSchema.index({ sessionId: 1, createdAt: -1 });

export default mongoose.model("ChatMessage", chatMessageSchema);
