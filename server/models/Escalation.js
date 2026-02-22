import mongoose from "mongoose";

const EscalationSchema = new mongoose.Schema({
  botId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Bot",
    required: true
  },

  question: {
    type: String,
    required: true
  },

  experienceIds: [
    {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Experience"
    }
  ],

  reason: {
    type: String,
    default: "Repeated negative feedback"
  },

  status: {
    type: String,
    enum: ["open", "resolved"],
    default: "open"
  },
  resolvedAt: {
    type: Date
  },
  resolvedExperienceId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Experience"
  },

  createdAt: {
    type: Date,
    default: Date.now
  }
});

export default mongoose.model("Escalation", EscalationSchema);
