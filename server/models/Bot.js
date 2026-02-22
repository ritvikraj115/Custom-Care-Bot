import mongoose from "mongoose";

const botSchema = new mongoose.Schema({
  tenantId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true
  },
  name: { type: String, required: true },
  description: String,
  websiteUrl: String,
  facebookUrl: String,
  instagramUrl: String,
  publicAccess: { type: Boolean, default: true },
  purpose: {
    type: String,
    enum: [
      "Customer Support",
      "Internal Knowledge Base",
      "Sales Assistant",
      "HR / Employee Onboarding",
      "IT Helpdesk",
      "Training & Education",
      "Healthcare Assistant",
      "Finance / Banking Assistant",
      "Other"
    ]
  },

  isActive: { type: Boolean, default: true }
}, { timestamps: true });

export default mongoose.model("Bot", botSchema);


