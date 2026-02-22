import mongoose from "mongoose";

const documentSchema = new mongoose.Schema(
  {
    clientId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Client",
      required: true
    },
    botId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Bot",
      required: true
    },
    fileName: String,
    filePath: String,
    fileSize: Number,
    contentHash: {
      type: String,
      index: true
    }
  },
  { timestamps: true }
);

documentSchema.index({
  clientId: 1,
  botId: 1,
  contentHash: 1
});

export default mongoose.model("Document", documentSchema);

