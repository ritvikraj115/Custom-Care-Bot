import mongoose from "mongoose";

const clientSchema = new mongoose.Schema(
  {
    companyName: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    passwordHash: { type: String, required: true },
    industry: { type: String }
  },
  { timestamps: true }
);

export default mongoose.model("Client", clientSchema);

