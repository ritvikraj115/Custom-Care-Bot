import mongoose from "mongoose";

const RetrievedChunkSchema = new mongoose.Schema(
    {
        chunkId: {
            type: String,
            required: true
        },
        similarity: {
            type: Number,
            required: true
        }
    },
    { _id: false }
);

const ExperienceSchema = new mongoose.Schema(
    {
        // --------------------
        // Scope
        // --------------------
        botId: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "Bot",
            required: true,
            index: true
        },

        clientId: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "Client",
            required: true,
            index: true
        },

        sessionId: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "ChatSession",
            index: true
        },

        // --------------------
        // Core interaction
        // --------------------
        question: {
            type: String,
            required: true
        },

        answer: {
            type: String,
            required: true
        },



        // --------------------
        // Retrieval trace
        // --------------------
        retrievedChunks: {
            type: [RetrievedChunkSchema],
            default: []
        },

        avgChunkSimilarity: {
            type: Number,
            required: true
        },

        // --------------------
        // Learning signal
        // --------------------
        feedbackScore: {
            type: Number,
            default: 0,
            index: true
        },
        feedbackReason: {
            type: String,
            default: ""
        },

        feedbackAt: {
            type: Date
        },

        resolvedByOwner: {
            type: Boolean,
            default: false
        },
        negativeCount: { type: Number, default: 0 },
        status: {
            type: String,
            enum: ["active", "escalated"],
            default: "active"
        },
        semanticId: {
            type: mongoose.Schema.Types.ObjectId,
            index: true
        },

        retrievalVariant: {
            type: String,
            enum: ["primary", "secondary", "owner"],
            default: "primary"
        }
    },
    {
        timestamps: true
    }
);

export default mongoose.model("Experience", ExperienceSchema);
