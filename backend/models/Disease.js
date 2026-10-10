const mongoose = require("mongoose");

const diseaseSchema = new mongoose.Schema(
    {
        diseaseName: {
            type: String,
            required: true,
            unique: true,
            trim: true
        },
        crop: {
            type: String,
            required: true
        },
        status: {
            type: String,
            enum: ["Healthy", "Diseased"],
            required: true
        },
        description: {
            type: String,
            required: true
        },
        symptoms: {
            type: [String],
            required: true
        },
        causes: {
            type: [String],
            required: true
        },
        prevention: {
            type: [String],
            required: true
        },
        management: {
            type: [String],
            required: true
        }
    },
    { timestamps: true }
);

module.exports = mongoose.model("Disease", diseaseSchema);
