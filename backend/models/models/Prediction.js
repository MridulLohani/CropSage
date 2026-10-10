const mongoose = require("mongoose");

const predictionSchema = new mongoose.Schema(
    {
        user: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "User",
            required: true
        },
        imageName: {
            type: String,
            required: true
        },
        prediction: {
            type: String,
            required: true
        },
        confidence: {
            type: Number,
            required: true
        },
        topPredictions: [
            {
                class: {
                    type: String,
                    required: true
                },
                confidence: {
                    type: Number,
                    required: true
                }
            }
        ]
    },
    { timestamps: true }
);

module.exports = mongoose.model("Prediction", predictionSchema);
