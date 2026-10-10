const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

const Prediction = require("../models/Prediction");
const protect = require("../middleware/authMiddleware");

const router = express.Router();


// Store uploaded image in memory
const upload = multer({
    storage: multer.memoryStorage()
});


// ========================================
// PREDICT CROP DISEASE
// ========================================

router.post(
    "/predict",
    protect,
    upload.single("image"),
    async (req, res) => {

        try {

            // Check image
            if (!req.file) {

                return res.status(400).json({
                    message: "Please upload an image"
                });
            }


            // Create form data
            const formData = new FormData();

            formData.append(
                "image",
                req.file.buffer,
                {
                    filename: req.file.originalname,
                    contentType: req.file.mimetype
                }
            );


            // Send image to Python ML API
            const mlResponse = await axios.post(
                `${process.env.ML_API_URL}/predict`,
                formData,
                {
                    headers: {
                        ...formData.getHeaders()
                    }
                }
            );


            const result = mlResponse.data;


            // Save prediction to MongoDB
            const prediction = await Prediction.create({

                user: req.user.id,

                imageName: req.file.originalname,

                prediction: result.prediction,

                confidence: result.confidence,

                topPredictions: result.top_predictions

            });


            // Send response to frontend
            res.status(201).json({

                message: "Prediction successful",

                prediction: {
                    id: prediction._id,

                    imageName: prediction.imageName,

                    disease: prediction.prediction,

                    confidence: prediction.confidence,

                    topPredictions: prediction.topPredictions,

                    createdAt: prediction.createdAt
                }

            });

        } catch (error) {

            console.error(
                "Prediction error:",
                error.message
            );


            res.status(500).json({

                message: "Prediction failed",

                error: error.message

            });
        }
    }
);


// ========================================
// GET USER PREDICTION HISTORY
// ========================================

router.get(
    "/history",
    protect,
    async (req, res) => {

        try {

            const predictions =
                await Prediction.find({
                    user: req.user.id
                })
                .sort({
                    createdAt: -1
                });


            res.json({
                predictions
            });

        } catch (error) {

            console.error(error);

            res.status(500).json({

                message:
                    "Failed to retrieve prediction history"

            });
        }
    }
);


module.exports = router;