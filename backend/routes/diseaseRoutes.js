const express = require("express");
const Disease = require("../models/Disease");

const router = express.Router();

// Get all disease advisory information
router.get("/", async (req, res) => {
    try {
        const diseases = await Disease.find().sort({
            diseaseName: 1
        });

        res.json({
            count: diseases.length,
            diseases
        });
    } catch (error) {
        console.error("Error fetching diseases:", error.message);

        res.status(500).json({
            message: "Failed to retrieve disease information"
        });
    }
});

// Get advisory information for a particular disease
router.get("/:diseaseName", async (req, res) => {
    try {
        const disease = await Disease.findOne({
            diseaseName: req.params.diseaseName
        });

        if (!disease) {
            return res.status(404).json({
                message: "Disease information not found"
            });
        }

        res.json({
            disease
        });
    } catch (error) {
        console.error("Error fetching disease:", error.message);

        res.status(500).json({
            message: "Failed to retrieve disease information"
        });
    }
});

module.exports = router;