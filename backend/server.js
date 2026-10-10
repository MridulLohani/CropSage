const express = require("express");
const authRoutes = require("./routes/authRoutes");
const cors = require("cors");
require("dotenv").config();
const predictionRoutes = require("./routes/predictionRoutes");
const connectDB = require("./config/db");

const app = express();


// Connect MongoDB
connectDB();


// Middleware
app.use(cors());
app.use(express.json());
app.use("/api/auth", authRoutes);
app.use("/api/predictions", predictionRoutes);
// Health check
app.get("/api/health", (req, res) => {
    res.json({
        status: "CropSage backend is running"
    });
});


// Server
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
    console.log(`CropSage backend running on port ${PORT}`);
});