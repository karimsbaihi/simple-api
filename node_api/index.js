const express = require("express");
const fileUpload = require("express-fileupload");
const axios = require("axios");
const FormData = require("form-data");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(fileUpload());
app.use(express.static("public")); // Serve frontend

// ------------------- Serve HTML Interface -------------------
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// ------------------- Health Check -------------------
app.get("/health", async (req, res) => {
  try {
    const response = await axios.get("https://tiny-imagenet-service.onrender.com/"); // Flask root route
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: "Model service unavailable" });
  }
});

// ------------------- Tiny ImageNet Prediction -------------------
app.post("/predict", async (req, res) => {
  try {
    if (!req.files || !req.files.image) {
      return res.status(400).json({ error: "No image uploaded" });
    }

    const file = req.files.image;
    const formData = new FormData();
    // Flask expects "file"
    formData.append("file", file.data, file.name);

    const response = await axios.post(
      "https://tiny-imagenet-service.onrender.com/predict",
      formData,
      { headers: formData.getHeaders() }
    );

    res.json(response.data);
  } catch (err) {
    console.error("Prediction error:", err.message);
    res.status(500).json({ error: "Prediction failed" });
  }
});

// ------------------- Simple Routes -------------------

// Root route
app.get("/", (req, res) => {
  res.json({ message: "Simple API with calculations and image classification" });
});

// Calculation routes
app.get("/sum/:a/:b", (req, res) => {
  const a = parseInt(req.params.a);
  const b = parseInt(req.params.b);
  res.json({ result: a + b });
});

app.get("/subtract/:a/:b", (req, res) => {
  const a = parseInt(req.params.a);
  const b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  res.json({ result: a - b });
});

app.get("/multiply/:a/:b", (req, res) => {
  const a = parseInt(req.params.a);
  const b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  res.json({ result: a * b });
});

app.get("/divide/:a/:b", (req, res) => {
  const a = parseInt(req.params.a);
  const b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  if (b === 0) return res.status(400).json({ error: "Division by zero" });
  res.json({ result: a / b });
});

app.get("/modulus/:a/:b", (req, res) => {
  const a = parseInt(req.params.a);
  const b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  res.json({ result: a % b });
});

app.get("/power/:a/:b", (req, res) => {
  const a = parseInt(req.params.a);
  const b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  res.json({ result: Math.pow(a, b) });
});

// ------------------- Start Server -------------------
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`🌍 Web interface: http://localhost:${PORT}`);
});

