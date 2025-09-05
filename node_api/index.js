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

// ------------------- Health Check for Python Service -------------------
app.get("/health", async (req, res) => {
  try {
    const response = await axios.get("https://tiny-imagenet-service.onrender.com/health");
    res.json({ node: "healthy", python_service: response.data });
  } catch (error) {
    console.error("Health check error:", error.message);
    res.status(500).json({ error: "Python service unavailable" });
  }
});

// ------------------- Tiny ImageNet Prediction -------------------
app.post("/predict", async (req, res) => {
  try {
    if (!req.files || !req.files.image) {
      console.log("No files received");
      return res.status(400).json({ error: "No image uploaded" });
    }

    const file = req.files.image;
    console.log("File received:", file.name, "Size:", file.size);

    const formData = new FormData();
    formData.append("image", file.data, file.name);

    console.log("Forwarding image to Python service...");

    const response = await axios.post(
      "https://tiny-imagenet-service.onrender.com/predict",
      formData,
      { headers: formData.getHeaders() }
    );

    console.log("Received response from Python service:", response.data);
    res.json(response.data);

  } catch (err) {
    console.error("Prediction error:", err.message);
    res.status(500).json({ error: "Prediction failed" });
  }
});

// ------------------- Simple Math Routes -------------------
app.get("/sum/:a/:b", (req, res) => {
  const a = parseInt(req.params.a), b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  res.json({ result: a + b });
});

app.get("/subtract/:a/:b", (req, res) => {
  const a = parseInt(req.params.a), b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  res.json({ result: a - b });
});

app.get("/multiply/:a/:b", (req, res) => {
  const a = parseInt(req.params.a), b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  res.json({ result: a * b });
});

app.get("/divide/:a/:b", (req, res) => {
  const a = parseInt(req.params.a), b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  if (b === 0) return res.status(400).json({ error: "Division by zero" });
  res.json({ result: a / b });
});

app.get("/modulus/:a/:b", (req, res) => {
  const a = parseInt(req.params.a), b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  res.json({ result: a % b });
});

app.get("/power/:a/:b", (req, res) => {
  const a = parseInt(req.params.a), b = parseInt(req.params.b);
  if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
  res.json({ result: Math.pow(a, b) });
});

// ------------------- Start Server -------------------
app.listen(PORT, () => {
  console.log(`🚀 Node API running on port ${PORT}`);
  console.log(`🌍 Web interface: http://localhost:${PORT}`);
});
