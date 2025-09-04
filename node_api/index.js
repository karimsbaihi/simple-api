const express = require("express");
const fileUpload = require("express-fileupload");
const axios = require("axios");
const FormData = require("form-data");

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(fileUpload()); // enable file uploads

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

// ------------------- Tiny ImageNet Prediction Route -------------------

app.post("/predict", async (req, res) => {
  try {
    if (!req.files || !req.files.image) {
      return res.status(400).json({ error: "No image uploaded" });
    }

    const file = req.files.image;
    const formData = new FormData();
    formData.append("file", file.data, file.name);

    // Replace this URL:
    const response = await axios.post(
      "https://tiny-imagenet-service.onrender.com/predict",
      formData,
      { headers: formData.getHeaders() }
    );


    res.json(response.data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Prediction failed" });
  }
});

// ------------------- Start Server -------------------
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
