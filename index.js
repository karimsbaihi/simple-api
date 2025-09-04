const express = require("express");
const app = express();
const PORT = process.env.PORT;

app.use(express.json());

// Simple route
app.get("/", (req, res) => {
  res.json({ message: "simple API with simple calculations" });
});

// Example dynamic route
app.get("/sum/:a/:b", (req, res) => {
  const a = parseInt(req.params.a);
  const b = parseInt(req.params.b);
  res.json({ result: a + b });
});
app.get("/divide/:a/:b", (req, res) => {
    const a = parseInt(req.params.a);
    const b = parseInt(req.params.b);
    if (isNaN(a) || isNaN(b)) return res.status(400).json({ error: "Invalid numbers" });
    if (b === 0) return res.status(400).json({ error: "Division by zero" });
    res.json({ result: a / b });
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
    res.json({ result: Math.pow(a, b)});
});
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
