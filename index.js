const express = require("express");
const app = express();
const PORT = process.env.PORT || 3000;

// Simple route
app.get("/", (req, res) => {
  res.json({ message: "Hello Karim! Your API is live 🚀" });
});

// Example dynamic route
app.get("/sum/:a/:b", (req, res) => {
  const a = parseInt(req.params.a);
  const b = parseInt(req.params.b);
  res.json({ result: a + b });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
