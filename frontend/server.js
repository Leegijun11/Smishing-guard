const express = require("express");
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, "..", ".env") });

const app = express();
const PORT = process.env.FRONTEND_PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.post("/api/analyze", async (req, res) => {
  const text = (req.body && req.body.text) || "";

  try {
    const backendRes = await fetch(`${BACKEND_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await backendRes.json();
    res.status(backendRes.status).json(data);
  } catch (err) {
    res.status(502).json({
      detail: `백엔드 서버(${BACKEND_URL})에 연결할 수 없습니다: ${err.message}`,
    });
  }
});

app.get("/api/health", async (req, res) => {
  try {
    const backendRes = await fetch(`${BACKEND_URL}/health`);
    const data = await backendRes.json();
    res.status(backendRes.status).json(data);
  } catch (err) {
    res.status(502).json({ status: "backend_unreachable" });
  }
});

app.listen(PORT, () => {
  console.log(`Smishing Guard 프론트엔드 실행 중: http://127.0.0.1:${PORT}`);
  console.log(`백엔드 프록시 대상: ${BACKEND_URL}`);
});
