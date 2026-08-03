const textArea = document.getElementById("smsText");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorMsg = document.getElementById("errorMsg");
const loadingSection = document.getElementById("loading");
const resultSection = document.getElementById("resultSection");
const healthStatus = document.getElementById("healthStatus");

const verdictBadge = document.getElementById("verdictBadge");
const summaryText = document.getElementById("summaryText");
const scamTypeText = document.getElementById("scamTypeText");
const reasonsList = document.getElementById("reasonsList");
const actionGuideList = document.getElementById("actionGuideList");
const bertBars = document.getElementById("bertBars");

const VERDICT_STYLE = {
  "위험": "result__badge--danger",
  "주의": "result__badge--warning",
  "안전": "result__badge--safe",
};

function setLoading(isLoading) {
  loadingSection.hidden = !isLoading;
  analyzeBtn.disabled = isLoading;
  if (isLoading) {
    resultSection.hidden = true;
    errorMsg.hidden = true;
  }
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
  resultSection.hidden = true;
}

function fillList(el, items) {
  el.innerHTML = "";
  (items || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  });
}

function renderBertBars(distribution) {
  bertBars.innerHTML = "";
  (distribution || []).slice(0, 3).forEach((d) => {
    const row = document.createElement("div");
    row.className = "bar";

    const label = document.createElement("span");
    label.textContent = d.label_ko;

    const track = document.createElement("div");
    track.className = "bar__track";
    const fill = document.createElement("div");
    fill.className = "bar__fill";
    fill.style.width = `${Math.round(d.score * 100)}%`;
    track.appendChild(fill);

    const pct = document.createElement("span");
    pct.textContent = `${Math.round(d.score * 100)}%`;

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(pct);
    bertBars.appendChild(row);
  });
}

function renderResult(data) {
  verdictBadge.textContent = data.verdict;
  verdictBadge.className = `result__badge ${VERDICT_STYLE[data.verdict] || ""}`;

  summaryText.textContent = data.summary || "";
  scamTypeText.textContent = `${data.scam_type_ko} (${data.scam_type})`;

  fillList(reasonsList, data.reasons);
  fillList(actionGuideList, data.action_guide);
  renderBertBars(data.bert_prediction && data.bert_prediction.distribution);

  resultSection.hidden = false;
}

async function analyze() {
  const text = textArea.value.trim();
  if (!text) {
    showError("분석할 문자 내용을 입력해주세요.");
    return;
  }

  setLoading(true);
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "분석 중 오류가 발생했습니다.");
    }

    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    healthStatus.textContent = data.agent_ready
      ? "✅ 백엔드 정상 연결됨"
      : "⏳ 백엔드 준비 중 (모델 로딩)...";
  } catch {
    healthStatus.textContent = "❌ 백엔드 서버에 연결할 수 없습니다.";
  }
}

analyzeBtn.addEventListener("click", analyze);
textArea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    analyze();
  }
});

checkHealth();
