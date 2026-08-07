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
  위험: "result__badge--danger",
  주의: "result__badge--warning",
  안전: "result__badge--safe",
};

function setLoading(isLoading) {
  loadingSection.hidden = !isLoading;
  analyzeBtn.disabled = isLoading;

  if (isLoading) {
    analyzeBtn.textContent = "분석 중...";
    resultSection.hidden = true;
    errorMsg.hidden = true;
  } else {
    analyzeBtn.textContent = "분석하기";
  }
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
  resultSection.hidden = true;
}

function fillList(element, items = []) {
  element.innerHTML = "";

  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    element.appendChild(li);
  });
}

function renderBertBars(distribution = []) {
  bertBars.innerHTML = "";

  distribution.slice(0, 3).forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar";

    const label = document.createElement("span");
    label.textContent = item.label_ko;

    const track = document.createElement("div");
    track.className = "bar__track";

    const fill = document.createElement("div");
    fill.className = "bar__fill";
    fill.style.width = `${Math.round(item.score * 100)}%`;

    track.appendChild(fill);

    const percent = document.createElement("span");
    percent.textContent = `${Math.round(item.score * 100)}%`;

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(percent);

    bertBars.appendChild(row);
  });
}

function renderResult(data) {
  verdictBadge.textContent = data.verdict;
  verdictBadge.className = `result__badge ${VERDICT_STYLE[data.verdict] || ""}`;

  summaryText.textContent = data.summary || "";

  scamTypeText.textContent =
    data.scam_type_ko && data.scam_type
      ? `${data.scam_type_ko} (${data.scam_type})`
      : data.scam_type_ko || "-";

  fillList(reasonsList, data.reasons);
  fillList(actionGuideList, data.action_guide);

  renderBertBars(data.bert_prediction?.distribution);

  resultSection.hidden = false;
}

async function analyze() {
  const text = textArea.value.trim();

  if (!text) {
    showError("문자 내용을 입력해주세요.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "분석 중 오류가 발생했습니다.");
    }

    renderResult(data);
  } catch (error) {
    showError(error.message || "오류가 발생했습니다.");
  } finally {
    setLoading(false);
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();

    if (data.agent_ready) {
      healthStatus.textContent = "서버 정상 연결";
    } else {
      healthStatus.textContent = "모델 준비 중";
    }
  } catch {
    healthStatus.textContent = "서버에 연결할 수 없습니다.";
  }
}

analyzeBtn.addEventListener("click", analyze);

textArea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    analyze();
  }
});

checkHealth();