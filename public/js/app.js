const API_BASE = detectApiBase();

function detectApiBase() {
  const host = window.location.hostname;
  if (host.includes("netlify")) {
    return "/.netlify/functions/analyze";
  }
  if (host.includes("vercel") || host === "localhost" || host === "127.0.0.1") {
    return "/api/analyze";
  }
  return "/api/analyze";
}

const SENTIMENT_ICONS = {
  positive: "↑",
  negative: "↓",
  neutral: "→",
};

const elements = {
  textInput: document.getElementById("text-input"),
  charCount: document.getElementById("char-count"),
  analyzeBtn: document.getElementById("analyze-btn"),
  btnText: document.querySelector(".btn-text"),
  btnLoader: document.querySelector(".btn-loader"),
  resultsPanel: document.getElementById("results-panel"),
  emptyState: document.getElementById("empty-state"),
  sentimentBadge: document.getElementById("sentiment-badge"),
  sentimentIcon: document.getElementById("sentiment-icon"),
  sentimentLabel: document.getElementById("sentiment-label"),
  confidenceValue: document.getElementById("confidence-value"),
  consistencyValue: document.getElementById("consistency-value"),
  consistencyFill: document.getElementById("consistency-fill"),
  reliabilityValue: document.getElementById("reliability-value"),
  reliabilityFill: document.getElementById("reliability-fill"),
  modelsList: document.getElementById("models-list"),
  breakdownGrid: document.getElementById("breakdown-grid"),
  sampleChips: document.getElementById("sample-chips"),
};

elements.textInput.addEventListener("input", () => {
  elements.charCount.textContent = `${elements.textInput.value.length} / 5000`;
});

elements.analyzeBtn.addEventListener("click", analyze);
elements.textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    analyze();
  }
});

loadSamples();

async function loadSamples() {
  try {
    const res = await fetch(`${API_BASE.replace("/analyze", "/samples")}`);
    if (!res.ok) return;
    const data = await res.json();
    renderSampleChips(data.samples || []);
  } catch {
    renderSampleChips([
      "I absolutely love this product!",
      "This is terrible, worst ever.",
      "The meeting is scheduled for 3pm.",
    ]);
  }
}

function renderSampleChips(samples) {
  elements.sampleChips.innerHTML = samples
    .slice(0, 4)
    .map((text, i) => {
      const short = text.length > 40 ? text.slice(0, 40) + "…" : text;
      return `<button class="chip" data-index="${i}" title="${escapeHtml(text)}">${escapeHtml(short)}</button>`;
    })
    .join("");

  elements.sampleChips.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const idx = parseInt(chip.dataset.index, 10);
      elements.textInput.value = samples[idx];
      elements.charCount.textContent = `${elements.textInput.value.length} / 5000`;
      analyze();
    });
  });
}

async function analyze() {
  const text = elements.textInput.value.trim();
  if (!text) {
    showError("Please enter some text to analyze.");
    return;
  }

  setLoading(true);

  try {
    const res = await fetch(API_BASE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Analysis failed");
    }

    renderResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

function renderResults(data) {
  elements.emptyState.classList.add("hidden");
  elements.resultsPanel.classList.remove("hidden");
  elements.resultsPanel.classList.add("fade-in");

  const label = data.final_label;
  elements.sentimentBadge.className = `sentiment-badge ${label}`;
  elements.sentimentIcon.textContent = SENTIMENT_ICONS[label] || "—";
  elements.sentimentLabel.textContent = label;
  elements.confidenceValue.textContent = `${(data.final_confidence * 100).toFixed(1)}%`;

  setRing(elements.consistencyFill, elements.consistencyValue, data.consistency_score);
  setRing(elements.reliabilityFill, elements.reliabilityValue, data.reliability_score);

  elements.modelsList.innerHTML = data.models
    .map(
      (m) => `
      <div class="model-card">
        <div>
          <span class="model-name">${escapeHtml(m.model)}</span>
          <span class="model-label ${m.label}">${m.label}</span>
        </div>
        <span class="model-confidence">${(m.confidence * 100).toFixed(1)}%</span>
      </div>
    `
    )
    .join("");

  const b = data.breakdown;
  elements.breakdownGrid.innerHTML = `
    <div class="breakdown-item"><span>Models Used</span>${b.models_used}</div>
    <div class="breakdown-item"><span>Word Count</span>${b.word_count}</div>
    <div class="breakdown-item"><span>Positive Votes</span>${b.agreement.positive}</div>
    <div class="breakdown-item"><span>Negative Votes</span>${b.agreement.negative}</div>
    <div class="breakdown-item"><span>Neutral Votes</span>${b.agreement.neutral}</div>
    <div class="breakdown-item"><span>Characters</span>${b.text_length}</div>
  `;
}

function setRing(fillEl, valueEl, score) {
  const pct = Math.min(Math.max(score, 0), 100);
  fillEl.setAttribute("stroke-dasharray", `${pct}, 100`);
  valueEl.textContent = `${pct.toFixed(0)}%`;
}

function setLoading(loading) {
  elements.analyzeBtn.disabled = loading;
  elements.btnText.classList.toggle("hidden", loading);
  elements.btnLoader.classList.toggle("hidden", !loading);
}

function showError(message) {
  const existing = document.querySelector(".error-toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = "error-toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
