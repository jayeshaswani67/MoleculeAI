const titles = {
  dashboard: "Overview",
  analyzer: "Molecule analyzer",
  explain: "Explainability",
  uncertainty: "Uncertainty analysis",
  benchmark: "Model benchmark"
};

let lastAnalysis = null;

function showPanel(panel) {
  document.querySelectorAll(".panel").forEach(el => el.classList.remove("active"));
  document.getElementById(panel).classList.add("active");
  document.querySelectorAll(".nav-btn").forEach(btn => {
    const active = btn.dataset.panel === panel;
    btn.classList.toggle("nav-active", active);
    btn.classList.toggle("text-slate-600", !active);
  });
  document.getElementById("page-title").textContent = titles[panel] || "MoleculeAI";
}
document.querySelectorAll(".nav-btn").forEach(btn =>
  btn.addEventListener("click", () => showPanel(btn.dataset.panel))
);

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed.");
  return data;
}

function currentSmiles() {
  return document.getElementById("smiles-input").value.trim();
}

function setBusy(buttonId, busy, label) {
  const button = document.getElementById(buttonId);
  if (!button) return;
  button.disabled = busy;
  button.innerHTML = busy ? `<span class="spinner"></span> ${label}` : label;
  button.classList.toggle("opacity-60", busy);
}

function showError(id, message) {
  const el = document.getElementById(id);
  el.textContent = message;
  el.classList.remove("hidden");
}

async function analyzeMolecule() {
  const smiles = currentSmiles();
  const error = document.getElementById("analyze-error");
  error.classList.add("hidden");
  if (!smiles) return showError("analyze-error", "Enter a SMILES string first.");
  setBusy("analyze-btn", true, "Analyzing...");
  try {
    const data = await api("/api/analyze", { smiles });
    lastAnalysis = data;
    document.getElementById("analysis-result").classList.remove("hidden");
    document.getElementById("prediction-label").textContent = data.prediction_label;
    document.getElementById("prediction-pill").textContent = data.prediction === 1 ? "ACTIVE · BACE1" : "INACTIVE · BACE1";
    document.getElementById("prediction-pill").className = "h-fit px-3 py-2 rounded-full text-sm font-bold " +
      (data.prediction === 1 ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-700");
    const score = data.active_probability * 100;
    document.getElementById("active-score").textContent = `${score.toFixed(2)}%`;
    document.getElementById("active-bar").style.width = `${score}%`;
    document.getElementById("scientific-note").textContent = data.scientific_note;
    const labels = {
      molecular_weight: "Molecular weight (Da)",
      logp: "LogP",
      h_bond_donors: "H-bond donors",
      h_bond_acceptors: "H-bond acceptors",
      tpsa: "TPSA",
      rotatable_bonds: "Rotatable bonds"
    };
    document.getElementById("descriptor-grid").innerHTML = Object.entries(data.descriptors).map(([key, value]) =>
      `<div class="bg-white border border-slate-200 rounded-xl p-4"><div class="text-xs text-slate-500">${labels[key] || key}</div><div class="text-xl font-extrabold mt-1">${value}</div></div>`
    ).join("");
  } catch (err) {
    showError("analyze-error", err.message);
  } finally {
    setBusy("analyze-btn", false, "Run prediction");
  }
}

async function runExplain() {
  const output = document.getElementById("explain-output");
  const smiles = currentSmiles();
  if (!smiles) {
    output.innerHTML = `<p class="text-sm text-red-700">Enter a SMILES string in Molecule Analyzer first.</p>`;
    showPanel("analyzer");
    return;
  }
  output.innerHTML = `<div class="text-sm text-slate-500"><span class="spinner"></span> Generating explanation...</div>`;
  try {
    const data = await api("/api/explain", { smiles });
    const features = data.explanation?.features || [];
    output.innerHTML = `
      <div class="rounded-xl bg-slate-50 p-4 mb-4 text-sm">
        <div class="font-bold">${data.interpretation}</div>
        <div class="text-slate-500 mt-1">Method: ${data.explanation?.method || "Unavailable"} · Active score: ${(data.active_class_score * 100).toFixed(2)}%</div>
      </div>
      <h4 class="font-bold text-sm mb-3">Most influential model features</h4>
      <div class="space-y-2">${features.map(item => {
        const impact = Number(item.impact || 0);
        const width = Math.min(100, Math.abs(impact) * 400);
        const positive = impact >= 0;
        return `<div class="grid grid-cols-[minmax(0,1fr)_90px] gap-3 items-center text-xs">
          <div class="min-w-0"><div class="truncate font-semibold">${item.feature_name}</div><div class="h-1.5 bg-slate-100 rounded-full mt-1 overflow-hidden"><div class="h-full ${positive ? "bg-teal-600" : "bg-rose-500"}" style="width:${width}%"></div></div></div>
          <div class="text-right font-mono ${positive ? "text-teal-700" : "text-rose-600"}">${impact >= 0 ? "+" : ""}${impact.toFixed(5)}</div>
        </div>`;
      }).join("")}</div>
      <p class="text-xs text-slate-500 mt-4">${data.scientific_note || ""}</p>`;
  } catch (err) {
    output.innerHTML = `<p class="text-sm text-red-700">${escapeHtml(err.message)}</p>`;
  }
}

async function runUncertainty() {
  const output = document.getElementById("uncertainty-output");
  const smiles = currentSmiles();
  if (!smiles) {
    output.innerHTML = `<p class="text-sm text-red-700">Enter a SMILES string in Molecule Analyzer first.</p>`;
    showPanel("analyzer");
    return;
  }
  output.innerHTML = `<div class="text-sm text-slate-500"><span class="spinner"></span> Assessing reliability...</div>`;
  try {
    const data = await api("/api/uncertainty", { smiles });
    const tree = data.random_forest_tree_disagreement;
    const sim = data.chemical_similarity;
    const cards = [
      ["Probability entropy", Number(data.probability_entropy).toFixed(4), "Higher values indicate more probability uncertainty."],
      ["Probability margin", Number(data.probability_margin).toFixed(4), "Difference between active and inactive scores."],
      ["Nearest dataset similarity", sim?.available ? `${(sim.nearest_similarity * 100).toFixed(1)}%` : "Unavailable", data.chemical_space_status],
      ["Tree disagreement", tree ? Number(tree.standard_deviation).toFixed(4) : "Unavailable", tree ? `${tree.number_of_trees} trees · standard deviation of active scores` : "This model does not expose individual tree predictions."]
    ];
    output.innerHTML = `
      <div class="grid sm:grid-cols-2 gap-3">${cards.map(c => `<div class="border border-slate-200 rounded-xl p-4"><div class="text-xs text-slate-500">${c[0]}</div><div class="text-2xl font-extrabold mt-2">${c[1]}</div><div class="text-xs text-slate-500 mt-2">${c[2]}</div></div>`).join("")}</div>
      <div class="mt-4 bg-amber-50 border border-amber-200 text-amber-900 rounded-xl p-4 text-sm"><b>${data.uncertainty_interpretation}</b><p class="mt-2">${data.scientific_note}</p></div>`;
  } catch (err) {
    output.innerHTML = `<p class="text-sm text-red-700">${escapeHtml(err.message)}</p>`;
  }
}

async function runBenchmark() {
  const output = document.getElementById("benchmark-output");
  setBusy("benchmark-btn", true, "Running benchmark...");
  output.innerHTML = `<p class="text-sm text-slate-500"><span class="spinner"></span> Training and evaluating models. Please wait...</p>`;
  try {
    const data = await api("/api/benchmark", {});
    const rows = data.results || [];
    if (!rows.length) throw new Error("No benchmark results were returned.");
    const columns = ["model", "accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"];
    const header = columns.map(c => `<th class="text-left p-3 text-xs uppercase text-slate-500 whitespace-nowrap">${c.replace("_", " ")}</th>`).join("");
    const body = rows.map(row => `<tr class="border-t border-slate-100">${columns.map(c => {
      const value = row[c];
      return `<td class="p-3 text-sm whitespace-nowrap">${typeof value === "number" ? value.toFixed(4) : (value ?? "—")}</td>`;
    }).join("")}</tr>`).join("");
    output.innerHTML = `<table class="w-full"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table><p class="text-xs text-slate-500 mt-4">Validation-set results. Select models using validation data; keep the final test set untouched until final evaluation.</p>`;
  } catch (err) {
    output.innerHTML = `<p class="text-sm text-red-700 bg-red-50 p-3 rounded-lg">${escapeHtml(err.message)}</p>`;
  } finally {
    setBusy("benchmark-btn", false, "Run benchmark");
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[char]));
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const dot = document.getElementById("health-dot");
    const label = document.getElementById("health-text");
    dot.className = "w-2 h-2 rounded-full " + (data.model_available ? "bg-emerald-500" : "bg-amber-400");
    label.textContent = data.model_available ? "API connected · model found" : "API connected · model missing";
  } catch {
    document.getElementById("health-dot").className = "w-2 h-2 rounded-full bg-red-500";
    document.getElementById("health-text").textContent = "API unavailable";
  }
}
checkHealth();
