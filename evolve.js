const grid = document.getElementById("grid");
const runSelect = document.getElementById("run-select");
const runName = document.getElementById("run-name");
const genLabel = document.getElementById("gen-label");
const prevBtn = document.getElementById("prev-gen");
const nextBtn = document.getElementById("next-gen");
const breedBtn = document.getElementById("breed");
const restartBtn = document.getElementById("restart-run");
const nOffspring = document.getElementById("n-offspring");
const elitism = document.getElementById("elitism");
const selCount = document.getElementById("sel-count");

let view = null;              // last generation payload received
const selected = new Set();
let busy = false;

async function api(path, body) {
  const opts = body === undefined
    ? { cache: "no-store" }
    : { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function atLatest() {
  return view && view.generation === view.n_generations - 1;
}

function updateControls() {
  const latest = atLatest();
  genLabel.textContent = view
    ? `Generation ${view.generation} of ${view.n_generations - 1}`
    : "no run loaded";
  prevBtn.disabled = !view || view.generation === 0;
  nextBtn.disabled = !view || latest;
  breedBtn.disabled = !view || !latest || selected.size === 0 || busy;
  restartBtn.disabled = !view || busy;
  selCount.textContent = selected.size ? `${selected.size} selected` : "";
}

function render(payload) {
  view = payload;
  selected.clear();
  runName.textContent = payload.name + " — " + payload.dir;
  nOffspring.value = payload.settings.n_offspring;
  elitism.checked = payload.settings.elitism;
  grid.innerHTML = "";
  const latest = atLatest();
  for (const c of payload.candidates) {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.id = c.id;
    card.innerHTML = c.svg + `<span class="badge">${c.origin}</span>`;
    // browsing an old generation: show what was selected back then, read-only
    if (!latest && payload.selected.includes(c.id)) card.classList.add("was-selected");
    if (latest) {
      card.addEventListener("click", () => {
        if (selected.has(c.id)) { selected.delete(c.id); card.classList.remove("selected"); }
        else { selected.add(c.id); card.classList.add("selected"); }
        updateControls();
      });
    }
    grid.appendChild(card);
  }
  updateControls();
}

async function loadRuns(autoOpen) {
  const data = await api("/api/runs");
  runSelect.innerHTML = '<option value="">— pick a run —</option>';
  for (const r of data.runs) {
    const opt = document.createElement("option");
    opt.value = r.name;
    opt.textContent = `${r.name} (${r.generations} gen)`;
    runSelect.appendChild(opt);
  }
  if (view) runSelect.value = view.name;
  else if (autoOpen && data.runs.length === 1) await openRun(data.runs[0].name);
}

async function openRun(name) {
  render(await api("/api/runs/" + name));
  runSelect.value = name;
}

async function withBusy(fn) {
  if (busy) return;
  busy = true;
  updateControls();
  try {
    await fn();
  } catch (e) {
    alert(e.message);
  } finally {
    busy = false;
    updateControls();
  }
}

breedBtn.addEventListener("click", () => withBusy(async () => {
  const payload = await api(`/api/runs/${view.name}/breed`, {
    selected: [...selected],
    n_offspring: Math.max(1, Math.min(200, Number(nOffspring.value) || 16)),
    elitism: elitism.checked,
  });
  render(payload);
  await loadRuns(false);
}));

restartBtn.addEventListener("click", () => withBusy(async () => {
  if (!confirm(`Restart run "${view.name}" from generation 0? All bred generations will be deleted.`)) return;
  render(await api(`/api/runs/${view.name}/restart`, {}));
  await loadRuns(false);
}));

document.getElementById("new-run").addEventListener("click", () => withBusy(async () => {
  const name = prompt("Run name ([A-Za-z0-9_-]):");
  if (!name) return;
  const outDir = prompt("Output directory (blank = default runs/):") || null;
  render(await api("/api/runs", { name, out_dir: outDir }));
  await loadRuns(false);
}));

runSelect.addEventListener("change", () => {
  if (runSelect.value) withBusy(() => openRun(runSelect.value));
});

prevBtn.addEventListener("click", () => withBusy(async () => {
  render(await api(`/api/runs/${view.name}/gen/${view.generation - 1}`));
}));

nextBtn.addEventListener("click", () => withBusy(async () => {
  render(await api(`/api/runs/${view.name}/gen/${view.generation + 1}`));
}));

loadRuns(true);
