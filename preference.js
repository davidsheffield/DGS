const stage = document.getElementById("stage");
const markA = document.getElementById("mark-a");
const markB = document.getElementById("mark-b");
const sizeSel = document.getElementById("size");
const count = document.getElementById("count");
const sizeCounts = document.getElementById("size-counts");
const modeLabel = document.getElementById("mode-label");
const confirmBtn = document.getElementById("confirm-toggle");

let current = null;          // the duel on screen
let sizeVal = "40px";        // fixed display scale for this session
let busy = false;
let confirmMode = false;     // "Confirm" toggle: ask confirm duels instead of scheduled ones
let sizeOrder = [];          // all known size buckets, for the per-size count strip
let votesBySize = {};        // size -> vote count, kept in sync from duel responses

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

function applySize() {
  stage.style.setProperty("--h", sizeVal);
  stage.dataset.size = sizeVal;
}

function renderSizeCounts() {
  sizeCounts.textContent = sizeOrder
    .map((s) => `${s === sizeVal ? "*" : ""}${s}:${votesBySize[s] || 0}`)
    .join("  ");
}

function render(duel) {
  current = duel;
  markA.innerHTML = duel.a.svg;
  markB.innerHTML = duel.b.svg;
  count.textContent = `${duel.n_votes} vote${duel.n_votes === 1 ? "" : "s"}`;
  modeLabel.textContent = duel.mode == null ? "" :
    duel.mode === "axis" && duel.axis != null ? `axis duel · PC${duel.axis + 1}`
                                              : `${duel.mode} duel`;
  votesBySize[duel.size] = duel.n_votes;
  renderSizeCounts();
}

async function vote(winner) {
  if (busy || !current) return;
  busy = true;
  try {
    // one round-trip: log the vote, get the next duel back (from the updated model)
    render(await api("/api/vote", {
      duel_id: current.duel_id, winner, size: sizeVal,
      mode: confirmMode ? "confirm" : undefined,
    }));
  } catch (e) {
    alert(e.message);
  } finally {
    busy = false;
  }
}

confirmBtn.addEventListener("click", () => {
  confirmMode = !confirmMode;
  confirmBtn.classList.toggle("active", confirmMode);
});

document.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); vote("a"); }
  else if (e.key === "ArrowRight") { e.preventDefault(); vote("b"); }
  else if (e.key === "ArrowDown") { e.preventDefault(); vote("tie"); }
});

async function loadNext() {
  // duels are locked to the size they were issued for (each bucket has its
  // own model server-side), so switching buckets needs a fresh duel, not
  // just a rescale of the one on screen.
  if (busy) return;
  busy = true;
  try {
    render(await api("/api/next?size=" + sizeVal + (confirmMode ? "&mode=confirm" : "")));
  } catch (e) {
    alert(e.message);
  } finally {
    busy = false;
  }
}

sizeSel.addEventListener("change", () => {
  sizeVal = sizeSel.value;
  applySize();
  loadNext();
});

(async function start() {
  const st = await api("/api/status");
  sizeSel.innerHTML = "";
  for (const s of st.sizes) {
    const opt = document.createElement("option");
    opt.value = opt.textContent = s;
    sizeSel.appendChild(opt);
  }
  sizeOrder = st.sizes;
  votesBySize = st.votes_by_size || {};
  sizeVal = st.default_size;
  sizeSel.value = sizeVal;
  applySize();
  await loadNext();
})();
