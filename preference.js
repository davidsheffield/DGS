const stage = document.getElementById("stage");
const markA = document.getElementById("mark-a");
const markB = document.getElementById("mark-b");
const sizeSel = document.getElementById("size");
const count = document.getElementById("count");

let current = null;          // the duel on screen
let sizeVal = "40px";        // fixed display scale for this session
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

function applySize() {
  stage.style.setProperty("--h", sizeVal);
  stage.dataset.size = sizeVal;
}

function render(duel) {
  current = duel;
  markA.innerHTML = duel.a.svg;
  markB.innerHTML = duel.b.svg;
  count.textContent = `${duel.n_votes} vote${duel.n_votes === 1 ? "" : "s"}`;
}

async function vote(winner) {
  if (busy || !current) return;
  busy = true;
  try {
    // one round-trip: log the vote, get the next duel back (from the updated model)
    render(await api("/api/vote",
      { duel_id: current.duel_id, winner, size: sizeVal }));
  } catch (e) {
    alert(e.message);
  } finally {
    busy = false;
  }
}

document.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); vote("a"); }
  else if (e.key === "ArrowRight") { e.preventDefault(); vote("b"); }
  else if (e.key === "ArrowDown") { e.preventDefault(); vote("tie"); }
});

// changing the scale resizes the marks on screen; it's logged with the next vote
sizeSel.addEventListener("change", () => { sizeVal = sizeSel.value; applySize(); });

(async function start() {
  const st = await api("/api/status");
  sizeSel.innerHTML = "";
  for (const s of st.sizes) {
    const opt = document.createElement("option");
    opt.value = opt.textContent = s;
    sizeSel.appendChild(opt);
  }
  sizeVal = st.default_size;
  sizeSel.value = sizeVal;
  applySize();
  render(await api("/api/next?size=" + sizeVal));
})();
