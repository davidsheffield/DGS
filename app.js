const stage = document.getElementById("stage");
const imgA = document.getElementById("img-a");
const imgB = document.getElementById("img-b");

let current = null;
let next = null;
let busy = false;

async function fetchPair() {
  const r = await fetch("/api/pair", { cache: "no-store" });
  return await r.json();
}

function preload(pair) {
  return new Promise((resolve) => {
    let remaining = 2;
    const done = () => { if (--remaining === 0) resolve(pair); };
    const a = new Image();
    const b = new Image();
    a.onload = a.onerror = done;
    b.onload = b.onerror = done;
    a.src = "/samples/" + pair.a;
    b.src = "/samples/" + pair.b;
    pair._a = a;
    pair._b = b;
  });
}

function render(pair) {
  stage.dataset.size = pair.size;
  imgA.src = "/samples/" + pair.a;
  imgB.src = "/samples/" + pair.b;
}

async function loadNextInBackground() {
  const p = await fetchPair();
  next = await preload(p);
}

async function advance() {
  if (!next) {
    const p = await fetchPair();
    next = await preload(p);
  }
  current = next;
  next = null;
  render(current);
  loadNextInBackground();
}

async function submitVote(winner) {
  if (busy || !current) return;
  busy = true;
  const payload = {
    pair_id: current.pair_id,
    a: current.a,
    b: current.b,
    size: current.size,
    winner,
  };
  try {
    await fetch("/api/vote", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await advance();
  } finally {
    busy = false;
  }
}

document.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    submitVote("a");
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    submitVote("b");
  }
});

advance();
