// User study client: routes are shown unlabelled ("A"/"B") in a random order per trial.
"use strict";

function participantId() {
  let id = null;
  try { id = localStorage.getItem("wayfinder-pid"); } catch (_) { /* storage blocked */ }
  if (!id) {
    id = "p-" + Math.random().toString(36).slice(2, 10);
    try { localStorage.setItem("wayfinder-pid", id); } catch (_) { /* ignore */ }
  }
  return id;
}

const pid = participantId();
const css = getComputedStyle(document.documentElement);
const color = (name) => css.getPropertyValue(name).trim();
let trials = [];
let index = 0;
let current = null;    // { order: ["profile","fastest"] or reverse, started }
let map = null;
let drawn = [];

function showError(msg) {
  const el = document.getElementById("error");
  el.textContent = msg || "";
  el.hidden = !msg;
}

function ensureMap() {
  if (map) return;
  map = L.map("map").setView([17.422, 78.42], 15);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
}

function showTrial() {
  const t = trials[index];
  document.getElementById("bar").style.width = `${(100 * index) / trials.length}%`;
  document.getElementById("counter").textContent = `Trip ${index + 1} of ${trials.length}`;
  document.getElementById("scenario").textContent = t.scenario;
  document.getElementById("endpoints").textContent = `From near ${t.origin.near} to near ${t.destination.near}.`;
  const order = Math.random() < 0.5 ? ["profile", "fastest"] : ["fastest", "profile"];
  current = { order, started: performance.now() };
  for (const l of drawn) map.removeLayer(l);
  drawn = [];
  const colors = [color("--route"), color("--route-b")];
  const dashes = [null, "10 8"];
  order.forEach((kind, i) => {
    const line = L.polyline(t.routes[kind].latlon, { color: colors[i], weight: 6, opacity: 0.85, dashArray: dashes[i] })
      .addTo(map).bindTooltip(`Route ${"AB"[i]}`, { sticky: true });
    drawn.push(line);
  });
  drawn.push(L.marker([t.origin.lat, t.origin.lon], { title: "Start" }).addTo(map).bindTooltip("Start"));
  drawn.push(L.marker([t.destination.lat, t.destination.lon], { title: "End" }).addTo(map).bindTooltip("End"));
  map.fitBounds(L.featureGroup(drawn.slice(0, 2)).getBounds(), { padding: [30, 30] });
}

async function choose(letter) {
  const t = trials[index];
  const kind = current.order["AB".indexOf(letter)];
  const body = {
    participant_id: pid, trial_id: t.trial_id, choice: kind, shown_first: current.order[0],
    seconds: (performance.now() - current.started) / 1000,
  };
  try {
    const res = await fetch("/api/study/response", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    showError("");
  } catch (err) {
    showError("Could not save your answer: " + (err.message || err));
    return;
  }
  index += 1;
  if (index >= trials.length) {
    document.getElementById("trial").hidden = true;
    document.getElementById("done").hidden = false;
    document.getElementById("pid").textContent = pid;
    document.getElementById("bar").style.width = "100%";
  } else {
    showTrial();
  }
}

document.querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", () => choose(b.dataset.pick)));
document.getElementById("start").addEventListener("click", async () => {
  try {
    const res = await fetch("/api/study/trials");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    trials = data.trials;
  } catch (err) {
    showError("Could not load the study: " + (err.message || err));
    return;
  }
  document.getElementById("intro").hidden = true;
  document.getElementById("trial").hidden = false;
  ensureMap();
  map.invalidateSize();
  showTrial();
});
