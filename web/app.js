// Routes That Fit You — map client. Talks to the FastAPI service (/api/meta, /api/route).
"use strict";

const SLIDERS = [
  { key: "time", label: "Walking time", help: "Prefer shorter, faster routes" },
  { key: "junction", label: "Junction load", help: "Avoid turns and busy junctions" },
  { key: "traffic", label: "Traffic exposure", help: "Stay off major roads (proxy from road class)" },
  { key: "canopy", label: "Canopy", help: "Walk near tree cover (proximity, not measured shade)" },
  { key: "climb", label: "Climb", help: "Avoid uphill grades and stairs" },
];
const PRESET_LABELS = { fastest: "Fastest", shade: "Shade seeker", quiet: "Quiet streets", easy: "Easy on the knees" };

const state = { origin: null, destination: null, weights: { time: 1, junction: 0, traffic: 0, canopy: 0, climb: 0 } };
const css = getComputedStyle(document.documentElement);
const color = (name) => css.getPropertyValue(name).trim();

const map = L.map("map", { zoomControl: true }).setView([17.422, 78.42], 14);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let markers = { origin: null, destination: null };
let layers = { route: null, fastest: null };

function buildSliders() {
  const box = document.getElementById("sliders");
  for (const s of SLIDERS) {
    const div = document.createElement("div");
    div.className = "slider";
    div.innerHTML = `<label for="w-${s.key}">${s.label}</label><output id="o-${s.key}"></output>
      <input type="range" id="w-${s.key}" min="0" max="1" step="0.05">
      <small>${s.help}</small>`;
    box.appendChild(div);
    const input = div.querySelector("input");
    input.value = state.weights[s.key];
    div.querySelector("output").textContent = Number(input.value).toFixed(2);
    input.addEventListener("input", () => {
      state.weights[s.key] = Number(input.value);
      div.querySelector("output").textContent = Number(input.value).toFixed(2);
      scheduleRoute();
    });
  }
}

function setWeights(w) {
  for (const s of SLIDERS) {
    state.weights[s.key] = w[s.key] ?? 0;
    document.getElementById(`w-${s.key}`).value = state.weights[s.key];
    document.getElementById(`o-${s.key}`).textContent = state.weights[s.key].toFixed(2);
  }
  scheduleRoute();
}

function buildPresets(presets) {
  const box = document.getElementById("presets");
  for (const [name, w] of Object.entries(presets)) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = PRESET_LABELS[name] || name;
    b.addEventListener("click", () => setWeights(w));
    box.appendChild(b);
  }
}

function placeMarker(kind, latlng) {
  state[kind] = latlng;
  if (markers[kind]) {
    markers[kind].setLatLng(latlng);
  } else {
    markers[kind] = L.marker(latlng, { draggable: true, title: kind === "origin" ? "Start" : "Destination" })
      .addTo(map).bindTooltip(kind === "origin" ? "Start" : "Destination");
    markers[kind].on("dragend", (e) => { state[kind] = e.target.getLatLng(); scheduleRoute(); });
  }
}

map.on("click", (e) => {
  if (!state.origin) placeMarker("origin", e.latlng);
  else if (!state.destination) placeMarker("destination", e.latlng);
  else placeMarker("destination", e.latlng);
  scheduleRoute();
});

let timer = null;
function scheduleRoute() {
  clearTimeout(timer);
  timer = setTimeout(requestRoute, 250);
}

const fmt = {
  m: (v) => (v >= 1000 ? `${(v / 1000).toFixed(2)} km` : `${v.toFixed(0)} m`),
  min: (s) => `${(s / 60).toFixed(1)} min`,
  pct: (v) => `${(100 * v).toFixed(0)}%`,
  num: (v, d = 2) => Number(v).toFixed(d),
};

function statsRows(m, f) {
  const rows = [
    ["Length", fmt.m(m.length_m), f && fmt.m(f.length_m)],
    ["Walking time", fmt.min(m.walk_time_s), f && fmt.min(f.walk_time_s)],
    ["Canopy along route", fmt.pct(m.canopy_share), f && fmt.pct(f.canopy_share)],
    ["Traffic exposure (0–1)", fmt.num(m.traffic), f && fmt.num(f.traffic)],
    ["Climb", `${m.climb_m.toFixed(0)} m`, f && `${f.climb_m.toFixed(0)} m`],
    ["Stairs", fmt.m(m.stairs_m), f && fmt.m(f.stairs_m)],
    ["Junction load / km", fmt.num(m.turn_load_per_km, 1), f && fmt.num(f.turn_load_per_km, 1)],
  ];
  const head = f ? "<tr><th></th><th>yours</th><th>fastest</th></tr>" : "";
  return head + rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td>${f ? `<td>${r[2]}</td>` : ""}</tr>`).join("");
}

function showError(msg) {
  const el = document.getElementById("error");
  el.textContent = msg || "";
  el.hidden = !msg;
}

let latestRequest = 0;   // only the newest request may draw: an older, slower reply must not overwrite it
async function requestRoute() {
  if (!state.origin || !state.destination) return;
  const requestId = ++latestRequest;
  const algorithm = document.getElementById("algorithm").value;
  const body = {
    origin: { lat: state.origin.lat, lon: state.origin.lng },
    destination: { lat: state.destination.lat, lon: state.destination.lng },
    weights: state.weights,
    algorithm,
    W: Number(document.getElementById("W").value),
    compare_fastest: document.getElementById("compare").checked,
    verify: document.getElementById("verify").checked,
  };
  try {
    const res = await fetch("/api/route", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (requestId !== latestRequest) return;   // superseded by a newer slider position
    if (!res.ok) throw new Error(data.detail || res.statusText);
    showError("");
    draw(data);
  } catch (err) {
    if (requestId === latestRequest) showError(String(err.message || err));
  }
}

function draw(data) {
  for (const k of Object.keys(layers)) if (layers[k]) { map.removeLayer(layers[k]); layers[k] = null; }
  if (data.fastest) {
    layers.fastest = L.polyline(data.fastest.latlon, { color: color("--fastest"), weight: 5, opacity: 0.8, dashArray: "8 8" })
      .addTo(map).bindTooltip("Fastest route", { sticky: true });
  }
  layers.route = L.polyline(data.route.latlon, { color: color("--route"), weight: 6, opacity: 0.95 })
    .addTo(map).bindTooltip("Your route", { sticky: true });
  markers.origin.setLatLng([data.origin.lat, data.origin.lon]);
  markers.destination.setLatLng([data.destination.lat, data.destination.lon]);

  document.getElementById("result").hidden = false;
  document.getElementById("route-stats").innerHTML = statsRows(data.route.metrics, data.fastest && data.fastest.metrics);
  const s = data.search;
  const rows = [
    ["Algorithm", s.algorithm + (s.guide ? ` (${s.guide} guide)` : "")],
    ["Guarantee", s.W === 1 ? "best route (exact optimum)" : `within ${s.W.toFixed(2)} × the best route`],
    ["States expanded", s.expanded.toLocaleString()],
    ["States generated", s.generated.toLocaleString()],
    ["Latency", `${s.latency_ms.toFixed(0)} ms`],
  ];
  if (data.optimal) {
    rows.push(["Achieved", `${data.optimal.suboptimality.toFixed(4)} × optimal`]);
    rows.push(["Dijkstra would expand", data.optimal.dijkstra_expanded.toLocaleString()]);
  }
  document.getElementById("search-stats").innerHTML = rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("");
}

function updateWVisibility() {
  const a = document.getElementById("algorithm").value;
  document.getElementById("w-row").style.display = a === "focal" || a === "wastar" ? "" : "none";
}
document.getElementById("algorithm").addEventListener("change", () => { updateWVisibility(); scheduleRoute(); });
document.getElementById("W").addEventListener("input", (e) => {
  document.getElementById("w-out").textContent = Number(e.target.value).toFixed(2);
  scheduleRoute();
});
for (const id of ["compare", "verify"]) document.getElementById(id).addEventListener("change", scheduleRoute);

async function init() {
  buildSliders();
  try {
    const meta = await (await fetch("/api/meta")).json();
    const b = meta.bounds;
    map.fitBounds([[b.south, b.west], [b.north, b.east]]);
    buildPresets(meta.presets);
    document.getElementById("algorithm").value = meta.default_algorithm;
    document.getElementById("W").value = meta.default_W;
    document.getElementById("w-out").textContent = Number(meta.default_W).toFixed(2);
    if (meta.guide !== "learned") {
      document.querySelector('#algorithm option[value="focal"]').textContent = `Focal Search, ${meta.guide} guide (within W × best)`;
    }
  } catch (err) {
    showError("Could not reach the routing service: " + err);
  }
  updateWVisibility();
}
init();
