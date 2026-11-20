const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const stages = [
  {
    title: "Start with separate white regions.",
    description: "This is the last observed mask. Every connected white region is a spoke object; colors show the identities it already holds.",
    inspector: "Observed objects",
    next: "Predict the next position →",
  },
  {
    title: "Move every spoke with orbital motion.",
    description: "Each object advances for the measured time gap. Radius stays fixed; longitude changes using orbital speed at that radius. Its track ID travels into the prediction.",
    inspector: "Predicted objects",
    next: "Bring in the new mask →",
  },
  {
    title: "Bring in the new observation.",
    description: "The new white shapes are detections only. They do not have track IDs yet. The tracker must decide which prediction belongs with which detection.",
    inspector: "Two sets to compare",
    next: "Compare all pairs →",
  },
  {
    title: "Score every possible pair.",
    description: "Select any row, or filter to one spoke. Lower costs are better: nearby centers score well, while stronger aligned-shape similarity reduces the cost.",
    inspector: "Candidate pair",
    next: "Start greedy matching →",
  },
  {
    title: "Take the best remaining pair first.",
    description: "Step through the sorted comparisons. Accept a pair only if neither side is already matched; otherwise skip it. That is the greedy part.",
    inspector: "Greedy decision",
    next: "Show propagated tracks →",
  },
  {
    title: "Pass each identity to its match.",
    description: "Each accepted observation inherits its prediction’s track ID and color. Surplus observations receive fresh IDs; unmatched predictions end here.",
    inspector: "Transition complete",
    next: "Explore this transition again ↺",
  },
];

const state = {
  metadata: null,
  data: null,
  stage: 0,
  selectedPair: null,
  focusedObject: null,
  predictionFilter: -1,
  observationFilter: -1,
  cursor: 0,
  decisions: new Map(),
  matchedPredictions: new Map(),
  matchedObservations: new Map(),
  lastDecision: null,
  autoTimer: null,
  loadToken: 0,
};
const transitionCache = new Map();
const pointSetCache = new WeakMap();
const pairKey = (candidate) => `${candidate.prediction}:${candidate.observation}`;
const rgb = (color) => `rgb(${color.join(",")})`;
const pad = (value) => String(value).padStart(2, "0");

function elapsed(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
}

function currentMetadata() {
  return state.metadata.transitions[state.data.index];
}

function stopAutoStep() {
  if (state.autoTimer !== null) window.clearInterval(state.autoTimer);
  state.autoTimer = null;
  $("#auto-compare").textContent = "Auto step";
  $("#auto-compare").setAttribute("aria-pressed", "false");
}

function resetMatching() {
  stopAutoStep();
  state.cursor = 0;
  state.decisions = new Map();
  state.matchedPredictions = new Map();
  state.matchedObservations = new Map();
  state.lastDecision = null;
  state.selectedPair = state.data.candidates[0] ?? null;
}

async function loadTransition(index) {
  const token = ++state.loadToken;
  stopAutoStep();
  $("#demo-content").style.opacity = ".45";
  $("#loading").hidden = false;
  $("#loading").textContent = "Loading this transition…";
  let data = transitionCache.get(index);
  try {
    if (!data) {
      const file = state.metadata.transitions[index].file;
      const response = await fetch(`data/${file}`);
      if (!response.ok) throw new Error(`Could not load ${file} (${response.status})`);
      data = await response.json();
      transitionCache.set(index, data);
    }
    if (token !== state.loadToken) return;
    state.data = data;
    state.selectedPair = data.candidates[0] ?? null;
    state.focusedObject = null;
    state.predictionFilter = -1;
    state.observationFilter = -1;
    resetMatching();
    populateFilters();
    setStage(0);
    $("#demo-content").hidden = false;
    $("#demo-content").style.opacity = "1";
    $("#loading").hidden = true;
  } catch (error) {
    if (token !== state.loadToken) return;
    $("#loading").textContent = `${error.message}. Run the demo through the local HTTP server and regenerate demo/data if needed.`;
  }
}

function setStage(stage) {
  if (!state.data) return;
  stopAutoStep();
  if (stage === 4 && state.stage !== 4) {
    resetMatching();
    state.predictionFilter = -1;
    state.observationFilter = -1;
    $("#prediction-filter").value = "-1";
    $("#observation-filter").value = "-1";
  }
  if (stage === 3 && !state.selectedPair) state.selectedPair = state.data.candidates[0] ?? null;
  state.stage = stage;
  render();
}

function filteredCandidates() {
  return state.data.candidates.filter((candidate) =>
    (state.predictionFilter < 0 || candidate.prediction === state.predictionFilter) &&
    (state.observationFilter < 0 || candidate.observation === state.observationFilter)
  );
}

function decisionReason(candidate) {
  const decision = state.decisions.get(pairKey(candidate));
  if (decision) return decision;
  if (state.stage === 3) return { accepted: null, reason: "A score only—no match has been accepted at this step." };
  if (candidate.rank === state.cursor + 1) return { accepted: null, reason: "This is the next pair waiting for a decision." };
  return { accepted: null, reason: "This pair is still waiting in the cost-sorted queue." };
}

function renderInspector() {
  const inspectorTitle = stages[state.stage].inspector;
  $("#inspector-title").textContent = inspectorTitle;
  $("#object-summary").innerHTML = "";
  $("#pair-card").hidden = state.stage < 3 || state.stage === 5;
  $("#candidate-tools").hidden = state.stage !== 3 && state.stage !== 4;
  $("#greedy-controls").hidden = state.stage !== 4;
  $("#final-summary").hidden = state.stage !== 5;
  $("#decision-badge").hidden = true;

  if (state.stage < 3) {
    const items = state.stage === 0 ? state.data.previous : state.stage === 1 ? state.data.predictions : state.data.observations;
    $("#inspector-description").textContent = state.stage === 0
      ? `${items.length} connected regions, each with an existing track ID.`
      : state.stage === 1
        ? `${items.length} predictions after ${elapsed(state.data.dt_seconds)}. The faded layer shows where they started.`
        : `${state.data.predictions.length} colored predictions → ${state.data.observations.length} white detections. That produces ${state.data.candidates.length} pairs to score.`;
    const fragment = document.createDocumentFragment();
    items.forEach((item, index) => {
      const chip = document.createElement("span");
      chip.className = "object-chip";
      const dot = document.createElement("i");
      dot.style.background = state.stage === 2 ? "#fff" : rgb(item.color);
      chip.append(dot, document.createTextNode(state.stage === 2 ? `O${index + 1}` : `${item.track_id} · P${index + 1}`));
      fragment.append(chip);
    });
    $("#object-summary").append(fragment);
    return;
  }

  if (state.stage === 5) {
    const meta = currentMetadata();
    const predictionCount = state.data.predictions.length;
    $("#inspector-description").textContent = "The right panel renders the observed shapes with their newly assigned identities.";
    $("#final-summary").innerHTML = `
      <div class="result-row"><span>Accepted matches</span><strong>${meta.matches}</strong></div>
      <div class="result-row"><span>New track IDs</span><strong>${meta.new_tracks}</strong></div>
      <div class="result-row"><span>Tracks ending here</span><strong>${meta.ended_tracks}</strong></div>
      <div class="identity-example">Observation → identity<span>${predictionCount} predictions had the chance to pass their IDs forward. Compare the white mask on the left with the colored result on the right.</span></div>`;
    return;
  }

  const candidate = state.selectedPair;
  if (!candidate) return;
  const prediction = state.data.predictions[candidate.prediction];
  const observation = state.data.observations[candidate.observation];
  const decision = decisionReason(candidate);
  $("#inspector-description").textContent = state.stage === 3
    ? `${state.data.candidates.length} total pairs, ordered from lowest to highest cost.`
    : `${state.matchedPredictions.size} accepted · ${[...state.decisions.values()].filter(d => !d.accepted).length} skipped · ${state.data.candidates.length - state.cursor} waiting`;

  $("#pair-rank").textContent = `RANK #${candidate.rank} / ${state.data.candidates.length}`;
  $("#pair-cost").textContent = `cost ${candidate.cost.toFixed(5)}`;
  $("#pair-names").innerHTML = `
    <span><i style="background:${rgb(prediction.color)}"></i>P${candidate.prediction + 1} · ${prediction.track_id}</span>
    <b>→</b>
    <span><i style="background:white"></i>O${candidate.observation + 1}</span>`;
  $("#pair-distance").textContent = candidate.distance.toFixed(5);
  $("#pair-shape").textContent = `${(candidate.shape * 100).toFixed(2)}%`;
  $("#decision-reason").textContent = decision.reason;

  if (state.stage === 4 && state.decisions.has(pairKey(candidate))) {
    $("#decision-badge").hidden = false;
    $("#decision-badge").textContent = decision.accepted ? "ACCEPTED" : "SKIPPED";
    $("#decision-badge").classList.toggle("skipped", !decision.accepted);
  }

  if (state.stage === 4) {
    $("#matching-progress").max = state.data.candidates.length;
    $("#matching-progress").value = state.cursor;
    $("#matching-progress-label").textContent = `${state.cursor} / ${state.data.candidates.length} considered`;
    $("#compare-next").disabled = state.cursor >= state.data.candidates.length;
    $("#compare-next").textContent = state.cursor >= state.data.candidates.length ? "All pairs considered" : `Consider #${state.cursor + 1} →`;
    $("#auto-compare").disabled = state.cursor >= state.data.candidates.length;
    $("#finish-matching").disabled = state.cursor >= state.data.candidates.length;
    $("#match-tally").innerHTML = `<span><b>${state.matchedPredictions.size}</b> accepted</span><span>${state.data.observations.length - state.matchedObservations.size} observations currently unmatched</span>`;
  }
}

function renderCandidateList() {
  if (state.stage !== 3 && state.stage !== 4) return;
  const candidates = filteredCandidates();
  $("#candidate-count").textContent = `${candidates.length} of ${state.data.candidates.length} pairs · sorted by cost`;
  const list = $("#candidate-list");
  list.innerHTML = "";
  const fragment = document.createDocumentFragment();
  if (!candidates.length) {
    const empty = document.createElement("p");
    empty.className = "empty-list";
    empty.textContent = "No pair matches both filters.";
    fragment.append(empty);
  }
  candidates.forEach((candidate) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "candidate-row";
    if (state.selectedPair && pairKey(candidate) === pairKey(state.selectedPair)) row.classList.add("is-selected");
    if (state.stage === 4 && candidate.rank === state.cursor + 1) row.classList.add("is-next");
    const decision = state.decisions.get(pairKey(candidate));
    row.innerHTML = `<span>#${candidate.rank}</span><b>P${candidate.prediction + 1} → O${candidate.observation + 1}</b><small>${candidate.cost.toFixed(4)}</small>`;
    if (decision) {
      const status = document.createElement("em");
      status.textContent = decision.accepted ? "✓ accepted" : `× ${decision.accepted ? "" : "skipped"}`;
      if (!decision.accepted) status.className = "skip";
      row.append(status);
    }
    row.addEventListener("click", () => {
      state.selectedPair = candidate;
      render();
    });
    fragment.append(row);
  });
  list.append(fragment);
}

function drawItems(canvas, items, options = {}) {
  const { width, height } = state.data;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.fillStyle = "#000";
  context.fillRect(0, 0, width, height);

  if (options.ghost) {
    context.fillStyle = "rgba(190,200,180,.13)";
    for (const ghost of options.ghost) {
      const path = new Path2D();
      for (const pixel of ghost.points) path.rect(pixel % width, Math.floor(pixel / width), 1, 1);
      context.fill(path);
    }
  }

  for (const [index, item] of items.entries()) {
    const isSelected = options.selectedIndex === index;
    const assignedIndex = options.matched?.get(index);
    const assignedTrack = assignedIndex === undefined ? null : options.trackSource[assignedIndex];
    const isMatchedObservation = assignedTrack !== null;
    let alpha = 1;
    if (options.selectedIndex !== undefined && options.selectedIndex >= 0 && !isSelected) {
      alpha = isMatchedObservation ? .8 : .16;
    }
    const useColor = options.colorMode === "tracks" || (options.matched && isMatchedObservation);
    context.globalAlpha = alpha;
    context.fillStyle = useColor ? rgb(assignedTrack ? assignedTrack.color : item.color) : "#fff";
    const path = new Path2D();
    for (const pixel of item.points) path.rect(pixel % width, Math.floor(pixel / width), 1.05, 1.05);
    context.fill(path);

    const [x, y] = item.label_xy;
    if (x < 0 || x >= width || y < 0 || y >= height) continue;
    const label = options.labelPrefix
      ? `${options.labelPrefix}${index + 1}`
      : isMatchedObservation || options.colorMode === "tracks"
        ? (assignedTrack ? assignedTrack.track_id : item.track_id)
        : `O${index + 1}`;
    context.font = "bold 15px ui-monospace, monospace";
    const textWidth = context.measureText(label).width;
    const tagX = Math.max(0, Math.min(x - textWidth / 2 - 5, width - textWidth - 10));
    const tagY = Math.max(0, Math.min(y - 10, height - 20));
    context.fillStyle = useColor ? rgb(assignedTrack ? assignedTrack.color : item.color) : "#fff";
    context.fillRect(tagX, tagY, textWidth + 10, 20);
    context.fillStyle = "#070907";
    context.fillText(label, tagX + 5, tagY + 14);

    if (isSelected && item.points.length) {
      let minX = width, minY = height, maxX = 0, maxY = 0;
      for (const pixel of item.points) {
        const px = pixel % width, py = Math.floor(pixel / width);
        minX = Math.min(minX, px); maxX = Math.max(maxX, px);
        minY = Math.min(minY, py); maxY = Math.max(maxY, py);
      }
      context.globalAlpha = 1;
      context.strokeStyle = "#d7fa65";
      context.lineWidth = 3;
      context.setLineDash([8, 5]);
      context.strokeRect(Math.max(1, minX - 5), Math.max(1, minY - 5), Math.min(width - minX + 8, maxX - minX + 12), Math.min(height - minY + 8, maxY - minY + 12));
      context.setLineDash([]);
    }
  }
  context.globalAlpha = 1;
}

function renderCanvases() {
  const data = state.data;
  const stage = state.stage;
  const left = $("#left-canvas");
  const right = $("#right-canvas");
  const twoPanels = stage !== 0;
  $("#right-panel").hidden = !twoPanels;
  $("#canvas-panels").classList.toggle("single-panel", !twoPanels);
  $("#canvas-panels").classList.toggle("stacked-panels", stage >= 2);
  const pair = state.selectedPair;
  const predictionIndex = stage >= 3 && stage <= 4 && pair ? pair.prediction : -1;
  const observationIndex = stage >= 3 && stage <= 4 && pair ? pair.observation : -1;
  $("#visual-note").textContent = stage === 1
    ? "Faint white = previous position; colors = predicted position"
    : stage === 3 || stage === 4
      ? "Click a shape or pair row to inspect it · dashed outline = selected pair"
      : stage === 5
        ? "Observed geometry stays unchanged; only identity is propagated"
        : "Each label is a connected component";

  if (stage === 0) {
    $("#left-label").textContent = "OBSERVED MASK";
    $("#left-meta").textContent = data.previous_name;
    drawItems(left, data.previous, { colorMode: "tracks" });
  } else if (stage === 1) {
    $("#left-label").textContent = "PREVIOUS OBSERVATION";
    $("#left-meta").textContent = data.previous_name;
    $("#right-label").textContent = "PREDICTION";
    $("#right-meta").textContent = `+ ${elapsed(data.dt_seconds)}`;
    drawItems(left, data.previous, { colorMode: "tracks" });
    drawItems(right, data.predictions, { colorMode: "tracks", ghost: data.previous, labelPrefix: "P" });
  } else {
    if (stage === 5) {
      $("#left-label").textContent = "NEW OBSERVED MASK";
      $("#left-meta").textContent = data.current_name;
      $("#right-label").textContent = "ASSIGNED TRACKS";
      $("#right-meta").textContent = data.current_name;
      drawItems(left, data.observations, { colorMode: "raw", labelPrefix: "O" });
      drawItems(right, data.observations, { colorMode: "tracks" });
    } else {
      $("#left-label").textContent = "PREDICTED SPOKES";
      $("#left-meta").textContent = `${data.predictions.length} objects`;
      $("#right-label").textContent = "NEW OBSERVED MASK";
      $("#right-meta").textContent = `${data.observations.length} objects`;
      const matched = stage === 4 ? state.matchedObservations : null;
      drawItems(left, data.predictions, { colorMode: "tracks", selectedIndex: predictionIndex, labelPrefix: "P" });
      drawItems(right, data.observations, {
        colorMode: "raw",
        selectedIndex: observationIndex,
        matched,
        trackSource: data.predictions,
        labelPrefix: "O",
      });
    }
  }
}

function render() {
  if (!state.data) return;
  $$(".stage-button").forEach((button, index) => {
    button.classList.toggle("is-active", index === state.stage);
    button.classList.toggle("is-complete", index < state.stage);
    if (index === state.stage) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  $("#stage-kicker").textContent = `STEP ${state.stage + 1} OF 6`;
  $("#stage-title").textContent = stages[state.stage].title;
  $("#stage-description").textContent = stages[state.stage].description;
  $("#previous-step").disabled = state.stage === 0;
  $("#next-step").textContent = stages[state.stage].next;
  const meta = currentMetadata();
  $("#transition-summary").innerHTML = `
    <div><strong>${meta.previous_frame} → ${meta.current_frame}</strong><small>observation pair</small></div>
    <div><strong>${elapsed(state.data.dt_seconds)}</strong><small>time between frames</small></div>
    <div><strong>${state.data.predictions.length} × ${state.data.observations.length}</strong><small>predictions × detections</small></div>
    <div><strong>${state.data.candidates.length}</strong><small>candidate pairs</small></div>`;
  renderCanvases();
  renderInspector();
  renderCandidateList();
}

function stepOnePair() {
  if (state.cursor >= state.data.candidates.length) {
    stopAutoStep();
    return false;
  }
  const candidate = state.data.candidates[state.cursor];
  state.cursor += 1;
  let accepted = false;
  let reason;
  if (state.matchedPredictions.has(candidate.prediction)) {
    const match = state.matchedPredictions.get(candidate.prediction);
    reason = `Skipped: P${candidate.prediction + 1} was already matched to O${match + 1}.`;
  } else if (state.matchedObservations.has(candidate.observation)) {
    const match = state.matchedObservations.get(candidate.observation);
    reason = `Skipped: O${candidate.observation + 1} was already matched to P${match + 1}.`;
  } else {
    accepted = true;
    state.matchedPredictions.set(candidate.prediction, candidate.observation);
    state.matchedObservations.set(candidate.observation, candidate.prediction);
    const track = state.data.predictions[candidate.prediction].track_id;
    reason = `Accepted as the best remaining pair. O${candidate.observation + 1} inherits ${track}.`;
  }
  const decision = { accepted, reason };
  state.decisions.set(pairKey(candidate), decision);
  state.lastDecision = { candidate, ...decision };
  state.selectedPair = candidate;
  return true;
}

function canvasClick(event, kind) {
  if (!state.data) return;
  const canvas = event.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) * canvas.width / rect.width);
  const y = Math.floor((event.clientY - rect.top) * canvas.height / rect.height);
  const pixel = y * canvas.width + x;
  let items, type;
  if (kind === "left") {
    if (state.stage === 0 || state.stage === 1) { items = state.data.previous; type = "previous"; }
    else { items = state.data.predictions; type = "prediction"; }
  } else {
    if (state.stage === 1) { items = state.data.predictions; type = "prediction"; }
    else { items = state.data.observations; type = "observation"; }
  }
  const hit = items.findIndex(item => {
    if (!pointSetCache.has(item)) pointSetCache.set(item, new Set(item.points));
    return pointSetCache.get(item).has(pixel);
  });
  if (hit < 0) return;
  if (type === "prediction" && (state.stage === 3 || state.stage === 4)) {
    state.predictionFilter = hit;
    $("#prediction-filter").value = String(hit);
    state.observationFilter = -1;
    $("#observation-filter").value = "-1";
    state.selectedPair = state.data.candidates.find(c => c.prediction === hit) ?? null;
    render();
  } else if (type === "observation" && (state.stage === 3 || state.stage === 4)) {
    state.observationFilter = hit;
    $("#observation-filter").value = String(hit);
    state.predictionFilter = -1;
    $("#prediction-filter").value = "-1";
    state.selectedPair = state.data.candidates.find(c => c.observation === hit) ?? null;
    render();
  } else {
    const item = items[hit];
    $("#inspector-title").textContent = type === "observation" ? `Detection O${hit + 1}` : `${item.track_id} · region ${hit + 1}`;
    $("#inspector-description").textContent = `${item.area} mask pixels · centroid ${item.lon.toFixed(2)}° longitude, ${Math.round(item.rad).toLocaleString()} km radius.`;
  }
}

function populateFilters() {
  $("#prediction-filter").innerHTML = `<option value="-1">All (${state.data.predictions.length})</option>` + state.data.predictions.map((item, index) => `<option value="${index}">P${index + 1} · ${item.track_id}</option>`).join("");
  $("#observation-filter").innerHTML = `<option value="-1">All (${state.data.observations.length})</option>` + state.data.observations.map((_, index) => `<option value="${index}">O${index + 1}</option>`).join("");
  $("#prediction-filter").value = String(state.predictionFilter);
  $("#observation-filter").value = String(state.observationFilter);
}

$("#transition-select").addEventListener("change", event => loadTransition(Number(event.target.value)));
$$('.stage-button').forEach(button => button.addEventListener("click", () => setStage(Number(button.dataset.stage))));
$("#previous-step").addEventListener("click", () => setStage(Math.max(0, state.stage - 1)));
$("#next-step").addEventListener("click", () => {
  if (state.stage < 5) setStage(state.stage + 1);
  else { resetMatching(); setStage(0); }
});
$("#prediction-filter").addEventListener("change", event => {
  state.predictionFilter = Number(event.target.value);
  state.observationFilter = -1;
  $("#observation-filter").value = "-1";
  state.selectedPair = filteredCandidates()[0] ?? null;
  render();
});
$("#observation-filter").addEventListener("change", event => {
  state.observationFilter = Number(event.target.value);
  state.predictionFilter = -1;
  $("#prediction-filter").value = "-1";
  state.selectedPair = filteredCandidates()[0] ?? null;
  render();
});
$("#compare-next").addEventListener("click", () => { stopAutoStep(); stepOnePair(); render(); });
$("#auto-compare").addEventListener("click", () => {
  if (state.autoTimer !== null) { stopAutoStep(); return; }
  $("#auto-compare").textContent = "Stop";
  $("#auto-compare").setAttribute("aria-pressed", "true");
  state.autoTimer = window.setInterval(() => {
    if (!stepOnePair()) return;
    render();
    if (state.cursor >= state.data.candidates.length) stopAutoStep();
  }, 350);
});
$("#finish-matching").addEventListener("click", () => {
  stopAutoStep();
  while (stepOnePair()) { /* run the remaining greedy decisions */ }
  setStage(5);
});
$("#left-canvas").addEventListener("click", event => canvasClick(event, "left"));
$("#right-canvas").addEventListener("click", event => canvasClick(event, "right"));

async function initialize() {
  try {
    const response = await fetch("data/index.json");
    if (!response.ok) throw new Error(`Could not load demo data (${response.status})`);
    state.metadata = await response.json();
    const select = $("#transition-select");
    select.innerHTML = "";
    state.metadata.transitions.forEach(transition => {
      const option = document.createElement("option");
      option.value = String(transition.index);
      option.textContent = `Frames ${pad(transition.previous_frame)} → ${pad(transition.current_frame)} · ${transition.predictions}/${transition.observations} objects`;
      select.append(option);
    });
    select.value = "0";
    await loadTransition(0);
  } catch (error) {
    $("#loading").textContent = `${error.message}. Start the demo with: python3 -m http.server 8000`;
  }
}

initialize();
