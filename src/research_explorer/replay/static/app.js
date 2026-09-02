"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  run: null,
  events: [],
  evaluations: [],
  artifacts: [],
  scenes: [],
  sceneIdx: 0,
  playhead: 0,
  playing: false,
  timer: null,
  currentTab: "scene",
};

const ROLE_META = {
  self: { label: "Self-Assessment", color: "#4fc3f7", icon: "S" },
  peer: { label: "Peer Review", color: "#66bb6a", icon: "P" },
  virgin: { label: "Virgin Judge", color: "#ce93d8", icon: "V" },
  structural: { label: "Structural Analyst", color: "#ffb74d", icon: "R" },
  moderator: { label: "Moderator", color: "#ffca28", icon: "M" },
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function htmlEl(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function fmtTs(ts) {
  try {
    return new Date(ts).toLocaleTimeString();
  } catch {
    return ts;
  }
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function renderPayload(payload) {
  const parts = [];
  for (const [k, v] of Object.entries(payload || {})) {
    const val = typeof v === "object" ? JSON.stringify(v) : String(v);
    parts.push(`<b>${esc(k)}</b>=${esc(val)}`);
  }
  return parts.join("  ");
}

function scoreClass(v) {
  if (v >= 0.7) return "good";
  if (v >= 0.4) return "warn";
  return "bad";
}

function deltaClass(v) {
  if (v > 0) return "pos";
  if (v < 0) return "neg";
  return "zero";
}

function buildScenes(evaluations, events) {
  const skipped = (events || []).filter((e) => e.type === "evaluation_skipped");
  const skippedByAgent = new Map();
  for (const ev of skipped) {
    const key = `${ev.payload.agent_id}:${ev.payload.oleada}:${ev.payload.turn}`;
    skippedByAgent.set(key, ev);
  }

  const scenes = [];
  const evalIndices = new Map();
  for (let i = 0; i < evaluations.length; i++) {
    const ev = evaluations[i];
    evalIndices.set(`${ev.agent_id}:${ev.oleada}:${ev.turn}`, i);
  }

  const allItems = [];
  for (const [key, ev] of skippedByAgent) {
    allItems.push({ kind: "skipped", key, oleada: ev.payload.oleada, turn: ev.payload.turn, agent_id: ev.payload.agent_id, reason: ev.payload.reason });
  }
  for (let i = 0; i < evaluations.length; i++) {
    const ev = evaluations[i];
    allItems.push({ kind: "eval", key: `${ev.agent_id}:${ev.oleada}:${ev.turn}`, oleada: ev.oleada, turn: ev.turn, agent_id: ev.agent_id, evalIdx: i });
  }
  allItems.sort((a, b) => a.oleada - b.oleada || a.turn - b.turn || (a.kind === "eval" ? 0 : -1));

  let lastModeratorKey = "";
  for (const item of allItems) {
    const modKey = `${item.oleada}:${item.turn}:${item.agent_id}`;
    if (modKey !== lastModeratorKey) {
      lastModeratorKey = modKey;
      if (item.kind === "skipped") {
        scenes.push({
          role: "moderator",
          speaker: "Moderator",
          target: item.agent_id,
          wave: item.oleada,
          turn: item.turn,
          title: `Oleada ${item.oleada} \u00b7 Turn ${item.turn} \u2014 Skipped`,
          body: `Agent ${item.agent_id} produced no new evidence (${item.reason}). Quality unchanged.`,
          scores: {},
        });
      } else {
        const ev = evaluations[item.evalIdx];
        scenes.push({
          role: "moderator",
          speaker: "Moderator",
          target: ev.agent_id || "",
          wave: ev.oleada,
          turn: ev.turn,
          title: `Oleada ${ev.oleada} \u00b7 Turn ${ev.turn}`,
          body: `Evaluating agent: ${ev.agent_id || "(unknown)"}`,
          scores: { Q: ev.q, deltaQ: ev.delta_q },
          evalIdx: item.evalIdx,
        });
      }
    }

    if (item.kind === "eval") {
      const ev = evaluations[item.evalIdx];
      scenes.push({
        role: "self",
        speaker: ev.agent_id,
        target: ev.agent_id || "",
        wave: ev.oleada,
        turn: ev.turn,
        title: `Self-Assessment \u2014 ${ev.agent_id}`,
        body: ev.self_assessment ? ev.self_assessment.reasoning || "" : "",
        scores: {
          S: ev.self_assessment ? ev.self_assessment.score : 0,
          P: ev.peers ? ev.peers.aggregated_score : 0,
          J: ev.virgin_judge ? ev.virgin_judge.score : 0,
          R: ev.structural ? ev.structural.r : 0,
          Q: ev.q,
          deltaQ: ev.delta_q,
        },
        evalIdx: item.evalIdx,
      });

      if (ev.peers && ev.peers.votes && ev.peers.votes.length) {
        for (const v of ev.peers.votes) {
          scenes.push({
            role: "peer",
            speaker: v.voter_id,
            target: ev.agent_id || "",
            wave: ev.oleada,
            turn: ev.turn,
            title: `Peer Review \u2014 ${v.voter_id}`,
            body: v.reasoning || "",
            scores: {
              S: ev.self_assessment ? ev.self_assessment.score : 0,
              P: ev.peers.aggregated_score,
              J: ev.virgin_judge ? ev.virgin_judge.score : 0,
              R: ev.structural ? ev.structural.r : 0,
              Q: ev.q,
              deltaQ: ev.delta_q,
            },
            peerScore: v.score,
            evalIdx: item.evalIdx,
          });
        }
      }

      if (ev.virgin_judge) {
        scenes.push({
          role: "virgin",
          speaker: "Virgin Judge",
          target: ev.agent_id || "",
          wave: ev.oleada,
          turn: ev.turn,
          title: "Virgin Judge",
          body: [
            ev.virgin_judge.coverage ? `Coverage: ${ev.virgin_judge.coverage}` : "",
            ev.virgin_judge.gaps ? `Gaps: ${ev.virgin_judge.gaps}` : "",
          ].filter(Boolean).join("\n"),
          scores: {
            S: ev.self_assessment ? ev.self_assessment.score : 0,
            P: ev.peers ? ev.peers.aggregated_score : 0,
            J: ev.virgin_judge.score,
            R: ev.structural ? ev.structural.r : 0,
            Q: ev.q,
            deltaQ: ev.delta_q,
          },
          evalIdx: item.evalIdx,
        });
      }

      if (ev.structural) {
        const s = ev.structural;
        scenes.push({
          role: "structural",
          speaker: "Structural Analyst",
          target: ev.agent_id || "",
          wave: ev.oleada,
          turn: ev.turn,
          title: "Structural Analyst",
          body: [
            `Coverage: ${Number(s.coverage).toFixed(3)}`,
            `Diversity: ${Number(s.diversity).toFixed(3)}`,
            `Depth: ${Number(s.depth).toFixed(3)}`,
            `Coherence: ${Number(s.coherence).toFixed(3)}`,
          ].join("\n"),
          scores: {
            S: ev.self_assessment ? ev.self_assessment.score : 0,
            P: ev.peers ? ev.peers.aggregated_score : 0,
            J: ev.virgin_judge ? ev.virgin_judge.score : 0,
            R: s.r,
            Q: ev.q,
            deltaQ: ev.delta_q,
          },
          evalIdx: item.evalIdx,
        });
      }
    }
  }
  return scenes;
}

function collectSeats(scenes, sceneIdx) {
  const current = scenes[sceneIdx];
  if (!current) return [];
  const seen = new Map();
  for (let i = 0; i <= sceneIdx; i++) {
    const s = scenes[i];
    if (s.role === "moderator") {
      if (!seen.has("moderator")) seen.set("moderator", { role: "moderator", speaker: s.speaker });
    } else {
      const key = s.speaker;
      if (!seen.has(key)) seen.set(key, { role: s.role, speaker: s.speaker });
    }
  }
  const seats = Array.from(seen.values());
  const moderatorSeat = seats.find((s) => s.role === "moderator");
  const others = seats.filter((s) => s.role !== "moderator");
  const result = [...others];
  if (moderatorSeat) result.push(moderatorSeat);
  return result;
}

function renderSceneRoom() {
  const container = $("#scene-container");
  const emptyEl = $("#scene-empty");
  if (!state.scenes.length) {
    container.classList.add("hidden");
    emptyEl.classList.remove("hidden");
    return;
  }
  emptyEl.classList.add("hidden");
  container.classList.remove("hidden");

  const scene = state.scenes[state.sceneIdx];
  if (!scene) return;

  const info = $("#scene-info");
  info.textContent = "";
  const infoItems = [
    { label: "Scene", value: `${state.sceneIdx + 1} / ${state.scenes.length}` },
    { label: "Target", value: scene.target },
    { label: "Wave", value: scene.wave != null ? scene.wave : "\u2014" },
    { label: "Turn", value: scene.turn != null ? scene.turn : "\u2014" },
  ];
  for (const item of infoItems) {
    const span = el("span");
    span.appendChild(el("span", "si-label", item.label + ": "));
    span.appendChild(el("span", "si-value", item.value));
    info.appendChild(span);
  }
  if (scene.scores.Q != null) {
    const qSpan = el("span");
    qSpan.appendChild(el("span", "si-label", "Q: "));
    qSpan.appendChild(el("span", "si-q", Number(scene.scores.Q).toFixed(4)));
    info.appendChild(qSpan);
  }
  if (scene.scores.deltaQ != null) {
    const dSpan = el("span");
    dSpan.appendChild(el("span", "si-label", "\u0394Q: "));
    dSpan.appendChild(el("span", `si-delta ${deltaClass(scene.scores.deltaQ)}`, Number(scene.scores.deltaQ).toFixed(4)));
    info.appendChild(dSpan);
  }

  const table = $("#scene-table");
  table.textContent = "";
  const seats = collectSeats(state.scenes, state.sceneIdx);
  for (let i = 0; i < seats.length; i++) {
    const s = seats[i];
    const meta = ROLE_META[s.role] || ROLE_META.moderator;
    const isActive = s.speaker === scene.speaker && scene.role !== "moderator";
    const isSpeaker = s.speaker === scene.speaker;
    const classes = ["seat"];
    if (s.role === "moderator") classes.push("moderator");
    if (isSpeaker && s.role !== "moderator") classes.push("speaker");
    else if (isActive) classes.push("active");
    else classes.push("idle");

    const seat = el("div", classes.join(" "));
    const icon = el("span", "seat-icon", meta.icon);
    seat.appendChild(icon);
    seat.appendChild(el("span", "seat-name", s.speaker));
    seat.appendChild(el("span", "seat-role", meta.label));
    table.appendChild(seat);
    if (i === seats.length - 2 && seats.length > 2) {
      table.appendChild(el("hr", "seat-divider"));
    }
  }

  const speechRole = $("#speech-role");
  const speechTitle = $("#speech-title");
  const speechBody = $("#speech-body");
  const speechScores = $("#speech-scores");

  const meta = ROLE_META[scene.role] || ROLE_META.moderator;
  speechRole.textContent = meta.label;
  speechRole.style.color = meta.color;
  speechTitle.textContent = scene.title;
  speechBody.textContent = scene.body || "(no content)";

  speechScores.textContent = "";
  const scoreKeys = [
    { key: "S", label: "Self (S)" },
    { key: "P", label: "Peers (P)" },
    { key: "J", label: "Virgin (J)" },
    { key: "R", label: "Structural (R)" },
  ];
  for (const sk of scoreKeys) {
    const val = scene.scores[sk.key];
    const box = el("div", "speech-score");
    box.appendChild(el("div", "ss-label", sk.label));
    const vEl = el("div", `ss-value ${val != null ? scoreClass(val) : ""}`, val != null ? Number(val).toFixed(3) : "\u2014");
    box.appendChild(vEl);
    speechScores.appendChild(box);
  }
}

async function loadRuns() {
  const runs = await fetchJSON("/api/runs");
  const list = $("#run-list");
  list.textContent = "";
  if (!runs.length) {
    list.appendChild(el("p", "empty", "No runs recorded yet."));
    return;
  }
  for (const r of runs) {
    const card = el("div", "run-card");
    card.appendChild(el("div", "title", `[${r.status}] ${r.seed_paper_id}`));
    card.appendChild(el("div", "subtitle", r.seed_query || "(no query)"));
    const metrics = [];
    if (r.best_quality != null) metrics.push(`Q=${Number(r.best_quality).toFixed(3)}`);
    metrics.push(`${r.event_count} events`);
    metrics.push(`${r.evaluation_count} evaluations`);
    card.appendChild(el("div", "metrics", metrics.join(" \u00a0\u00b7\u00a0 ")));
    card.appendChild(el("div", `status-${r.status.length ? r.status : "unknown"}`, fmtTs(r.started_at)));
    card.addEventListener("click", () => openRoom(r.run_id));
    list.appendChild(card);
  }
}

async function openRoom(runId) {
  const run = await fetchJSON(`/api/runs/${runId}`);
  const [events, evaluations, artifacts] = await Promise.all([
    fetchJSON(`/api/runs/${runId}/events`),
    fetchJSON(`/api/runs/${runId}/evaluations`),
    fetchJSON(`/api/runs/${runId}/artifacts`),
  ]);
  state.run = run;
  state.events = events;
  state.evaluations = evaluations;
  state.artifacts = artifacts;
  state.scenes = buildScenes(evaluations, events);
  state.sceneIdx = 0;
  state.playhead = 0;
  state.playing = false;
  stopPlay();
  $("#selector-view").classList.add("hidden");
  $("#room-view").classList.remove("hidden");

  const sceneMax = Math.max(0, state.scenes.length - 1);
  const eventMax = Math.max(0, events.length - 1);
  const max = Math.max(sceneMax, eventMax);
  $("#seek-slider").max = max;
  $("#seek-slider").value = 0;
  funcRenderMeta(run);
  renderSceneRoom();
  renderTimeline();
  renderNarratives();
  renderEvaluations();
  updateSeekLabel();
}

function funcRenderMeta(run) {
  const meta = $("#run-meta");
  meta.textContent = "";
  meta.appendChild(el("div", "", `Seed: ${run.seed_paper_id}`));
  meta.appendChild(el("div", "", `Query: ${run.seed_query || "(none)"}`));
  const line = [];
  if (run.best_quality != null) line.push(`best Q = ${Number(run.best_quality).toFixed(4)}`);
  line.push(`status: ${run.status}`);
  line.push(`started: ${fmtTs(run.started_at)}`);
  meta.appendChild(el("div", "q", line.join("   \u00b7   ")));
  $("#run-badge").textContent = run.run_id;
  $("#run-badge").classList.remove("hidden");
}

function renderEvent(ev, isCurrent) {
  const li = el("li", `event-item${isCurrent ? " current" : ""}`);
  const head = el("div", "ev-head");
  head.appendChild(el("span", "ev-seq", `#${ev.seq}`));
  head.appendChild(el("span", "ev-type", ev.type));
  head.appendChild(el("span", "ev-ts", fmtTs(ev.ts)));
  li.appendChild(head);
  li.appendChild(el("div", "ev-payload", renderPayload(ev.payload)));
  return li;
}

function renderTimeline() {
  const list = $("#event-list");
  list.textContent = "";
  const visible = state.events.slice(0, state.playhead + 1);
  if (!visible.length) {
    list.appendChild(el("li", "empty", "No events yet."));
    return;
  }
  visible.forEach((ev, i) => list.appendChild(renderEvent(ev, i === visible.length - 1)));
  list.scrollTop = list.scrollHeight;
}

function updateSeekLabel() {
  const max = Math.max(state.scenes.length - 1, state.events.length - 1, 0);
  $("#seek-label").textContent = `${state.playhead} / ${max}`;
}

function syncPlayheadFromScene() {
  if (state.currentTab === "scene") {
    state.playhead = state.sceneIdx;
    $("#seek-slider").value = state.playhead;
    updateSeekLabel();
  } else {
    state.sceneIdx = Math.min(state.playhead, state.scenes.length - 1);
  }
}

function togglePlay() {
  state.playing = !state.playing;
  $("#play-btn").innerHTML = state.playing ? "\u23f8 Pause" : "\u25b6 Play";
  if (state.playing) startPlay();
  else stopPlay();
}

function startPlay() {
  stopPlay();
  const speed = parseFloat($("#speed-select").value) || 1;
  const sceneMax = state.scenes.length - 1;
  const eventMax = state.events.length - 1;
  state.timer = setInterval(() => {
    const atEnd = state.currentTab === "scene"
      ? state.sceneIdx >= sceneMax
      : state.playhead >= eventMax;
    if (atEnd) {
      state.playing = false;
      $("#play-btn").innerHTML = "\u25b6 Play";
      stopPlay();
      return;
    }
    advanceStep();
  }, 1000 / speed);
}

function advanceStep() {
  if (state.currentTab === "scene") {
    if (state.sceneIdx < state.scenes.length - 1) {
      state.sceneIdx++;
      state.playhead = state.sceneIdx;
      renderSceneRoom();
    }
  } else {
    if (state.playhead < state.events.length - 1) {
      state.playhead++;
      renderTimeline();
    }
  }
  $("#seek-slider").value = state.playhead;
  updateSeekLabel();
}

function retreatStep() {
  if (state.currentTab === "scene") {
    if (state.sceneIdx > 0) {
      state.sceneIdx--;
      state.playhead = state.sceneIdx;
      renderSceneRoom();
    }
  } else {
    if (state.playhead > 0) {
      state.playhead--;
      renderTimeline();
    }
  }
  $("#seek-slider").value = state.playhead;
  updateSeekLabel();
}

function stopPlay() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

function renderNarratives() {
  const container = $("#narrative-list");
  container.textContent = "";
  const nars = state.artifacts.filter((a) => a.kind === "narrative");
  const errs = state.artifacts.filter((a) => a.kind === "error");
  if (!nars.length && !errs.length) {
    container.appendChild(el("p", "empty", "No narrative snapshot recorded for this run."));
    return;
  }
  for (const a of nars) {
    const card = el("div", "nar-card");
    card.appendChild(el("h3", "", `${a.name}  \u00b7  ${fmtTs(a.created_at)}`));
    const body = document.createElement("pre");
    body.textContent = "(empty narrative)";
    fetchJSON(`/api/artifacts/${a.artifact_id}`)
      .then((full) => {
        body.textContent = full.content || "(empty narrative)";
      })
      .catch(() => {
        body.textContent = "(failed to load narrative)";
      });
    card.appendChild(body);
    container.appendChild(card);
  }
  for (const a of errs) {
    const card = el("div", "nar-card");
    card.appendChild(el("h3", "", `${a.name}  (run error)`));
    fetchJSON(`/api/artifacts/${a.artifact_id}`).then((full) => {
      card.appendChild(el("div", "ev-payload", full.content || ""));
    });
    container.appendChild(card);
  }
}

function evalComponent(label, value) {
  const box = el("div", "eval-component");
  box.appendChild(el("div", "label", label));
  box.appendChild(el("div", "value", Number(value).toFixed(3)));
  return box;
}

function renderEvaluations() {
  const container = $("#evaluation-list");
  container.textContent = "";
  if (!state.evaluations.length) {
    container.appendChild(el("p", "empty", "No evaluation records for this run."));
    return;
  }
  for (const ev of state.evaluations) {
    const card = el("div", "eval-card");
    const head = el("div", "eval-head");
    head.appendChild(el("span", "agent", ev.agent_id));
    head.appendChild(el("span", "qscore", `Q=${Number(ev.q).toFixed(3)}`));
    head.appendChild(el("span", "muted", `oleada ${ev.oleada}  \u00b7  turn ${ev.turn}  \u00b7  \u0394Q ${Number(ev.delta_q).toFixed(3)}  \u00b7  ${fmtTs(ev.timestamp)}`));
    card.appendChild(head);

    const comps = el("div", "eval-components");
    comps.appendChild(evalComponent("S self", ev.self_assessment.score));
    comps.appendChild(evalComponent("P peers", ev.peers.aggregated_score));
    comps.appendChild(evalComponent("J virgin", ev.virgin_judge.score));
    comps.appendChild(evalComponent("R structural", ev.structural.r));
    card.appendChild(comps);

    card.appendChild(htmlEl("div", "eval-reasoning", `Self: <span class="reason">${esc(ev.self_assessment.reasoning)}</span>`));

    const peerBox = htmlEl("div", "eval-reasoning");
    peerBox.appendChild(
      el("span", "", `Peers (${ev.peers.num_votes}, ${esc(ev.peers.aggregation_method)}): `)
    );
    card.appendChild(peerBox);
    if (ev.peers.votes && ev.peers.votes.length) {
      const voteList = el("div", "vote-list");
      for (const v of ev.peers.votes) {
        voteList.appendChild(
          htmlEl("div", "vote", `<span class="voter">${esc(v.voter_id)}</span> score=${Number(v.score).toFixed(3)} \u2014 <span class="reason">${esc(v.reasoning)}</span>`)
        );
      }
      card.appendChild(voteList);
    }

    card.appendChild(
      htmlEl("div", "eval-reasoning", `Virgin judge: <span class="reason">${esc(ev.virgin_judge.coverage)}</span>`)
    );
    if (ev.virgin_judge.gaps) {
      card.appendChild(
        htmlEl("div", "eval-reasoning", `Gaps: <span class="reason">${esc(ev.virgin_judge.gaps)}</span>`)
      );
    }
    const s = ev.structural;
    card.appendChild(
      htmlEl(
        "div",
        "eval-reasoning",
        `Structural: cover=${Number(s.coverage).toFixed(3)} diversity=${Number(s.diversity).toFixed(3)} depth=${Number(s.depth).toFixed(3)} coherence=${Number(s.coherence).toFixed(3)}`
      )
    );
    if (ev.new_papers && ev.new_papers.length) {
      card.appendChild(
        htmlEl("div", "eval-reasoning", `New papers this turn: ${ev.new_papers.map(esc).join(", ")}`)
      );
    }
    container.appendChild(card);
  }
}

$("#back-btn").addEventListener("click", () => {
  stopPlay();
  $("#room-view").classList.add("hidden");
  $("#selector-view").classList.remove("hidden");
  $("#run-badge").classList.add("hidden");
});

async function initApp() {
  await loadRuns();
  try {
    const cfg = await fetchJSON("/api/config");
    if (cfg.default_run_id) {
      await openRoom(cfg.default_run_id);
    }
  } catch {
  }
}

initApp();

$("#play-btn").addEventListener("click", togglePlay);
$("#prev-btn").addEventListener("click", () => retreatStep());
$("#next-btn").addEventListener("click", () => advanceStep());
$("#seek-slider").addEventListener("input", (e) => {
  const val = parseInt(e.target.value, 10) || 0;
  state.playhead = val;
  if (state.currentTab === "scene") {
    state.sceneIdx = Math.min(val, state.scenes.length - 1);
    renderSceneRoom();
  } else {
    renderTimeline();
  }
  updateSeekLabel();
});
$("#speed-select").addEventListener("change", () => {
  if (state.playing) startPlay();
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    state.currentTab = tab.dataset.tab;
    for (const p of document.querySelectorAll(".tab-panel")) p.classList.add("hidden");
    $(`#tab-${state.currentTab}`).classList.remove("hidden");
    syncPlayheadFromScene();
    updateSeekLabel();
    if (state.currentTab === "timeline") renderTimeline();
    if (state.currentTab === "scene") renderSceneRoom();
  });
});
