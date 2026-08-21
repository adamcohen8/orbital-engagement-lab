import {
  DUEL_CAMERA_MODES,
  duelPlotFrame,
  referenceRelativePair,
  toggleDuelCameraMode,
} from "./plot-model.js";

const elements = Object.fromEntries([
  "landing-view", "game-view", "connection-pill", "create-tab", "join-tab", "create-form", "join-form",
  "create-name", "join-name", "join-code", "setup-error", "room-code", "round-label", "role-label", "time-label",
  "auto-time", "player-one-card", "player-two-card", "range-label", "speed-label", "dv-label", "phase-overlay",
  "phase-kicker", "phase-title", "phase-detail", "copy-invite", "own-connection-dot", "own-connection-label",
  "opponent-connection-dot", "opponent-connection-label", "ri-canvas", "rc-canvas", "camera-toggle", "toast",
].map((id) => [id, document.getElementById(id)]));

const state = {
  roomCode: "",
  token: "",
  playerId: "",
  socket: null,
  snapshot: null,
  reconnectTimer: null,
  heartbeatTimer: null,
  reconnectAttempts: 0,
  sequence: 0,
  controls: { r: 0, i: 0, c: 0 },
  pressedKeys: new Set(),
  trail: [],
  lastTrailTick: -1,
  lastRoundIndex: null,
  cameraMode: DUEL_CAMERA_MODES.REFERENCE,
};

const keyBindings = {
  KeyW: ["r", 1], KeyS: ["r", -1],
  KeyD: ["i", 1], KeyA: ["i", -1],
  ArrowRight: ["c", 1], ArrowLeft: ["c", -1],
};

restoreSession();
populateInviteFromUrl();
wireSetup();
wireControls();
paintCameraMode();
window.addEventListener("resize", drawPlots);
requestAnimationFrame(renderLoop);

function wireSetup() {
  elements["create-tab"].addEventListener("click", () => selectTab("create"));
  elements["join-tab"].addEventListener("click", () => selectTab("join"));
  elements["create-form"].addEventListener("submit", async (event) => {
    event.preventDefault();
    const rounds = Number(new FormData(event.currentTarget).get("rounds"));
    await submitRoom("/api/rooms", { name: elements["create-name"].value, regulation_rounds: rounds });
  });
  elements["join-form"].addEventListener("submit", async (event) => {
    event.preventDefault();
    const room = cleanRoomCode(elements["join-code"].value);
    await submitRoom(`/api/rooms/${room}/join`, { name: elements["join-name"].value });
  });
  elements["room-code"].addEventListener("click", copyInvite);
  elements["copy-invite"].addEventListener("click", copyInvite);
}

function selectTab(tab) {
  const create = tab === "create";
  elements["create-tab"].classList.toggle("active", create);
  elements["create-tab"].setAttribute("aria-selected", String(create));
  elements["join-tab"].classList.toggle("active", !create);
  elements["join-tab"].setAttribute("aria-selected", String(!create));
  elements["create-form"].classList.toggle("hidden", !create);
  elements["join-form"].classList.toggle("hidden", create);
  elements["setup-error"].textContent = "";
}

async function submitRoom(path, body) {
  elements["setup-error"].textContent = "";
  document.querySelectorAll(".primary-button").forEach((button) => { button.disabled = true; });
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to enter room.");
    enterRoom(result.room_code, result.reconnect_token, result.player.id);
  } catch (error) {
    elements["setup-error"].textContent = error.message;
  } finally {
    document.querySelectorAll(".primary-button").forEach((button) => { button.disabled = false; });
  }
}

function enterRoom(roomCode, token, playerId) {
  document.body.classList.add("duel-active");
  state.roomCode = roomCode;
  state.token = token;
  state.playerId = playerId;
  sessionStorage.setItem("rpo-duel-session", JSON.stringify({ roomCode, token, playerId }));
  const url = new URL(window.location.href);
  url.searchParams.set("room", roomCode);
  history.replaceState(null, "", url);
  elements["landing-view"].classList.add("hidden");
  elements["game-view"].classList.remove("hidden");
  elements["room-code"].textContent = roomCode;
  connectSocket();
}

function connectSocket() {
  if (!state.roomCode || !state.token || state.socket?.readyState === WebSocket.OPEN) return;
  clearTimeout(state.reconnectTimer);
  setConnectionPill("CONNECTING", "warn");
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${scheme}//${location.host}/ws?room=${encodeURIComponent(state.roomCode)}&token=${encodeURIComponent(state.token)}`);
  state.socket = socket;
  socket.addEventListener("open", () => {
    state.reconnectAttempts = 0;
    setConnectionPill("LIVE", "good");
    sendControls(true);
    clearInterval(state.heartbeatTimer);
    state.heartbeatTimer = setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "ping" }));
    }, 15000);
  });
  socket.addEventListener("message", (event) => {
    let message;
    try { message = JSON.parse(event.data); } catch { return; }
    if (message.type === "snapshot") acceptSnapshot(message);
    if (message.type === "error") showToast(message.error || "Server rejected a message.");
  });
  socket.addEventListener("close", (event) => {
    if (state.socket !== socket) return;
    clearInterval(state.heartbeatTimer);
    state.heartbeatTimer = null;
    neutralizeControls(false);
    if (event.code === 4003 || event.code === 1008) {
      setConnectionPill("REJOIN CLOSED", "warn");
      return;
    }
    scheduleReconnect();
  });
  socket.addEventListener("error", () => socket.close());
}

function scheduleReconnect() {
  setConnectionPill("RECONNECTING", "warn");
  const delay = Math.min(750 * 2 ** state.reconnectAttempts, 8000);
  state.reconnectAttempts += 1;
  clearTimeout(state.reconnectTimer);
  state.reconnectTimer = setTimeout(connectSocket, delay);
}

function acceptSnapshot(snapshot) {
  state.snapshot = snapshot;
  state.playerId = snapshot.you?.id || state.playerId;
  const roundIndex = snapshot.series?.round_index ?? null;
  if (roundIndex !== state.lastRoundIndex) {
    state.trail = [];
    state.lastTrailTick = -1;
    state.lastRoundIndex = roundIndex;
  }
  const round = snapshot.series?.round;
  if (round && round.tick !== state.lastTrailTick) {
    state.trail.push({ ...referenceRelativePair(round), tick: round.tick });
    if (state.trail.length > 300) state.trail.shift();
    state.lastTrailTick = round.tick;
  }
  updateUi(snapshot);
}

function updateUi(snapshot) {
  const series = snapshot.series;
  const players = snapshot.players || [];
  elements["room-code"].textContent = snapshot.room_code;
  elements["round-label"].textContent = series ? `${series.round_index} / ${series.regulation_rounds}` : "— / —";
  updatePlayerCard(elements["player-one-card"], players[0], series?.score?.[players[0]?.id] ?? 0);
  updatePlayerCard(elements["player-two-card"], players[1], series?.score?.[players[1]?.id] ?? 0);

  const role = ownRole(series);
  elements["role-label"].textContent = role ? role.toUpperCase() : "WAITING";
  elements["time-label"].textContent = formatSimTime(series?.round?.time_remaining_s ?? 18000);
  elements["range-label"].textContent = series ? `${formatDistance(series.round.range_km)}` : "—";
  elements["speed-label"].textContent = series ? `${(series.round.relative_speed_km_s * 1000).toFixed(2)} m/s` : "—";
  elements["dv-label"].textContent = role ? `${series.round.delta_v_remaining_m_s[role].toFixed(2)} m/s` : "—";
  const auto = snapshot.speed || { speed_multiple: 100, reason: "coasting" };
  elements["auto-time"].querySelector("strong").textContent = `${auto.speed_multiple}x`;
  elements["auto-time"].querySelector("small").textContent = auto.reason.replaceAll("_", " ").toUpperCase();
  elements["auto-time"].classList.toggle("maneuvering", auto.speed_multiple === 10);

  const own = players.find((player) => player.id === state.playerId);
  const opponent = players.find((player) => player.id !== state.playerId);
  updateConnectionLine("own", own, "YOU");
  updateConnectionLine("opponent", opponent, "OPPONENT");
  const controllable = snapshot.phase === "active" && own?.connected && !series?.round?.terminal;
  document.querySelectorAll(".thrust-button").forEach((button) => { button.disabled = !controllable; });
  if (!controllable && Object.values(state.controls).some(Boolean)) neutralizeControls(true);
  updateOverlay(snapshot, own, opponent);
  drawPlots();
}

function updateOverlay(snapshot, own, opponent) {
  const overlay = elements["phase-overlay"];
  const series = snapshot.series;
  overlay.classList.toggle("hidden", snapshot.phase === "active");
  elements["copy-invite"].classList.toggle("hidden", snapshot.phase !== "waiting");
  if (snapshot.phase === "waiting") {
    setOverlay("ROOM READY", "Waiting for opponent", "Share the invite link or room code to begin.");
  } else if (snapshot.phase === "countdown") {
    setOverlay("GET READY", String(Math.max(1, Math.ceil(snapshot.phase_remaining_ms / 1000))), roleBrief(series, state.playerId));
  } else if (snapshot.phase === "round_complete") {
    const winner = snapshot.players.find((player) => player.id === series.round_summaries.at(-1)?.winner_player_id);
    setOverlay("ROUND COMPLETE", `${winner?.name || "Player"} wins`, `${series.round.terminal_reason} Next geometry in ${Math.ceil(snapshot.phase_remaining_ms / 1000)}…`);
  } else if (snapshot.phase === "complete") {
    const winner = snapshot.players.find((player) => player.id === series.match_winner_player_id);
    setOverlay("MATCH COMPLETE", series.match_draw ? "Draw" : `${winner?.name || "Player"} wins`, scoreSentence(snapshot.players, series.score));
  } else if (!own?.connected || !opponent?.connected) {
    setOverlay("CONNECTION", "Reconnecting", "Your spacecraft is coasting while this device reconnects.");
  }
}

function setOverlay(kicker, title, detail) {
  elements["phase-kicker"].textContent = kicker;
  elements["phase-title"].textContent = title;
  elements["phase-detail"].textContent = detail;
}

function wireControls() {
  window.addEventListener("keydown", (event) => {
    if (event.code === "KeyC" && !event.repeat && !isTyping()) {
      event.preventDefault();
      toggleCamera();
      return;
    }
    const binding = keyBindings[event.code];
    if (!binding || event.repeat || isTyping()) return;
    event.preventDefault();
    state.pressedKeys.add(event.code);
    updateAxisFromInputs(binding[0]);
  });
  elements["camera-toggle"].addEventListener("click", toggleCamera);
  window.addEventListener("keyup", (event) => {
    const binding = keyBindings[event.code];
    if (!binding) return;
    event.preventDefault();
    state.pressedKeys.delete(event.code);
    updateAxisFromInputs(binding[0]);
  });
  for (const button of document.querySelectorAll(".thrust-button")) {
    const axis = button.dataset.axis;
    const value = Number(button.dataset.value);
    const press = (event) => {
      event.preventDefault();
      if (button.disabled) return;
      button.setPointerCapture?.(event.pointerId);
      button.dataset.pressed = "true";
      state.controls[axis] = value;
      sendControls();
      paintButtons();
    };
    const release = (event) => {
      event.preventDefault();
      delete button.dataset.pressed;
      const otherPressed = [...document.querySelectorAll(`.thrust-button[data-axis="${axis}"][data-pressed="true"]`)].at(-1);
      state.controls[axis] = otherPressed ? Number(otherPressed.dataset.value) : keyboardAxisValue(axis);
      sendControls();
      paintButtons();
    };
    button.addEventListener("pointerdown", press);
    button.addEventListener("pointerup", release);
    button.addEventListener("pointercancel", release);
    button.addEventListener("lostpointercapture", release);
  }
  window.addEventListener("blur", () => neutralizeControls(true));
  document.addEventListener("visibilitychange", () => { if (document.hidden) neutralizeControls(true); });
}

function toggleCamera() {
  state.cameraMode = toggleDuelCameraMode(state.cameraMode);
  paintCameraMode();
  drawPlots();
}

function paintCameraMode() {
  const pairMode = state.cameraMode === DUEL_CAMERA_MODES.CURRENT_PAIR;
  elements["camera-toggle"].textContent = pairMode ? "C · PAIR VIEW" : "C · REFERENCE VIEW";
  elements["camera-toggle"].setAttribute("aria-pressed", String(pairMode));
  elements["game-view"].dataset.cameraMode = state.cameraMode;
}

function updateAxisFromInputs(axis) {
  const touchPressed = [...document.querySelectorAll(`.thrust-button[data-axis="${axis}"][data-pressed="true"]`)].at(-1);
  state.controls[axis] = touchPressed ? Number(touchPressed.dataset.value) : keyboardAxisValue(axis);
  sendControls();
  paintButtons();
}

function keyboardAxisValue(axis) {
  let value = 0;
  for (const code of state.pressedKeys) {
    const binding = keyBindings[code];
    if (binding?.[0] === axis) value += binding[1];
  }
  return Math.max(-1, Math.min(1, value));
}

function neutralizeControls(send = true) {
  state.pressedKeys.clear();
  document.querySelectorAll(".thrust-button").forEach((button) => { delete button.dataset.pressed; });
  state.controls = { r: 0, i: 0, c: 0 };
  if (send) sendControls();
  paintButtons();
}

function sendControls(force = false) {
  if (state.socket?.readyState !== WebSocket.OPEN) return;
  if (!force && state.snapshot?.phase !== "active") return;
  state.sequence += 1;
  state.socket.send(JSON.stringify({ type: "input", sequence: state.sequence, controls: state.controls }));
}

function paintButtons() {
  for (const button of document.querySelectorAll(".thrust-button")) {
    button.classList.toggle("active", state.controls[button.dataset.axis] === Number(button.dataset.value));
  }
}

function drawPlots() {
  const round = state.snapshot?.series?.round;
  const frame = duelPlotFrame(round, state.trail, state.cameraMode);
  drawPlot(elements["ri-canvas"], frame, "i_km", "r_km", "I", "R");
  drawPlot(elements["rc-canvas"], frame, "c_km", "r_km", "C", "R");
}

function drawPlot(canvas, frame, xKey, yKey, xLabel, yLabel) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width * dpr);
  const height = Math.round(rect.height * dpr);
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  const pairMode = frame.cameraMode === DUEL_CAMERA_MODES.CURRENT_PAIR;
  const centerX = frame.cameraCenter[xKey];
  const centerY = frame.cameraCenter[yKey];
  const points = pairMode
    ? [frame.target, frame.chaser]
    : [
        ...frame.targetTrail,
        ...frame.chaserTrail,
        ...frame.targetProjection,
        ...frame.chaserProjection,
        frame.target,
        frame.chaser,
      ];
  const captureRadiusKm = Number(state.snapshot?.series?.round?.capture_range_km) || .1;
  const extent = Math.max(
    pairMode ? .12 : 1,
    ...points.flatMap((sample) => [
      Math.abs((sample?.[xKey] || 0) - centerX),
      Math.abs((sample?.[yKey] || 0) - centerY),
    ]),
    Math.abs(frame.target[xKey] - centerX) + captureRadiusKm,
    Math.abs(frame.target[yKey] - centerY) + captureRadiusKm,
  );
  const span = niceExtent(extent * 1.22);
  const pad = Math.max(22, Math.min(w, h) * .1);
  const mapX = (value) => pad + (((value - centerX) + span) / (2 * span)) * (w - pad * 2);
  const mapY = (value) => h - pad - (((value - centerY) + span) / (2 * span)) * (h - pad * 2);

  if (!pairMode) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(30,38,50,.95)";
    for (let index = -2; index <= 2; index += 1) {
      const value = index * span / 2;
      ctx.beginPath(); ctx.moveTo(mapX(value), pad); ctx.lineTo(mapX(value), h - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, mapY(value)); ctx.lineTo(w - pad, mapY(value)); ctx.stroke();
    }
    ctx.strokeStyle = "rgba(90,104,124,.95)";
    ctx.beginPath(); ctx.moveTo(mapX(0), pad); ctx.lineTo(mapX(0), h - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, mapY(0)); ctx.lineTo(w - pad, mapY(0)); ctx.stroke();
    ctx.fillStyle = "rgba(170,184,204,.9)";
    ctx.beginPath(); ctx.arc(mapX(0), mapY(0), 3, 0, Math.PI * 2); ctx.fill();
  }

  const captureRadiusX = Math.max(3, Math.abs(mapX(frame.target[xKey] + captureRadiusKm) - mapX(frame.target[xKey])));
  const captureRadiusY = Math.max(3, Math.abs(mapY(frame.target[yKey] + captureRadiusKm) - mapY(frame.target[yKey])));
  ctx.fillStyle = "rgba(245,92,92,.08)";
  ctx.strokeStyle = "rgba(245,92,92,.72)";
  ctx.beginPath();
  ctx.ellipse(mapX(frame.target[xKey]), mapY(frame.target[yKey]), captureRadiusX, captureRadiusY, 0, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();

  if (!pairMode) {
    drawPath(ctx, frame.targetTrail, xKey, yKey, mapX, mapY, "rgba(245,92,92,.7)", 1.5);
    drawPath(ctx, frame.chaserTrail, xKey, yKey, mapX, mapY, "rgba(245,205,92,.72)", 1.5);
    drawPath(ctx, frame.targetProjection, xKey, yKey, mapX, mapY, "rgba(245,92,92,.95)", 2, [8, 6]);
    drawPath(ctx, frame.chaserProjection, xKey, yKey, mapX, mapY, "rgba(96,174,224,.95)", 2, [8, 6]);
  }

  drawSatellite(ctx, frame.target, xKey, yKey, mapX, mapY, "#f55c5c", "T");
  drawSatellite(ctx, frame.chaser, xKey, yKey, mapX, mapY, "#f5cd5c", "C");
  ctx.fillStyle = "rgba(170,184,204,.92)";
  ctx.font = "11px Menlo, Consolas, monospace";
  if (!pairMode) {
    ctx.fillText(`+${xLabel}`, w - pad - 12, mapY(0) - 6);
    ctx.fillText(`+${yLabel}`, mapX(0) + 6, pad + 8);
    drawProjectionLegend(ctx, w, pad);
  }
  ctx.fillText(pairMode ? "PAIR · SATELLITES ONLY" : "REFERENCE ORBIT · HCW COAST", pad, 14);
  ctx.fillText(`${span >= 10 ? span.toFixed(0) : span.toFixed(1)} km`, pad, h - 8);
}

function drawProjectionLegend(ctx, width, pad) {
  if (width < 260) return;
  const startX = Math.max(pad + 150, width - pad - 132);
  const entries = [
    { color: "rgba(245,92,92,.95)", label: "T HCW" },
    { color: "rgba(96,174,224,.95)", label: "C HCW" },
  ];
  entries.forEach((entry, index) => {
    const x = startX + index * 68;
    ctx.save();
    ctx.strokeStyle = entry.color;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(x, 11); ctx.lineTo(x + 17, 11); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = entry.color;
    ctx.font = "9px Menlo, Consolas, monospace";
    ctx.fillText(entry.label, x + 21, 14);
  });
}

function drawPath(ctx, points, xKey, yKey, mapX, mapY, color, width, dash = []) {
  if (points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  points.forEach((sample, index) => {
    const x = mapX(sample[xKey]);
    const y = mapY(sample[yKey]);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawSatellite(ctx, point, xKey, yKey, mapX, mapY, color, label) {
  const x = mapX(point[xKey]);
  const y = mapY(point[yKey]);
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
  ctx.save();
  ctx.globalAlpha = .4;
  ctx.strokeStyle = color;
  ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
  ctx.fillStyle = color;
  ctx.font = "bold 10px Menlo, Consolas, monospace";
  ctx.fillText(label, x + 8, y - 8);
}

function renderLoop() {
  if (!elements["game-view"].classList.contains("hidden")) drawPlots();
  requestAnimationFrame(renderLoop);
}

function updatePlayerCard(card, player, score) {
  card.querySelector(".player-name").textContent = player?.name || "WAITING";
  card.querySelector("b").textContent = score;
  card.querySelector(".status-dot").classList.toggle("connected", Boolean(player?.connected));
}

function updateConnectionLine(prefix, player, label) {
  elements[`${prefix}-connection-dot`].classList.toggle("connected", Boolean(player?.connected));
  elements[`${prefix}-connection-label`].textContent = `${label} · ${player ? (player.connected ? "CONNECTED" : "DISCONNECTED · COASTING") : "WAITING"}`;
}

function ownRole(series) {
  if (!series) return null;
  return series.roles.chaser === state.playerId ? "chaser" : series.roles.target === state.playerId ? "target" : null;
}

function roleBrief(series, playerId) {
  const role = ownRole(series);
  if (!role) return "Stand by.";
  const opponentRole = role === "chaser" ? "target" : "chaser";
  return `You are ${role.toUpperCase()}. Opponent is ${opponentRole.toUpperCase()}.`;
}

function scoreSentence(players, score) {
  return players.map((player) => `${player.name} ${score[player.id] || 0}`).join(" · ");
}

function formatSimTime(seconds) {
  const total = Math.max(0, Math.ceil(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return [hours, minutes, secs].map((value) => String(value).padStart(2, "0")).join(":");
}

function formatDistance(km) {
  if (!Number.isFinite(km)) return "—";
  return km < 1 ? `${(km * 1000).toFixed(0)} m` : `${km.toFixed(km < 10 ? 2 : 1)} km`;
}

function niceExtent(value) {
  const power = 10 ** Math.floor(Math.log10(Math.max(value, .1)));
  const normalized = value / power;
  return (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * power;
}

async function copyInvite() {
  const url = new URL(window.location.origin);
  url.searchParams.set("room", state.roomCode);
  try { await navigator.clipboard.writeText(url.toString()); showToast("Invite link copied"); }
  catch { showToast(`Room code: ${state.roomCode}`); }
}

function setConnectionPill(label, tone) {
  elements["connection-pill"].textContent = label;
  elements["connection-pill"].dataset.tone = tone;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => elements.toast.classList.remove("visible"), 1800);
}

function populateInviteFromUrl() {
  const code = cleanRoomCode(new URLSearchParams(location.search).get("room"));
  if (code && !state.token) {
    elements["join-code"].value = code;
    selectTab("join");
  }
}

function restoreSession() {
  try {
    const saved = JSON.parse(sessionStorage.getItem("rpo-duel-session") || "null");
    const roomFromUrl = cleanRoomCode(new URLSearchParams(location.search).get("room"));
    if (saved?.token && saved?.roomCode && (!roomFromUrl || roomFromUrl === saved.roomCode)) enterRoom(saved.roomCode, saved.token, saved.playerId);
  } catch { sessionStorage.removeItem("rpo-duel-session"); }
}

function cleanRoomCode(value) { return String(value || "").trim().toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8); }
function isTyping() { return ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName); }
