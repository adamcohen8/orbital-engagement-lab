import {
  DUEL_CAMERA_MODES,
  DUEL_VISUAL_TIMING,
  captureRingStyle,
  duelPlotFrame,
  duelPlotSpan,
  interpolateDuelRound,
  referenceRelativePair,
  toggleDuelCameraMode,
} from "./plot-model.js";
import {
  frameConventionDisplayAxisSign,
  frameConventionDisplayValue,
  frameConventionFromSearch,
  frameConventionLabel,
  nextFrameConvention,
  urlWithFrameConvention,
} from "./frame-convention.js";

const previewMode = new URLSearchParams(location.search).get("preview");
if (previewMode === "mobile-landscape" || previewMode === "mobile-portrait") {
  document.body.classList.add(`${previewMode}-preview`);
  syncMobilePreviewScale();
  window.addEventListener("resize", syncMobilePreviewScale);
}

function syncMobilePreviewScale() {
  const portrait = previewMode === "mobile-portrait";
  const previewWidth = portrait ? 390 : 667;
  const previewHeight = portrait ? 844 : 375;
  const scale = Math.min(1, window.innerWidth / previewWidth, window.innerHeight / previewHeight);
  document.documentElement.style.setProperty("--mobile-preview-scale", scale.toFixed(4));
}

const DUEL_MUSIC_SOURCE = "/assets/39_perigee_afterburner_demo.wav";
const AUTO_TIME_STATUS_LABELS = Object.freeze({
  coasting: "COAST",
  neutral_cooldown: "COOL",
  maneuvering: "BURN",
});
const HOSTED_LEVEL_SELECTOR_URL = document.querySelector('meta[name="oel-level-selector-url"]')?.content.trim()
  || "https://orbital-engagement-lab.vercel.app/";
const duelMusic = new Audio(DUEL_MUSIC_SOURCE);
duelMusic.loop = true;
duelMusic.preload = "none";
duelMusic.volume = 0.65;

const elements = Object.fromEntries([
  "landing-view", "game-view", "create-tab", "join-tab", "computer-tab", "create-form", "join-form",
  "computer-form", "create-name", "join-name", "computer-name", "join-code", "setup-error", "room-code", "round-label", "role-label", "time-label",
  "auto-time", "player-one-card", "player-two-card", "range-label", "speed-label", "dv-label", "phase-overlay",
  "phase-kicker", "phase-title", "phase-detail", "copy-invite", "match-actions", "play-again", "return-lobby",
  "ri-canvas", "rc-canvas", "camera-toggle", "music-toggle", "level-selector-link", "frame-convention-button", "frame-convention-label", "command-line", "toast",
].map((id) => [id, document.getElementById(id)]));

const state = {
  roomCode: "",
  token: "",
  playerId: "",
  socket: null,
  snapshot: null,
  reconnectTimer: null,
  heartbeatTimer: null,
  lastPongAt: 0,
  reconnectAttempts: 0,
  sequence: 0,
  controls: { r: 0, i: 0, c: 0 },
  pressedKeys: new Set(),
  trail: [],
  renderFrames: [],
  plotViewports: new Map(),
  lastTrailTick: -1,
  lastRoundIndex: null,
  cameraMode: DUEL_CAMERA_MODES.REFERENCE,
  frameConvention: frameConventionFromSearch(location.search),
  musicEnabled: true,
  musicRequested: false,
  musicAvailable: true,
};
const touchPointers = new Map();

const keyBindings = {
  KeyW: ["r", 1], KeyS: ["r", -1],
  KeyD: ["i", 1], KeyA: ["i", -1],
  ArrowRight: ["c", 1], ArrowLeft: ["c", -1],
};

restoreSession();
populateInviteFromUrl();
paintFrameConvention();
configureLevelSelectorLink();
wireSetup();
wireControls();
paintCameraMode();
paintMusicMode();
duelMusic.addEventListener("error", () => {
  state.musicAvailable = false;
  paintMusicMode();
});
window.addEventListener("resize", () => drawPlots());
requestAnimationFrame(renderLoop);

function configureLevelSelectorLink() {
  const localHost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
  const destination = localHost ? "/trainer/" : HOSTED_LEVEL_SELECTOR_URL;
  elements["level-selector-link"].href = urlWithFrameConvention(
    destination,
    state.frameConvention,
    window.location.href,
  ).href;
}

function paintFrameConvention() {
  const label = frameConventionLabel(state.frameConvention);
  const inTrackDisplaySign = frameConventionDisplayAxisSign(state.frameConvention, "i");
  document.body.dataset.frameConvention = state.frameConvention;
  elements["frame-convention-button"].textContent = `Frame: ${label}`;
  elements["frame-convention-button"].setAttribute(
    "aria-label",
    `Frame convention: ${label}. Activate to switch.`,
  );
  elements["frame-convention-label"].textContent = `Frame: ${label}`;
  elements["command-line"].textContent = inTrackDisplaySign > 0
    ? "W/S Radial  A −I / D +I  Left/Right Cross-Track"
    : "W/S Radial  A +I / D −I  Left/Right Cross-Track";
  for (const button of document.querySelectorAll('.thrust-button[data-axis="i"]')) {
    const spatialSign = button.dataset.spatialDirection === "right" ? 1 : -1;
    const canonicalValue = spatialSign * inTrackDisplaySign;
    button.dataset.value = String(canonicalValue);
    button.querySelector("[data-thrust-label]").textContent = canonicalValue > 0 ? "+I" : "−I";
  }
}

function toggleFrameConvention() {
  neutralizeControls(false);
  state.frameConvention = nextFrameConvention(state.frameConvention);
  const currentUrl = urlWithFrameConvention(window.location.href, state.frameConvention);
  window.history.replaceState(window.history.state, "", currentUrl.href);
  paintFrameConvention();
  configureLevelSelectorLink();
  showToast(`Frame convention: ${frameConventionLabel(state.frameConvention)}`);
}

function wireSetup() {
  elements["create-tab"].addEventListener("click", () => selectTab("create"));
  elements["join-tab"].addEventListener("click", () => selectTab("join"));
  elements["computer-tab"].addEventListener("click", () => selectTab("computer"));
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
  elements["computer-form"].addEventListener("submit", async (event) => {
    event.preventDefault();
    const rounds = Number(new FormData(event.currentTarget).get("rounds"));
    await submitRoom("/api/rooms", {
      name: elements["computer-name"].value,
      regulation_rounds: rounds,
      opponent: "computer",
    });
  });
  elements["room-code"].addEventListener("click", copyInvite);
  elements["copy-invite"].addEventListener("click", copyInvite);
  elements["play-again"].addEventListener("click", requestRematch);
  elements["return-lobby"].addEventListener("click", returnToLobby);
  elements["frame-convention-button"].addEventListener("click", toggleFrameConvention);
}

function selectTab(tab) {
  const create = tab === "create";
  const join = tab === "join";
  const computer = tab === "computer";
  elements["create-tab"].classList.toggle("active", create);
  elements["create-tab"].setAttribute("aria-selected", String(create));
  elements["join-tab"].classList.toggle("active", join);
  elements["join-tab"].setAttribute("aria-selected", String(join));
  elements["computer-tab"].classList.toggle("active", computer);
  elements["computer-tab"].setAttribute("aria-selected", String(computer));
  elements["create-form"].classList.toggle("hidden", !create);
  elements["join-form"].classList.toggle("hidden", !join);
  elements["computer-form"].classList.toggle("hidden", !computer);
  elements["setup-error"].textContent = "";
}

async function submitRoom(path, body) {
  requestMusicStart();
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
  requestMusicStart();
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
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(
    `${scheme}//${location.host}/ws?room=${encodeURIComponent(state.roomCode)}`,
    ["oel-rpo-duel-v1", `oel-token.${state.token}`],
  );
  state.socket = socket;
  socket.addEventListener("open", () => {
    state.reconnectAttempts = 0;
    state.lastPongAt = Date.now();
    sendControls(true);
    clearInterval(state.heartbeatTimer);
    state.heartbeatTimer = setInterval(() => {
      if (socket.readyState !== WebSocket.OPEN) return;
      if (Date.now() - state.lastPongAt > 35000) {
        socket.close(4002, "Heartbeat timeout.");
        return;
      }
      socket.send(JSON.stringify({ type: "ping" }));
    }, 15000);
  });
  socket.addEventListener("message", (event) => {
    let message;
    try { message = JSON.parse(event.data); } catch { return; }
    if (message.type === "snapshot") acceptSnapshot(message);
    if (message.type === "pong") state.lastPongAt = Date.now();
    if (message.type === "error") showToast(message.error || "Server rejected a message.");
  });
  socket.addEventListener("close", (event) => {
    if (state.socket !== socket) return;
    clearInterval(state.heartbeatTimer);
    state.heartbeatTimer = null;
    neutralizeControls(false);
    if ([4002, 4003, 4008, 1008].includes(event.code)) {
      returnToLanding("The saved Duel session is no longer joinable.");
      return;
    }
    if (state.reconnectAttempts >= 5) {
      returnToLanding("Unable to restore the saved Duel session.");
      return;
    }
    scheduleReconnect();
  });
  socket.addEventListener("error", () => socket.close());
}

function scheduleReconnect() {
  const delay = Math.min(750 * 2 ** state.reconnectAttempts, 8000);
  state.reconnectAttempts += 1;
  clearTimeout(state.reconnectTimer);
  state.reconnectTimer = setTimeout(connectSocket, delay);
}

function returnToLanding(message = "") {
  const socket = state.socket;
  state.socket = null;
  clearTimeout(state.reconnectTimer);
  clearInterval(state.heartbeatTimer);
  state.reconnectTimer = null;
  state.heartbeatTimer = null;
  socket?.close(1000, "Client left room.");
  neutralizeControls(false);
  sessionStorage.removeItem("rpo-duel-session");
  state.roomCode = "";
  state.token = "";
  state.playerId = "";
  state.snapshot = null;
  state.sequence = 0;
  state.lastPongAt = 0;
  state.reconnectAttempts = 0;
  state.trail = [];
  state.renderFrames = [];
  state.plotViewports.clear();
  state.lastTrailTick = -1;
  state.lastRoundIndex = null;
  state.cameraMode = DUEL_CAMERA_MODES.REFERENCE;
  document.body.classList.remove("duel-active");
  elements["game-view"].classList.add("hidden");
  elements["landing-view"].classList.remove("hidden");
  elements["play-again"].disabled = false;
  elements["play-again"].textContent = "Play Again";
  paintCameraMode();
  selectTab("create");
  const url = new URL(window.location.href);
  url.searchParams.delete("room");
  history.replaceState(null, "", url);
  elements["setup-error"].textContent = message;
}

function acceptSnapshot(snapshot) {
  recordRenderSnapshot(snapshot, performance.now());
  state.snapshot = snapshot;
  state.playerId = snapshot.you?.id || state.playerId;
  const roundIndex = snapshot.series?.round_index ?? null;
  if (roundIndex !== state.lastRoundIndex) {
    state.trail = [];
    state.plotViewports.clear();
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
  elements["role-label"].dataset.role = role || "waiting";
  elements["time-label"].textContent = formatSimTime(series?.round?.time_remaining_s ?? 18000);
  elements["range-label"].textContent = series ? `${formatDistance(series.round.range_km)}` : "—";
  elements["speed-label"].textContent = series ? `${(series.round.relative_speed_km_s * 1000).toFixed(2)} m/s` : "—";
  elements["dv-label"].textContent = role ? `${series.round.delta_v_remaining_m_s[role].toFixed(2)} m/s` : "—";
  const auto = snapshot.speed || { speed_multiple: 100, reason: "coasting" };
  elements["auto-time"].querySelector("strong").textContent = `${auto.speed_multiple}x`;
  elements["auto-time"].querySelector("small").textContent = AUTO_TIME_STATUS_LABELS[auto.reason]
    || auto.reason.replaceAll("_", " ").toUpperCase();
  elements["auto-time"].classList.toggle("maneuvering", auto.speed_multiple === 10);

  const own = players.find((player) => player.id === state.playerId);
  const opponent = players.find((player) => player.id !== state.playerId);
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
  elements["copy-invite"].classList.toggle("hidden", snapshot.phase !== "waiting" || snapshot.match_mode === "computer");
  elements["match-actions"].classList.toggle("hidden", snapshot.phase !== "complete");
  if (snapshot.phase === "waiting") {
    setOverlay("ROOM READY", "Waiting for opponent", "Share the invite link or room code to begin.");
  } else if (snapshot.phase === "countdown") {
    setOverlay("GET READY", String(Math.max(1, Math.ceil(snapshot.phase_remaining_ms / 1000))), roleBrief(series, state.playerId));
  } else if (snapshot.phase === "round_complete") {
    const winner = snapshot.players.find((player) => player.id === series.round_summaries.at(-1)?.winner_player_id);
    setOverlay("ROUND COMPLETE", `${winner?.name || "Player"} wins`, `${series.round.terminal_reason} Next geometry in ${Math.ceil(snapshot.phase_remaining_ms / 1000)}…`);
  } else if (snapshot.phase === "complete") {
    const winner = snapshot.players.find((player) => player.id === series.match_winner_player_id);
    const waiting = snapshot.rematch?.your_ready && snapshot.match_mode !== "computer";
    const score = scoreSentence(snapshot.players, series.score);
    setOverlay(
      waiting ? "REMATCH READY" : "MATCH COMPLETE",
      series.match_draw ? "Draw" : `${winner?.name || "Player"} wins`,
      waiting ? `${score} · Waiting for opponent…` : score,
    );
    elements["play-again"].disabled = Boolean(snapshot.rematch?.your_ready);
    elements["play-again"].textContent = waiting ? "Waiting…" : "Play Again";
  } else if (!own?.connected || !opponent?.connected) {
    setOverlay("CONNECTION", "Reconnecting", "Your spacecraft is coasting while this device reconnects.");
  }
}

function setOverlay(kicker, title, detail) {
  elements["phase-kicker"].textContent = kicker;
  elements["phase-title"].textContent = title;
  elements["phase-detail"].textContent = detail;
}

function requestRematch() {
  if (state.socket?.readyState !== WebSocket.OPEN || state.snapshot?.phase !== "complete") return;
  elements["play-again"].disabled = true;
  state.socket.send(JSON.stringify({ type: "rematch" }));
}

function returnToLobby() {
  const ownName = state.snapshot?.you?.name || "";
  const socket = state.socket;
  state.socket = null;
  socket?.close(1000, "Returned to lobby");
  clearTimeout(state.reconnectTimer);
  clearInterval(state.heartbeatTimer);
  state.reconnectTimer = null;
  state.heartbeatTimer = null;
  neutralizeControls(false);
  sessionStorage.removeItem("rpo-duel-session");
  state.roomCode = "";
  state.token = "";
  state.playerId = "";
  state.snapshot = null;
  state.sequence = 0;
  state.lastPongAt = 0;
  state.reconnectAttempts = 0;
  state.trail = [];
  state.renderFrames = [];
  state.plotViewports.clear();
  state.lastTrailTick = -1;
  state.lastRoundIndex = null;
  state.cameraMode = DUEL_CAMERA_MODES.REFERENCE;
  document.body.classList.remove("duel-active");
  elements["game-view"].classList.add("hidden");
  elements["landing-view"].classList.remove("hidden");
  elements["create-name"].value = ownName;
  elements["join-name"].value = ownName;
  elements["computer-name"].value = ownName;
  elements["join-code"].value = "";
  elements["play-again"].disabled = false;
  elements["play-again"].textContent = "Play Again";
  paintCameraMode();
  selectTab("create");
  const url = new URL(window.location.href);
  url.searchParams.delete("room");
  history.replaceState(null, "", url);
}

function wireControls() {
  ["selectstart", "dragstart", "contextmenu"].forEach((type) => {
    document.addEventListener(type, (event) => {
      if (!document.body.classList.contains("duel-active")) return;
      event.preventDefault();
    }, { capture: true });
  });
  window.addEventListener("keydown", (event) => {
    resumeMusic();
    if (event.code === "KeyM" && !event.repeat && !isTyping()) {
      event.preventDefault();
      toggleMusic();
      return;
    }
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
  elements["music-toggle"].addEventListener("click", toggleMusic);
  window.addEventListener("pointerdown", resumeMusic, { passive: true });
  window.addEventListener("keyup", (event) => {
    const binding = keyBindings[event.code];
    if (!binding) return;
    event.preventDefault();
    state.pressedKeys.delete(event.code);
    updateAxisFromInputs(binding[0]);
  });
  for (const button of document.querySelectorAll(".thrust-button")) {
    const axis = button.dataset.axis;
    const press = (event) => {
      event.preventDefault();
      if (button.disabled) return;
      const value = Number(button.dataset.value);
      touchPointers.set(event.pointerId, button);
      button.setPointerCapture?.(event.pointerId);
      button.dataset.pressed = "true";
      state.controls[axis] = value;
      sendControls();
      paintButtons();
    };
    button.addEventListener("pointerdown", press);
    button.addEventListener("lostpointercapture", releaseTouchPointer);
    button.addEventListener("click", (event) => {
      if (event.detail !== 0 || button.disabled) return;
      const value = Number(button.dataset.value);
      state.controls[axis] = value;
      sendControls();
      paintButtons();
      setTimeout(() => {
        if (state.controls[axis] !== value) return;
        state.controls[axis] = keyboardAxisValue(axis);
        sendControls();
        paintButtons();
      }, 150);
    });
  }
  window.addEventListener("pointerup", releaseTouchPointer, { capture: true });
  window.addEventListener("pointercancel", releaseTouchPointer, { capture: true });
  window.addEventListener("blur", () => neutralizeControls(true));
  document.addEventListener("visibilitychange", () => { if (document.hidden) neutralizeControls(true); });
}

function requestMusicStart() {
  state.musicRequested = true;
  resumeMusic();
}

function resumeMusic() {
  if (!state.musicRequested || !state.musicEnabled || !state.musicAvailable || !duelMusic.paused) return;
  duelMusic.play().catch(() => {});
}

function toggleMusic() {
  state.musicEnabled = !state.musicEnabled;
  if (state.musicEnabled) requestMusicStart();
  else duelMusic.pause();
  paintMusicMode();
}

function paintMusicMode() {
  const button = elements["music-toggle"];
  button.disabled = !state.musicAvailable;
  button.textContent = state.musicAvailable ? `M · MUSIC ${state.musicEnabled ? "ON" : "OFF"}` : "M · MUSIC N/A";
  button.setAttribute("aria-pressed", String(state.musicAvailable && state.musicEnabled));
}

function toggleCamera() {
  state.cameraMode = toggleDuelCameraMode(state.cameraMode);
  state.plotViewports.clear();
  paintCameraMode();
  drawPlots();
}

function paintCameraMode() {
  const pairMode = state.cameraMode === DUEL_CAMERA_MODES.CURRENT_PAIR;
  const projectionMode = state.cameraMode === DUEL_CAMERA_MODES.CURRENT_PROJECTIONS;
  elements["camera-toggle"].textContent = pairMode
    ? "C · PAIR VIEW"
    : projectionMode
      ? "C · PROJECTION VIEW"
      : "C · REFERENCE VIEW";
  elements["camera-toggle"].setAttribute("aria-pressed", String(state.cameraMode !== DUEL_CAMERA_MODES.REFERENCE));
  elements["game-view"].dataset.cameraMode = state.cameraMode;
}

function releaseTouchPointer(event) {
  const button = touchPointers.get(event.pointerId);
  if (!button) return;
  event.preventDefault();
  touchPointers.delete(event.pointerId);
  const axis = button.dataset.axis;
  const stillPressed = [...touchPointers.values()].some((candidate) => candidate === button);
  if (stillPressed) return;
  delete button.dataset.pressed;
  const otherPressed = [...document.querySelectorAll(`.thrust-button[data-axis="${axis}"][data-pressed="true"]`)].at(-1);
  state.controls[axis] = otherPressed ? Number(otherPressed.dataset.value) : keyboardAxisValue(axis);
  sendControls();
  paintButtons();
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
    if (binding?.[0] === axis) {
      const inputSign = axis === "i" ? frameConventionDisplayAxisSign(state.frameConvention, "i") : 1;
      value += binding[1] * inputSign;
    }
  }
  return Math.max(-1, Math.min(1, value));
}

function neutralizeControls(send = true) {
  state.pressedKeys.clear();
  touchPointers.clear();
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

function drawPlots(now = performance.now()) {
  const round = visualRound(now);
  const frame = duelPlotFrame(round, visualTrail(round), state.cameraMode);
  drawPlot(elements["ri-canvas"], frame, "i_km", "r_km", "I", "R", now);
  drawPlot(elements["rc-canvas"], frame, "c_km", "r_km", "C", "R", now);
}

function drawPlot(canvas, frame, xKey, yKey, xLabel, yLabel, now) {
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
  const floatingMode = frame.cameraMode !== DUEL_CAMERA_MODES.REFERENCE;
  const captureRadiusKm = Number(state.snapshot?.series?.round?.capture_range_km) || .1;
  const targetViewport = {
    centerX: frame.cameraCenter[xKey],
    centerY: frame.cameraCenter[yKey],
    span: duelPlotSpan(frame, xKey, yKey, captureRadiusKm),
  };
  const viewport = floatingMode
    ? smoothPlotViewport(`${xKey}:${yKey}`, targetViewport, now)
    : targetViewport;
  const { centerX, centerY, span } = viewport;
  const xSign = frameConventionDisplayAxisSign(state.frameConvention, xKey);
  const ySign = frameConventionDisplayAxisSign(state.frameConvention, yKey);
  const displayCenterX = frameConventionDisplayValue(state.frameConvention, xKey, centerX);
  const displayCenterY = frameConventionDisplayValue(state.frameConvention, yKey, centerY);
  const pad = 0;
  const mapX = (value) => pad + (((frameConventionDisplayValue(state.frameConvention, xKey, value) - displayCenterX) + span) / (2 * span)) * (w - pad * 2);
  const mapY = (value) => h - pad - (((frameConventionDisplayValue(state.frameConvention, yKey, value) - displayCenterY) + span) / (2 * span)) * (h - pad * 2);

  ctx.lineWidth = 1;
  for (let index = -2; index <= 2; index += 1) {
    const xValue = centerX + index * span / 2;
    const yValue = centerY + index * span / 2;
    ctx.strokeStyle = index === 0 ? "rgba(48,60,76,.95)" : "rgba(30,38,50,.95)";
    ctx.beginPath(); ctx.moveTo(mapX(xValue), pad); ctx.lineTo(mapX(xValue), h - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, mapY(yValue)); ctx.lineTo(w - pad, mapY(yValue)); ctx.stroke();
  }
  if (!floatingMode) {
    ctx.strokeStyle = "rgba(90,104,124,.95)";
    ctx.beginPath(); ctx.moveTo(mapX(0), pad); ctx.lineTo(mapX(0), h - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, mapY(0)); ctx.lineTo(w - pad, mapY(0)); ctx.stroke();
    ctx.fillStyle = "rgba(170,184,204,.9)";
    ctx.beginPath(); ctx.arc(mapX(0), mapY(0), 3, 0, Math.PI * 2); ctx.fill();
  }

  const captureRadiusX = Math.max(3, Math.abs(mapX(frame.target[xKey] + captureRadiusKm) - mapX(frame.target[xKey])));
  const captureRadiusY = Math.max(3, Math.abs(mapY(frame.target[yKey] + captureRadiusKm) - mapY(frame.target[yKey])));
  const captureStyle = captureRingStyle(ownRole(state.snapshot?.series));
  ctx.fillStyle = captureStyle.fill;
  ctx.strokeStyle = captureStyle.stroke;
  ctx.beginPath();
  ctx.ellipse(mapX(frame.target[xKey]), mapY(frame.target[yKey]), captureRadiusX, captureRadiusY, 0, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();

  if (!floatingMode) {
    drawPath(ctx, frame.targetTrail, xKey, yKey, mapX, mapY, "rgba(245,92,92,.7)", 1.5);
    drawPath(ctx, frame.chaserTrail, xKey, yKey, mapX, mapY, "rgba(245,205,92,.72)", 1.5);
  }
  drawPath(ctx, frame.targetProjection, xKey, yKey, mapX, mapY, "rgba(245,92,92,.95)", 2, [8, 6]);
  drawPath(ctx, frame.chaserProjection, xKey, yKey, mapX, mapY, "rgba(96,174,224,.95)", 2, [8, 6]);

  drawSatellite(ctx, frame.target, xKey, yKey, mapX, mapY, "#f55c5c", "T");
  drawSatellite(ctx, frame.chaser, xKey, yKey, mapX, mapY, "#f5cd5c", "C");
  ctx.fillStyle = "rgba(170,184,204,.92)";
  ctx.font = "11px Menlo, Consolas, monospace";
  if (!floatingMode) {
    ctx.textAlign = xSign > 0 ? "right" : "left";
    ctx.fillText(`+${xLabel}`, xSign > 0 ? w - pad - 4 : pad + 4, mapY(0) - 6);
    ctx.textAlign = "left";
    ctx.fillText(`+${yLabel}`, mapX(0) + 6, ySign > 0 ? pad + 12 : h - pad - 4);
  }
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

function renderLoop(now) {
  if (!elements["game-view"].classList.contains("hidden")) drawPlots(now);
  requestAnimationFrame(renderLoop);
}

function updatePlayerCard(card, player, score) {
  card.querySelector(".player-name").textContent = player?.name || "WAITING";
  card.querySelector("b").textContent = score;
  card.querySelector(".status-dot").classList.toggle("connected", Boolean(player?.connected));
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

function recordRenderSnapshot(snapshot, receivedAtMs) {
  const previous = state.renderFrames.at(-1);
  const sameRound = previous?.snapshot?.series?.round_index === snapshot.series?.round_index;
  const previousTick = previous?.snapshot?.series?.round?.tick;
  const currentTick = snapshot.series?.round?.tick;
  const continuous = Boolean(
    previous
    && previous.snapshot.phase === "active"
    && snapshot.phase === "active"
    && previous.snapshot.speed?.speed_multiple === 200
    && snapshot.speed?.speed_multiple === 200
    && sameRound
    && Number.isFinite(previousTick)
    && Number.isFinite(currentTick)
    && currentTick > previousTick
    && receivedAtMs - previous.receivedAtMs <= DUEL_VISUAL_TIMING.max_interpolation_gap_ms
  );
  if (!continuous) {
    state.renderFrames = [];
    state.plotViewports.clear();
  }
  state.renderFrames.push({ snapshot, receivedAtMs });
  if (state.renderFrames.length > 5) state.renderFrames.shift();
}

function visualRound(now) {
  const authoritative = state.snapshot?.series?.round;
  if (!authoritative || state.renderFrames.length < 2) return authoritative;
  const renderAt = now - DUEL_VISUAL_TIMING.render_delay_ms;
  const first = state.renderFrames[0];
  if (renderAt <= first.receivedAtMs) return first.snapshot.series.round;
  for (let index = 1; index < state.renderFrames.length; index += 1) {
    const current = state.renderFrames[index];
    if (renderAt > current.receivedAtMs) continue;
    const previous = state.renderFrames[index - 1];
    const interval = current.receivedAtMs - previous.receivedAtMs;
    const alpha = interval > 0 ? (renderAt - previous.receivedAtMs) / interval : 1;
    return interpolateDuelRound(previous.snapshot.series.round, current.snapshot.series.round, alpha);
  }
  return state.renderFrames.at(-1).snapshot.series.round;
}

function visualTrail(round) {
  if (!round || !Number.isFinite(round.tick)) return state.trail;
  const trail = state.trail.filter((sample) => sample.tick <= round.tick);
  const current = { ...referenceRelativePair(round), tick: round.tick };
  if (trail.at(-1)?.tick === current.tick) trail[trail.length - 1] = current;
  else trail.push(current);
  return trail;
}

function smoothPlotViewport(key, target, now) {
  const previous = state.plotViewports.get(key);
  if (!previous || now - previous.updatedAtMs > DUEL_VISUAL_TIMING.max_interpolation_gap_ms) {
    const initial = { ...target, updatedAtMs: now };
    state.plotViewports.set(key, initial);
    return initial;
  }
  const elapsed = Math.max(0, now - previous.updatedAtMs);
  const alpha = 1 - Math.exp(-elapsed / DUEL_VISUAL_TIMING.camera_smoothing_ms);
  const smoothed = {
    centerX: previous.centerX + (target.centerX - previous.centerX) * alpha,
    centerY: previous.centerY + (target.centerY - previous.centerY) * alpha,
    span: previous.span + (target.span - previous.span) * alpha,
    updatedAtMs: now,
  };
  state.plotViewports.set(key, smoothed);
  return smoothed;
}

async function copyInvite() {
  const url = urlWithFrameConvention(window.location.origin, state.frameConvention);
  url.searchParams.set("room", state.roomCode);
  try { await navigator.clipboard.writeText(url.toString()); showToast("Invite link copied"); }
  catch { showToast(`Room code: ${state.roomCode}`); }
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
