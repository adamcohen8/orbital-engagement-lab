import {
  buildChallengeRecord,
  createPursuitArcadeSession,
  DEFAULT_PURSUIT_CHALLENGE,
  ellipticLinearCoastStates,
  gameTickDtS,
  validateAttemptPacket,
} from "./competition/arcade-engine.js";

const MU = 398600.4418;
const TARGET_A_KM = 7000;
const MEAN_MOTION = Math.sqrt(MU / TARGET_A_KM ** 3);
const ORBIT_PERIOD_S = (2 * Math.PI) / MEAN_MOTION;
const MAX_ACCEL_KM_S2 = 1.0e-5;
const FIXED_DT_S = 0.1;
const MAX_STEPS_PER_FRAME = 32;
const MAX_GHOST_DRAW_POINTS = 120;
const TUTORIAL_TARGET_PATH_POINTS = 181;
const SPEED_OPTIONS = [1, 2, 5, 10, 25, 50, 100, 200];
const MANEUVER_CONTROL_SPEED = 10;
const TRAIL_LIMIT = 1200;
const MIN_PLOT_SPAN_KM = 0.005;
const PLOT_SCALE_MARGIN = 1.2;
const SATELLITE_SPRITE_DIAMETER_KM = 0.006;
const SATELLITE_DOT_THRESHOLD_PX = 4;
const SATELLITE_ICON_THRESHOLD_PX = 18;
const SATELLITE_ICON_SIZE_PX = 20;
const SATELLITE_MAX_SIZE_PX = 72;
const TARGET_MARKER = "#f55c5c";
const CHASER_MARKER = "#f5cd5c";
const BUILD_ID = "unified-mobile-shell-2026-06-17m";
const ARCADE_BUILD_ID = `${BUILD_ID}-competition-local`;
const ARCADE_CHALLENGE_RECORD = buildChallengeRecord(DEFAULT_PURSUIT_CHALLENGE);
const LEADERBOARD_REFRESH_MS = 30000;
const PLAUSIBLE_ANALYTICS_SCRIPT_SRC = "https://plausible.io/js/script.js";
const VERCEL_ANALYTICS_SCRIPT_SRC = "/_vercel/insights/script.js";
const ANALYTICS_LOCAL_HOSTNAMES = new Set(["", "localhost", "127.0.0.1", "::1"]);
const PREVIEW_DEV_HOSTNAMES = new Set(["", "localhost", "127.0.0.1", "::1"]);
const PRIMER_AMPLITUDES_KM = { r: 0.65, i: 0.75, c: 0.65 };
const MUSIC_TRACKS = {
  selector: "./assets/01_insert_coin_to_orbit.wav",
  tutorial: "./assets/10_training_grid_sunrise.wav",
  sandbox: "./assets/04_docking_bay_neon.wav",
  arcade: "./assets/21_pursuit_arcade_overdrive_no_siren_demo.wav",
  arcadeBoss: "./assets/28_high_shred_boss_riff.wav",
};

const el = {
  shell: document.querySelector(".trainer-shell"),
  levelSelector: document.querySelector("#levelSelector"),
  selectorMusicButton: document.querySelector("#selectorMusicButton"),
  selectorViewButton: document.querySelector("#selectorViewButton"),
  selectorPreviewTitle: document.querySelector("#selectorPreviewTitle"),
  selectorPreviewBudget: document.querySelector("#selectorPreviewBudget"),
  selectorPreviewObjective: document.querySelector("#selectorPreviewObjective"),
  selectorPreviewBrief: document.querySelector("#selectorPreviewBrief"),
  selectorPreviewCriteria: document.querySelector("#selectorPreviewCriteria"),
  selectorPreviewNotes: document.querySelector("#selectorPreviewNotes"),
  leaderboardPanel: document.querySelector("#leaderboardPanel"),
  leaderboardList: document.querySelector("#leaderboardList"),
  leaderboardMeta: document.querySelector("#leaderboardMeta"),
  leaderboardRefresh: document.querySelector("#leaderboardRefresh"),
  pauseButton: document.querySelector("#pauseButton"),
  resetButton: document.querySelector("#resetButton"),
  levelSelectButton: document.querySelector("#levelSelectButton"),
  musicButton: document.querySelector("#musicButton"),
  viewButton: document.querySelector("#viewButton"),
  modeLabel: document.querySelector("#modeLabel"),
  objectiveTitle: document.querySelector("#objectiveTitle"),
  objectiveText: document.querySelector("#objectiveText"),
  riTitle: document.querySelector("#riTitle"),
  riSubtitle: document.querySelector("#riSubtitle"),
  rcTitle: document.querySelector("#rcTitle"),
  rcSubtitle: document.querySelector("#rcSubtitle"),
  riPanel: document.querySelector("#riPanel"),
  rcPanel: document.querySelector("#rcPanel"),
  riCanvas: document.querySelector("#riCanvas"),
  rcCanvas: document.querySelector("#rcCanvas"),
  rangeMetric: document.querySelector("#rangeMetric"),
  speedMetric: document.querySelector("#speedMetric"),
  dvMetric: document.querySelector("#dvMetric"),
  timeMetric: document.querySelector("#timeMetric"),
  topRangeMetric: document.querySelector("#topRangeMetric"),
  topSpeedMetric: document.querySelector("#topSpeedMetric"),
  topDvMetric: document.querySelector("#topDvMetric"),
  hudLine: document.querySelector("#hudLine"),
  coachHint: document.querySelector("#coachHint"),
  commandLine: document.querySelector("#commandLine"),
  footerLine: document.querySelector("#footerLine"),
  levelLabel: document.querySelector("#levelLabel"),
  speedMultiple: document.querySelector("#speedMultiple"),
  rMeter: document.querySelector("#rMeter"),
  iMeter: document.querySelector("#iMeter"),
  cMeter: document.querySelector("#cMeter"),
  sandboxPanel: document.querySelector("#sandboxPanel"),
  presetSelect: document.querySelector("#presetSelect"),
  rangeSlider: document.querySelector("#rangeSlider"),
  driftSlider: document.querySelector("#driftSlider"),
  applySandbox: document.querySelector("#applySandbox"),
  randomSandbox: document.querySelector("#randomSandbox"),
  debriefPanel: document.querySelector("#debriefPanel"),
  debriefTitle: document.querySelector("#debriefTitle"),
  debriefText: document.querySelector("#debriefText"),
  leaderboardForm: document.querySelector("#leaderboardForm"),
  leaderboardUsername: document.querySelector("#leaderboardUsername"),
  leaderboardEmail: document.querySelector("#leaderboardEmail"),
  leaderboardSubmit: document.querySelector("#leaderboardSubmit"),
  leaderboardStatus: document.querySelector("#leaderboardStatus"),
  downloadLink: document.querySelector("#downloadLink"),
  mobileSpeedButtons: Array.from(document.querySelectorAll("[data-mobile-speed]")),
};

const levelOptions = [
  {
    id: "tutorial",
    mode: "primer",
    title: "Level 0 - Tutorial",
    budget: `Time: 18000s   Chaser dV: ${formatSpeedMS(12.0)}   Speed Gate: ${formatSpeedMS(0.3)}`,
    objective:
      "Learn what R, I, and C mean by creating six small target orbits, then use short pulse-and-coast translations to settle near a passive target.",
    brief:
      "The yellow satellite is you. R is radial, I is in-track, and C is cross-track. The simulation pauses for each guided stage until you hold the requested control.",
    criteria: [
      "Complete the +I and -I guided orbit demonstrations.",
      "After +I, increase the speed multiple to 10x.",
      "Complete the +R and -R guided orbit demonstrations.",
      "Complete the +C and -C guided orbit demonstrations.",
      `Get within ${formatDistanceKm(0.25)} of the passive target below ${formatSpeedMS(0.3)}.`,
    ],
    notes: [
      "This level teaches the controls before introducing natural-motion matching, keepout constraints, or target evasion.",
      "Use short pulses followed by coasting rather than continuous thrust.",
      "RI shows in-track versus radial motion; RC shows cross-track versus radial motion.",
    ],
  },
  {
    id: "sandbox",
    mode: "sandbox",
    title: "Sandbox",
    budget: "Time: 20000s",
    objective: "Experiment with RIC translation controls and relative orbital motion without pass/fail goals.",
    brief:
      "Edit the starting RIC state in the setup panel, then maneuver freely. Delta-v used remains visible, but there is no delta-v budget.",
    criteria: ["No pass/fail objective; experiment freely."],
    notes: [
      "Use this mode to demonstrate how initial relative state changes relative motion.",
      "Circular-orbit HCW prediction is shown for the browser preview.",
    ],
  },
  {
    id: "pursuitArcade",
    mode: "arcade",
    title: "Pursuit Arcade",
    budget: `Time: 12000s   Chaser dV: ${formatSpeedMS(3.0)}   Goal: ${formatDistanceKm(0.1)}`,
    objective: "Chase an evading target using RIC translation controls in a browser-native two-body arcade model.",
    brief:
      "The arcade mode propagates target and chaser in ECI under central Earth gravity, maps your RIC commands into the target frame, records inputs by simulation tick, and validates the replay locally.",
    criteria: [
      `Reach the ${formatDistanceKm(0.1)} goal circle.`,
      "Clear as many pursuit rounds as possible.",
      `Stay inside the ${formatSpeedMS(3.0)} chaser delta-v budget.`,
    ],
    notes: [
      "Beta competition prototype: browser play uses a deterministic two-body engine, not the full downloadable OEL engine.",
      "Standalone and multi-round arcade attempts can be replay-validated locally; hosted leaderboard submissions are validated before scoring.",
      "Static RI and RC plots can be generated from recomputed replay history.",
    ],
  },
];

const tutorialStages = [
  {
    id: "plusI",
    title: "Pulse +I",
    text: "Hold D for a short in-track pulse, then coast and watch the RI curve.",
    axis: "i",
    sign: 1,
    targetDv: 0.25,
  },
  {
    id: "speed",
    title: "Speed 10x",
    text: "Tap the up arrow until the speed readout reaches 10x.",
    speedTarget: 10,
  },
  {
    id: "minusI",
    title: "Pulse -I",
    text: "Hold A for the matching backward in-track pulse.",
    axis: "i",
    sign: -1,
    targetDv: 0.25,
  },
  {
    id: "plusR",
    title: "Pulse +R",
    text: "Hold W to push radially away from the target reference line.",
    axis: "r",
    sign: 1,
    targetDv: 0.25,
  },
  {
    id: "minusR",
    title: "Pulse -R",
    text: "Hold S to create the opposite radial response.",
    axis: "r",
    sign: -1,
    targetDv: 0.25,
  },
  {
    id: "plusC",
    title: "Pulse +C",
    text: "Hold the right arrow to leave the orbital plane.",
    axis: "c",
    sign: 1,
    targetDv: 0.25,
  },
  {
    id: "minusC",
    title: "Pulse -C",
    text: "Hold the left arrow to make the cross-track mirror path.",
    axis: "c",
    sign: -1,
    targetDv: 0.25,
  },
  {
    id: "final",
    title: "Final Approach",
    text: `Use small pulses and coast into the green ${formatDistanceKm(0.25)} circle below ${formatSpeedMS(0.3)}.`,
    final: true,
  },
];

const primerStages = [
  {
    id: "radial",
    axis: "r",
    title: "Radial Axis",
    text: "Away from Earth through the target.",
    hint: "Higher or lower circular orbits map to up/down motion on R.",
    eciPlane: "rc",
    localPlane: "ri",
    localSubtitle: "Radial offset in RI",
    eciSubtitle: "Orbit radius changes",
  },
  {
    id: "inTrack",
    axis: "i",
    title: "In-Track Axis",
    text: "Along the target's direction of motion.",
    hint: "Ahead or behind on the same circular orbit maps to left/right motion on I.",
    eciPlane: "rc",
    localPlane: "ri",
    localSubtitle: "Phase offset in RI",
    eciSubtitle: "Same orbit, phase shift",
  },
  {
    id: "crossTrack",
    axis: "c",
    title: "Cross-Track Axis",
    text: "Out of the target's orbital plane.",
    hint: "Different orbital-plane slopes map to left/right motion on C.",
    eciPlane: "ri",
    localPlane: "rc",
    localSubtitle: "Plane offset in RC",
    eciSubtitle: "Inclination sweep",
  },
];

const presets = {
  behind: { r: 0, i: -1.2, c: 0, rd: 0, id: 0, cd: 0 },
  radial: { r: 1.0, i: -0.25, c: 0, rd: 0, id: 0, cd: 0 },
  crossTrack: { r: 0, i: -0.6, c: 0.9, rd: 0, id: 0, cd: 0 },
  close: { r: 0.12, i: -0.32, c: 0.04, rd: 0, id: 0.00005, cd: 0 },
};

const keys = new Set();
const touch = new Set();

const state = {
  mode: "selector",
  running: false,
  passed: false,
  speedIndex: 0,
  sim: makeState(presets.behind),
  trail: [],
  ghost: [],
  tutorialTargetPath: [],
  activeStage: 0,
  stageStart: null,
  stageDv: 0,
  closestKm: Infinity,
  finalReason: "",
  lastFrameMs: 0,
  stepAccumulatorS: 0,
  cameraRuleMode: "full_trajectory",
  primerStage: 0,
  primerTimeS: 0,
  selectedLevel: 0,
  musicEnabled: true,
  musicStartRequested: false,
  arcadeSession: null,
  arcadeSnapshot: null,
  arcadeValidation: null,
  arcadeAttemptPacket: null,
  arcadeSeed: 4242,
  arcadeTransition: null,
  arcadeReferenceStateEci: null,
  arcadeTargetRel: { r: 0, i: 0, c: 0, rd: 0, id: 0, cd: 0, t: 0 },
  arcadeChaserTargetRel: { r: 0, i: 0, c: 0, rd: 0, id: 0, cd: 0, t: 0 },
  leaderboardEntries: [],
  leaderboardLastFetchMs: 0,
  leaderboardLoading: false,
  targetTrail: [],
  targetGhost: [],
  viewPreference: "auto",
  activeView: "desktop",
};

const music = createMusicPlayer(MUSIC_TRACKS.selector);
music.loop = true;
music.volume = 0.65;

const analytics = {
  enabled: false,
  plausibleEnabled: false,
  vercelEnabled: false,
  provider: "",
  domain: "",
  trackedOnce: new Set(),
};

function initAnalytics() {
  const provider = metaContent("oel-analytics-provider").toLowerCase();
  const domain = metaContent("oel-analytics-domain");
  const providers = new Set(
    provider
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean),
  );
  analytics.provider = provider;
  analytics.domain = domain;
  if (!analyticsAllowedOnCurrentHost()) return;
  if (providers.has("plausible") && domain) initPlausibleAnalytics(domain);
  if (providers.has("vercel") && vercelAnalyticsAllowedOnCurrentHost()) initVercelAnalytics();
  analytics.enabled = analytics.plausibleEnabled || analytics.vercelEnabled;
}

function initPlausibleAnalytics(domain) {
  analytics.plausibleEnabled = true;
  window.plausible =
    window.plausible ||
    function plausibleQueue() {
      window.plausible.q = window.plausible.q || [];
      window.plausible.q.push(arguments);
    };
  if (document.querySelector(`script[src="${PLAUSIBLE_ANALYTICS_SCRIPT_SRC}"]`)) return;
  const script = document.createElement("script");
  script.defer = true;
  script.dataset.domain = domain;
  script.src = PLAUSIBLE_ANALYTICS_SCRIPT_SRC;
  document.head.appendChild(script);
}

function initVercelAnalytics() {
  analytics.vercelEnabled = true;
  window.va =
    window.va ||
    function vercelAnalyticsQueue() {
      window.vaq = window.vaq || [];
      window.vaq.push(arguments);
    };
  const scriptSrc = metaContent("oel-vercel-analytics-script") || VERCEL_ANALYTICS_SCRIPT_SRC;
  if (document.querySelector(`script[src="${scriptSrc}"]`)) return;
  const script = document.createElement("script");
  script.defer = true;
  script.src = scriptSrc;
  document.head.appendChild(script);
}

function metaContent(name) {
  return document.querySelector(`meta[name="${name}"]`)?.content.trim() || "";
}

function analyticsAllowedOnCurrentHost() {
  if (!["http:", "https:"].includes(window.location.protocol)) return false;
  return !ANALYTICS_LOCAL_HOSTNAMES.has(window.location.hostname);
}

function vercelAnalyticsAllowedOnCurrentHost() {
  const hostname = window.location.hostname;
  const hostRules = metaContent("oel-vercel-analytics-hosts")
    .split(",")
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  if (hostRules.length <= 0) return hostname.endsWith(".vercel.app");
  const normalizedHost = hostname.toLowerCase();
  return hostRules.some((rule) => {
    if (rule === "*") return true;
    if (rule.startsWith(".")) return normalizedHost.endsWith(rule);
    return normalizedHost === rule;
  });
}

function trackEvent(name, props = {}) {
  if (!analytics.enabled) return;
  const cleanProps = analyticsProps(props);
  if (analytics.plausibleEnabled && typeof window.plausible === "function") {
    window.plausible(name, { props: cleanProps });
  }
  if (analytics.vercelEnabled && typeof window.va === "function") {
    window.va("event", { name, data: cleanProps });
  }
}

function trackEventOnce(name, props = {}) {
  if (analytics.trackedOnce.has(name)) return;
  analytics.trackedOnce.add(name);
  trackEvent(name, props);
}

function initializeViewPreference() {
  const params = new URLSearchParams(window.location.search);
  const requested = params.get("view");
  const saved = readLocalPreference("oelPreviewViewPreference");
  state.viewPreference = normalizeViewPreference(requested || saved || "auto");
  applyViewPreference();
}

function readLocalPreference(key) {
  try {
    return window.localStorage?.getItem(key) || "";
  } catch {
    return "";
  }
}

function writeLocalPreference(key, value) {
  try {
    window.localStorage?.setItem(key, value);
  } catch {
    // Private/restricted browser contexts can block localStorage.
  }
}

function normalizeViewPreference(value) {
  return ["auto", "mobile", "desktop"].includes(value) ? value : "auto";
}

function detectedView() {
  const narrow = window.matchMedia("(max-width: 760px), (max-height: 620px)").matches;
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  return narrow || coarse ? "mobile" : "desktop";
}

function applyViewPreference() {
  const activeView = state.viewPreference === "auto" ? detectedView() : state.viewPreference;
  state.activeView = activeView;
  document.body.classList.toggle("mobile-view", activeView === "mobile");
  document.body.classList.toggle("desktop-view", activeView === "desktop");
  el.shell.classList.toggle("mobile-view", activeView === "mobile");
  el.shell.classList.toggle("desktop-view", activeView === "desktop");
  syncViewButtons();
  syncMusicButton();
  updateDebugState();
  draw();
}

function syncViewButtons() {
  const viewLabel = state.viewPreference === "desktop" ? "Computer" : state.viewPreference[0].toUpperCase() + state.viewPreference.slice(1);
  const label = `View: ${viewLabel}`;
  if (el.viewButton) {
    const mobileCameraButton = state.activeView === "mobile";
    el.viewButton.textContent = mobileCameraButton ? "Toggle Camera" : label;
    el.viewButton.disabled = mobileCameraButton && state.mode !== "sandbox" && state.mode !== "arcade";
    el.viewButton.setAttribute(
      "aria-label",
      mobileCameraButton ? "Toggle camera framing." : `${label}. Active layout ${state.activeView}.`,
    );
  }
  if (el.selectorViewButton) {
    el.selectorViewButton.textContent = label;
    el.selectorViewButton.setAttribute("aria-label", `${label}. Active layout ${state.activeView}.`);
  }
}

function cycleViewPreference() {
  const next = state.viewPreference === "auto" ? "mobile" : state.viewPreference === "mobile" ? "desktop" : "auto";
  state.viewPreference = next;
  writeLocalPreference("oelPreviewViewPreference", next);
  const url = new URL(window.location.href);
  if (next === "auto") url.searchParams.delete("view");
  else url.searchParams.set("view", next);
  window.history.replaceState(null, "", url);
  applyViewPreference();
  trackEvent("view_toggle", { preference: next, active_view: state.activeView });
}

function launchInitialLevelFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const requested = params.get("level") || params.get("mode");
  if (!requested) return false;
  const normalized = requested.toLowerCase();
  const idx = levelOptions.findIndex(
    (option) => option.id.toLowerCase() === normalized || option.mode.toLowerCase() === normalized,
  );
  if (idx < 0) return false;
  selectLevel(idx);
  launchSelectedLevel("url");
  return true;
}

function analyticsProps(props) {
  const clean = {};
  Object.entries(props).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    clean[key] = String(value);
  });
  return clean;
}

function createMusicPlayer(src) {
  if (typeof Audio === "function") {
    return new Audio(src);
  }
  return {
    currentSrc: "",
    loop: false,
    paused: true,
    src: new URL(src, window.location.href).href,
    volume: 0,
    load() {},
    pause() {
      this.paused = true;
    },
    play() {
      this.paused = false;
      return Promise.resolve();
    },
  };
}

function makeState(seed) {
  return {
    r: seed.r,
    i: seed.i,
    c: seed.c,
    rd: seed.rd,
    id: seed.id,
    cd: seed.cd,
    t: 0,
    dv: 0,
  };
}

function currentControls() {
  const r = axisValue("w", "s", "rPlus", "rMinus");
  const i = axisValue("d", "a", "iPlus", "iMinus");
  const c = axisValue("arrowright", "arrowleft", "cPlus", "cMinus");
  const mag = Math.hypot(r, i, c);
  if (mag > 1) {
    return { r: r / mag, i: i / mag, c: c / mag };
  }
  return { r, i, c };
}

function axisValue(plusKey, minusKey, plusTouch, minusTouch) {
  const plus = keys.has(plusKey) || touch.has(plusTouch);
  const minus = keys.has(minusKey) || touch.has(minusTouch);
  return Number(plus) - Number(minus);
}

function resetState(seed = presets.behind) {
  state.sim = makeState(seed);
  state.trail = [samplePoint()];
  state.targetTrail = [];
  state.targetGhost = [];
  state.stageStart = { ...state.sim };
  state.closestKm = rangeKm();
  state.stageDv = 0;
  state.passed = false;
  state.finalReason = "";
  state.stepAccumulatorS = 0;
  el.debriefPanel.classList.add("hidden");
  setLeaderboardFormVisible(false);
  updateGhost();
  draw();
}

function showLevelSelector(options = {}) {
  const previousMode = state.mode;
  state.mode = "selector";
  state.running = false;
  state.passed = false;
  state.stepAccumulatorS = 0;
  keys.clear();
  touch.clear();
  el.debriefPanel.classList.add("hidden");
  setLeaderboardFormVisible(false);
  el.shell.classList.add("selector-mode");
  el.shell.classList.remove("primer-mode");
  setMusicTrackForMode("selector");
  renderLevelSelector();
  syncMusicButton();
  updateDebugState();
  if (options.track && previousMode !== "selector") {
    trackEvent("level_select_return", { from: previousMode, source: options.source || "unknown" });
  }
}

function renderLevelSelector() {
  document.querySelectorAll("[data-level-option]").forEach((button) => {
    const idx = levelOptions.findIndex((option) => option.id === button.dataset.levelOption);
    const active = idx === state.selectedLevel;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", active ? "true" : "false");
  });
  const option = levelOptions[state.selectedLevel] || levelOptions[0];
  el.selectorPreviewTitle.textContent = option.title;
  el.selectorPreviewBudget.textContent = option.budget;
  el.selectorPreviewObjective.textContent = option.objective;
  el.selectorPreviewBrief.textContent = option.brief;
  replaceList(el.selectorPreviewCriteria, option.criteria);
  replaceList(el.selectorPreviewNotes, option.notes);
  const showLeaderboard = option.id === "pursuitArcade";
  el.leaderboardPanel.classList.toggle("hidden", !showLeaderboard);
  if (showLeaderboard) refreshLeaderboard();
}

function replaceList(listEl, items) {
  listEl.replaceChildren();
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    listEl.appendChild(li);
  });
}

function refreshLeaderboard(options = {}) {
  if (!el.leaderboardPanel || state.leaderboardLoading) return;
  const force = Boolean(options.force);
  const now = Date.now();
  if (!force && state.leaderboardLastFetchMs && now - state.leaderboardLastFetchMs < LEADERBOARD_REFRESH_MS) {
    renderLeaderboard();
    return;
  }
  if (!["http:", "https:"].includes(window.location.protocol)) {
    state.leaderboardEntries = [];
    renderLeaderboard("Leaderboard appears on the hosted site.");
    return;
  }
  state.leaderboardLoading = true;
  renderLeaderboard("Loading scores...");
  fetch(`/api/leaderboard?challenge=${encodeURIComponent(ARCADE_CHALLENGE_RECORD.challenge_id)}&limit=10`)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((payload) => {
      state.leaderboardEntries = Array.isArray(payload.entries) ? payload.entries : [];
      state.leaderboardLastFetchMs = Date.now();
      renderLeaderboard();
      trackEvent("arcade_leaderboard_view", { entries: state.leaderboardEntries.length });
    })
    .catch((error) => {
      renderLeaderboard(`Leaderboard unavailable: ${error instanceof Error ? error.message : String(error)}`);
    })
    .finally(() => {
      state.leaderboardLoading = false;
    });
}

function renderLeaderboard(message = "") {
  if (!el.leaderboardList || !el.leaderboardMeta) return;
  el.leaderboardList.replaceChildren();
  if (state.leaderboardEntries.length <= 0) {
    const li = document.createElement("li");
    li.className = "leaderboard-empty";
    li.textContent = "No scores yet.";
    el.leaderboardList.appendChild(li);
  } else {
    state.leaderboardEntries.forEach((entry, idx) => {
      const li = document.createElement("li");
      li.className = "leaderboard-row";
      const rank = document.createElement("span");
      rank.className = "leaderboard-rank";
      rank.textContent = String(idx + 1).padStart(2, "0");
      const name = document.createElement("span");
      name.className = "leaderboard-name";
      name.textContent = String(entry.username || "anonymous");
      const score = document.createElement("span");
      score.className = "leaderboard-score";
      score.textContent = Number(entry.score || 0).toLocaleString();
      const rounds = document.createElement("span");
      rounds.className = "leaderboard-rounds";
      rounds.textContent = `${Number(entry.metrics?.rounds_cleared || 0)} rd`;
      li.append(rank, name, score, rounds);
      el.leaderboardList.appendChild(li);
    });
  }
  if (message) {
    el.leaderboardMeta.textContent = message;
  } else if (state.leaderboardLastFetchMs) {
    el.leaderboardMeta.textContent = `Updated ${new Date(state.leaderboardLastFetchMs).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    })}`;
  } else {
    el.leaderboardMeta.textContent = "";
  }
}

function selectLevel(index) {
  state.selectedLevel = Math.max(0, Math.min(index, levelOptions.length - 1));
  renderLevelSelector();
}

function launchSelectedLevel(source = "selector") {
  const option = levelOptions[state.selectedLevel] || levelOptions[0];
  setMode(option.mode);
  if (option.id === "sandbox") {
    trackEvent("sandbox_start", { source });
  } else if (option.id === "pursuitArcade") {
    trackEvent("arcade_start", { source });
  } else {
    trackEvent("tutorial_start", { source, entry: option.mode });
  }
  playMusicFromGesture();
}

function setMode(mode) {
  state.mode = mode;
  el.shell.classList.remove("selector-mode");
  state.running = false;
  state.speedIndex = 0;
  state.cameraRuleMode = mode === "sandbox" || mode === "arcade" ? "full_trajectory" : "default";
  setMusicTrackForMode(initialMusicTrackForMode(mode));
  state.activeStage = 0;
  state.stageStart = null;
  state.stageDv = 0;
  state.passed = false;
  state.primerTimeS = 0;
  if (mode === "primer") state.primerStage = 0;
  if (mode === "arcade") {
    startArcadeSession();
  } else {
    state.arcadeSession = null;
    state.arcadeSnapshot = null;
    state.arcadeValidation = null;
    const seed = mode === "sandbox" ? sandboxSeed() : mode === "primer" ? primerSample() : presets.behind;
    resetState(seed);
  }
  updateMissionText();
}

function activePrimerStage() {
  return primerStages[state.primerStage] || primerStages[primerStages.length - 1];
}

function primerSample() {
  const stage = activePrimerStage();
  const phase = Math.sin(state.primerTimeS * 1.05);
  const sample = { r: 0, i: 0, c: 0, rd: 0, id: 0, cd: 0, t: state.primerTimeS };
  sample[stage.axis] = (PRIMER_AMPLITUDES_KM[stage.axis] || 0.6) * phase;
  return sample;
}

function advancePrimer() {
  if (state.mode !== "primer") return false;
  if (state.primerStage < primerStages.length - 1) {
    state.primerStage += 1;
    state.primerTimeS = 0;
    updateMissionText();
    updateGhost();
    draw();
    return true;
  }
  trackEvent("primer_complete", { step_count: primerStages.length });
  setMode("tutorial");
  return true;
}

function sandboxSeed() {
  const base = { ...presets[el.presetSelect.value] };
  const range = Number(el.rangeSlider.value);
  const drift = Number(el.driftSlider.value) / 1000;
  const norm = Math.hypot(base.r, base.i, base.c) || 1;
  base.r = (base.r / norm) * range;
  base.i = (base.i / norm) * range;
  base.c = (base.c / norm) * range;
  base.id += drift;
  return base;
}

function randomSandboxSeed() {
  const range = 0.25 + Math.random() * 2.9;
  const theta = Math.random() * Math.PI * 2;
  const z = Math.random() * 2 - 1;
  const plane = Math.sqrt(Math.max(1 - z * z, 0));
  return {
    r: range * plane * Math.cos(theta),
    i: range * plane * Math.sin(theta),
    c: range * z,
    rd: (Math.random() - 0.5) * 0.0004,
    id: (Math.random() - 0.5) * 0.0004,
    cd: (Math.random() - 0.5) * 0.0004,
  };
}

function startArcadeSession() {
  state.arcadeSeed = (state.arcadeSeed + 101) % 4294967296;
  state.arcadeSession = createPursuitArcadeSession(ARCADE_CHALLENGE_RECORD.config, {
    seed: state.arcadeSeed,
    ...arcadeDevStartOptions(),
  });
  state.arcadeValidation = null;
  state.arcadeAttemptPacket = null;
  state.arcadeTransition = null;
  state.passed = false;
  state.finalReason = "";
  state.stepAccumulatorS = 0;
  el.debriefPanel.classList.add("hidden");
  setLeaderboardFormVisible(false);
  syncArcadeSnapshot();
}

function arcadeDevStartOptions() {
  if (!previewDevParamsAllowed()) return {};
  const params = new URLSearchParams(window.location.search);
  const firstBossRound = Math.max(Math.floor(Number(ARCADE_CHALLENGE_RECORD.config.arcade?.boss_round_interval || 0)), 1);
  const requestedRound = params.has("boss") ? firstBossRound : Math.floor(Number(params.get("round") || 1));
  if (!Number.isFinite(requestedRound) || requestedRound <= 1) return {};
  return { startRoundIndex: requestedRound };
}

function previewDevParamsAllowed() {
  return window.location.protocol === "file:" || PREVIEW_DEV_HOSTNAMES.has(window.location.hostname);
}

function syncArcadeSnapshot() {
  if (!state.arcadeSession) return;
  state.arcadeSnapshot = state.arcadeSession.snapshot();
  state.arcadeTransition = state.arcadeSnapshot.round_transition || null;
  state.arcadeReferenceStateEci = arcadeEciBlockToState(state.arcadeSnapshot.target_reference_state_eci);
  state.arcadeChaserTargetRel = arcadeRicBlockToSim(state.arcadeSnapshot.relative_ric, state.arcadeSnapshot.time_s);
  state.arcadeTargetRel = arcadeRicBlockToSim(
    lastArcadeHistoryBlock(state.arcadeSnapshot.history, "target_reference_ric"),
    state.arcadeSnapshot.time_s,
  );
  const chaserReference = lastArcadeHistoryBlock(state.arcadeSnapshot.history, "chaser_reference_ric");
  state.sim = arcadeRicBlockToSim(chaserReference || state.arcadeSnapshot.relative_ric, state.arcadeSnapshot.time_s);
  state.closestKm = Number(state.arcadeSnapshot.closest_range_km || state.arcadeSnapshot.range_km || Infinity);
  state.trail = arcadeHistoryToTrail(state.arcadeSnapshot.history || [], "chaser_reference_ric");
  state.targetTrail = arcadeHistoryToTrail(state.arcadeSnapshot.history || [], "target_reference_ric");
  state.passed = Boolean(state.arcadeSnapshot.terminal);
  if (state.arcadeTransition) {
    state.running = false;
  }
  if (state.arcadeSnapshot.terminal) {
    state.running = false;
    state.finalReason = state.arcadeSnapshot.terminal_reason || "";
  }
  setMusicTrackForMode(arcadeMusicTrackKey());
}

function lastArcadeHistoryBlock(history, key) {
  if (!Array.isArray(history) || history.length === 0) return null;
  return history[history.length - 1]?.[key] || null;
}

function arcadeRicBlockToSim(rel, timeS = 0) {
  rel = rel || {};
  return {
    r: Number(rel.r_km || 0),
    i: Number(rel.i_km || 0),
    c: Number(rel.c_km || 0),
    rd: Number(rel.rd_km_s || 0),
    id: Number(rel.id_km_s || 0),
    cd: Number(rel.cd_km_s || 0),
    t: Number(timeS || 0),
    dv: Number(state.arcadeSnapshot?.player_delta_v_m_s || 0),
  };
}

function arcadeEciBlockToState(block) {
  if (!block || !Array.isArray(block.r_km) || !Array.isArray(block.v_km_s)) return null;
  return {
    r: block.r_km.map(Number),
    v: block.v_km_s.map(Number),
  };
}

function arcadeHistoryToTrail(history, key = "relative_ric") {
  return history.slice(-TRAIL_LIMIT).map((sample) => {
    const rel = sample[key] || sample.relative_ric || {};
    return {
      r: Number(rel.r_km || 0),
      i: Number(rel.i_km || 0),
      c: Number(rel.c_km || 0),
      rd: Number(rel.rd_km_s || 0),
      id: Number(rel.id_km_s || 0),
      cd: Number(rel.cd_km_s || 0),
      t: Number(sample.time_s || 0),
    };
  });
}

function arcadeStep() {
  if (!state.arcadeSession || !state.running || state.passed) return;
  state.arcadeSession.setControls(currentControls());
  state.arcadeSession.step(1);
  syncArcadeSnapshot();
  if (state.arcadeTransition) {
    showArcadeRoundTransition();
  } else if (state.arcadeSnapshot?.terminal) {
    showArcadeDebrief();
  }
}

function step(dt, forceRun = false) {
  if (state.mode === "arcade") {
    arcadeStep();
    return;
  }
  if ((!state.running && !forceRun) || state.passed) return;
  const u = currentControls();
  const ar = u.r * MAX_ACCEL_KM_S2;
  const ai = u.i * MAX_ACCEL_KM_S2;
  const ac = u.c * MAX_ACCEL_KM_S2;
  const n = MEAN_MOTION;
  const rdd = 3 * n * n * state.sim.r + 2 * n * state.sim.id + ar;
  const idd = -2 * n * state.sim.rd + ai;
  const cdd = -n * n * state.sim.c + ac;
  state.sim.rd += rdd * dt;
  state.sim.id += idd * dt;
  state.sim.cd += cdd * dt;
  state.sim.r += state.sim.rd * dt;
  state.sim.i += state.sim.id * dt;
  state.sim.c += state.sim.cd * dt;
  state.sim.t += dt;
  state.sim.dv += Math.hypot(ar, ai, ac) * dt * 1000;
  state.closestKm = Math.min(state.closestKm, rangeKm());
  state.trail.push(samplePoint());
  if (state.trail.length > TRAIL_LIMIT) state.trail.shift();
  updateTutorial(dt, u);
}

function updateTutorial(dt, u) {
  if (state.mode !== "tutorial" || state.passed) return;
  const stage = tutorialStages[state.activeStage];
  if (!stage) return;
  if (stage.speedTarget) {
    maybeCompleteSpeedStage(stage);
    return;
  }
  if (stage.final) {
    const slowEnough = relativeSpeedKmS() <= 0.0003;
    if (rangeKm() <= 0.25 && slowEnough) {
      state.passed = true;
      state.running = false;
      state.finalReason = "Goal reached under the speed limit.";
      showDebrief(true);
    }
    return;
  }
  const value = u[stage.axis] || 0;
  if (Math.sign(value) === stage.sign) {
    state.stageDv += Math.abs(value) * MAX_ACCEL_KM_S2 * dt * 1000;
  }
  if (state.stageDv >= stage.targetDv) {
    completeStage();
  }
}

function maybeCompleteSpeedStage(stage = activeTutorialStage()) {
  if (!stage || !stage.speedTarget) return false;
  if (SPEED_OPTIONS[state.speedIndex] < stage.speedTarget) return false;
  completeStage();
  return true;
}

function activeTutorialStage() {
  if (state.mode !== "tutorial") return null;
  return tutorialStages[state.activeStage] || null;
}

function tutorialInputMatches(stage = activeTutorialStage(), controls = currentControls()) {
  if (!stage || !stage.axis || stage.final || stage.speedTarget) return false;
  const value = controls[stage.axis] || 0;
  return Math.sign(value) === stage.sign && Math.abs(value) > 0.5;
}

function simulationShouldRun() {
  if (state.mode === "primer") return false;
  if (state.mode === "arcade") return state.running;
  return state.running || tutorialInputMatches();
}

function currentStepDtS() {
  const baseDtS = state.mode === "arcade" ? ARCADE_CHALLENGE_RECORD.config.dt_s : FIXED_DT_S;
  return gameTickDtS({ baseDtS, speedMultiple: currentSpeedMultiple() });
}

function currentSpeedMultiple() {
  return SPEED_OPTIONS[state.speedIndex];
}

function speedOptionIndex(value) {
  let bestIdx = 0;
  let bestError = Infinity;
  SPEED_OPTIONS.forEach((option, idx) => {
    const error = Math.abs(option - value);
    if (error < bestError) {
      bestIdx = idx;
      bestError = error;
    }
  });
  return bestIdx;
}

function hasManeuverInput(controls = currentControls()) {
  return Math.hypot(controls.r, controls.i, controls.c) > 1.0e-9;
}

function applyManeuverSpeedLimit(controls = currentControls()) {
  if (currentSpeedMultiple() <= MANEUVER_CONTROL_SPEED || !hasManeuverInput(controls)) return false;
  state.speedIndex = speedOptionIndex(MANEUVER_CONTROL_SPEED);
  state.stepAccumulatorS = 0;
  updateMissionText();
  return true;
}

function refreshInputState() {
  if (state.mode === "arcade" && state.arcadeSession && !state.arcadeSnapshot?.terminal) {
    state.arcadeSession.setControls(currentControls());
    syncArcadeSnapshot();
  }
  applyManeuverSpeedLimit();
  updateGhost();
  draw();
}

function completeStage() {
  state.activeStage += 1;
  state.stageDv = 0;
  const enteringFinalApproach =
    state.mode === "tutorial" && Boolean(tutorialStages[state.activeStage] && tutorialStages[state.activeStage].final);
  if (state.activeStage < tutorialStages.length - 1) {
    resetState(presets.behind);
  } else {
    resetState({ r: 0, i: -0.8, c: 0, rd: 0, id: 0, cd: 0 });
  }
  state.running = enteringFinalApproach;
  updateMissionText();
}

function samplePoint() {
  return {
    r: state.sim.r,
    i: state.sim.i,
    c: state.sim.c,
    rd: state.sim.rd,
    id: state.sim.id,
    cd: state.sim.cd,
    t: state.sim.t,
  };
}

function rangeKm() {
  if (state.mode === "arcade" && state.arcadeTransition?.clear_range_km !== undefined) {
    return Number(state.arcadeTransition.clear_range_km || 0);
  }
  if (state.mode === "arcade" && state.arcadeSnapshot) {
    return Number(state.arcadeSnapshot.range_km || 0);
  }
  return Math.hypot(state.sim.r, state.sim.i, state.sim.c);
}

function currentArcadeGoalRangeKm() {
  return Number(
    state.arcadeSnapshot?.goal_range_km ??
      state.arcadeSession?.config?.goal_range_km ??
      ARCADE_CHALLENGE_RECORD.config.goal_range_km,
  );
}

function relativeSpeedKmS() {
  if (state.mode === "arcade" && state.arcadeSnapshot) {
    return Number(state.arcadeSnapshot.relative_speed_km_s || 0);
  }
  return Math.hypot(state.sim.rd, state.sim.id, state.sim.cd);
}

function updateMissionText() {
  if (state.mode === "selector") {
    el.shell.classList.add("selector-mode");
    el.shell.classList.remove("mode-arcade", "mode-sandbox", "mode-tutorial", "primer-mode");
    renderLevelSelector();
    return;
  }
  el.shell.classList.remove("selector-mode");
  el.shell.classList.toggle("primer-mode", state.mode === "primer");
  el.shell.classList.toggle("mode-arcade", state.mode === "arcade");
  el.shell.classList.toggle("mode-sandbox", state.mode === "sandbox");
  el.shell.classList.toggle("mode-tutorial", state.mode === "tutorial");
  el.modeLabel.textContent = state.mode === "sandbox" ? "Sandbox" : "Tutorial";
  if (state.mode === "primer") {
    const stage = activePrimerStage();
    el.levelLabel.textContent = "RIC FRAME PRIMER";
    el.objectiveTitle.textContent = stage.title;
    if (el.objectiveText) {
      el.objectiveText.textContent = stage.text;
    }
  } else if (state.mode === "sandbox") {
    el.levelLabel.textContent = "RPO SANDBOX";
    el.objectiveTitle.textContent = "Free Flight";
    if (el.objectiveText) {
      el.objectiveText.textContent = "";
    }
  } else if (state.mode === "arcade") {
    el.levelLabel.textContent = "PURSUIT ARCADE";
    const snap = state.arcadeSnapshot;
    const roundLabel = snap?.is_boss_round ? `Boss Round ${snap.round_index}` : `Round ${snap?.round_index || 1}`;
    el.objectiveTitle.textContent = roundLabel;
    if (el.objectiveText) {
      el.objectiveText.textContent = `Reach ${formatDistanceKm(currentArcadeGoalRangeKm())}`;
    }
  } else {
    el.levelLabel.textContent = "LEVEL 0 - TUTORIAL";
    const stage = tutorialStages[state.activeStage] || tutorialStages[tutorialStages.length - 1];
    el.objectiveTitle.textContent = stage.title;
    if (el.objectiveText) {
      el.objectiveText.textContent = "";
    }
  }
  el.pauseButton.disabled = false;
  if (state.mode === "arcade") {
    el.pauseButton.textContent = state.arcadeTransition ? "Next Round" : state.running ? "Running" : "Start";
    el.pauseButton.disabled = Boolean(state.running && !state.arcadeTransition);
  } else {
    el.pauseButton.textContent = state.mode === "primer" ? primerAdvanceLabel() : state.running ? "Pause" : "Start";
  }
  el.resetButton.textContent = state.mode === "primer" ? "Replay" : "Reset";
  el.sandboxPanel.classList.toggle("hidden", state.mode !== "sandbox" || state.running);
  updatePlotTitles();
  syncViewButtons();
  syncMusicButton();
}

function primerAdvanceLabel() {
  return state.primerStage >= primerStages.length - 1 ? "Start Tutorial" : "Next";
}

function updatePlotTitles() {
  if (state.mode === "primer") {
    const stage = activePrimerStage();
    const riIsEci = stage.eciPlane === "ri";
    el.riTitle.textContent = riIsEci ? "ECI Orbit" : "RI Plane";
    el.riSubtitle.textContent = riIsEci ? stage.eciSubtitle : stage.localSubtitle;
    el.rcTitle.textContent = riIsEci ? "RC Plane" : "ECI Orbit";
    el.rcSubtitle.textContent = riIsEci ? stage.localSubtitle : stage.eciSubtitle;
    return;
  }
  el.riTitle.textContent = "RI Plane";
  el.riSubtitle.textContent = "In-Track vs Radial";
  el.rcTitle.textContent = "RC Plane";
  el.rcSubtitle.textContent = "Cross-Track vs Radial";
}

function syncMusicButton() {
  const musicPrefix = state.activeView === "mobile" ? "" : "M ";
  el.musicButton.textContent = state.musicEnabled ? `${musicPrefix}Music: ON` : `${musicPrefix}Music: OFF`;
  el.musicButton.classList.toggle("active", state.musicEnabled);
  el.musicButton.setAttribute("aria-pressed", String(state.musicEnabled));
  el.selectorMusicButton.textContent = state.musicEnabled ? "Music: ON" : "Music: OFF";
  el.selectorMusicButton.classList.toggle("active", state.musicEnabled);
  el.selectorMusicButton.setAttribute("aria-pressed", String(state.musicEnabled));
}

function playMusicFromGesture() {
  if (!state.musicEnabled || !music.paused) return;
  state.musicStartRequested = true;
  music.play().catch(() => {
    // Browser audio policies require a click or key press; the next gesture will retry.
  });
}

function setMusicTrackForMode(mode) {
  const nextSrc = new URL(MUSIC_TRACKS[mode] || MUSIC_TRACKS.tutorial, window.location.href).href;
  if (music.currentSrc === nextSrc || music.src === nextSrc) return;
  const shouldResume = state.musicEnabled && state.musicStartRequested && !music.paused;
  music.pause();
  music.src = nextSrc;
  music.load();
  if (shouldResume) {
    playMusicFromGesture();
  }
}

function initialMusicTrackForMode(mode) {
  if (mode === "sandbox") return "sandbox";
  if (mode === "arcade") return "arcade";
  return "tutorial";
}

function arcadeMusicTrackKey() {
  return state.arcadeSnapshot?.is_boss_round ? "arcadeBoss" : "arcade";
}

function toggleMusic() {
  if (state.musicEnabled && music.paused && !state.musicStartRequested) {
    playMusicFromGesture();
    syncMusicButton();
    return;
  }
  state.musicEnabled = !state.musicEnabled;
  trackEvent("music_toggle", { enabled: state.musicEnabled, mode: state.mode });
  if (state.musicEnabled) {
    playMusicFromGesture();
  } else {
    music.pause();
  }
  syncMusicButton();
}

function isEditableControlTarget(target) {
  return target instanceof Element && Boolean(target.closest("input, select, textarea, [contenteditable='true']"));
}

function updateGhost() {
  if (state.mode === "selector" || state.mode === "primer") {
    state.ghost = [];
    state.tutorialTargetPath = [];
    return;
  }
  if (state.mode === "arcade") {
    updateArcadeGhosts();
    state.tutorialTargetPath = [];
    return;
  }
  state.ghost = predictGhost(livePredictionSeed(), ORBIT_PERIOD_S, MAX_GHOST_DRAW_POINTS);
  state.tutorialTargetPath = [];
  if (state.mode !== "tutorial") return;
  const stage = tutorialStages[state.activeStage];
  if (!stage || !stage.axis || stage.final) return;
  if (!state.stageStart) state.stageStart = { ...state.sim };
  const seed = { ...state.stageStart };
  const dvKmS = (stage.targetDv / 1000) * stage.sign;
  seed[`${stage.axis}d`] += dvKmS;
  state.tutorialTargetPath = predictGhost(seed, ORBIT_PERIOD_S, TUTORIAL_TARGET_PATH_POINTS);
}

function livePredictionSeed() {
  return { ...state.sim };
}

function updateArcadeGhosts() {
  const targetSeed = { ...state.arcadeTargetRel };
  const chaserTargetSeed = { ...state.arcadeChaserTargetRel };
  if (state.arcadeSnapshot?.is_boss_round && state.arcadeReferenceStateEci) {
    const times = ghostTimes(arcadeProjectionHorizonS(), MAX_GHOST_DRAW_POINTS);
    state.targetGhost = ellipticLinearCoastStates(simToArcadeRicBlock(targetSeed), times, state.arcadeReferenceStateEci, MU)
      .map(arcadeCoastPointToSim);
    state.ghost = ellipticLinearCoastStates(simToArcadeRicBlock(state.sim), times, state.arcadeReferenceStateEci, MU)
      .map(arcadeCoastPointToSim);
    return;
  }
  state.targetGhost = predictGhost(targetSeed, ORBIT_PERIOD_S, MAX_GHOST_DRAW_POINTS);
  const relativeGhost = predictGhost(chaserTargetSeed, ORBIT_PERIOD_S, MAX_GHOST_DRAW_POINTS);
  state.ghost = relativeGhost.map((point, idx) => addRicPoint(point, state.targetGhost[idx] || targetSeed));
}

function arcadeProjectionHorizonS() {
  const aKm = Number(state.arcadeSession?.config?.target_coes?.a_km || TARGET_A_KM);
  const mu = Number(state.arcadeSession?.config?.mu_km3_s2 || MU);
  return (2 * Math.PI) * Math.sqrt(Math.max(aKm, 1) ** 3 / mu);
}

function ghostTimes(horizonS, samples) {
  const count = Math.max(Math.floor(samples), 2);
  return Array.from({ length: count }, (_, idx) => (horizonS * idx) / (count - 1));
}

function simToArcadeRicBlock(seed) {
  return {
    r_km: Number(seed?.r || 0),
    i_km: Number(seed?.i || 0),
    c_km: Number(seed?.c || 0),
    rd_km_s: Number(seed?.rd || 0),
    id_km_s: Number(seed?.id || 0),
    cd_km_s: Number(seed?.cd || 0),
  };
}

function arcadeCoastPointToSim(point) {
  return {
    r: Number(point?.r || 0),
    i: Number(point?.i || 0),
    c: Number(point?.c || 0),
    rd: Number(point?.rd || 0),
    id: Number(point?.id || 0),
    cd: Number(point?.cd || 0),
    t: Number(point?.t || 0),
  };
}

function addRicPoint(a, b) {
  return {
    r: Number(a.r || 0) + Number(b.r || 0),
    i: Number(a.i || 0) + Number(b.i || 0),
    c: Number(a.c || 0) + Number(b.c || 0),
    rd: Number(a.rd || 0) + Number(b.rd || 0),
    id: Number(a.id || 0) + Number(b.id || 0),
    cd: Number(a.cd || 0) + Number(b.cd || 0),
    t: Number(a.t || 0),
  };
}

function predictGhost(seed, horizonS, samples) {
  const ghost = [];
  const count = Math.max(Math.floor(samples), 2);
  for (let idx = 0; idx < count; idx += 1) {
    const t = count <= 1 ? 0 : (horizonS * idx) / (count - 1);
    ghost.push(cwCoastPoint(seed, t));
  }
  return ghost;
}

function cwCoastPoint(seed, tS) {
  const x = seed.r;
  const y = seed.i;
  const z = seed.c;
  const xd = seed.rd;
  const yd = seed.id;
  const zd = seed.cd;
  const n = MEAN_MOTION;
  const t = Number(tS);
  if (Math.abs(n) <= 1.0e-12) {
    return { r: x + xd * t, i: y + yd * t, c: z + zd * t, rd: xd, id: yd, cd: zd };
  }
  const nt = n * t;
  const cosNt = Math.cos(nt);
  const sinNt = Math.sin(nt);
  return {
    r: (4 - 3 * cosNt) * x + (sinNt / n) * xd + ((2 * (1 - cosNt)) / n) * yd,
    i: 6 * (sinNt - nt) * x + y - ((2 * (1 - cosNt)) / n) * xd + (((4 * sinNt - 3 * nt) / n) * yd),
    c: cosNt * z + (sinNt / n) * zd,
    rd: 3 * n * sinNt * x + cosNt * xd + 2 * sinNt * yd,
    id: -6 * n * (1 - cosNt) * x - 2 * sinNt * xd + (4 * cosNt - 3) * yd,
    cd: -n * sinNt * z + cosNt * zd,
  };
}

function integrateCopy(s, u, dt) {
  const n = MEAN_MOTION;
  const ar = u.r * MAX_ACCEL_KM_S2;
  const ai = u.i * MAX_ACCEL_KM_S2;
  const ac = u.c * MAX_ACCEL_KM_S2;
  const rdd = 3 * n * n * s.r + 2 * n * s.id + ar;
  const idd = -2 * n * s.rd + ai;
  const cdd = -n * n * s.c + ac;
  s.rd += rdd * dt;
  s.id += idd * dt;
  s.cd += cdd * dt;
  s.r += s.rd * dt;
  s.i += s.id * dt;
  s.c += s.cd * dt;
}

function draw() {
  if (state.mode === "selector") {
    updateDebugState();
    return;
  }
  updateHud();
  updateDebugState();
  drawPlot(el.riCanvas, "i", "r", "ri");
  drawPlot(el.rcCanvas, "c", "r", "rc");
}

function updateDebugState() {
  window.__OEL_RPO_PREVIEW_DEBUG__ = {
    buildId: BUILD_ID,
    mode: state.mode,
    running: state.running,
    activeStage: state.activeStage,
    primerStage: state.primerStage,
    selectedLevel: state.selectedLevel,
    viewPreference: state.viewPreference,
    activeView: state.activeView,
    speedMultiple: currentSpeedMultiple(),
    cameraRuleMode: state.cameraRuleMode,
    musicSrc: music.currentSrc || music.src,
    analytics: {
      enabled: analytics.enabled,
      provider: analytics.provider,
      domain: analytics.domain,
    },
    controls: currentControls(),
    sim: { ...state.sim },
    arcade: state.arcadeSnapshot
      ? {
          score: state.arcadeSnapshot.score,
          goalRangeKm: state.arcadeSnapshot.goal_range_km,
          roundIndex: state.arcadeSnapshot.round_index,
          terminal: state.arcadeSnapshot.terminal,
          inputEventCount: state.arcadeSnapshot.input_events.length,
          validation: state.arcadeValidation?.status || "",
          challengeHash: ARCADE_CHALLENGE_RECORD.config_hash,
        }
      : null,
    livePredictionSeed: livePredictionSeed(),
    ghostHead: state.ghost.slice(0, 8).map((point) => ({ ...point })),
    tutorialTargetHead: state.tutorialTargetPath.slice(0, 8).map((point) => ({ ...point })),
  };
}

function updateHud() {
  if (state.mode === "selector") return;
  if (state.mode === "primer") {
    const stage = activePrimerStage();
    const stepText = `Step ${state.primerStage + 1}/${primerStages.length}`;
    el.rangeMetric.textContent = stepText;
    el.speedMetric.textContent = stage.title;
    el.dvMetric.textContent = "RIC";
    el.timeMetric.textContent = `${Math.round(state.primerTimeS)} s`;
    el.topRangeMetric.textContent = `INFO ${stepText}`;
    el.topSpeedMetric.textContent = `INFO ${stage.title}`;
    el.topDvMetric.textContent = "";
    el.hudLine.textContent = `${stepText}   ${stage.title}`;
    el.coachHint.textContent = currentCoachHint();
    el.commandLine.textContent = "";
    el.footerLine.textContent = `${primerAdvanceLabel()} to continue   Esc Level Select`;
    el.speedMultiple.textContent = "1x";
    syncMobileSpeedButtons();
    el.rMeter.value = 0;
    el.iMeter.value = 0;
    el.cMeter.value = 0;
    updateMissionText();
    return;
  }
  const rangeText = formatDistanceKm(rangeKm());
  const speedText = formatSpeedKmS(relativeSpeedKmS());
  const arcadePlayerDvUsed = Number(state.arcadeSnapshot?.player_delta_v_m_s || state.sim.dv || 0);
  const arcadeTargetDvUsed = Number(state.arcadeSnapshot?.target_delta_v_m_s || 0);
  const arcadePlayerDvBudget = Number(state.arcadeSnapshot?.max_delta_v_m_s ?? DEFAULT_PURSUIT_CHALLENGE.max_delta_v_m_s ?? 0);
  const arcadeTargetDvBudget = Number(
    state.arcadeSnapshot?.max_target_delta_v_m_s ?? DEFAULT_PURSUIT_CHALLENGE.max_target_delta_v_m_s ?? 0,
  );
  const arcadePlayerDvRemaining = Math.max(arcadePlayerDvBudget - arcadePlayerDvUsed, 0);
  const arcadeTargetDvRemaining = Math.max(arcadeTargetDvBudget - arcadeTargetDvUsed, 0);
  const arcadeTimeRemainingS = Math.max(
    Number(state.arcadeSnapshot?.remaining_time_s || 0) - Number(state.arcadeSnapshot?.time_s || 0),
    0,
  );
  const dvText = state.mode === "arcade" ? formatSpeedMS(arcadePlayerDvRemaining) : formatSpeedMS(state.sim.dv);
  const targetDvText = formatSpeedMS(arcadeTargetDvRemaining);
  const arcadeTimeText = formatClockTime(arcadeTimeRemainingS);
  const timeText = `${Math.round(state.sim.t)} s`;
  el.rangeMetric.textContent = rangeText;
  el.speedMetric.textContent = speedText;
  el.dvMetric.textContent = dvText;
  el.timeMetric.textContent = timeText;
  if (state.mode === "arcade") {
    const metricGap = "\u00a0".repeat(state.activeView === "mobile" ? 3 : 6);
    const timeLabel = state.activeView === "mobile" ? "Time" : "Time Remaining";
    el.topRangeMetric.textContent = `Score ${state.arcadeSnapshot?.score || 0}${metricGap}${timeLabel} ${arcadeTimeText}`;
    el.topSpeedMetric.textContent = `Target dV ${targetDvText}`;
    el.topDvMetric.textContent = `Chaser dV ${dvText}`;
  } else {
    el.topRangeMetric.textContent = `INFO Range ${rangeText}`;
    el.topSpeedMetric.textContent = `INFO Rel Speed ${speedText}`;
    el.topDvMetric.textContent = `INFO Delta-v ${dvText}`;
  }
  el.hudLine.textContent = `T=${state.sim.t.toFixed(1).padStart(7, " ")}s   Range=${rangeText}   Rel Speed=${speedText}`;
  el.coachHint.textContent = currentCoachHint();
  el.commandLine.textContent = commandStatusLine();
  const spaceAction = state.mode === "arcade" ? "Space Start" : "Space Pause";
  el.footerLine.textContent = `Speed ${SPEED_OPTIONS[state.speedIndex].toFixed(
    0,
  )}x  Up/Down Speed  ${spaceAction}  R Reset  Esc Level Select`;
  el.speedMultiple.textContent = `${SPEED_OPTIONS[state.speedIndex]}x`;
  syncMobileSpeedButtons();
  const u = currentControls();
  el.rMeter.value = u.r;
  el.iMeter.value = u.i;
  el.cMeter.value = u.c;
  updateMissionText();
}

function currentCoachHint() {
  if (state.mode === "primer") {
    return activePrimerStage().hint;
  }
  if (state.mode === "sandbox") {
    const label = state.cameraRuleMode === "full_trajectory" ? "Full Trajectory" : "Satellites Only";
    const cameraLabel = state.activeView === "mobile" ? "Camera" : "C Camera";
    if (state.activeView === "mobile") return `${cameraLabel}: ${label}.`;
    return `Use small pulses, then coast and watch the target-centered RIC motion. ${cameraLabel}: ${label}.`;
  }
  if (state.mode === "arcade") {
    const snap = state.arcadeSnapshot;
    const label = state.cameraRuleMode === "full_trajectory" ? "Full Trajectory" : "Satellites Only";
    if (state.arcadeTransition) {
      const tr = state.arcadeTransition;
      return `Round ${tr.cleared_round_index} clear. +${tr.round_score.toLocaleString()} points, ${Math.round(
        tr.next_time_budget_s,
      )} s for round ${tr.next_round_index}.`;
    }
    if (snap?.terminal) return `${snap.terminal_reason} Local validator score: ${snap.score}.`;
    const cameraLabel = state.activeView === "mobile" ? "Camera" : "C Camera";
    if (state.activeView === "mobile") return `${cameraLabel}: ${label}.`;
    return `${snap?.is_boss_round ? "Boss round. " : ""}Clear rounds to tighten the goal and grow the score. ${cameraLabel}: ${label}.`;
  }
  const stage = tutorialStages[state.activeStage] || tutorialStages[tutorialStages.length - 1];
  if (stage.final) {
    return `Guided burns complete. Settle gently into the green ${formatDistanceKm(0.25)} circle. Keep pulses short.`;
  }
  if (stage.speedTarget) {
    if (state.activeView === "mobile") {
      return `Want to go faster? Tap ${stage.speedTarget}x in the speed row. Current speed: ${SPEED_OPTIONS[state.speedIndex]}x.`;
    }
    return `Want to go faster? Hit the up arrow key to increase the speed multiple. Current speed: ${SPEED_OPTIONS[state.speedIndex]}x.`;
  }
  const progress = Math.min(state.stageDv, stage.targetDv || 0);
  if (state.activeView === "mobile") {
    return `${tutorialStageInstruction(stage)} Burn ${formatBurnProgressMS(progress, stage.targetDv || 0)}.`;
  }
  return `${tutorialStageInstruction(stage)} Burn progress: ${formatSpeedMS(progress)}/${formatSpeedMS(stage.targetDv || 0)}.`;
}

function commandStatusLine() {
  if (state.mode === "primer") {
    return "";
  }
  if (state.mode === "arcade") return "W/S R  A/D I  Left/Right C  Space Start  R Reset";
  return "W/S R  A/D I  Left/Right C  C Camera  M Music";
}

function tutorialStageInstruction(stage) {
  if (state.activeView !== "mobile") return stage.text;
  const label = `${stage.sign > 0 ? "+" : "-"}${String(stage.axis || "").toUpperCase()}`;
  if (stage.id === "plusI") return `Hold ${label}, then coast.`;
  if (stage.id === "minusI") return `Hold ${label}, then coast.`;
  if (stage.id === "plusR") return `Hold ${label}, then coast.`;
  if (stage.id === "minusR") return `Hold ${label}, then coast.`;
  if (stage.id === "plusC") return `Hold ${label}, then coast.`;
  if (stage.id === "minusC") return `Hold ${label}, then coast.`;
  return stage.text;
}

function formatBurnProgressMS(progressMps, targetMps) {
  return `${Math.round(progressMps * 1000)}/${Math.round(targetMps * 1000)} mm/s`;
}

function syncMobileSpeedButtons() {
  el.mobileSpeedButtons.forEach((button) => {
    const active = Number(button.dataset.mobileSpeed || 0) === currentSpeedMultiple();
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function drawPlot(canvas, xAxis, yAxis, plane) {
  const ctx = canvas.getContext("2d");
  const { width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  if (state.mode === "primer") {
    const stage = activePrimerStage();
    if (stage.eciPlane === plane) {
      drawPrimerEci(ctx, width, height, stage);
    } else {
      drawPrimerRic(ctx, width, height, xAxis, yAxis, stage);
    }
    return;
  }
  const cameraCenter = cameraCenterFor(xAxis, yAxis);
  const scale = plotScale(width, height, xAxis, yAxis, cameraCenter);
  const toPx = (p) => ({
    x: width / 2 + (p[xAxis] - cameraCenter[xAxis]) * scale,
    y: height / 2 - (p[yAxis] - cameraCenter[yAxis]) * scale,
  });

  const targetState = state.mode === "arcade" ? state.arcadeTargetRel : { r: 0, i: 0, c: 0 };
  drawGrid(ctx, width, height, scale);
  drawRings(ctx, toPx, scale, xAxis, yAxis, targetState);
  drawPath(ctx, state.tutorialTargetPath, toPx, "rgba(92, 240, 132, 0.92)", true, 3);
  if (state.mode === "arcade") {
    drawPath(ctx, state.targetGhost, toPx, "rgba(245, 92, 92, 0.55)", true, 2);
    drawPath(ctx, state.targetTrail, toPx, "rgba(245, 92, 92, 0.9)", false, 2);
  }
  drawPath(ctx, state.ghost, toPx, "rgba(135, 150, 172, 0.95)", true, 2);
  drawPath(ctx, state.trail, toPx, "rgba(245, 205, 92, 0.95)", false);

  const target = toPx(targetState);
  const chaser = toPx(state.sim);
  drawVector(ctx, chaser, state.sim, xAxis, yAxis, "velocity");
  drawThrustVector(ctx, chaser, xAxis, yAxis);
  drawSpacecraftMarker(ctx, target, "target", { scale, fallbackRadius: 6 });
  drawSpacecraftMarker(ctx, chaser, "chaser", { scale, fallbackRadius: 7 });
  ctx.fillStyle = "rgba(170, 180, 195, 0.92)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText(`${axisLabel(xAxis)} km`, width - 58, height / 2 + 22);
  ctx.save();
  ctx.fillText(`${axisLabel(yAxis)} km`, width / 2 + 8, 24);
  ctx.restore();
}

function drawPrimerRic(ctx, width, height, xAxis, yAxis, stage) {
  const sample = primerSample();
  const scale = Math.min(width, height) / 2.5;
  const toPx = (p) => ({
    x: width / 2 + (p[xAxis] || 0) * scale,
    y: height / 2 - (p[yAxis] || 0) * scale,
  });

  drawGrid(ctx, width, height, scale);
  drawPrimerAxis(ctx, width, height, xAxis, yAxis, stage.axis);

  const target = toPx({ r: 0, i: 0, c: 0 });
  const chaser = toPx(sample);
  ctx.strokeStyle = "rgba(245, 205, 92, 0.38)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(target.x, target.y, 0.25 * scale, 0, Math.PI * 2);
  ctx.stroke();

  drawSpacecraftMarker(ctx, target, "target", { scale, fallbackRadius: 6, forceIcon: true });
  drawSpacecraftMarker(ctx, chaser, "chaser", { scale, fallbackRadius: 7, forceIcon: true });

  ctx.fillStyle = "rgba(245, 92, 92, 0.95)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText("Target", target.x + 10, target.y - 10);
  ctx.fillStyle = "rgba(245, 205, 92, 0.95)";
  ctx.fillText("Chaser", chaser.x + 10, chaser.y + 18);
}

function drawPrimerAxis(ctx, width, height, xAxis, yAxis, activeAxis) {
  const axisColors = {
    r: "rgba(150, 235, 170, 0.94)",
    i: "rgba(245, 205, 92, 0.94)",
    c: "rgba(96, 190, 245, 0.94)",
  };
  const xColor = activeAxis === xAxis ? axisColors[xAxis] : "rgba(90, 104, 124, 0.95)";
  const yColor = activeAxis === yAxis ? axisColors[yAxis] : "rgba(90, 104, 124, 0.95)";
  ctx.save();
  ctx.lineWidth = 2;
  drawArrow(ctx, 36, height / 2, width - 36, height / 2, xColor);
  drawArrow(ctx, width / 2, height - 32, width / 2, 32, yColor);
  ctx.font = "13px Menlo, Consolas, monospace";
  ctx.fillStyle = xColor;
  ctx.fillText(`+${axisLabel(xAxis)}`, width - 54, height / 2 - 10);
  ctx.fillStyle = yColor;
  ctx.fillText(`+${axisLabel(yAxis)}`, width / 2 + 10, 42);
  ctx.restore();
}

function drawPrimerEci(ctx, width, height, stage) {
  const center = { x: width / 2, y: height / 2 + 4 };
  const orbitScale = Math.min(width, height) * 0.34;
  const t = state.primerTimeS;
  const phase = Math.sin(t * 1.05);
  const targetTheta = -0.35;
  const radialOffset = stage.id === "radial" ? 0.14 * phase : 0;
  const phaseOffset = stage.id === "inTrack" ? 0.34 * phase : 0;
  const inclinationDeg = stage.id === "crossTrack" ? 10 * phase : 0;

  if (stage.id === "crossTrack") {
    drawCrossTrackSideView(ctx, width, height, inclinationDeg);
    return;
  }

  drawEciCircle(ctx, center, orbitScale, "rgba(96, 174, 224, 0.62)");
  drawEarth(ctx, center, orbitScale);
  const chaserRadius = 1 + radialOffset;
  const chaserTheta = targetTheta + phaseOffset;
  if (stage.id === "radial") {
    drawEciCircle(ctx, center, orbitScale * chaserRadius, "rgba(245, 205, 92, 0.66)");
  } else {
    drawEciCircle(ctx, center, orbitScale, "rgba(245, 205, 92, 0.34)", true);
  }
  const target = projectEciCircular(1, targetTheta, center, orbitScale);
  const chaser = projectEciCircular(chaserRadius, chaserTheta, center, orbitScale);
  drawSatellite(ctx, target, "target", "Target", -56, 22);
  drawSatellite(ctx, chaser, "chaser", "Chaser", 10, -14);

  if (stage.id === "radial") {
    ctx.fillStyle = "rgba(170, 184, 204, 0.92)";
    ctx.font = "12px Menlo, Consolas, monospace";
    ctx.fillText(`Chaser radius ${(1 + radialOffset).toFixed(2)}x target orbit`, 18, height - 18);
  }
}

function drawEciCircle(ctx, center, radiusPx, color, dashed = false) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  if (dashed) ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radiusPx, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function projectEciCircular(radius, theta, center, scale) {
  return {
    x: center.x + radius * Math.cos(theta) * scale,
    y: center.y + radius * Math.sin(theta) * scale,
  };
}

function drawCrossTrackSideView(ctx, width, height, inclinationDeg) {
  const center = { x: width / 2, y: height / 2 + 6 };
  const earthRadius = Math.max(22, Math.min(width, height) * 0.07);
  const halfSpan = Math.min(width, height) * 0.36;
  const baseSlope = -0.5;
  const chaserSlope = baseSlope - inclinationDeg / 22;
  const targetLine = lineSegment(center, halfSpan, baseSlope);
  const chaserLine = lineSegment(center, halfSpan, chaserSlope);
  const target = pointAlongLine(targetLine, 0.86);
  const chaser = pointAlongLine(chaserLine, 0.86);

  drawEarth(ctx, center, earthRadius / 0.16);
  drawOrbitLine(ctx, targetLine, "rgba(96, 174, 224, 0.72)");
  drawOrbitLine(ctx, chaserLine, "rgba(245, 205, 92, 0.78)");
  drawSatellite(ctx, target, "target", "Target", -60, 24);
  drawSatellite(ctx, chaser, "chaser", "Chaser", 12, -12);

  ctx.fillStyle = "rgba(170, 184, 204, 0.92)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText(`Side view: chaser inclination ${inclinationDeg.toFixed(1)} deg`, 18, height - 18);
}

function lineSegment(center, halfSpan, slope) {
  const dx = halfSpan / Math.sqrt(1 + slope * slope);
  const dy = slope * dx;
  return {
    a: { x: center.x - dx, y: center.y - dy },
    b: { x: center.x + dx, y: center.y + dy },
  };
}

function pointAlongLine(line, fraction) {
  return {
    x: line.a.x + (line.b.x - line.a.x) * fraction,
    y: line.a.y + (line.b.y - line.a.y) * fraction,
  };
}

function drawOrbitLine(ctx, line, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(line.a.x, line.a.y);
  ctx.lineTo(line.b.x, line.b.y);
  ctx.stroke();
  ctx.restore();
}

function drawEarth(ctx, center, orbitScale) {
  const radius = Math.max(24, orbitScale * 0.16);
  const gradient = ctx.createRadialGradient(center.x - radius * 0.35, center.y - radius * 0.35, 4, center.x, center.y, radius);
  gradient.addColorStop(0, "rgb(95, 170, 210)");
  gradient.addColorStop(0.65, "rgb(28, 82, 132)");
  gradient.addColorStop(1, "rgb(9, 28, 58)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(220, 240, 255, 0.55)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawSatellite(ctx, point, role, label, labelOffsetX, labelOffsetY) {
  drawSpacecraftMarker(ctx, point, role, { scale: 1, fallbackRadius: 7, forceIcon: true });
  ctx.fillStyle = "rgba(230, 235, 242, 0.95)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText(label, point.x + labelOffsetX, point.y + labelOffsetY);
}

function satelliteMarkerSizePx(scalePxPerKm) {
  const rawPx = Math.abs(Number(scalePxPerKm || 0)) * SATELLITE_SPRITE_DIAMETER_KM;
  if (!Number.isFinite(rawPx) || rawPx < SATELLITE_DOT_THRESHOLD_PX) return 0;
  if (rawPx < SATELLITE_ICON_THRESHOLD_PX) return SATELLITE_ICON_SIZE_PX;
  return Math.round(Math.max(SATELLITE_ICON_SIZE_PX, Math.min(rawPx, SATELLITE_MAX_SIZE_PX)));
}

function drawSpacecraftMarker(ctx, point, role, options = {}) {
  const color = role === "target" ? TARGET_MARKER : CHASER_MARKER;
  const fallbackRadius = options.fallbackRadius || 7;
  const size = options.forceIcon ? SATELLITE_ICON_SIZE_PX : satelliteMarkerSizePx(options.scale);
  if (size <= 0) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(point.x, point.y, fallbackRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(245, 235, 242, 0.64)";
    ctx.lineWidth = 1;
    ctx.stroke();
    return;
  }

  ctx.save();
  ctx.translate(point.x, point.y);
  const s = size / 128;
  ctx.scale(s, s);
  ctx.translate(-64, -64);
  drawSpacecraftSprite(ctx, color);
  ctx.restore();

  const dotRadius = size < 30 ? 2 : 3;
  ctx.fillStyle = "rgba(235, 248, 255, 0.96)";
  ctx.beginPath();
  ctx.arc(point.x, point.y, dotRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(point.x, point.y, dotRadius + 2, 0, Math.PI * 2);
  ctx.stroke();
}

function drawSpacecraftSprite(ctx, accent) {
  ctx.save();
  ctx.shadowColor = accent;
  ctx.shadowBlur = 10;
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(7, 91);
  ctx.lineTo(50, 72);
  ctx.moveTo(78, 56);
  ctx.lineTo(122, 41);
  ctx.stroke();
  ctx.shadowBlur = 0;

  drawPanel(ctx, [
    [18, 38],
    [48, 51],
    [44, 78],
    [14, 65],
  ]);
  drawPanel(ctx, [
    [80, 51],
    [112, 64],
    [106, 91],
    [76, 78],
  ]);

  ctx.fillStyle = "rgba(8, 18, 26, 0.92)";
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  roundRectPath(ctx, 49, 48, 30, 32, 8);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(64, 64, 10, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = "rgba(220, 242, 250, 0.82)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(64, 64, 7, 0.2 * Math.PI, 1.8 * Math.PI);
  ctx.stroke();
  ctx.strokeStyle = "rgba(80, 110, 130, 0.72)";
  ctx.beginPath();
  ctx.moveTo(49, 58);
  ctx.lineTo(79, 58);
  ctx.moveTo(49, 69);
  ctx.lineTo(79, 69);
  ctx.stroke();
  ctx.fillStyle = "rgba(235, 248, 255, 0.94)";
  ctx.beginPath();
  ctx.arc(64, 64, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = accent;
  ctx.beginPath();
  ctx.moveTo(64, 56);
  ctx.lineTo(64, 72);
  ctx.moveTo(56, 64);
  ctx.lineTo(72, 64);
  ctx.stroke();
  ctx.restore();
}

function drawPanel(ctx, points) {
  ctx.fillStyle = "rgba(8, 30, 42, 0.76)";
  ctx.strokeStyle = "rgba(70, 190, 245, 0.82)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach(([x, y], idx) => {
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function roundRectPath(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(Math.floor(rect.width * dpr), 1);
  const height = Math.max(Math.floor(rect.height * dpr), 1);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: rect.width, height: rect.height };
}

function cameraCenterFor(xAxis, yAxis) {
  if (state.mode === "arcade") {
    if (state.cameraRuleMode === "full_trajectory") return { r: 0, i: 0, c: 0 };
    return {
      r: (Number(state.sim.r || 0) + Number(state.arcadeTargetRel.r || 0)) / 2,
      i: (Number(state.sim.i || 0) + Number(state.arcadeTargetRel.i || 0)) / 2,
      c: (Number(state.sim.c || 0) + Number(state.arcadeTargetRel.c || 0)) / 2,
    };
  }
  if (state.mode === "sandbox" && new Set([xAxis, yAxis]).has("r")) {
    return { r: 0, i: 0, c: 0 };
  }
  if (new Set([xAxis, yAxis]).has("c") && new Set([xAxis, yAxis]).has("r")) {
    return { r: 0, i: 0, c: 0 };
  }
  return {
    r: state.sim.r / 2,
    i: state.sim.i / 2,
    c: state.sim.c / 2,
  };
}

function plotScale(width, height, xAxis, yAxis, cameraCenter) {
  let values;
  if (state.mode === "arcade" && state.cameraRuleMode === "current_pair") {
    values = [state.sim, state.arcadeTargetRel];
  } else if (state.mode === "sandbox" && state.cameraRuleMode === "current_pair") {
    values = [state.sim, { r: 0, i: 0, c: 0 }];
  } else {
    values = [...state.trail, ...state.ghost, ...state.tutorialTargetPath, { r: 0, i: 0, c: 0 }];
  }
  if (state.mode === "arcade") {
    if (state.cameraRuleMode === "current_pair") {
      values.push(state.arcadeTargetRel);
    } else {
      values.push(...state.targetTrail, ...state.targetGhost, state.arcadeTargetRel);
    }
  }
  if (state.mode === "tutorial") values.push({ r: 0.25, i: 0.25, c: 0.25 });
  const span = values.reduce(
    (max, p) =>
      Math.max(
        max,
        Math.abs((p[xAxis] || 0) - (cameraCenter[xAxis] || 0)),
        Math.abs((p[yAxis] || 0) - (cameraCenter[yAxis] || 0)),
      ),
    MIN_PLOT_SPAN_KM,
  );
  const padded = Math.max(span * PLOT_SCALE_MARGIN, MIN_PLOT_SPAN_KM);
  return Math.min(width, height) / (2 * padded);
}

function drawGrid(ctx, width, height, scale) {
  ctx.strokeStyle = "rgba(30, 38, 50, 0.95)";
  ctx.lineWidth = 1;
  const step = niceStep(80 / scale);
  for (let xKm = -20; xKm <= 20; xKm += step) {
    const x = width / 2 + xKm * scale;
    if (x < 0 || x > width) continue;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let yKm = -20; yKm <= 20; yKm += step) {
    const y = height / 2 - yKm * scale;
    if (y < 0 || y > height) continue;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(90, 104, 124, 0.95)";
  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}

function niceStep(raw) {
  if (raw <= 0.05) return 0.05;
  if (raw <= 0.1) return 0.1;
  if (raw <= 0.25) return 0.25;
  if (raw <= 0.5) return 0.5;
  if (raw <= 1) return 1;
  if (raw <= 2) return 2;
  return 5;
}

function drawRings(ctx, toPx, scale, xAxis, yAxis, targetState = { r: 0, i: 0, c: 0 }) {
  const target = toPx(targetState);
  ctx.strokeStyle = "rgba(190, 68, 68, 0.72)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(target.x, target.y, 0.025 * scale, 0, Math.PI * 2);
  ctx.stroke();
  if (state.mode === "tutorial" || state.mode === "arcade") {
    const goalRange = state.mode === "arcade" ? currentArcadeGoalRangeKm() : 0.25;
    ctx.strokeStyle = "rgba(78, 178, 112, 0.86)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(target.x, target.y, goalRange * scale, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (xAxis === "i" && yAxis === "r") {
    ctx.fillStyle = "rgba(245, 205, 92, 0.9)";
    ctx.font = "12px Menlo, Consolas, monospace";
    ctx.fillText("Target", target.x + 10, target.y - 10);
  }
}

function drawPath(ctx, points, toPx, color, dashed, width = 2) {
  if (points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  if (dashed) ctx.setLineDash([8, 8]);
  ctx.beginPath();
  points.forEach((p, idx) => {
    const px = toPx(p);
    if (idx === 0) ctx.moveTo(px.x, px.y);
    else ctx.lineTo(px.x, px.y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawVector(ctx, origin, sim, xAxis, yAxis, kind) {
  const scale = kind === "velocity" ? 75000 : 1;
  const vx = sim[`${xAxis}d`] * scale;
  const vy = sim[`${yAxis}d`] * scale;
  drawArrow(ctx, origin.x, origin.y, origin.x + vx, origin.y - vy, "rgba(245, 205, 92, 0.9)");
}

function drawThrustVector(ctx, origin, xAxis, yAxis) {
  const u = currentControls();
  const scale = 42;
  const vx = u[xAxis] * scale;
  const vy = u[yAxis] * scale;
  if (Math.hypot(vx, vy) < 1) return;
  drawArrow(ctx, origin.x, origin.y, origin.x + vx, origin.y - vy, "rgba(92, 220, 160, 0.95)");
}

function drawArrow(ctx, x1, y1, x2, y2, color) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 8 * Math.cos(angle - 0.45), y2 - 8 * Math.sin(angle - 0.45));
  ctx.lineTo(x2 - 8 * Math.cos(angle + 0.45), y2 - 8 * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();
}

function axisLabel(axis) {
  if (axis === "r") return "R";
  if (axis === "i") return "I";
  return "C";
}

function formatDistanceKm(valueKm, sigFigs = 4) {
  const value = Number(valueKm);
  if (!Number.isFinite(value)) return "--";
  const magnitude = Math.abs(value);
  if (magnitude >= 1.0) return `${formatSigFig(value, sigFigs)} km`;
  if (magnitude >= 1.0e-3) return `${formatSigFig(value * 1000.0, sigFigs)} m`;
  return `${formatSigFig(value * 1.0e6, sigFigs)} mm`;
}

function formatSpeedKmS(valueKmS, sigFigs = 4) {
  const value = Number(valueKmS);
  if (!Number.isFinite(value)) return "--";
  return formatSpeedMS(value * 1000.0, sigFigs);
}

function formatSpeedMS(valueMS, sigFigs = 4) {
  const value = Number(valueMS);
  if (!Number.isFinite(value)) return "--";
  const magnitude = Math.abs(value);
  if (magnitude >= 1000.0) return `${formatSigFig(value / 1000.0, sigFigs)} km/s`;
  if (magnitude >= 1.0) return `${formatSigFig(value, sigFigs)} m/s`;
  return `${formatSigFig(value * 1000.0, sigFigs)} mm/s`;
}

function formatClockTime(valueS) {
  const totalSeconds = Math.max(Math.ceil(Number(valueS) || 0), 0);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatSigFig(value, sigFigs = 4) {
  const normalizedSigFigs = Math.max(Math.floor(Number(sigFigs) || 4), 1);
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  if (numeric === 0) return "0";
  const decimals = Math.max(normalizedSigFigs - Math.floor(Math.log10(Math.abs(numeric))) - 1, 0);
  return numeric.toFixed(decimals);
}

function showDebrief(passed) {
  setLeaderboardFormVisible(false);
  el.debriefPanel.classList.remove("hidden");
  el.debriefTitle.textContent = passed ? "Tutorial complete." : "Attempt ended.";
  el.debriefText.textContent = `${state.finalReason} Closest approach ${formatDistanceKm(
    state.closestKm,
  )}, delta-v ${formatSpeedMS(state.sim.dv)}.`;
  trackEvent(passed ? "tutorial_complete" : "tutorial_end", completionAnalyticsProps(passed));
}

function showArcadeDebrief() {
  if (!state.arcadeSession) return;
  const snap = state.arcadeSnapshot || state.arcadeSession.snapshot();
  const attempt = state.arcadeSession.attemptPacket({
    challengeRecord: ARCADE_CHALLENGE_RECORD,
    username: "LOCAL_PLAYER",
    client_build_hash: ARCADE_BUILD_ID,
  });
  const validation = validateAttemptPacket(attempt, ARCADE_CHALLENGE_RECORD);
  state.arcadeValidation = validation;
  state.arcadeAttemptPacket = attempt;
  prepareLeaderboardSubmission(validation);
  if ((snap.round_summaries || []).length > 0) {
    el.debriefPanel.classList.remove("hidden");
    el.debriefTitle.textContent = "Arcade run ended.";
    el.debriefText.textContent = `${snap.terminal_reason || "Attempt complete."} Total score ${Number(
      snap.score || 0,
    ).toLocaleString()} after ${(snap.round_summaries || []).length} cleared rounds. Final range ${formatDistanceKm(
      Number(snap.range_km || 0),
    )}, chaser delta-v ${formatSpeedMS(Number(snap.player_delta_v_m_s || 0))}. Local validation: ${validation.status}.`;
    trackEvent("arcade_attempt_complete", {
      result: "ended",
      validation: validation.status,
      score_bucket: snap.score > 0 ? "positive" : "zero",
    });
    return;
  }
  el.debriefPanel.classList.remove("hidden");
  const valid = validation.status === "valid";
  el.debriefTitle.textContent = snap.passed ? "Arcade rendezvous complete." : "Arcade attempt ended.";
  el.debriefText.textContent = `${snap.terminal_reason || "Attempt complete."} Local validation: ${
    validation.status
  }. Score ${validation.canonical_score || 0}, closest ${formatDistanceKm(
    validation.canonical_metrics?.closest_range_km || snap.closest_range_km || 0,
  )}, delta-v ${formatSpeedMS(
    Number(validation.canonical_metrics?.player_delta_v_m_s || snap.player_delta_v_m_s || 0),
  )}.${valid ? "" : ` ${validation.errors.join(" ")}`}`;
  trackEvent("arcade_attempt_complete", {
    result: snap.passed ? "success" : "ended",
    validation: validation.status,
    score_bucket: validation.canonical_score > 0 ? "positive" : "zero",
  });
}

function setLeaderboardFormVisible(visible, statusText = "") {
  if (!el.leaderboardForm) return;
  el.leaderboardForm.classList.toggle("hidden", !visible);
  el.leaderboardStatus.textContent = statusText;
  el.leaderboardSubmit.disabled = false;
}

function prepareLeaderboardSubmission(validation) {
  const canSubmit = ["valid", "suspicious"].includes(validation.status);
  setLeaderboardFormVisible(state.mode === "arcade", canSubmit ? "" : "This attempt cannot be submitted.");
  el.leaderboardSubmit.disabled = !canSubmit;
}

async function submitLeaderboardAttempt(event) {
  event.preventDefault();
  if (!state.arcadeAttemptPacket || !state.arcadeValidation) return;
  if (!["http:", "https:"].includes(window.location.protocol)) {
    el.leaderboardStatus.textContent = "Leaderboard submission is available after the hosted deploy.";
    return;
  }
  const username = String(el.leaderboardUsername.value || "LOCAL_PLAYER").trim() || "LOCAL_PLAYER";
  const email = String(el.leaderboardEmail.value || "").trim();
  const attempt = { ...state.arcadeAttemptPacket, username, email };
  el.leaderboardSubmit.disabled = true;
  el.leaderboardStatus.textContent = "Submitting...";
  try {
    const response = await fetch("/api/submit-attempt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, attempt }),
    });
    const payload = await response.json();
    if (!response.ok) {
      el.leaderboardStatus.textContent = payload.status
        ? `Rejected: ${payload.status}. ${(payload.errors || []).join(" ")}`
        : `Submission failed: ${payload.error || response.statusText}`;
      trackEvent("arcade_leaderboard_submit", { result: "rejected", status: payload.status || response.status });
      return;
    }
    const emailNote =
      email && payload.email_status === "sent"
        ? " Verification email sent."
        : email && payload.email_status === "not_configured"
          ? " Email receipts are not configured yet."
          : email && payload.email_status === "failed"
            ? ` Email failed: ${payload.email_error || "provider error"}.`
            : "";
    const ownershipNote =
      payload.ownership_status === "pending_verification"
        ? " Verify your email to reserve this username."
        : payload.ownership_status === "verified_owner"
          ? " Username verified."
          : payload.ownership_status === "locked"
            ? " Username reserved; attempt saved but leaderboard not updated."
            : "";
    el.leaderboardStatus.textContent = payload.leaderboard_updated
      ? `Submitted. Score ${Number(payload.score || 0).toLocaleString()} is on the leaderboard.`
      : `Submitted. Score ${Number(payload.score || 0).toLocaleString()} did not beat your best.`;
    el.leaderboardStatus.textContent += emailNote + ownershipNote;
    refreshLeaderboard({ force: true });
    trackEvent("arcade_leaderboard_submit", { result: "accepted", status: payload.status || "valid" });
  } catch (error) {
    el.leaderboardStatus.textContent = `Submission failed: ${error instanceof Error ? error.message : String(error)}`;
    trackEvent("arcade_leaderboard_submit", { result: "error" });
  } finally {
    el.leaderboardSubmit.disabled = false;
  }
}

function showArcadeRoundTransition() {
  const tr = state.arcadeTransition;
  if (!tr) return;
  setLeaderboardFormVisible(false);
  el.debriefPanel.classList.remove("hidden");
  el.debriefTitle.textContent = `Round ${tr.cleared_round_index} cleared.`;
  el.debriefText.textContent = `Round score ${tr.round_score.toLocaleString()}. Total score ${tr.total_score.toLocaleString()}. Bonus time ${Math.round(
    tr.bonus_time_s,
  )} s. Clear range ${formatDistanceKm(Number(tr.clear_range_km || 0))} against a ${formatDistanceKm(
    Number(tr.goal_range_km || 0),
  )} goal. Round ${tr.next_round_index}${tr.next_is_boss ? " is a boss round" : ""} starts with ${Math.round(
    tr.next_time_budget_s,
  )} s and a ${formatDistanceKm(tr.next_goal_range_km)} goal.`;
}

function completionAnalyticsProps(passed) {
  return {
    result: passed ? "success" : "ended",
    time_bucket: bucketSeconds(state.sim.t),
    dv_bucket: bucketDvMps(state.sim.dv),
    closest_range_bucket: bucketRangeKm(state.closestKm),
  };
}

function bucketSeconds(value) {
  if (value < 60) return "under_1min";
  if (value < 300) return "1_5min";
  if (value < 600) return "5_10min";
  if (value < 1200) return "10_20min";
  return "20min_plus";
}

function bucketDvMps(value) {
  if (value < 2) return "under_2mps";
  if (value < 5) return "2_5mps";
  if (value < 10) return "5_10mps";
  return "10mps_plus";
}

function bucketRangeKm(value) {
  if (value < 0.05) return "under_50m";
  if (value < 0.1) return "50_100m";
  if (value < 0.25) return "100_250m";
  if (value < 1) return "250m_1km";
  return "1km_plus";
}

function frame(nowMs) {
  if (!state.lastFrameMs) state.lastFrameMs = nowMs;
  const elapsedS = Math.min((nowMs - state.lastFrameMs) / 1000, 0.2);
  state.lastFrameMs = nowMs;
  if (state.mode === "selector") {
    updateDebugState();
    queueFrame(frame);
    return;
  }
  if (state.mode === "primer") {
    state.primerTimeS += elapsedS;
    updateGhost();
    draw();
    queueFrame(frame);
    return;
  }
  const controls = currentControls();
  applyManeuverSpeedLimit(controls);
  const shouldRun = simulationShouldRun();
  if (shouldRun && !state.passed) {
    state.stepAccumulatorS += elapsedS * currentSpeedMultiple();
  } else {
    state.stepAccumulatorS = 0;
  }
  let steps = 0;
  const stepDtS = currentStepDtS();
  while (state.stepAccumulatorS >= stepDtS && steps < MAX_STEPS_PER_FRAME) {
    step(stepDtS, shouldRun);
    state.stepAccumulatorS -= stepDtS;
    steps += 1;
  }
  if (steps >= MAX_STEPS_PER_FRAME) {
    state.stepAccumulatorS = 0;
  }
  updateGhost();
  draw();
  queueFrame(frame);
}

function queueFrame(callback) {
  if (typeof window.requestAnimationFrame === "function") {
    window.requestAnimationFrame(callback);
    return;
  }
  window.setTimeout(() => callback(Date.now()), 16);
}

function togglePause() {
  if (advancePrimer()) return;
  if (state.passed) return;
  if (state.mode === "arcade" && state.arcadeTransition && state.arcadeSession) {
    state.arcadeSession.continueNextRound();
    state.speedIndex = speedOptionIndex(1);
    state.cameraRuleMode = "full_trajectory";
    state.stepAccumulatorS = 0;
    state.arcadeTransition = null;
    el.debriefPanel.classList.add("hidden");
    syncArcadeSnapshot();
    state.running = true;
    updateMissionText();
    return;
  }
  if (state.mode === "arcade") {
    if (!state.running) state.running = true;
    updateMissionText();
    return;
  }
  state.running = !state.running;
  updateMissionText();
}

function toggleCameraRuleMode() {
  if (state.mode !== "sandbox" && state.mode !== "arcade") return;
  state.cameraRuleMode = state.cameraRuleMode === "full_trajectory" ? "current_pair" : "full_trajectory";
  updateGhost();
  draw();
}

function resetCurrent() {
  if (state.mode === "primer") {
    state.primerStage = 0;
    state.primerTimeS = 0;
    resetState(primerSample());
  } else if (state.mode === "tutorial") {
    state.activeStage = 0;
    state.speedIndex = 0;
    resetState(presets.behind);
  } else if (state.mode === "arcade") {
    startArcadeSession();
  } else {
    resetState(sandboxSeed());
  }
  state.running = false;
  updateMissionText();
}

function bindCommandButton(button, handler) {
  if (!button) return;
  let suppressClickUntil = 0;
  button.addEventListener("pointerdown", (event) => {
    if (typeof event.button === "number" && event.button !== 0) return;
    event.preventDefault();
    suppressClickUntil = Date.now() + 500;
    handler();
  });
  button.addEventListener("click", (event) => {
    if (Date.now() < suppressClickUntil) {
      event.preventDefault();
      return;
    }
    handler();
  });
}

function suppressMobileSelectionEvents() {
  const protectedSelectors = [
    ".touch-controls",
    ".mobile-speed-controls",
    ".hud-actions",
    ".plot-panel",
    "canvas",
  ].join(",");
  const shouldSuppress = (event) => {
    if (state.activeView !== "mobile") return false;
    if (isEditableControlTarget(event.target)) return false;
    return Boolean(event.target?.closest?.(protectedSelectors));
  };
  ["selectstart", "dragstart", "contextmenu"].forEach((type) => {
    document.addEventListener(
      type,
      (event) => {
        if (!shouldSuppress(event)) return;
        event.preventDefault();
      },
      { capture: true },
    );
  });
}

function bindEvents() {
  suppressMobileSelectionEvents();
  document.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (key === "escape") {
      event.preventDefault();
      if (state.mode !== "selector") showLevelSelector({ track: true, source: "keyboard" });
      return;
    }
    if (isEditableControlTarget(event.target)) return;
    if (
      [
        "w",
        "a",
        "s",
        "d",
        "arrowleft",
        "arrowright",
        "arrowup",
        "arrowdown",
        " ",
        "enter",
        "r",
        "m",
        "c",
        "v",
      ].includes(key)
    ) {
      event.preventDefault();
    }
    if (key === "v") {
      cycleViewPreference();
      return;
    }
    if (key === "m") {
      toggleMusic();
      return;
    }
    if (state.mode === "selector") {
      if (key === "arrowdown" || key === "s") selectLevel(state.selectedLevel + 1);
      else if (key === "arrowup" || key === "w") selectLevel(state.selectedLevel - 1);
      else if (key === " " || key === "enter") launchSelectedLevel("keyboard");
      return;
    }
    if (state.mode === "primer") {
      playMusicFromGesture();
      if (key === " " || key === "enter" || key === "arrowright") advancePrimer();
      else if (key === "r") resetCurrent();
      return;
    }
    if (key === "c" && (state.mode === "sandbox" || state.mode === "arcade")) {
      playMusicFromGesture();
      toggleCameraRuleMode();
      return;
    } else {
      playMusicFromGesture();
    }
    if (key === " ") togglePause();
    else if (key === "r") resetCurrent();
    else if (key === "arrowup") {
      state.speedIndex = Math.min(state.speedIndex + 1, SPEED_OPTIONS.length - 1);
      maybeCompleteSpeedStage();
    } else if (key === "arrowdown") {
      state.speedIndex = Math.max(state.speedIndex - 1, 0);
      maybeCompleteSpeedStage();
    } else {
      keys.add(key);
      refreshInputState();
    }
    if (key === "arrowup" || key === "arrowdown" || key === " " || key === "r") refreshInputState();
  });
  document.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if (keys.delete(key) || ["w", "a", "s", "d", "arrowleft", "arrowright"].includes(key)) {
      refreshInputState();
    }
  });
  document.querySelectorAll("[data-touch]").forEach((button) => {
    const value = button.dataset.touch;
    const start = (event) => {
      event.preventDefault();
      playMusicFromGesture();
      touch.add(value);
      refreshInputState();
    };
    const stop = () => {
      touch.delete(value);
      refreshInputState();
    };
    button.addEventListener("pointerdown", start);
    button.addEventListener("touchstart", start, { passive: false });
    button.addEventListener("pointerup", stop);
    button.addEventListener("pointerleave", stop);
    button.addEventListener("pointercancel", stop);
    button.addEventListener("touchend", stop);
    button.addEventListener("touchcancel", stop);
  });
  bindCommandButton(el.pauseButton, () => {
    playMusicFromGesture();
    togglePause();
  });
  bindCommandButton(el.resetButton, () => {
    playMusicFromGesture();
    resetCurrent();
  });
  bindCommandButton(el.levelSelectButton, () => {
    playMusicFromGesture();
    showLevelSelector({ track: true, source: "button" });
  });
  bindCommandButton(el.musicButton, toggleMusic);
  bindCommandButton(el.selectorMusicButton, toggleMusic);
  bindCommandButton(el.viewButton, () => {
    if (state.activeView === "mobile") {
      playMusicFromGesture();
      toggleCameraRuleMode();
      return;
    }
    cycleViewPreference();
  });
  bindCommandButton(el.selectorViewButton, cycleViewPreference);
  [el.riPanel, el.rcPanel].forEach((panel) => {
    if (!panel) return;
    let suppressClickUntil = 0;
    const toggle = (event) => {
      if (state.mode !== "sandbox" && state.mode !== "arcade") return;
      if (state.activeView === "mobile") return;
      event.preventDefault();
      event.stopPropagation();
      playMusicFromGesture();
      toggleCameraRuleMode();
    };
    const suppressFollowUp = (event) => {
      event.preventDefault();
      event.stopPropagation();
    };
    panel.addEventListener("pointerdown", (event) => {
      if (typeof event.button === "number" && event.button !== 0) return;
      suppressClickUntil = Date.now() + 700;
      toggle(event);
    });
    panel.addEventListener("mousedown", (event) => {
      if (typeof event.button === "number" && event.button !== 0) return;
      if (Date.now() < suppressClickUntil) {
        suppressFollowUp(event);
        return;
      }
      suppressClickUntil = Date.now() + 700;
      toggle(event);
    });
    panel.addEventListener("click", (event) => {
      if (Date.now() < suppressClickUntil) {
        suppressFollowUp(event);
        return;
      }
      toggle(event);
    });
  });
  el.mobileSpeedButtons.forEach((button) => {
    button.addEventListener("click", () => {
      playMusicFromGesture();
      state.speedIndex = speedOptionIndex(Number(button.dataset.mobileSpeed || currentSpeedMultiple()));
      maybeCompleteSpeedStage();
      refreshInputState();
    });
  });
  document.querySelectorAll("[data-level-option]").forEach((button) => {
    button.addEventListener("pointerenter", () => {
      const idx = levelOptions.findIndex((option) => option.id === button.dataset.levelOption);
      if (idx >= 0) selectLevel(idx);
    });
    button.addEventListener("focus", () => {
      const idx = levelOptions.findIndex((option) => option.id === button.dataset.levelOption);
      if (idx >= 0) selectLevel(idx);
    });
    button.addEventListener("click", () => {
      const idx = levelOptions.findIndex((option) => option.id === button.dataset.levelOption);
      if (idx >= 0) selectLevel(idx);
      launchSelectedLevel("selector_click");
    });
  });
  el.downloadLink.addEventListener("click", () => {
    trackEvent("download_click", { source: "debrief", mode: state.mode });
  });
  el.leaderboardForm.addEventListener("submit", submitLeaderboardAttempt);
  el.leaderboardRefresh.addEventListener("click", () => refreshLeaderboard({ force: true }));
  el.applySandbox.addEventListener("click", () => {
    playMusicFromGesture();
    resetState(sandboxSeed());
    state.running = false;
    updateMissionText();
  });
  el.randomSandbox.addEventListener("click", () => {
    playMusicFromGesture();
    resetState(randomSandboxSeed());
    state.running = false;
    updateMissionText();
  });
  window.addEventListener("resize", draw);
  const viewQuery = window.matchMedia("(max-width: 760px), (max-height: 620px)");
  const handleViewQueryChange = () => {
    if (state.viewPreference === "auto") applyViewPreference();
  };
  if (viewQuery.addEventListener) viewQuery.addEventListener("change", handleViewQueryChange);
  else if (viewQuery.addListener) viewQuery.addListener(handleViewQueryChange);
}

bindEvents();
initializeViewPreference();
initAnalytics();
trackEventOnce("preview_view", { build: BUILD_ID });
if (!launchInitialLevelFromUrl()) showLevelSelector();
queueFrame(frame);
