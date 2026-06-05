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
const BUILD_ID = "hcw-sandbox-2026-06-05";
const PRIMER_AMPLITUDES_KM = { r: 0.65, i: 0.75, c: 0.65 };
const MUSIC_TRACKS = {
  selector: "./assets/01_insert_coin_to_orbit.wav",
  tutorial: "./assets/10_training_grid_sunrise.wav",
  sandbox: "./assets/04_docking_bay_neon.wav",
};

const el = {
  shell: document.querySelector(".trainer-shell"),
  levelSelector: document.querySelector("#levelSelector"),
  selectorMusicButton: document.querySelector("#selectorMusicButton"),
  selectorPreviewTitle: document.querySelector("#selectorPreviewTitle"),
  selectorPreviewBudget: document.querySelector("#selectorPreviewBudget"),
  selectorPreviewObjective: document.querySelector("#selectorPreviewObjective"),
  selectorPreviewBrief: document.querySelector("#selectorPreviewBrief"),
  selectorPreviewCriteria: document.querySelector("#selectorPreviewCriteria"),
  selectorPreviewNotes: document.querySelector("#selectorPreviewNotes"),
  tutorialMode: document.querySelector("#tutorialMode"),
  sandboxMode: document.querySelector("#sandboxMode"),
  pauseButton: document.querySelector("#pauseButton"),
  resetButton: document.querySelector("#resetButton"),
  levelSelectButton: document.querySelector("#levelSelectButton"),
  musicButton: document.querySelector("#musicButton"),
  modeLabel: document.querySelector("#modeLabel"),
  objectiveTitle: document.querySelector("#objectiveTitle"),
  objectiveText: document.querySelector("#objectiveText"),
  riTitle: document.querySelector("#riTitle"),
  riSubtitle: document.querySelector("#riSubtitle"),
  rcTitle: document.querySelector("#rcTitle"),
  rcSubtitle: document.querySelector("#rcSubtitle"),
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
};

const levelOptions = [
  {
    id: "tutorial",
    mode: "primer",
    title: "Level 0 - Tutorial",
    budget: "Time: 18000s   Chaser dV: 12.000 m/s   Speed Gate: 0.300 m/s",
    objective:
      "Learn what R, I, and C mean by creating six small target orbits, then use short pulse-and-coast translations to settle near a passive target.",
    brief:
      "The red satellite is you. R is radial, I is in-track, and C is cross-track. The simulation pauses for each guided stage until you hold the requested control.",
    criteria: [
      "Complete the +I and -I guided orbit demonstrations.",
      "After +I, increase the speed multiple to 10x.",
      "Complete the +R and -R guided orbit demonstrations.",
      "Complete the +C and -C guided orbit demonstrations.",
      "Get within 250 m of the passive target below 0.3 m/s.",
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
    text: "Use small pulses and coast into the green 250 m circle below 0.3 m/s.",
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
};

const music = createMusicPlayer(MUSIC_TRACKS.selector);
music.loop = true;
music.volume = 0.65;

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
  state.stageStart = { ...state.sim };
  state.closestKm = rangeKm();
  state.stageDv = 0;
  state.passed = false;
  state.finalReason = "";
  state.stepAccumulatorS = 0;
  el.debriefPanel.classList.add("hidden");
  updateGhost();
  draw();
}

function showLevelSelector() {
  state.mode = "selector";
  state.running = false;
  state.passed = false;
  state.stepAccumulatorS = 0;
  keys.clear();
  touch.clear();
  el.debriefPanel.classList.add("hidden");
  el.shell.classList.add("selector-mode");
  el.shell.classList.remove("primer-mode");
  setMusicTrackForMode("selector");
  renderLevelSelector();
  syncMusicButton();
  updateDebugState();
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
}

function replaceList(listEl, items) {
  listEl.replaceChildren();
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    listEl.appendChild(li);
  });
}

function selectLevel(index) {
  state.selectedLevel = Math.max(0, Math.min(index, levelOptions.length - 1));
  renderLevelSelector();
}

function launchSelectedLevel() {
  const option = levelOptions[state.selectedLevel] || levelOptions[0];
  setMode(option.mode);
  playMusicFromGesture();
}

function setMode(mode) {
  state.mode = mode;
  el.shell.classList.remove("selector-mode");
  state.running = false;
  state.speedIndex = 0;
  state.cameraRuleMode = mode === "sandbox" ? "full_trajectory" : "default";
  setMusicTrackForMode(mode === "sandbox" ? "sandbox" : "tutorial");
  state.activeStage = 0;
  state.stageStart = null;
  state.stageDv = 0;
  state.passed = false;
  state.primerTimeS = 0;
  if (mode === "primer") state.primerStage = 0;
  el.tutorialMode.classList.toggle("active", mode === "tutorial" || mode === "primer");
  el.sandboxMode.classList.toggle("active", mode === "sandbox");
  const seed = mode === "sandbox" ? sandboxSeed() : mode === "primer" ? primerSample() : presets.behind;
  resetState(seed);
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

function step(dt, forceRun = false) {
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
  return state.running || tutorialInputMatches();
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
  return Math.hypot(state.sim.r, state.sim.i, state.sim.c);
}

function relativeSpeedKmS() {
  return Math.hypot(state.sim.rd, state.sim.id, state.sim.cd);
}

function updateMissionText() {
  if (state.mode === "selector") {
    el.shell.classList.add("selector-mode");
    renderLevelSelector();
    return;
  }
  el.shell.classList.remove("selector-mode");
  el.shell.classList.toggle("primer-mode", state.mode === "primer");
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
  } else {
    el.levelLabel.textContent = "LEVEL 0 - TUTORIAL";
    const stage = tutorialStages[state.activeStage] || tutorialStages[tutorialStages.length - 1];
    el.objectiveTitle.textContent = stage.title;
    if (el.objectiveText) {
      el.objectiveText.textContent = "";
    }
  }
  el.pauseButton.textContent = state.mode === "primer" ? primerAdvanceLabel() : state.running ? "Pause" : "Start";
  el.resetButton.textContent = state.mode === "primer" ? "Replay" : "Reset";
  el.sandboxPanel.classList.toggle("hidden", state.mode !== "sandbox" || state.running);
  updatePlotTitles();
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
  el.riSubtitle.textContent = "In-track vs radial";
  el.rcTitle.textContent = "RC Plane";
  el.rcSubtitle.textContent = "Cross-track vs radial";
}

function syncMusicButton() {
  el.musicButton.textContent = state.musicEnabled ? "M Music: ON" : "M Music: OFF";
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

function toggleMusic() {
  if (state.musicEnabled && music.paused && !state.musicStartRequested) {
    playMusicFromGesture();
    syncMusicButton();
    return;
  }
  state.musicEnabled = !state.musicEnabled;
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
    speedMultiple: currentSpeedMultiple(),
    cameraRuleMode: state.cameraRuleMode,
    musicSrc: music.currentSrc || music.src,
    controls: currentControls(),
    sim: { ...state.sim },
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
    el.rMeter.value = 0;
    el.iMeter.value = 0;
    el.cMeter.value = 0;
    updateMissionText();
    return;
  }
  const rangeText = `${rangeKm().toFixed(3)} km`;
  const speedText = `${(relativeSpeedKmS() * 1000).toFixed(3)} m/s`;
  const dvText = `${state.sim.dv.toFixed(2)} m/s`;
  const timeText = `${Math.round(state.sim.t)} s`;
  el.rangeMetric.textContent = rangeText;
  el.speedMetric.textContent = speedText;
  el.dvMetric.textContent = dvText;
  el.timeMetric.textContent = timeText;
  el.topRangeMetric.textContent = `INFO Range ${rangeText}`;
  el.topSpeedMetric.textContent = `INFO Rel Speed ${speedText}`;
  el.topDvMetric.textContent = `INFO Delta-v ${dvText}`;
  el.hudLine.textContent = `T=${state.sim.t.toFixed(1).padStart(7, " ")}s   Range=${rangeText}   Rel Speed=${speedText}`;
  el.coachHint.textContent = currentCoachHint();
  el.commandLine.textContent = commandStatusLine();
  el.footerLine.textContent = `Speed ${SPEED_OPTIONS[state.speedIndex].toFixed(
    0,
  )}x   Up/Down Speed   Space Pause   . Step   R Reset   Esc Level Select`;
  el.speedMultiple.textContent = `${SPEED_OPTIONS[state.speedIndex]}x`;
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
    return `Use small pulses, then coast and watch the target-centered RIC motion. C Camera: ${label}.`;
  }
  const stage = tutorialStages[state.activeStage] || tutorialStages[tutorialStages.length - 1];
  if (stage.final) {
    return "Guided burns complete. Settle gently into the green 250 m circle. Keep pulses short.";
  }
  if (stage.speedTarget) {
    return `Want to go faster? Hit the up arrow key to increase the speed multiple. Current speed: ${SPEED_OPTIONS[state.speedIndex]}x.`;
  }
  const progress = Math.min(state.stageDv, stage.targetDv || 0);
  return `${stage.text} Burn progress: ${progress.toFixed(2)}/${(stage.targetDv || 0).toFixed(2)} m/s.`;
}

function commandStatusLine() {
  if (state.mode === "primer") {
    return "";
  }
  const u = currentControls();
  const simState = state.running ? "RUNNING" : "PAUSED";
  const camera = state.mode === "sandbox" ? "  C Camera" : "";
  return `W/S Radial +/-R  A/D In-Track +/-I  Left/Right Cross-Track +/-C${camera}  M Music   ${simState}  R=${u.r.toFixed(
    0,
  )} I=${u.i.toFixed(0)} C=${u.c.toFixed(0)} Throttle=1.00`;
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

  drawGrid(ctx, width, height, scale);
  drawRings(ctx, toPx, scale, xAxis, yAxis);
  drawPath(ctx, state.tutorialTargetPath, toPx, "rgba(92, 240, 132, 0.92)", true, 3);
  drawPath(ctx, state.ghost, toPx, "rgba(135, 150, 172, 0.95)", true, 2);
  drawPath(ctx, state.trail, toPx, "rgba(215, 86, 86, 0.95)", false);

  const target = toPx({ r: 0, i: 0, c: 0 });
  const chaser = toPx(state.sim);
  drawVector(ctx, chaser, state.sim, xAxis, yAxis, "velocity");
  drawThrustVector(ctx, chaser, xAxis, yAxis);
  ctx.fillStyle = "#f5cd5c";
  ctx.beginPath();
  ctx.arc(target.x, target.y, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#f55c5c";
  ctx.beginPath();
  ctx.arc(chaser.x, chaser.y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(245, 235, 242, 0.6)";
  ctx.lineWidth = 1;
  ctx.stroke();
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

  ctx.fillStyle = "#f5cd5c";
  ctx.beginPath();
  ctx.arc(target.x, target.y, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#f55c5c";
  ctx.beginPath();
  ctx.arc(chaser.x, chaser.y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(245, 235, 242, 0.72)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = "rgba(245, 205, 92, 0.92)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText("Target", target.x + 10, target.y - 10);
  ctx.fillStyle = "rgba(245, 92, 92, 0.95)";
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
    drawEciCircle(ctx, center, orbitScale * chaserRadius, "rgba(245, 92, 92, 0.66)");
  } else {
    drawEciCircle(ctx, center, orbitScale, "rgba(245, 92, 92, 0.34)", true);
  }
  const target = projectEciCircular(1, targetTheta, center, orbitScale);
  const chaser = projectEciCircular(chaserRadius, chaserTheta, center, orbitScale);
  drawSatellite(ctx, target, "#f5cd5c", "Target", -56, 22);
  drawSatellite(ctx, chaser, "#f55c5c", "Chaser", 10, -14);

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
  drawOrbitLine(ctx, chaserLine, "rgba(245, 92, 92, 0.78)");
  drawSatellite(ctx, target, "#f5cd5c", "Target", -60, 24);
  drawSatellite(ctx, chaser, "#f55c5c", "Chaser", 12, -12);

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

function drawSatellite(ctx, point, color, label, labelOffsetX, labelOffsetY) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point.x, point.y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(245, 235, 242, 0.78)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = "rgba(230, 235, 242, 0.95)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText(label, point.x + labelOffsetX, point.y + labelOffsetY);
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
  const values =
    state.mode === "sandbox" && state.cameraRuleMode === "current_pair"
      ? [state.sim, { r: 0, i: 0, c: 0 }]
      : [...state.trail, ...state.ghost, ...state.tutorialTargetPath, { r: 0, i: 0, c: 0 }];
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

function drawRings(ctx, toPx, scale, xAxis, yAxis) {
  const target = toPx({ r: 0, i: 0, c: 0 });
  ctx.strokeStyle = "rgba(190, 68, 68, 0.72)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(target.x, target.y, 0.025 * scale, 0, Math.PI * 2);
  ctx.stroke();
  if (state.mode === "tutorial") {
    ctx.strokeStyle = "rgba(78, 178, 112, 0.86)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(target.x, target.y, 0.25 * scale, 0, Math.PI * 2);
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

function showDebrief(passed) {
  el.debriefPanel.classList.remove("hidden");
  el.debriefTitle.textContent = passed ? "Tutorial complete." : "Attempt ended.";
  el.debriefText.textContent = `${state.finalReason} Closest approach ${state.closestKm.toFixed(
    3,
  )} km, delta-v ${state.sim.dv.toFixed(2)} m/s.`;
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
  while (state.stepAccumulatorS >= FIXED_DT_S && steps < MAX_STEPS_PER_FRAME) {
    step(FIXED_DT_S, shouldRun);
    state.stepAccumulatorS -= FIXED_DT_S;
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
  state.running = !state.running;
  updateMissionText();
}

function toggleCameraRuleMode() {
  if (state.mode !== "sandbox") return;
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
  } else {
    resetState(sandboxSeed());
  }
  state.running = false;
  updateMissionText();
}

function bindEvents() {
  document.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (key === "escape") {
      event.preventDefault();
      if (state.mode !== "selector") showLevelSelector();
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
      ].includes(key)
    ) {
      event.preventDefault();
    }
    if (key === "m") {
      toggleMusic();
      return;
    }
    if (state.mode === "selector") {
      if (key === "arrowdown" || key === "s") selectLevel(state.selectedLevel + 1);
      else if (key === "arrowup" || key === "w") selectLevel(state.selectedLevel - 1);
      else if (key === " " || key === "enter") launchSelectedLevel();
      return;
    }
    if (state.mode === "primer") {
      playMusicFromGesture();
      if (key === " " || key === "enter" || key === "arrowright") advancePrimer();
      else if (key === "r") resetCurrent();
      return;
    }
    if (key === "c" && state.mode === "sandbox") {
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
    button.addEventListener("pointerup", stop);
    button.addEventListener("pointerleave", stop);
    button.addEventListener("pointercancel", stop);
  });
  el.pauseButton.addEventListener("click", () => {
    playMusicFromGesture();
    togglePause();
  });
  el.resetButton.addEventListener("click", () => {
    playMusicFromGesture();
    resetCurrent();
  });
  el.levelSelectButton.addEventListener("click", () => {
    playMusicFromGesture();
    showLevelSelector();
  });
  el.tutorialMode.addEventListener("click", () => {
    setMode("primer");
    playMusicFromGesture();
  });
  el.sandboxMode.addEventListener("click", () => {
    setMode("sandbox");
    playMusicFromGesture();
  });
  el.musicButton.addEventListener("click", toggleMusic);
  el.selectorMusicButton.addEventListener("click", toggleMusic);
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
      launchSelectedLevel();
    });
  });
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
}

bindEvents();
showLevelSelector();
queueFrame(frame);
