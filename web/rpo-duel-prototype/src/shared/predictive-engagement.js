export const PREDICTIVE_ENGAGEMENT_POLICY = deepFreeze({
  schema_version: "rpo-duel.computer-policy.v1",
  prediction_model: "HCW",
  horizon_s: 1800,
  step_s: 30,
  pulse_duration_s: 60,
  decision_interval_s: 120,
  capture_margin_m: 20,
  acceleration_fractions: [0.5, 1],
});

export function selectInterceptAction(stateRicSi, options = {}) {
  const state = validateState(stateRicSi);
  const config = validateOptions(options);
  const targetAction = options.target_acceleration_ric_m_s2 === undefined
    ? [0, 0, 0]
    : validateVector(options.target_acceleration_ric_m_s2, "target_acceleration_ric_m_s2");
  const actions = candidateActions(state, config.max_acceleration_m_s2, config.acceleration_fractions);
  const predictions = actions.map((action) => predict(
    state,
    subtract(action, targetAction),
    config,
  ));

  const passive = predictions[0];
  const passiveThreshold = Math.max(config.capture_radius_m - config.capture_margin_m, 0);
  if (passive.closest_range_m <= passiveThreshold) {
    return result(actions[0], passive, "passive_intercept_coast");
  }

  let capturing = null;
  for (let index = 0; index < predictions.length; index += 1) {
    const prediction = predictions[index];
    if (prediction.capture_time_s === null) continue;
    const key = [
      norm(actions[index]) * config.pulse_duration_s,
      prediction.capture_time_s || 0,
      prediction.closest_range_m,
    ];
    if (!capturing || compareKeys(key, capturing.key) < 0) capturing = { index, prediction, key };
  }
  if (capturing) {
    const action = actions[capturing.index];
    const phase = norm(action) > 0 ? "intercept_burn" : "intercept_coast";
    return result(action, capturing.prediction, phase);
  }

  let best = null;
  for (let index = 0; index < predictions.length; index += 1) {
    const prediction = predictions[index];
    const key = [
      prediction.closest_range_m,
      prediction.closest_time_s,
      norm(actions[index]) * config.pulse_duration_s,
    ];
    if (!best || compareKeys(key, best.key) < 0) best = { index, prediction, key };
  }
  const action = actions[best.index];
  const phase = norm(action) > 0 ? "intercept_search_burn" : "intercept_search_coast";
  return result(action, best.prediction, phase);
}

export function selectEvasionAction(stateRicSi, options = {}) {
  const state = validateState(stateRicSi);
  const config = validateOptions(options);
  const opponentMaximum = nonnegative(
    options.opponent_max_acceleration_m_s2,
    "opponent_max_acceleration_m_s2",
  );
  const targetActions = candidateActions(state, config.max_acceleration_m_s2, config.acceleration_fractions);
  const pursuerActions = candidateActions(
    state.map((value) => -value),
    opponentMaximum,
    config.acceleration_fractions,
  );

  let targetBest = null;
  for (let targetIndex = 0; targetIndex < targetActions.length; targetIndex += 1) {
    const targetAction = targetActions[targetIndex];
    let pursuerBest = null;
    for (const pursuerAction of pursuerActions) {
      const prediction = predict(state, subtract(targetAction, pursuerAction), config);
      const key = pursuerOutcomeKey(prediction);
      if (!pursuerBest || compareKeys(key, pursuerBest.key) < 0) pursuerBest = { prediction, key };
    }
    const key = evaderOutcomeKey(pursuerBest.prediction, targetAction, config.pulse_duration_s);
    if (!targetBest || compareKeys(key, targetBest.key) > 0) {
      targetBest = { targetIndex, prediction: pursuerBest.prediction, key };
    }
  }

  const action = targetActions[targetBest.targetIndex];
  const phase = norm(action) > 0 ? "predictive_evasion_burn" : "predictive_evasion_coast";
  return result(action, targetBest.prediction, phase);
}

function predict(state, relativeAcceleration, config) {
  let current = state.slice();
  let elapsed = 0;
  let closestRange = norm(current.slice(0, 3));
  let closestTime = 0;
  let captureTime = closestRange <= config.capture_radius_m ? 0 : null;
  while (elapsed < config.horizon_s - 1.0e-12) {
    let dt = Math.min(config.step_s, config.horizon_s - elapsed);
    if (elapsed < config.pulse_duration_s && config.pulse_duration_s < elapsed + dt) {
      dt = config.pulse_duration_s - elapsed;
    }
    const action = elapsed < config.pulse_duration_s - 1.0e-12
      ? relativeAcceleration
      : ZERO_VECTOR;
    current = propagateHcwConstantAcceleration(
      current,
      action,
      config.mean_motion_rad_s,
      dt,
    );
    elapsed += dt;
    const range = norm(current.slice(0, 3));
    if (range < closestRange) {
      closestRange = range;
      closestTime = elapsed;
    }
    if (captureTime === null && range <= config.capture_radius_m) captureTime = elapsed;
  }
  return {
    closest_range_m: closestRange,
    closest_time_s: closestTime,
    capture_time_s: captureTime,
  };
}

export function propagateHcwConstantAcceleration(stateRicSi, accelerationRicMps2, meanMotionRadS, dtS) {
  const state = validateState(stateRicSi);
  const acceleration = validateVector(accelerationRicMps2, "acceleration_ric_m_s2");
  const n = positive(meanMotionRadS, "mean_motion_rad_s");
  const dt = positive(dtS, "dt_s");
  const nt = n * dt;
  const cosine = Math.cos(nt);
  const sine = Math.sin(nt);
  const n2 = n * n;
  const [r, i, c, rd, id, cd] = state;
  const [ar, ai, ac] = acceleration;

  return [
    (4 - 3 * cosine) * r + sine / n * rd + 2 * (1 - cosine) / n * id
      + (1 - cosine) / n2 * ar + 2 * (nt - sine) / n2 * ai,
    6 * (sine - nt) * r + i - 2 * (1 - cosine) / n * rd + (4 * sine - 3 * nt) / n * id
      - 2 * (nt - sine) / n2 * ar + (4 * (1 - cosine) / n2 - 1.5 * dt * dt) * ai,
    cosine * c + sine / n * cd + (1 - cosine) / n2 * ac,
    3 * n * sine * r + cosine * rd + 2 * sine * id
      + sine / n * ar + 2 * (1 - cosine) / n * ai,
    -6 * n * (1 - cosine) * r - 2 * sine * rd + (4 * cosine - 3) * id
      - 2 * (1 - cosine) / n * ar + (4 * sine / n - 3 * dt) * ai,
    -n * sine * c + cosine * cd + sine / n * ac,
  ];
}

function candidateActions(state, maximum, fractions) {
  const actions = [[0, 0, 0]];
  if (maximum <= 0) return actions;
  const directions = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  ];
  for (const vector of [
    state.slice(0, 3),
    state.slice(3, 6),
    state.slice(0, 3).map((value, index) => value + state[index + 3] * 120),
  ]) {
    const magnitude = norm(vector);
    if (magnitude > 1.0e-12) {
      const unit = vector.map((value) => value / magnitude);
      directions.push(unit, unit.map((value) => -value));
    }
  }
  const unique = [];
  for (const direction of directions) {
    if (!unique.some((existing) => dot(direction, existing) > 1 - 1.0e-10)) unique.push(direction);
  }
  for (const fraction of fractions) {
    for (const direction of unique) {
      actions.push(direction.map((value) => maximum * fraction * value));
    }
  }
  return actions;
}

function pursuerOutcomeKey(prediction) {
  const captures = prediction.capture_time_s !== null;
  return [
    captures ? 0 : 1,
    captures ? prediction.capture_time_s : prediction.closest_range_m,
    prediction.closest_range_m,
  ];
}

function evaderOutcomeKey(prediction, action, pulseDurationS) {
  const survives = prediction.capture_time_s === null;
  return [
    survives ? 1 : 0,
    survives ? prediction.closest_range_m : prediction.capture_time_s || 0,
    prediction.closest_time_s,
    -norm(action) * pulseDurationS,
  ];
}

function result(action, prediction, phase) {
  return {
    acceleration_ric_m_s2: action.slice(),
    predicted_closest_range_m: prediction.closest_range_m,
    predicted_closest_time_s: prediction.closest_time_s,
    predicted_capture_time_s: prediction.capture_time_s,
    phase,
  };
}

function validateOptions(options) {
  const fractions = Array.from(options.acceleration_fractions || []);
  const config = {
    mean_motion_rad_s: positive(options.mean_motion_rad_s, "mean_motion_rad_s"),
    max_acceleration_m_s2: nonnegative(options.max_acceleration_m_s2, "max_acceleration_m_s2"),
    horizon_s: positive(options.horizon_s, "horizon_s"),
    step_s: positive(options.step_s, "step_s"),
    pulse_duration_s: positive(options.pulse_duration_s, "pulse_duration_s"),
    capture_radius_m: positive(options.capture_radius_m, "capture_radius_m"),
    capture_margin_m: nonnegative(options.capture_margin_m ?? 0, "capture_margin_m"),
    acceleration_fractions: fractions.map((value) => positive(value, "acceleration_fractions")),
  };
  if (config.step_s > config.pulse_duration_s) throw new Error("step_s must be no greater than pulse_duration_s");
  if (config.pulse_duration_s > config.horizon_s) throw new Error("pulse_duration_s must be no greater than horizon_s");
  if (config.capture_margin_m >= config.capture_radius_m) {
    throw new Error("capture_margin_m must be smaller than capture_radius_m");
  }
  if (!fractions.length || config.acceleration_fractions.some((value) => value > 1)) {
    throw new Error("acceleration_fractions must contain values in (0, 1]");
  }
  return config;
}

function validateState(value) {
  const state = Array.from(value || [], Number);
  if (state.length !== 6 || state.some((item) => !Number.isFinite(item))) {
    throw new Error("state_ric_si must contain six finite values");
  }
  return state;
}

function validateVector(value, name) {
  const vector = Array.from(value || [], Number);
  if (vector.length !== 3 || vector.some((item) => !Number.isFinite(item))) {
    throw new Error(`${name} must contain three finite values`);
  }
  return vector;
}

function positive(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) throw new Error(`${name} must be finite and positive`);
  return number;
}

function nonnegative(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) throw new Error(`${name} must be finite and nonnegative`);
  return number;
}

function compareKeys(first, second) {
  for (let index = 0; index < first.length; index += 1) {
    if (first[index] < second[index]) return -1;
    if (first[index] > second[index]) return 1;
  }
  return 0;
}

function subtract(first, second) {
  return first.map((value, index) => value - second[index]);
}

function dot(first, second) {
  return first.reduce((total, value, index) => total + value * second[index], 0);
}

function norm(values) {
  return Math.hypot(...values);
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.freeze(value);
  Object.values(value).forEach(deepFreeze);
  return value;
}

const ZERO_VECTOR = Object.freeze([0, 0, 0]);
