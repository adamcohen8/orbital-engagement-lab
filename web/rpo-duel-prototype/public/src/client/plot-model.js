export const DUEL_CAMERA_MODES = Object.freeze({
  REFERENCE: "reference",
  CURRENT_PAIR: "current_pair",
  CURRENT_PROJECTIONS: "current_projections",
});

export const DUEL_VISUAL_TIMING = Object.freeze({
  render_delay_ms: 120,
  max_interpolation_gap_ms: 250,
  camera_smoothing_ms: 180,
});

export function captureRingStyle(playerRole) {
  return playerRole === "chaser"
    ? { fill: "rgba(150,235,170,.10)", stroke: "rgba(150,235,170,.82)" }
    : { fill: "rgba(245,92,92,.08)", stroke: "rgba(245,92,92,.72)" };
}

export function toggleDuelCameraMode(mode) {
  if (mode === DUEL_CAMERA_MODES.REFERENCE) return DUEL_CAMERA_MODES.CURRENT_PAIR;
  if (mode === DUEL_CAMERA_MODES.CURRENT_PAIR) return DUEL_CAMERA_MODES.CURRENT_PROJECTIONS;
  return DUEL_CAMERA_MODES.REFERENCE;
}

export function referenceRelativePair(round = {}) {
  const zero = relativeState();
  const target = normalizeRelative(round.target_reference_ric || zero);
  const chaser = normalizeRelative(round.chaser_reference_ric || addRelative(target, round.relative_ric));
  return { target, chaser };
}

export function interpolateDuelRound(previous = {}, current = {}, alpha = 1) {
  const amount = Math.max(0, Math.min(1, Number(alpha) || 0));
  const interpolated = { ...current };
  for (const key of ["target_reference_ric", "chaser_reference_ric", "relative_ric"]) {
    if (previous[key] && current[key]) interpolated[key] = interpolateRelative(previous[key], current[key], amount);
  }
  for (const key of ["tick", "time_s", "time_remaining_s", "range_km", "relative_speed_km_s"]) {
    if (Number.isFinite(previous[key]) && Number.isFinite(current[key])) {
      interpolated[key] = previous[key] + (current[key] - previous[key]) * amount;
    }
  }
  return interpolated;
}

export function hcwCoastProjection(initialState, {
  meanMotionRadS,
  horizonS,
  samples = 121,
} = {}) {
  const n = Number(meanMotionRadS);
  const horizon = Math.max(0, Number(horizonS) || 0);
  const count = Math.max(2, Math.floor(Number(samples) || 0));
  if (!Number.isFinite(n) || n <= 0 || horizon <= 0) return [];
  const initial = normalizeRelative(initialState);
  return Array.from({ length: count }, (_, index) => {
    const t = horizon * index / (count - 1);
    const nt = n * t;
    const cosine = Math.cos(nt);
    const sine = Math.sin(nt);
    return {
      r_km: (4 - 3 * cosine) * initial.r_km
        + sine / n * initial.rd_km_s
        + 2 * (1 - cosine) / n * initial.id_km_s,
      i_km: 6 * (sine - nt) * initial.r_km
        + initial.i_km
        - 2 * (1 - cosine) / n * initial.rd_km_s
        + (4 * sine - 3 * nt) / n * initial.id_km_s,
      c_km: cosine * initial.c_km + sine / n * initial.cd_km_s,
      rd_km_s: 3 * n * sine * initial.r_km
        + cosine * initial.rd_km_s
        + 2 * sine * initial.id_km_s,
      id_km_s: 6 * n * (cosine - 1) * initial.r_km
        - 2 * sine * initial.rd_km_s
        + (4 * cosine - 3) * initial.id_km_s,
      cd_km_s: -n * sine * initial.c_km + cosine * initial.cd_km_s,
      t_s: t,
    };
  });
}

export function duelPlotFrame(round, trail = [], cameraMode = DUEL_CAMERA_MODES.REFERENCE) {
  const current = referenceRelativePair(round);
  const origin = relativeState();
  const pairMode = cameraMode === DUEL_CAMERA_MODES.CURRENT_PAIR;
  const projectionMode = cameraMode === DUEL_CAMERA_MODES.CURRENT_PROJECTIONS;
  const meanMotion = Number(round?.reference_mean_motion_rad_s);
  const horizon = Number.isFinite(meanMotion) && meanMotion > 0
    ? 2 * Math.PI / meanMotion
    : 0;
  const targetTrail = pairMode ? [] : trail.map((sample) => sample.target).filter(Boolean);
  const chaserTrail = pairMode ? [] : trail.map((sample) => sample.chaser).filter(Boolean);
  const targetProjection = hcwCoastProjection(current.target, {
    meanMotionRadS: meanMotion,
    horizonS: horizon,
  });
  const chaserProjection = hcwCoastProjection(current.chaser, {
    meanMotionRadS: meanMotion,
    horizonS: horizon,
  });
  const projectionFramingPoints = [
    ...targetProjection,
    ...chaserProjection,
    current.target,
    current.chaser,
  ];
  const referenceFramingPoints = [
    origin,
    ...projectionFramingPoints,
    ...targetTrail,
    ...chaserTrail,
  ];
  const cameraCenter = pairMode
    ? midpointPosition(current.target, current.chaser)
    : projectionMode
      ? boundingCenter(projectionFramingPoints)
      : boundingCenter(referenceFramingPoints);
  return {
    cameraMode: pairMode
      ? DUEL_CAMERA_MODES.CURRENT_PAIR
      : projectionMode
        ? DUEL_CAMERA_MODES.CURRENT_PROJECTIONS
        : DUEL_CAMERA_MODES.REFERENCE,
    cameraCenter,
    target: current.target,
    chaser: current.chaser,
    targetTrail: pairMode || projectionMode ? [] : targetTrail,
    chaserTrail: pairMode || projectionMode ? [] : chaserTrail,
    targetProjection,
    chaserProjection,
    framingPoints: pairMode
      ? [current.target, current.chaser]
      : projectionMode
        ? projectionFramingPoints
        : referenceFramingPoints,
  };
}

export function duelPlotSpan(frame, xKey, yKey, captureRadiusKm = .1) {
  const floatingMode = frame.cameraMode !== DUEL_CAMERA_MODES.REFERENCE;
  const centerX = frame.cameraCenter[xKey];
  const centerY = frame.cameraCenter[yKey];
  const captureRadius = Math.max(0, Number(captureRadiusKm) || 0);
  const extent = Math.max(
    floatingMode ? .12 : 1,
    ...frame.framingPoints.flatMap((sample) => [
      Math.abs((sample?.[xKey] || 0) - centerX),
      Math.abs((sample?.[yKey] || 0) - centerY),
    ]),
    Math.abs(frame.target[xKey] - centerX) + captureRadius,
    Math.abs(frame.target[yKey] - centerY) + captureRadius,
  );
  const padded = extent * 1.22;
  return padded;
}

function midpointPosition(first, second) {
  return {
    r_km: (first.r_km + second.r_km) / 2,
    i_km: (first.i_km + second.i_km) / 2,
    c_km: (first.c_km + second.c_km) / 2,
  };
}

function boundingCenter(points) {
  const center = {};
  for (const key of ["r_km", "i_km", "c_km"]) {
    const values = points.map((point) => finite(point?.[key]));
    center[key] = (Math.min(...values) + Math.max(...values)) / 2;
  }
  return center;
}

function interpolateRelative(previous, current, alpha) {
  const first = normalizeRelative(previous);
  const second = normalizeRelative(current);
  return relativeState(
    lerp(first.r_km, second.r_km, alpha),
    lerp(first.i_km, second.i_km, alpha),
    lerp(first.c_km, second.c_km, alpha),
    lerp(first.rd_km_s, second.rd_km_s, alpha),
    lerp(first.id_km_s, second.id_km_s, alpha),
    lerp(first.cd_km_s, second.cd_km_s, alpha),
  );
}

function normalizeRelative(value = {}) {
  return relativeState(
    value.r_km,
    value.i_km,
    value.c_km,
    value.rd_km_s,
    value.id_km_s,
    value.cd_km_s,
  );
}

function addRelative(first, second = {}) {
  const relative = normalizeRelative(second);
  return relativeState(
    first.r_km + relative.r_km,
    first.i_km + relative.i_km,
    first.c_km + relative.c_km,
    first.rd_km_s + relative.rd_km_s,
    first.id_km_s + relative.id_km_s,
    first.cd_km_s + relative.cd_km_s,
  );
}

function relativeState(r = 0, i = 0, c = 0, rd = 0, id = 0, cd = 0) {
  return {
    r_km: finite(r),
    i_km: finite(i),
    c_km: finite(c),
    rd_km_s: finite(rd),
    id_km_s: finite(id),
    cd_km_s: finite(cd),
  };
}

function finite(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function lerp(first, second, alpha) {
  return first + (second - first) * alpha;
}
