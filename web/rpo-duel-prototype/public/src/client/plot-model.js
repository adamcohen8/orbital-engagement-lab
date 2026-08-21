export const DUEL_CAMERA_MODES = Object.freeze({
  REFERENCE: "reference",
  CURRENT_PAIR: "current_pair",
});

export function toggleDuelCameraMode(mode) {
  return mode === DUEL_CAMERA_MODES.CURRENT_PAIR
    ? DUEL_CAMERA_MODES.REFERENCE
    : DUEL_CAMERA_MODES.CURRENT_PAIR;
}

export function referenceRelativePair(round = {}) {
  const zero = relativeState();
  const target = normalizeRelative(round.target_reference_ric || zero);
  const chaser = normalizeRelative(round.chaser_reference_ric || addRelative(target, round.relative_ric));
  return { target, chaser };
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
  const pairMode = cameraMode === DUEL_CAMERA_MODES.CURRENT_PAIR;
  const cameraCenter = pairMode
    ? midpointPosition(current.target, current.chaser)
    : { r_km: 0, i_km: 0, c_km: 0 };
  const meanMotion = Number(round?.reference_mean_motion_rad_s);
  const horizon = Math.min(
    Math.max(0, Number(round?.time_remaining_s) || 0),
    Number.isFinite(meanMotion) && meanMotion > 0 ? Math.PI / meanMotion : 0,
  );
  return {
    cameraMode: pairMode ? DUEL_CAMERA_MODES.CURRENT_PAIR : DUEL_CAMERA_MODES.REFERENCE,
    cameraCenter,
    target: current.target,
    chaser: current.chaser,
    targetTrail: pairMode ? [] : trail.map((sample) => sample.target).filter(Boolean),
    chaserTrail: pairMode ? [] : trail.map((sample) => sample.chaser).filter(Boolean),
    targetProjection: pairMode ? [] : hcwCoastProjection(current.target, {
      meanMotionRadS: meanMotion,
      horizonS: horizon,
    }),
    chaserProjection: pairMode ? [] : hcwCoastProjection(current.chaser, {
      meanMotionRadS: meanMotion,
      horizonS: horizon,
    }),
  };
}

function midpointPosition(first, second) {
  return {
    r_km: (first.r_km + second.r_km) / 2,
    i_km: (first.i_km + second.i_km) / 2,
    c_km: (first.c_km + second.c_km) / 2,
  };
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
