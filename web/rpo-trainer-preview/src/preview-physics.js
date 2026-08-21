export const PREVIEW_MU_KM3_S2 = 398600.4418;
export const PREVIEW_TARGET_A_KM = 7000.0;
export const PREVIEW_MEAN_MOTION_RAD_S = Math.sqrt(PREVIEW_MU_KM3_S2 / PREVIEW_TARGET_A_KM ** 3);
export const PREVIEW_MAX_ACCEL_KM_S2 = 1.0e-5;
export const PREVIEW_FIXED_DT_S = 0.1;

export function stepHcwStateInPlace(
  state,
  controls = { r: 0, i: 0, c: 0 },
  dtS = PREVIEW_FIXED_DT_S,
  options = {},
) {
  const n = Number(options.mean_motion_rad_s ?? PREVIEW_MEAN_MOTION_RAD_S);
  const maxAccel = Number(options.max_accel_km_s2 ?? PREVIEW_MAX_ACCEL_KM_S2);
  const dt = Number(dtS);
  if (!Number.isFinite(n) || n <= 0) throw new Error("mean_motion_rad_s must be positive and finite.");
  if (!Number.isFinite(maxAccel) || maxAccel < 0) throw new Error("max_accel_km_s2 must be finite and nonnegative.");
  if (!Number.isFinite(dt) || dt <= 0) throw new Error("dt_s must be positive and finite.");

  const ar = Number(controls.r || 0) * maxAccel;
  const ai = Number(controls.i || 0) * maxAccel;
  const ac = Number(controls.c || 0) * maxAccel;
  const rdd = 3 * n * n * state.r + 2 * n * state.id + ar;
  const idd = -2 * n * state.rd + ai;
  const cdd = -n * n * state.c + ac;
  state.rd += rdd * dt;
  state.id += idd * dt;
  state.cd += cdd * dt;
  state.r += state.rd * dt;
  state.i += state.id * dt;
  state.c += state.cd * dt;
  if (Number.isFinite(state.t)) state.t += dt;
  return state;
}
export function propagateHcwState(initialState, durationS, dtS = PREVIEW_FIXED_DT_S) {
  const duration = Number(durationS);
  const dt = Number(dtS);
  if (!Number.isFinite(duration) || duration < 0) throw new Error("duration_s must be finite and nonnegative.");
  if (!Number.isFinite(dt) || dt <= 0) throw new Error("dt_s must be positive and finite.");
  const steps = Math.round(duration / dt);
  if (Math.abs(steps * dt - duration) > 1.0e-9) throw new Error("duration_s must be divisible by dt_s.");
  const state = {
    r: Number(initialState.r),
    i: Number(initialState.i),
    c: Number(initialState.c),
    rd: Number(initialState.rd),
    id: Number(initialState.id),
    cd: Number(initialState.cd),
    t: Number(initialState.t || 0),
  };
  for (let idx = 0; idx < steps; idx += 1) stepHcwStateInPlace(state, { r: 0, i: 0, c: 0 }, dt);
  return state;
}
