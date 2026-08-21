import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DUEL_CAMERA_MODES,
  duelPlotFrame,
  hcwCoastProjection,
  referenceRelativePair,
  toggleDuelCameraMode,
} from "../public/src/client/plot-model.js";

const relative = (r, i, c, rd = 0, id = 0, cd = 0) => ({
  r_km: r,
  i_km: i,
  c_km: c,
  rd_km_s: rd,
  id_km_s: id,
  cd_km_s: cd,
});

test("C camera toggle alternates reference and current-pair modes", () => {
  assert.equal(toggleDuelCameraMode(DUEL_CAMERA_MODES.REFERENCE), DUEL_CAMERA_MODES.CURRENT_PAIR);
  assert.equal(toggleDuelCameraMode(DUEL_CAMERA_MODES.CURRENT_PAIR), DUEL_CAMERA_MODES.REFERENCE);
});

test("reference-relative pair preserves target motion and supports legacy snapshots", () => {
  const target = relative(.2, -.1, .05, 0, .0001, 0);
  const chaser = relative(2, -4, .4, 0, .0002, 0);
  assert.deepEqual(referenceRelativePair({ target_reference_ric: target, chaser_reference_ric: chaser }), {
    target,
    chaser,
  });
  assert.deepEqual(referenceRelativePair({ relative_ric: relative(1, -2, .3) }).chaser, relative(1, -2, .3));
});

test("HCW coast projection starts at the supplied state and follows cross-track oscillation", () => {
  const n = .001;
  const initial = relative(0, 0, 2, 0, 0, 0);
  const projection = hcwCoastProjection(initial, {
    meanMotionRadS: n,
    horizonS: Math.PI / (2 * n),
    samples: 3,
  });
  assert.deepEqual(projection[0], { ...initial, t_s: 0 });
  assert.ok(Math.abs(projection.at(-1).c_km) < 1e-12);
  assert.ok(Math.abs(projection.at(-1).cd_km_s + .002) < 1e-12);
});

test("reference camera centers the target reference orbit and shows both HCW projections", () => {
  const round = {
    time_remaining_s: 1000,
    reference_mean_motion_rad_s: .001,
    target_reference_ric: relative(.4, -.2, .1, 0, .0001, 0),
    chaser_reference_ric: relative(4, -6, .8, 0, .0002, 0),
  };
  const trail = [{ target: relative(.3, -.1, .1), chaser: relative(3, -5, .7) }];
  const frame = duelPlotFrame(round, trail, DUEL_CAMERA_MODES.REFERENCE);
  assert.deepEqual(frame.cameraCenter, { r_km: 0, i_km: 0, c_km: 0 });
  assert.equal(frame.targetTrail.length, 1);
  assert.equal(frame.chaserTrail.length, 1);
  assert.equal(frame.targetProjection.length, 121);
  assert.equal(frame.chaserProjection.length, 121);
});

test("current-pair camera centers the satellite midpoint and suppresses trails and projections", () => {
  const round = {
    time_remaining_s: 1000,
    reference_mean_motion_rad_s: .001,
    target_reference_ric: relative(2, -4, .5),
    chaser_reference_ric: relative(6, 2, -.5),
  };
  const frame = duelPlotFrame(round, [{ target: relative(1, 1, 1), chaser: relative(2, 2, 2) }], DUEL_CAMERA_MODES.CURRENT_PAIR);
  assert.deepEqual(frame.cameraCenter, { r_km: 4, i_km: -1, c_km: 0 });
  assert.deepEqual(frame.targetTrail, []);
  assert.deepEqual(frame.chaserTrail, []);
  assert.deepEqual(frame.targetProjection, []);
  assert.deepEqual(frame.chaserProjection, []);
});
