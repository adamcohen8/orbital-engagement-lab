import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DUEL_CAMERA_MODES,
  DUEL_VISUAL_TIMING,
  captureRingStyle,
  duelPlotFrame,
  duelPlotSpan,
  hcwCoastProjection,
  interpolateDuelRound,
  referenceRelativePair,
  toggleDuelCameraMode,
} from "../public/src/client/plot-model.js";

test("capture ring is green for the Chaser and red for the Target", () => {
  assert.deepEqual(captureRingStyle("chaser"), {
    fill: "rgba(150,235,170,.10)",
    stroke: "rgba(150,235,170,.82)",
  });
  assert.deepEqual(captureRingStyle("target"), {
    fill: "rgba(245,92,92,.08)",
    stroke: "rgba(245,92,92,.72)",
  });
});

const relative = (r, i, c, rd = 0, id = 0, cd = 0) => ({
  r_km: r,
  i_km: i,
  c_km: c,
  rd_km_s: rd,
  id_km_s: id,
  cd_km_s: cd,
});

test("C camera toggle cycles reference, current-pair, and current-projections modes", () => {
  assert.equal(toggleDuelCameraMode(DUEL_CAMERA_MODES.REFERENCE), DUEL_CAMERA_MODES.CURRENT_PAIR);
  assert.equal(toggleDuelCameraMode(DUEL_CAMERA_MODES.CURRENT_PAIR), DUEL_CAMERA_MODES.CURRENT_PROJECTIONS);
  assert.equal(toggleDuelCameraMode(DUEL_CAMERA_MODES.CURRENT_PROJECTIONS), DUEL_CAMERA_MODES.REFERENCE);
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

test("visual interpolation blends authoritative RIC states without changing endpoints", () => {
  const previous = {
    tick: 100,
    time_s: 100,
    time_remaining_s: 900,
    range_km: 10,
    relative_speed_km_s: .001,
    target_reference_ric: relative(0, 0, 0, 0, 0, 0),
    chaser_reference_ric: relative(10, -4, 2, .01, -.02, .03),
  };
  const current = {
    ...previous,
    tick: 120,
    time_s: 120,
    time_remaining_s: 880,
    range_km: 8,
    relative_speed_km_s: .003,
    target_reference_ric: relative(2, 4, 6, .02, .04, .06),
    chaser_reference_ric: relative(8, 0, -2, -.01, .02, -.03),
  };
  assert.deepEqual(interpolateDuelRound(previous, current, 0).target_reference_ric, previous.target_reference_ric);
  assert.deepEqual(interpolateDuelRound(previous, current, 1).chaser_reference_ric, current.chaser_reference_ric);
  const midpoint = interpolateDuelRound(previous, current, .5);
  assert.equal(midpoint.tick, 110);
  assert.equal(midpoint.time_remaining_s, 890);
  assert.equal(midpoint.range_km, 9);
  assert.deepEqual(midpoint.target_reference_ric, relative(1, 2, 3, .01, .02, .03));
  assert.deepEqual(midpoint.chaser_reference_ric, relative(9, -2, 0, 0, 0, 0));
  assert.equal(DUEL_VISUAL_TIMING.render_delay_ms, 120);
});

test("reference camera tightly frames the origin, satellites, trails, and both HCW projections", () => {
  const round = {
    time_remaining_s: 1000,
    reference_mean_motion_rad_s: .001,
    target_reference_ric: relative(.4, -.2, .1, 0, .0001, 0),
    chaser_reference_ric: relative(4, -6, .8, 0, .0002, 0),
  };
  const trail = [{ target: relative(.3, -.1, .1), chaser: relative(3, -5, .7) }];
  const frame = duelPlotFrame(round, trail, DUEL_CAMERA_MODES.REFERENCE);
  assert.notDeepEqual(frame.cameraCenter, { r_km: 0, i_km: 0, c_km: 0 });
  assert.equal(frame.targetTrail.length, 1);
  assert.equal(frame.chaserTrail.length, 1);
  assert.equal(frame.targetProjection.length, 121);
  assert.equal(frame.chaserProjection.length, 121);
  assert.ok(frame.framingPoints.includes(frame.targetProjection.at(-1)));
  assert.ok(frame.framingPoints.includes(frame.chaserProjection.at(-1)));
  assert.ok(frame.framingPoints.includes(frame.targetTrail[0]));
  assert.ok(frame.framingPoints.includes(frame.chaserTrail[0]));
  assert.ok(frame.framingPoints.some((point) => point.r_km === 0 && point.i_km === 0 && point.c_km === 0));
  for (const [xKey, yKey] of [["i_km", "r_km"], ["c_km", "r_km"]]) {
    const span = duelPlotSpan(frame, xKey, yKey, .1);
    for (const point of frame.framingPoints) {
      assert.ok(Math.abs(point[xKey] - frame.cameraCenter[xKey]) < span);
      assert.ok(Math.abs(point[yKey] - frame.cameraCenter[yKey]) < span);
    }
  }
  const orbitalPeriodS = 2 * Math.PI / round.reference_mean_motion_rad_s;
  assert.ok(Math.abs(frame.targetProjection.at(-1).t_s - orbitalPeriodS) < 1e-9);
  assert.ok(Math.abs(frame.chaserProjection.at(-1).t_s - orbitalPeriodS) < 1e-9);
});

test("reference camera keeps a full-orbit projection even late in the round", () => {
  const meanMotion = .001;
  const frame = duelPlotFrame({
    time_remaining_s: 10,
    reference_mean_motion_rad_s: meanMotion,
    target_reference_ric: relative(0, 0, 0),
    chaser_reference_ric: relative(1, -2, .5),
  });
  assert.ok(Math.abs(frame.targetProjection.at(-1).t_s - 2 * Math.PI / meanMotion) < 1e-9);
  assert.ok(Math.abs(frame.chaserProjection.at(-1).t_s - 2 * Math.PI / meanMotion) < 1e-9);
});

test("current-pair camera frames only the satellites, suppresses trails, and still draws both projections", () => {
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
  assert.equal(frame.targetProjection.length, 121);
  assert.equal(frame.chaserProjection.length, 121);
  assert.deepEqual(frame.framingPoints, [frame.target, frame.chaser]);
  assert.ok(Math.abs(duelPlotSpan(frame, "i_km", "r_km", .1) - 3.782) < 1e-12);
  assert.deepEqual(frame.targetProjection[0], { ...round.target_reference_ric, t_s: 0 });
  assert.deepEqual(frame.chaserProjection[0], { ...round.chaser_reference_ric, t_s: 0 });
});

test("current-projections camera frames satellites and both HCW projections without the origin", () => {
  const round = {
    reference_mean_motion_rad_s: .001,
    target_reference_ric: relative(20, 30, 4, 0, .0001, 0),
    chaser_reference_ric: relative(24, 36, 6, 0, .0002, 0),
  };
  const frame = duelPlotFrame(
    round,
    [{ target: relative(-100, -100, -100), chaser: relative(-90, -90, -90) }],
    DUEL_CAMERA_MODES.CURRENT_PROJECTIONS,
  );
  assert.equal(frame.cameraMode, DUEL_CAMERA_MODES.CURRENT_PROJECTIONS);
  assert.deepEqual(frame.targetTrail, []);
  assert.deepEqual(frame.chaserTrail, []);
  assert.ok(frame.framingPoints.includes(frame.target));
  assert.ok(frame.framingPoints.includes(frame.chaser));
  assert.ok(frame.framingPoints.includes(frame.targetProjection.at(-1)));
  assert.ok(frame.framingPoints.includes(frame.chaserProjection.at(-1)));
  assert.notDeepEqual(frame.cameraCenter, { r_km: 0, i_km: 0, c_km: 0 });
  for (const [xKey, yKey] of [["i_km", "r_km"], ["c_km", "r_km"]]) {
    const span = duelPlotSpan(frame, xKey, yKey, .1);
    for (const point of frame.framingPoints) {
      assert.ok(Math.abs(point[xKey] - frame.cameraCenter[xKey]) < span);
      assert.ok(Math.abs(point[yKey] - frame.cameraCenter[yKey]) < span);
    }
  }
});
