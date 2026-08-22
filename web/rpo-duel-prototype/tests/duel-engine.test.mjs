import assert from "node:assert/strict";
import { test } from "node:test";

import {
  automaticSpeedState,
  createDuelRound,
  createDuelSeries,
  DUEL_PROTOTYPE_RULES,
  replayDuelRound,
  restoreDuelSeries,
} from "../src/shared/duel-engine.js";

test("prototype rules freeze the accepted round contract", () => {
  assert.equal(DUEL_PROTOTYPE_RULES.schema_version, "rpo-duel.prototype.v1");
  assert.equal(DUEL_PROTOTYPE_RULES.round_duration_s, 18000);
  assert.equal(DUEL_PROTOTYPE_RULES.capture_range_km, 0.1);
  assert.equal(DUEL_PROTOTYPE_RULES.capture_relative_speed_limit_km_s, null);
  assert.equal(DUEL_PROTOTYPE_RULES.chaser_delta_v_budget_m_s, 15);
  assert.equal(DUEL_PROTOTYPE_RULES.target_delta_v_budget_m_s, 5);
});

test("duel round is deterministic for the same seed and role inputs", () => {
  const events = [
    { tick: 2, role: "chaser", controls: { r: 0, i: 1, c: 0 }, sequence: 1 },
    { tick: 12, role: "chaser", controls: { r: 0, i: 0, c: 0 }, sequence: 2 },
    { tick: 25, role: "target", controls: { r: 1, i: 0, c: 0 }, sequence: 1 },
    { tick: 30, role: "target", controls: { r: 0, i: 0, c: 0 }, sequence: 2 },
  ];
  const first = replayDuelRound({ pairSeed: 991, inputEvents: events, finalTick: 60 });
  const second = replayDuelRound({ pairSeed: 991, inputEvents: events, finalTick: 60 });
  assert.deepEqual(first, second);
  assert.ok(first.delta_v_m_s.chaser > 0);
  assert.ok(first.delta_v_m_s.target > 0);
});

test("round snapshots expose both satellites in the propagated target-reference frame", () => {
  const round = createDuelRound({ pairSeed: 404 });
  const initial = round.snapshot();
  assert.deepEqual(initial.target_reference_ric, {
    r_km: 0, i_km: 0, c_km: 0, rd_km_s: 0, id_km_s: 0, cd_km_s: 0,
  });
  assert.deepEqual(initial.chaser_reference_ric, initial.relative_ric);
  assert.ok(initial.reference_mean_motion_rad_s > 0);
  assert.equal(initial.capture_range_km, DUEL_PROTOTYPE_RULES.capture_range_km);

  round.setControls("target", { r: 1, i: 0, c: 0 });
  round.step(10);
  const maneuvered = round.snapshot();
  assert.ok(Math.hypot(
    maneuvered.target_reference_ric.r_km,
    maneuvered.target_reference_ric.i_km,
    maneuvered.target_reference_ric.c_km,
  ) > 0);
  assert.notDeepEqual(maneuvered.chaser_reference_ric, maneuvered.relative_ric);
});

test("delta-v budgets are hard coast caps rather than automatic losses", () => {
  const rules = {
    ...DUEL_PROTOTYPE_RULES,
    round_duration_s: 100,
    chaser_delta_v_budget_m_s: 0.001,
    chaser_max_accel_km_s2: 0.001,
  };
  const round = createDuelRound({ pairSeed: 17, rules });
  round.setControls("chaser", { r: 0, i: 1, c: 0 });
  round.step(10);
  const snapshot = round.snapshot();
  assert.equal(snapshot.delta_v_m_s.chaser, 0.001);
  assert.equal(snapshot.delta_v_remaining_m_s.chaser, 0);
  assert.equal(snapshot.terminal, false);
  assert.equal(round.hasActiveManeuver("chaser"), false);
});

test("fractional computer commands preserve the policy acceleration fraction", () => {
  const round = createDuelRound({
    pairSeed: 18,
    rules: { ...DUEL_PROTOTYPE_RULES, capture_range_km: 0.000001 },
  });
  round.setControls("chaser", { r: 0.5, i: 0, c: 0 }, {
    source: "computer_policy",
    policyPhase: "intercept_burn",
  });
  round.step(1);
  const result = round.result();
  assert.equal(result.delta_v_m_s.chaser, 0.0075);
  assert.equal(result.input_events[0].source, "computer_policy");
  assert.equal(result.input_events[0].policy_phase, "intercept_burn");
});

test("guidance state uses each role as deputy relative to the opposing chief", () => {
  const round = createDuelRound({ pairSeed: 181 });
  const chaser = round.guidanceState("chaser").state_ric_si;
  const target = round.guidanceState("target").state_ric_si;
  const targetGuidance = round.guidanceState("target");
  const snapshotState = Object.values(round.snapshot().relative_ric).map((value) => value * 1000);
  chaser.forEach((value, index) => assert.ok(Math.abs(value - snapshotState[index]) < 1e-9));
  assert.ok(Math.abs(Math.hypot(...target.slice(0, 3)) - Math.hypot(...chaser.slice(0, 3))) < 1e-6);
  assert.ok(round.guidanceState("target").mean_motion_rad_s > 0);
  for (const row of targetGuidance.action_basis_to_game_ric) {
    assert.ok(Math.abs(Math.hypot(...row) - 1) < 1e-12);
  }
});

test("time expiration awards the round to the target", () => {
  const round = createDuelRound({
    pairSeed: 22,
    rules: { ...DUEL_PROTOTYPE_RULES, round_duration_s: 3 },
  });
  round.step(3);
  assert.equal(round.snapshot().terminal, true);
  assert.equal(round.snapshot().winner_role, "target");
});

test("capture ignores relative speed in the prototype", () => {
  const round = createDuelRound({
    pairSeed: 23,
    rules: { ...DUEL_PROTOTYPE_RULES, capture_range_km: 100 },
  });
  assert.equal(round.snapshot().terminal, true);
  assert.equal(round.snapshot().winner_role, "chaser");
});

test("series mirrors geometry within pairs and randomizes across pairs", () => {
  const rules = { ...DUEL_PROTOTYPE_RULES, capture_range_km: 100 };
  const series = createDuelSeries({
    playerIds: ["alpha", "bravo"],
    regulationRounds: 4,
    matchSeed: 20260820,
    rules,
  });

  series.step(1);
  series.advanceRound();
  series.step(1);
  series.advanceRound();
  series.step(1);
  series.advanceRound();
  series.step(1);

  const result = series.result();
  assert.equal(result.match_terminal, true);
  assert.equal(result.match_draw, true);
  assert.deepEqual(
    result.round_summaries[0].initial_geometry,
    result.round_summaries[1].initial_geometry,
  );
  assert.deepEqual(
    result.round_summaries[2].initial_geometry,
    result.round_summaries[3].initial_geometry,
  );
  assert.notDeepEqual(
    result.round_summaries[0].initial_geometry,
    result.round_summaries[2].initial_geometry,
  );
  assert.notEqual(result.round_summaries[0].roles.chaser, result.round_summaries[1].roles.chaser);
  assert.notEqual(result.round_summaries[2].roles.chaser, result.round_summaries[3].roles.chaser);
});

test("automatic time control uses maneuver rail and neutral cooldown", () => {
  const rules = DUEL_PROTOTYPE_RULES;
  assert.deepEqual(automaticSpeedState({ maneuvering: true, nowMs: 1000, lastManeuverMs: 1000, rules }), {
    speed_multiple: 10,
    reason: "maneuvering",
  });
  assert.deepEqual(automaticSpeedState({ maneuvering: false, nowMs: 1500, lastManeuverMs: 1000, rules }), {
    speed_multiple: 10,
    reason: "neutral_cooldown",
  });
  assert.deepEqual(automaticSpeedState({ maneuvering: false, nowMs: 2500, lastManeuverMs: 1000, rules }), {
    speed_multiple: 200,
    reason: "coasting",
  });
});

test("a serialized series can be restored without changing authoritative state", () => {
  const series = createDuelSeries({
    playerIds: ["alpha", "bravo"],
    regulationRounds: 2,
    matchSeed: 712,
  });
  series.setPlayerControls("alpha", { r: 0, i: 1, c: 0 }, { sequence: 1 });
  series.step(12);
  series.setPlayerControls("alpha", { r: 0, i: 0, c: 0 }, { sequence: 2 });
  series.step(8);

  const restored = restoreDuelSeries({ result: series.result() });
  assert.deepEqual(restored.snapshot(), series.snapshot());
  assert.deepEqual(restored.result().current_round.input_events, series.result().current_round.input_events);
});

test("a completed serialized round restores its winner and score", () => {
  const rules = {
    ...DUEL_PROTOTYPE_RULES,
    round_duration_s: 3,
    capture_range_km: 0.000001,
  };
  const series = createDuelSeries({
    playerIds: ["alpha", "bravo"],
    regulationRounds: 2,
    matchSeed: 713,
    rules,
  });
  series.step(3);
  const restored = restoreDuelSeries({ result: series.result(), rules });
  assert.deepEqual(restored.snapshot(), series.snapshot());
});
