import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  buildChallengeRecord,
  canonicalJson,
  createPursuitArcadeSession,
  createPursuitSession,
  DEFAULT_PURSUIT_CHALLENGE,
  ellipticLinearCoastStates,
  hashCanonicalJson,
  keplerianToEci,
  makeAttemptPacket,
  relativeRicState,
  runPursuitArcadeReplay,
  runPursuitReplay,
  stateFromRelativeRic,
  trajectoryPlotSvg,
  validateAttemptPacket,
} from "../src/competition/arcade-engine.js";
import { hashVerificationToken, verificationExpiryIso, verificationUrl } from "../api/_email.mjs";
import { isLeaderboardEligibleStatus, shouldReplaceLeaderboardScore } from "../api/_leaderboard.mjs";
import { canVerifyUsernameForEmail, decideOwnership, OWNERSHIP_STATUS } from "../api/_ownership.mjs";

test("canonical JSON and hash are stable across object key order", () => {
  const a = { b: 2, a: { d: 4, c: 3 } };
  const b = { a: { c: 3, d: 4 }, b: 2 };
  assert.equal(canonicalJson(a), canonicalJson(b));
  assert.equal(hashCanonicalJson(a), hashCanonicalJson(b));
});

test("email verification helpers hash tokens and build public links", () => {
  const hash = hashVerificationToken("abc123");
  assert.equal(hash, hashVerificationToken("abc123"));
  assert.notEqual(hash, hashVerificationToken("abc124"));
  assert.match(hash, /^[a-f0-9]{64}$/);
  assert.equal(
    verificationUrl({ headers: { host: "example.test", "x-forwarded-proto": "https" } }, "token with spaces"),
    "https://example.test/api/verify-email?token=token%20with%20spaces",
  );
  assert.equal(verificationExpiryIso(new Date("2026-01-01T00:00:00.000Z")), "2026-01-08T00:00:00.000Z");
});

test("username ownership decisions support verified email locking", () => {
  assert.deepEqual(decideOwnership({ player: {}, email: "" }), {
    status: OWNERSHIP_STATUS.UNCLAIMED,
    leaderboard_allowed: true,
    verification_allowed: false,
  });
  assert.deepEqual(decideOwnership({ player: {}, email: "ace@example.edu" }), {
    status: OWNERSHIP_STATUS.PENDING_VERIFICATION,
    leaderboard_allowed: true,
    verification_allowed: true,
  });
  const lockedPlayer = {
    email: "ace@example.edu",
    email_verified_at: "2026-01-01T00:00:00.000Z",
    username_locked_at: "2026-01-01T00:00:00.000Z",
  };
  assert.deepEqual(decideOwnership({ player: lockedPlayer, email: "ace@example.edu" }), {
    status: OWNERSHIP_STATUS.VERIFIED_OWNER,
    leaderboard_allowed: true,
    verification_allowed: true,
  });
  assert.deepEqual(decideOwnership({ player: lockedPlayer, email: "" }), {
    status: OWNERSHIP_STATUS.LOCKED,
    leaderboard_allowed: false,
    verification_allowed: false,
  });
  assert.deepEqual(decideOwnership({ player: lockedPlayer, email: "other@example.edu" }), {
    status: OWNERSHIP_STATUS.LOCKED,
    leaderboard_allowed: false,
    verification_allowed: false,
  });
  assert.equal(canVerifyUsernameForEmail({ player: {}, email: "ace@example.edu" }), true);
  assert.equal(canVerifyUsernameForEmail({ player: lockedPlayer, email: "ACE@example.edu" }), true);
  assert.equal(canVerifyUsernameForEmail({ player: lockedPlayer, email: "other@example.edu" }), false);
});

test("leaderboard promotion keeps only eligible better scores", () => {
  assert.equal(isLeaderboardEligibleStatus("valid"), true);
  assert.equal(isLeaderboardEligibleStatus("suspicious"), true);
  assert.equal(isLeaderboardEligibleStatus("invalid"), false);
  assert.equal(shouldReplaceLeaderboardScore(null, 10), true);
  assert.equal(shouldReplaceLeaderboardScore(10, 11), true);
  assert.equal(shouldReplaceLeaderboardScore(10, 10), false);
  assert.equal(shouldReplaceLeaderboardScore(10, 9), false);
});

test("RIC conversion round trips rotating-frame relative position and velocity", () => {
  const target = keplerianToEci(DEFAULT_PURSUIT_CHALLENGE.target_coes);
  const rel = {
    r_km: 0.4,
    i_km: -1.2,
    c_km: 0.25,
    rd_km_s: 0.0001,
    id_km_s: -0.00003,
    cd_km_s: 0.00002,
  };
  const chaser = stateFromRelativeRic(target, rel);
  const recovered = relativeRicState(target, chaser);
  Object.entries(rel).forEach(([key, value]) => {
    assert.ok(Math.abs(recovered[key] - value) < 1.0e-10, `${key}: ${recovered[key]} != ${value}`);
  });
});

test("elliptic linear coast matches HCW on a circular chief orbit", () => {
  const mu = DEFAULT_PURSUIT_CHALLENGE.mu_km3_s2;
  const aKm = DEFAULT_PURSUIT_CHALLENGE.target_coes.a_km;
  const chief = keplerianToEci(DEFAULT_PURSUIT_CHALLENGE.target_coes, mu);
  const seed = {
    r_km: 0.12,
    i_km: -0.8,
    c_km: 0.18,
    rd_km_s: 0.00004,
    id_km_s: -0.00002,
    cd_km_s: 0.00003,
  };
  const times = [0, 30, 120, 600, 1800];
  const coast = ellipticLinearCoastStates(seed, times, chief, mu);
  times.forEach((timeS, idx) => {
    const expected = cwState(seed, timeS, Math.sqrt(mu / aKm ** 3));
    assert.ok(Math.abs(coast[idx].r - expected.r) < 2.0e-5, `r at ${timeS}s`);
    assert.ok(Math.abs(coast[idx].i - expected.i) < 2.0e-5, `i at ${timeS}s`);
    assert.ok(Math.abs(coast[idx].c - expected.c) < 2.0e-5, `c at ${timeS}s`);
  });
});

test("replay is deterministic for the same seed and input events", () => {
  const record = buildChallengeRecord();
  const input_events = [
    { tick: 10, control: "iPlus", state: "down" },
    { tick: 30, control: "iPlus", state: "up" },
    { tick: 200, control: "rMinus", state: "down" },
    { tick: 215, control: "rMinus", state: "up" },
  ];
  const first = runPursuitReplay(record.config, { seed: 99, input_events });
  const second = runPursuitReplay(record.config, { seed: 99, input_events });
  assert.deepEqual(first.metrics, second.metrics);
  assert.equal(first.score, second.score);
  assert.equal(first.burn_markers.length > 0, true);
});

test("validator accepts a canonical attempt and rejects a tampered score", () => {
  const record = buildChallengeRecord();
  const result = runPursuitReplay(record.config, {
    seed: 7,
    input_events: [
      { tick: 3, control: "iPlus", state: "down" },
      { tick: 23, control: "iPlus", state: "up" },
    ],
  });
  const attempt = makeAttemptPacket({
    challengeRecord: record,
    username: "VALIDATOR",
    seed: 7,
    input_events: [
      { tick: 3, control: "iPlus", state: "down" },
      { tick: 23, control: "iPlus", state: "up" },
    ],
    result,
  });
  assert.equal(validateAttemptPacket(attempt, record).status, "valid");
  assert.equal(validateAttemptPacket({ ...attempt, claimed_score: attempt.claimed_score + 1 }, record).status, "invalid");
});

test("interactive session records tick inputs and validates its attempt packet", () => {
  const record = buildChallengeRecord();
  const session = createPursuitSession(record.config, { seed: 123 });
  session.setControl("iPlus", true);
  session.step(8);
  session.setControl("iPlus", false);
  session.step(4);
  const snapshot = session.snapshot();
  assert.equal(snapshot.input_events.length, 2);
  assert.equal(snapshot.input_events[0].tick, 0);
  assert.equal(snapshot.input_events[1].tick, 8);
  const attempt = session.attemptPacket({ challengeRecord: record, username: "SESSION" });
  assert.equal(validateAttemptPacket(attempt, record).status, "valid");
});

test("target defensive maneuver is visible relative to target reference", () => {
  const record = buildChallengeRecord();
  const config = {
    ...record.config,
    target_defense: {
      ...record.config.target_defense,
      trigger_range_km: 2.0,
    },
  };
  const session = createPursuitSession(config, { seed: 123 });
  session.step(20);
  const latest = session.snapshot().history.at(-1);
  assert.ok(latest.target_delta_v_m_s > 0);
  assert.ok(Math.hypot(latest.target_reference_ric.r_km, latest.target_reference_ric.i_km, latest.target_reference_ric.c_km) > 0);
  assert.ok(Math.hypot(latest.chaser_reference_ric.r_km, latest.chaser_reference_ric.i_km, latest.chaser_reference_ric.c_km) > 0);
});

test("arcade run clears rounds, awards score, and tightens the next goal", () => {
  const record = buildChallengeRecord({
    ...DEFAULT_PURSUIT_CHALLENGE,
    goal_range_km: 2.0,
    arcade: {
      ...DEFAULT_PURSUIT_CHALLENGE.arcade,
      goal_range_step_km: 0.25,
      min_goal_range_km: 0.25,
    },
  });
  const session = createPursuitArcadeSession(record.config, { seed: 321 });
  session.step(1);
  const transition = session.snapshot().round_transition;
  assert.ok(transition);
  assert.equal(transition.cleared_round_index, 1);
  assert.equal(transition.next_round_index, 2);
  assert.ok(transition.round_score > 0);
  assert.equal(transition.next_goal_range_km, 1.75);
  session.continueNextRound();
  const next = session.snapshot();
  assert.equal(next.round_index, 2);
  assert.equal(next.goal_range_km, 1.75);
  assert.equal(next.total_score, transition.total_score);
});

test("arcade multi-round attempt packet validates from recorded round inputs", () => {
  const record = buildChallengeRecord({
    ...DEFAULT_PURSUIT_CHALLENGE,
    goal_range_km: 20.0,
    arcade: {
      ...DEFAULT_PURSUIT_CHALLENGE.arcade,
      goal_range_step_km: 0.0,
      min_goal_range_km: 20.0,
    },
  });
  const session = createPursuitArcadeSession(record.config, { seed: 321 });
  session.step(1);
  session.continueNextRound();
  session.step(1);

  const attempt = session.attemptPacket({ challengeRecord: record, username: "ARCADE" });
  assert.equal(attempt.attempt_type, "arcade_run");
  assert.equal(attempt.round_attempts.length, 2);

  const replay = runPursuitArcadeReplay(record.config, {
    seed: attempt.seed,
    round_attempts: attempt.round_attempts,
  });
  assert.equal(replay.round_summaries.length, 2);

  const validation = validateAttemptPacket(attempt, record);
  assert.equal(validation.status, "valid");
  assert.equal(validation.canonical_score, attempt.claimed_score);
  assert.equal(validateAttemptPacket({ ...attempt, claimed_score: attempt.claimed_score + 1 }, record).status, "invalid");
});

test("arcade boss rounds are flagged for elliptic projection", () => {
  const record = buildChallengeRecord({
    ...DEFAULT_PURSUIT_CHALLENGE,
    arcade: {
      ...DEFAULT_PURSUIT_CHALLENGE.arcade,
      boss_round_interval: 1,
    },
  });
  const session = createPursuitArcadeSession(record.config, { seed: 654 });
  const snapshot = session.snapshot();
  assert.equal(snapshot.is_boss_round, true);
  assert.ok(snapshot.target_reference_state_eci);
});

test("arcade sessions can start directly on a selected round for preview testing", () => {
  const record = buildChallengeRecord();
  const session = createPursuitArcadeSession(record.config, { seed: 654, startRoundIndex: 5 });
  const snapshot = session.snapshot();
  assert.equal(snapshot.round_index, 5);
  assert.equal(snapshot.is_boss_round, true);
  assert.equal(snapshot.goal_range_km, 0.08);
  assert.equal(session.config.target_coes.a_km, DEFAULT_PURSUIT_CHALLENGE.arcade.boss.target_coes.a_km);
});

test("fixture validates against the built-in challenge record", () => {
  const record = buildChallengeRecord();
  const attempt = JSON.parse(readFileSync(new URL("../fixtures/sample-valid-attempt.json", import.meta.url), "utf8"));
  const validation = validateAttemptPacket(attempt, record);
  assert.equal(validation.status, "valid");
  assert.equal(validation.canonical_metrics.player_delta_v_m_s, 1.35);
});

test("trajectory plots are generated from recomputed replay history", () => {
  const record = buildChallengeRecord();
  const replay = runPursuitReplay(record.config, {
    seed: 42,
    input_events: [
      { tick: 5, control: "iPlus", state: "down" },
      { tick: 20, control: "iPlus", state: "up" },
    ],
  });
  const svg = trajectoryPlotSvg(replay, "RI");
  assert.match(svg, /<svg/);
  assert.match(svg, /burn-marker/);
  assert.match(svg, /RI Plane/);
});

function cwState(seed, tS, n) {
  const x = seed.r_km;
  const y = seed.i_km;
  const z = seed.c_km;
  const xd = seed.rd_km_s;
  const yd = seed.id_km_s;
  const zd = seed.cd_km_s;
  const nt = n * tS;
  const cosNt = Math.cos(nt);
  const sinNt = Math.sin(nt);
  return {
    r: (4 - 3 * cosNt) * x + (sinNt / n) * xd + ((2 * (1 - cosNt)) / n) * yd,
    i: 6 * (sinNt - nt) * x + y - ((2 * (1 - cosNt)) / n) * xd + (((4 * sinNt - 3 * nt) / n) * yd),
    c: cosNt * z + (sinNt / n) * zd,
  };
}
