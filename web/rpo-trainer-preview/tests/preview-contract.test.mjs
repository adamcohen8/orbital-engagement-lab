import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";

import { DEFAULT_PURSUIT_CHALLENGE } from "../src/competition/arcade-engine.js";
import { PREVIEW_LEVEL_CONTRACTS } from "../src/preview-contract.js";
import { propagateHcwState } from "../src/preview-physics.js";

const downloadable = JSON.parse(
  readFileSync(new URL("../fixtures/downloadable-game-contract.json", import.meta.url), "utf8"),
);
const trajectories = JSON.parse(
  readFileSync(new URL("../fixtures/oel-level0-reference-trajectories.json", import.meta.url), "utf8"),
);
const previewHtml = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const previewApp = readFileSync(new URL("../src/app.js", import.meta.url), "utf8");

function tutorialComparable(contract) {
  return {
    title: contract.title,
    max_time_s: contract.max_time_s,
    max_delta_v_m_s: contract.max_delta_v_m_s,
    goal_range_km: contract.goal_range_km,
    max_goal_speed_km_s: contract.max_goal_speed_km_s,
    guided_burn_delta_v_m_s: contract.guided_burn_delta_v_m_s,
    guided_speed_multiplier: contract.guided_speed_multiplier,
    learning_goal: contract.learning_goal,
    player_brief: contract.player_brief,
    pass_criteria: [...contract.pass_criteria],
    instructor_notes: [...contract.instructor_notes],
  };
}

function arcadeComparable(config) {
  return {
    challenge_id: config.challenge_id,
    title: config.title,
    mu_km3_s2: config.mu_km3_s2,
    dt_s: config.dt_s,
    max_time_s: config.max_time_s,
    max_player_accel_km_s2: config.max_player_accel_km_s2,
    max_delta_v_m_s: config.max_delta_v_m_s,
    max_target_delta_v_m_s: config.max_target_delta_v_m_s,
    goal_range_km: config.goal_range_km,
    difficulty: config.difficulty,
    target_coes: config.target_coes,
    chaser_initial_ric: config.chaser_initial_ric,
    target_defense: config.target_defense,
    arcade: config.arcade,
  };
}

function stateFromArray(values) {
  return {
    r: values[0],
    i: values[1],
    c: values[2],
    rd: values[3],
    id: values[4],
    cd: values[5],
    t: 0,
  };
}

test("Level 0 browser copy and budgets match the downloadable YAML contract", () => {
  assert.deepEqual(tutorialComparable(PREVIEW_LEVEL_CONTRACTS.tutorial), downloadable.tutorial);
});

test("Pursuit Arcade browser constants match the checked-in OEL scenario contract", () => {
  const actual = arcadeComparable(DEFAULT_PURSUIT_CHALLENGE);
  const expected = {
    ...downloadable.arcade,
    arcade: { ...downloadable.arcade.arcade },
  };
  const browserMu = actual.mu_km3_s2;
  const oelMu = expected.mu_km3_s2;
  delete actual.mu_km3_s2;
  delete expected.mu_km3_s2;
  delete expected.arcade.enabled;
  assert.deepEqual(actual, expected);
  assert.ok(
    Math.abs(browserMu - oelMu) <= 1.0e-3,
    `versioned browser mu ${browserMu} is not within the documented bound of OEL mu ${oelMu}`,
  );
});

test("the browser Sandbox truthfully identifies its reduced scope", () => {
  assert.equal(downloadable.sandbox.supports_target_orbit_edit, true);
  assert.equal(downloadable.sandbox.supports_target_eccentricity, true);
  assert.match(PREVIEW_LEVEL_CONTRACTS.sandbox.title, /Reduced/);
  assert.match(PREVIEW_LEVEL_CONTRACTS.sandbox.scope, /downloadable Sandbox/i);
  assert.match(PREVIEW_LEVEL_CONTRACTS.sandbox.scope, /target orbit and eccentricity/i);
});

test("Pursuit Arcade is explicitly identified as web-only", () => {
  assert.equal(PREVIEW_LEVEL_CONTRACTS.pursuit_arcade.web_only, true);
  assert.match(PREVIEW_LEVEL_CONTRACTS.pursuit_arcade.scope, /Web-only/);
  assert.match(PREVIEW_LEVEL_CONTRACTS.pursuit_arcade.scope, /not included in the downloadable launcher/i);
});

test("browser HCW paths remain within tolerance of OEL Level 0 two-body references", () => {
  trajectories.cases.forEach((testCase) => {
    const initial = stateFromArray(testCase.initial_relative_ric_km_km_s);
    testCase.samples.forEach((sample) => {
      const actual = propagateHcwState(initial, sample.time_s, trajectories.browser_step_s);
      const expected = stateFromArray(sample.relative_ric_km_km_s);
      for (const axis of ["r", "i", "c"]) {
        const errorKm = Math.abs(actual[axis] - expected[axis]);
        assert.ok(
          errorKm <= trajectories.position_tolerance_km,
          `${testCase.name} ${axis} at ${sample.time_s}s: ${errorKm} km exceeds ${trajectories.position_tolerance_km} km`,
        );
      }
    });
  });
});

test("RPO Duel appears in the selector as a hosted Beta destination", () => {
  assert.match(previewHtml, /data-level-option="rpoDuel"/);
  assert.match(previewHtml, /level-beta-badge">Beta</);
  assert.match(
    previewHtml,
    /name="oel-rpo-duel-url"\s+content="https:\/\/oel-rpo-duel\.oel-rpo-duel\.workers\.dev"/,
  );
  assert.match(previewApp, /id: "rpoDuel"/);
  assert.match(previewApp, /mode: "external"/);
  assert.match(previewApp, /externalUrl: RPO_DUEL_URL/);
  assert.match(previewApp, /window\.location\.assign\(option\.externalUrl\)/);
  assert.match(previewHtml, /Four focused experiences/);
  assert.match(previewHtml, /Hosted PvP \/ computer match/);
  assert.match(previewApp, /choose Play computer/);
  assert.match(previewApp, /automatic 200x coast and 10x maneuver time rails/);
  assert.doesNotMatch(previewApp, /automatic 100x coast/);
});
