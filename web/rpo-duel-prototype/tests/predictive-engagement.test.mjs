import assert from "node:assert/strict";
import { test } from "node:test";

import {
  PREDICTIVE_ENGAGEMENT_POLICY,
  selectEvasionAction,
  selectInterceptAction,
} from "../src/shared/predictive-engagement.js";

const common = {
  mean_motion_rad_s: 0.001078007612872506,
  horizon_s: PREDICTIVE_ENGAGEMENT_POLICY.horizon_s,
  step_s: PREDICTIVE_ENGAGEMENT_POLICY.step_s,
  pulse_duration_s: PREDICTIVE_ENGAGEMENT_POLICY.pulse_duration_s,
  capture_radius_m: 100,
  capture_margin_m: PREDICTIVE_ENGAGEMENT_POLICY.capture_margin_m,
  acceleration_fractions: PREDICTIVE_ENGAGEMENT_POLICY.acceleration_fractions,
};

const parityCases = [
  {
    name: "intercept search",
    policy: "intercept",
    state: [0, -1000, 0, 0, 0, 0],
    options: { max_acceleration_m_s2: 0.015 },
    expected: {
      acceleration_ric_m_s2: [-0.0075, 0, 0],
      predicted_closest_range_m: 406.3489047875341,
      predicted_closest_time_s: 1740,
      predicted_capture_time_s: null,
      phase: "intercept_search_burn",
    },
  },
  {
    name: "passive intercept",
    policy: "intercept",
    state: [200, 0, 0, -0.5, 0, 0],
    options: { max_acceleration_m_s2: 0.015 },
    expected: {
      acceleration_ric_m_s2: [0, 0, 0],
      predicted_closest_range_m: 89.82035950640548,
      predicted_closest_time_s: 360,
      predicted_capture_time_s: 270,
      phase: "intercept_coast",
    },
  },
  {
    name: "three-dimensional intercept",
    policy: "intercept",
    state: [350, -2200, 180, -0.08, 0.16, -0.03],
    options: { max_acceleration_m_s2: 0.015 },
    expected: {
      acceleration_ric_m_s2: [-0.015, 0, 0],
      predicted_closest_range_m: 1674.7311274591716,
      predicted_closest_time_s: 1080,
      predicted_capture_time_s: null,
      phase: "intercept_search_burn",
    },
  },
  {
    name: "bounded-response evasion",
    policy: "evasion",
    state: [0, 1000, 0, 0, 0, 0],
    options: { max_acceleration_m_s2: 0.0075, opponent_max_acceleration_m_s2: 0.015 },
    expected: {
      acceleration_ric_m_s2: [0, 0.0075, 0],
      predicted_closest_range_m: 821.3676757270817,
      predicted_closest_time_s: 600,
      predicted_capture_time_s: null,
      phase: "predictive_evasion_burn",
    },
  },
  {
    name: "three-dimensional evasion",
    policy: "evasion",
    state: [-350, 2200, -180, 0.08, -0.16, 0.03],
    options: { max_acceleration_m_s2: 0.0075, opponent_max_acceleration_m_s2: 0.015 },
    expected: {
      acceleration_ric_m_s2: [-0.0075, 0, 0],
      predicted_closest_range_m: 1989.9157996030212,
      predicted_closest_time_s: 420,
      predicted_capture_time_s: null,
      phase: "predictive_evasion_burn",
    },
  },
];

for (const fixture of parityCases) {
  test(`JavaScript ${fixture.name} matches the Python predictive policy fixture`, () => {
    const decide = fixture.policy === "intercept" ? selectInterceptAction : selectEvasionAction;
    const actual = decide(fixture.state, { ...common, ...fixture.options });
    assert.equal(actual.phase, fixture.expected.phase);
    assert.equal(actual.predicted_capture_time_s, fixture.expected.predicted_capture_time_s);
    assert.equal(actual.predicted_closest_time_s, fixture.expected.predicted_closest_time_s);
    assertVectorClose(actual.acceleration_ric_m_s2, fixture.expected.acceleration_ric_m_s2, 1e-12);
    assertClose(actual.predicted_closest_range_m, fixture.expected.predicted_closest_range_m, 1e-7);
  });
}

function assertVectorClose(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assertClose(value, expected[index], tolerance));
}

function assertClose(actual, expected, tolerance) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);
}
