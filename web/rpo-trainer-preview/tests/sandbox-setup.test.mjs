import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_SANDBOX_SETUP,
  SANDBOX_CHASER_FIELDS,
  SANDBOX_TARGET_FIELDS,
  sandboxRelativeSeed,
  sandboxTargetCoes,
  validateSandboxSetup,
} from "../src/sandbox-setup.js";

test("web Sandbox setup mirrors the downloadable field contract", () => {
  assert.deepEqual(
    SANDBOX_TARGET_FIELDS.map(({ label, unit, key }) => [label, unit, key]),
    [
      ["Semimajor Axis", "km", "target_a_km"],
      ["Eccentricity", "", "target_ecc"],
      ["Inclination", "deg", "target_inc_deg"],
      ["RAAN", "deg", "target_raan_deg"],
      ["Argument of Periapsis", "deg", "target_argp_deg"],
      ["True Anomaly", "deg", "target_true_anomaly_deg"],
    ],
  );
  assert.deepEqual(
    SANDBOX_CHASER_FIELDS.map(({ label, unit, key }) => [label, unit, key]),
    [
      ["Radial R", "km", "radial_km"],
      ["In-Track I", "km", "in_track_km"],
      ["Cross-Track C", "km", "cross_track_km"],
      ["Radial Rate dR", "m/s", "radial_rate_m_s"],
      ["In-Track Rate dI", "m/s", "in_track_rate_m_s"],
      ["Cross-Track Rate dC", "m/s", "cross_track_rate_m_s"],
    ],
  );
});

test("Sandbox setup validates the downloadable numeric bounds", () => {
  assert.deepEqual(validateSandboxSetup(DEFAULT_SANDBOX_SETUP), {
    value: { ...DEFAULT_SANDBOX_SETUP },
    error: "",
  });
  assert.match(validateSandboxSetup({ ...DEFAULT_SANDBOX_SETUP, target_a_km: 0 }).error, /must be positive/);
  assert.match(validateSandboxSetup({ ...DEFAULT_SANDBOX_SETUP, target_ecc: 1 }).error, /0 <= e < 1/);
  assert.match(validateSandboxSetup({ ...DEFAULT_SANDBOX_SETUP, target_inc_deg: 181 }).error, /0 <= i <= 180/);
  assert.match(validateSandboxSetup({ ...DEFAULT_SANDBOX_SETUP, radial_km: "nope" }).error, /Radial R/);
  assert.match(validateSandboxSetup({ ...DEFAULT_SANDBOX_SETUP, radial_km: "" }).error, /Radial R/);
});

test("Sandbox setup maps target COEs and chaser RIC rates without changing units", () => {
  const setup = {
    ...DEFAULT_SANDBOX_SETUP,
    target_a_km: 7200,
    target_ecc: 0.12,
    radial_km: 1.25,
    in_track_km: -4.5,
    cross_track_km: 0.4,
    radial_rate_m_s: 2,
    in_track_rate_m_s: -3,
    cross_track_rate_m_s: 0.5,
  };
  assert.deepEqual(sandboxTargetCoes(setup), {
    a_km: 7200,
    ecc: 0.12,
    inc_deg: 45,
    raan_deg: 0,
    argp_deg: 0,
    true_anomaly_deg: 0,
  });
  assert.deepEqual(sandboxRelativeSeed(setup), {
    r: 1.25,
    i: -4.5,
    c: 0.4,
    rd: 0.002,
    id: -0.003,
    cd: 0.0005,
  });
});
