import assert from "node:assert/strict";
import { test } from "node:test";

import { ellipticLinearCoastStates } from "../src/competition/arcade-engine.js";
import {
  sandboxEllipticLinearCoastStates,
  sandboxTargetStateAt,
} from "../src/sandbox-projection.js";
import { DEFAULT_SANDBOX_SETUP } from "../src/sandbox-setup.js";

function assertPointClose(actual, expected, tolerance = 1.0e-12) {
  for (const [actualKey, expectedKey] of [
    ["r", "r"],
    ["i", "i"],
    ["c", "c"],
    ["rd", "rd"],
    ["id", "id"],
    ["cd", "cd"],
  ]) {
    assert.ok(
      Math.abs(actual[actualKey] - expected[expectedKey]) <= tolerance,
      `${actualKey}: ${actual[actualKey]} != ${expected[expectedKey]}`,
    );
  }
}

test("elliptical Sandbox projection delegates to the numerical TH coast model", () => {
  const mu = 398600.4418;
  const setup = {
    ...DEFAULT_SANDBOX_SETUP,
    target_a_km: 7800,
    target_ecc: 0.18,
    target_true_anomaly_deg: 63,
  };
  const seed = {
    r: 0.4,
    i: -2.5,
    c: 0.2,
    rd: 0.0002,
    id: -0.0001,
    cd: 0.00005,
    t: 240,
    dv: 1.25,
  };
  const times = [0, 120, 900];
  const chief = sandboxTargetStateAt(setup, seed.t, mu);
  const expected = ellipticLinearCoastStates(
    {
      r_km: seed.r,
      i_km: seed.i,
      c_km: seed.c,
      rd_km_s: seed.rd,
      id_km_s: seed.id,
      cd_km_s: seed.cd,
    },
    times,
    chief,
    mu,
  );

  const actual = sandboxEllipticLinearCoastStates(setup, seed, times, mu);

  assert.equal(actual.length, expected.length);
  actual.forEach((point, idx) => {
    assertPointClose(point, expected[idx]);
    assert.equal(point.t, seed.t + times[idx]);
    assert.equal(point.dv, seed.dv);
  });
});
