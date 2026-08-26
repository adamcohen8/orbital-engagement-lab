import assert from "node:assert/strict";
import { test } from "node:test";

import {
  FRAME_CONVENTIONS,
  frameConventionDisplayAxisSign,
  frameConventionDisplayValue,
  frameConventionFromSearch,
  frameConventionLabel,
  nextFrameConvention,
  normalizeFrameConvention,
  urlWithFrameConvention,
} from "../public/src/client/frame-convention.js";

test("frame convention query values normalize fail-closed to OEL", () => {
  assert.equal(frameConventionFromSearch("?frame_convention=space_force"), FRAME_CONVENTIONS.SPACE_FORCE);
  assert.equal(frameConventionFromSearch("?frame_convention=sf"), FRAME_CONVENTIONS.SPACE_FORCE);
  assert.equal(frameConventionFromSearch("?frame_convention=unknown"), FRAME_CONVENTIONS.OEL_DEFAULT);
  assert.equal(normalizeFrameConvention(), FRAME_CONVENTIONS.OEL_DEFAULT);
  assert.equal(frameConventionLabel(FRAME_CONVENTIONS.SPACE_FORCE), "Space Force");
  assert.equal(frameConventionLabel(FRAME_CONVENTIONS.OEL_DEFAULT), "OEL");
  assert.equal(nextFrameConvention(FRAME_CONVENTIONS.OEL_DEFAULT), FRAME_CONVENTIONS.SPACE_FORCE);
  assert.equal(nextFrameConvention(FRAME_CONVENTIONS.SPACE_FORCE), FRAME_CONVENTIONS.OEL_DEFAULT);
});

test("Space Force mirrors only displayed in-track values", () => {
  assert.equal(frameConventionDisplayAxisSign(FRAME_CONVENTIONS.SPACE_FORCE, "i_km"), -1);
  assert.equal(frameConventionDisplayAxisSign(FRAME_CONVENTIONS.SPACE_FORCE, "id_km_s"), -1);
  assert.equal(frameConventionDisplayAxisSign(FRAME_CONVENTIONS.SPACE_FORCE, "r_km"), 1);
  assert.equal(frameConventionDisplayAxisSign(FRAME_CONVENTIONS.SPACE_FORCE, "c_km"), 1);
  assert.equal(frameConventionDisplayValue(FRAME_CONVENTIONS.SPACE_FORCE, "i_km", 12.5), -12.5);
  assert.equal(frameConventionDisplayValue(FRAME_CONVENTIONS.OEL_DEFAULT, "i_km", 12.5), 12.5);
});

test("selector, invite, and return URLs retain normalized frame convention", () => {
  const url = urlWithFrameConvention("/trainer/?room=ABC234", "space-force", "http://localhost:8787/");
  assert.equal(url.href, "http://localhost:8787/trainer/?room=ABC234&frame_convention=space_force");
});
