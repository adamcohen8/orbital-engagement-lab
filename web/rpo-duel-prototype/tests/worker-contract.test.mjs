import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const wrangler = JSON.parse(
  readFileSync(new URL("../wrangler.jsonc", import.meta.url), "utf8"),
);
const workerSource = readFileSync(new URL("../worker/index.js", import.meta.url), "utf8");
const roomSource = readFileSync(new URL("../src/shared/duel-room.js", import.meta.url), "utf8");
const assetsIgnore = readFileSync(new URL("../public/.assetsignore", import.meta.url), "utf8");
const clientSource = readFileSync(new URL("../public/src/client/app.js", import.meta.url), "utf8");
const stylesSource = readFileSync(new URL("../public/src/client/styles.css", import.meta.url), "utf8");

test("Cloudflare deployment binds one SQLite Durable Object namespace and static assets", () => {
  assert.equal(wrangler.name, "oel-rpo-duel");
  assert.equal(wrangler.main, "./worker/index.js");
  assert.equal(wrangler.assets.directory, "./public");
  assert.deepEqual(wrangler.assets.run_worker_first, ["/api/*", "/ws"]);
  assert.deepEqual(wrangler.durable_objects.bindings, [
    { name: "DUEL_ROOMS", class_name: "RpoDuelRoom" },
    { name: "DUEL_ADMISSION", class_name: "RpoDuelAdmission" },
  ]);
  assert.deepEqual(wrangler.exports.RpoDuelRoom, {
    type: "durable-object",
    storage: "sqlite",
  });
  assert.deepEqual(wrangler.exports.RpoDuelAdmission, {
    type: "durable-object",
    storage: "sqlite",
  });
});

test("deployed assets use a dedicated public directory", () => {
  assert.equal(assetsIgnore.trim(), ".assetsignore");
  for (const path of ["index.html", "src/client/app.js", "src/client/plot-model.js", "src/client/styles.css"]) {
    assert.equal(existsSync(new URL(`../public/${path}`, import.meta.url)), true, path);
  }
  for (const path of ["server", "tests", "worker", "src/shared", "package.json", "package-lock.json"]) {
    assert.equal(existsSync(new URL(`../public/${path}`, import.meta.url)), false, path);
  }
});

test("Worker contract enforces same-origin rooms, bounded payloads, persistence, and security headers", () => {
  assert.match(workerSource, /assertSameOrigin\(request\)/);
  assert.match(workerSource, /MAX_BODY_BYTES = 8192/);
  assert.match(roomSource, /max_message_bytes: 2048/);
  assert.match(workerSource, /state\.storage\.put\("room"/);
  assert.match(workerSource, /Content-Security-Policy/);
  assert.match(workerSource, /Permissions-Policy/);
  assert.match(workerSource, /DUEL_MAX_ACTIVE_ROOMS/);
  assert.match(workerSource, /DUEL_CREATE_RATE_LIMIT/);
});

test("client contract includes the C camera shortcut and reuses the trainer landscape layout", () => {
  assert.match(clientSource, /event\.code === "KeyC"/);
  const landscapeStart = stylesSource.indexOf("@media (max-height: 580px) and (orientation: landscape)");
  const landscapeEnd = stylesSource.indexOf("@media (prefers-reduced-motion", landscapeStart);
  assert.ok(landscapeStart >= 0 && landscapeEnd > landscapeStart);
  const landscape = stylesSource.slice(landscapeStart, landscapeEnd);
  assert.match(landscape, /\.game-view \{ position: static; display: grid;/);
  assert.match(landscape, /grid-template-columns: minmax\(0, 1fr\) clamp\(280px, 36vw, 360px\)/);
  assert.match(landscape, /grid-template-areas: "top top" "game game" "hud controls"/);
  assert.match(landscape, /\.plots \{ grid-template-columns: 1fr 1fr;/);
  assert.match(landscape, /\.duel-hud-panel \{ grid-area: hud;/);
  assert.match(landscape, /\.touch-controls \{ grid-area: controls;/);
  assert.match(landscape, /grid-template-columns: repeat\(3,/);
});
