import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const wrangler = JSON.parse(
  readFileSync(new URL("../wrangler.jsonc", import.meta.url), "utf8"),
);
const workerSource = readFileSync(new URL("../worker/index.js", import.meta.url), "utf8");
const roomSource = readFileSync(new URL("../src/shared/duel-room.js", import.meta.url), "utf8");
const policySource = readFileSync(new URL("../src/shared/predictive-engagement.js", import.meta.url), "utf8");
const assetsIgnore = readFileSync(new URL("../public/.assetsignore", import.meta.url), "utf8");
const clientSource = readFileSync(new URL("../public/src/client/app.js", import.meta.url), "utf8");
const stylesSource = readFileSync(new URL("../public/src/client/styles.css", import.meta.url), "utf8");
const indexSource = readFileSync(new URL("../public/index.html", import.meta.url), "utf8");
const musicAssetUrl = new URL("../public/assets/39_perigee_afterburner_demo.wav", import.meta.url);

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
  for (const path of ["index.html", "assets/39_perigee_afterburner_demo.wav", "src/client/app.js", "src/client/frame-convention.js", "src/client/plot-model.js", "src/client/styles.css"]) {
    assert.equal(existsSync(new URL(`../public/${path}`, import.meta.url)), true, path);
  }
  for (const path of ["server", "tests", "worker", "src/shared", "package.json", "package-lock.json"]) {
    assert.equal(existsSync(new URL(`../public/${path}`, import.meta.url)), false, path);
  }
});

test("RPO Duel wires the exact Perigee Afterburner cue as optional looping music", () => {
  const music = readFileSync(musicAssetUrl);
  assert.equal(createHash("sha256").update(music).digest("hex"), "739a2f88269a1b4f275b8066a03852bf2504eaa6c3fa68783c6493836ab420de");
  assert.match(clientSource, /\/assets\/39_perigee_afterburner_demo\.wav/);
  assert.match(clientSource, /duelMusic\.loop = true/);
  assert.match(clientSource, /duelMusic\.volume = 0\.65/);
  assert.match(clientSource, /event\.code === "KeyM"/);
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

test("computer opponent remains server-owned and uses both predictive policies", () => {
  assert.match(roomSource, /source: "computer_policy"/);
  assert.match(roomSource, /selectInterceptAction/);
  assert.match(roomSource, /selectEvasionAction/);
  assert.match(roomSource, /pulse_duration_s: 30/);
  assert.match(roomSource, /decision_interval_s: 120/);
  assert.match(roomSource, /minimum_range_improvement_m: 100/);
  assert.match(roomSource, /target_guard_range_m: 600/);
  assert.match(roomSource, /materiallyImprovesComputerOutcome/);
  assert.match(policySource, /prediction_model: "HCW"/);
  assert.match(policySource, /decision_interval_s: 120/);
});

test("client contract includes the C camera shortcut and reuses the trainer landscape layout", () => {
  assert.match(clientSource, /event\.code === "KeyC"/);
  assert.match(clientSource, /previewMode === "mobile-landscape" \|\| previewMode === "mobile-portrait"/);
  assert.match(stylesSource, /body\.mobile-portrait-preview\.duel-active \.app-shell/);
  const landscapeStart = stylesSource.indexOf("@media (max-height: 580px) and (orientation: landscape)");
  const landscapeEnd = stylesSource.indexOf("@media (prefers-reduced-motion", landscapeStart);
  assert.ok(landscapeStart >= 0 && landscapeEnd > landscapeStart);
  const landscape = stylesSource.slice(landscapeStart, landscapeEnd);
  assert.match(landscape, /\.game-view \{ position: static; display: grid;/);
  assert.match(landscape, /grid-template-columns: minmax\(0, 1fr\) clamp\(280px, 36vw, 360px\)/);
  assert.match(landscape, /grid-template-areas: "top top" "game game" "hud controls"/);
  assert.match(landscape, /\.plots \{ grid-template-columns: 1fr 1fr;/);
  assert.match(landscape, /\.duel-hud-panel \{ grid-area: hud;/);
  assert.match(landscape, /grid-template-areas: "room score" "toggles score"/);
  assert.match(landscape, /\.view-toggles \{ grid-area: toggles; justify-self: start;/);
  assert.match(landscape, /\.scoreboard \{ position: static; grid-area: score; align-self: stretch; display: grid;/);
  assert.match(landscape, /\.player-score b \{ grid-column: 1 \/ -1; grid-row: 2;/);
  assert.match(landscape, /\.auto-time-heading \{ grid-column: 1; grid-row: 1; display: flex;/);
  assert.match(landscape, /\.auto-time-heading span \{ width: auto; white-space: nowrap; \}/);
  assert.match(landscape, /\.auto-time > strong \{ grid-row: 2; grid-column: 1; justify-self: center;/);
  assert.match(landscape, /\.duel-objective \.objective-heading \{ grid-column: 1; grid-row: 1;[^}]*width: auto; white-space: nowrap;/);
  assert.match(landscape, /\.duel-objective > strong \{ grid-column: 1; grid-row: 2;/);
  assert.match(landscape, /\.touch-controls \{ grid-area: controls;/);
  assert.match(landscape, /grid-template-columns: repeat\(3,/);
});

test("role label follows the satellite role palette", () => {
  assert.match(clientSource, /elements\["role-label"\]\.dataset\.role = role \|\| "waiting"/);
  assert.match(stylesSource, /#role-label\[data-role="chaser"\] \{ color: var\(--chaser\); \}/);
  assert.match(stylesSource, /#role-label\[data-role="target"\] \{ color: var\(--target\); \}/);
});

test("portrait HUD stacks view toggles below the room and preserves full scoreboard names", () => {
  const portraitStart = stylesSource.indexOf("@media (max-width: 720px)");
  const portraitEnd = stylesSource.indexOf("@media (max-height: 580px)", portraitStart);
  assert.ok(portraitStart >= 0 && portraitEnd > portraitStart);
  const portrait = stylesSource.slice(portraitStart, portraitEnd);
  assert.match(portrait, /grid-template-areas: "room score" "toggles score"/);
  assert.match(portrait, /\.duel-objective \{ grid-template-columns: 6\.5ch minmax\(0, 1fr\)/);
  assert.match(portrait, /\.duel-objective \.objective-heading \{ grid-column: 1; grid-row: 1; width: 4ch; white-space: normal;/);
  assert.match(portrait, /\.auto-time \{ grid-column: 1 \/ -1; grid-row: 2;[^}]*grid-template-columns: 6\.5ch minmax\(0, 1fr\)/);
  assert.match(portrait, /\.auto-time-heading \{ grid-column: 1; grid-row: 1; display: grid;/);
  assert.match(portrait, /\.auto-time-heading span \{ width: 4ch; white-space: normal; \}/);
  assert.match(portrait, /\.auto-time > strong \{ grid-column: 2; grid-row: 1; align-self: center;/);
  assert.match(portrait, /\.view-toggles \{ grid-area: toggles; justify-self: start;/);
  assert.match(portrait, /\.player-score \.player-name \{ align-self: center; max-width: 80px;/);
  assert.match(portrait, /\.hint-line \{ display: none; \}/);
});

test("expanded metrics use concise labels in one shorter header row", () => {
  assert.doesNotMatch(indexSource, />INFO (?:Range|Rel Speed|Delta-v|Time Left)/);
  for (const label of ["Range", "Rel Speed", "Delta-v", "Time Left"]) {
    assert.match(indexSource, new RegExp(`<span>${label} <strong`));
  }
  const expandedStart = stylesSource.indexOf("@media (min-width: 1000px) and (min-height: 581px)");
  const expandedEnd = stylesSource.indexOf("@media (max-width: 920px)", expandedStart);
  const expanded = stylesSource.slice(expandedStart, expandedEnd);
  assert.match(expanded, /\.duel-top-bar \{ min-height: 70px;/);
  assert.match(expanded, /\.duel-top-metrics \{ flex-wrap: nowrap;/);
  assert.match(expanded, /\.duel-game-layout \{ top: 100px; \}/);
  assert.match(expanded, /\.plots \{ gap: 24px; \}/);
  assert.match(expanded, /\.plot-card canvas \{ left: 12px; right: 12px; top: 42px; bottom: 10px;/);
});

test("expanded auto-time status shares the heading row", () => {
  assert.match(indexSource, /class="auto-time-heading"><span>AUTO TIME<\/span><small>COAST<\/small><\/div><strong>200x<\/strong>/);
  assert.match(clientSource, /coasting: "COAST"/);
  assert.match(clientSource, /neutral_cooldown: "COOL"/);
  assert.match(clientSource, /maneuvering: "BURN"/);
  const expandedStart = stylesSource.indexOf("@media (min-width: 1000px) and (min-height: 581px)");
  const expandedEnd = stylesSource.indexOf("@media (max-width: 920px)", expandedStart);
  const expanded = stylesSource.slice(expandedStart, expandedEnd);
  assert.match(expanded, /grid-template-rows: 15px 30px; align-content: center;/);
  assert.match(expanded, /\.auto-time-heading \{ grid-column: 2; grid-row: 1; display: flex;/);
  assert.match(expanded, /white-space: nowrap;/);
  assert.match(expanded, /\.auto-time > strong \{ grid-column: 2; grid-row: 2;/);
});

test("expanded lower controls stay with the room code without redundant shortcut text", () => {
  assert.match(indexSource, /id="frame-convention-label">Frame: OEL<\/span>/);
  assert.match(clientSource, /document\.body\.dataset\.frameConvention = state\.frameConvention/);
  assert.match(clientSource, /urlWithFrameConvention\(window\.location\.origin, state\.frameConvention\)/);
  assert.match(indexSource, /id="command-line">W\/S Radial&nbsp;&nbsp; A −I \/ D \+I&nbsp;&nbsp; Left\/Right Cross-Track<\/div>/);
  assert.doesNotMatch(indexSource, /class="command-line"[^<]*(?:C Camera|M Music)/);
  const expandedStart = stylesSource.indexOf("@media (min-width: 1000px) and (min-height: 581px)");
  const expandedEnd = stylesSource.indexOf("@media (max-width: 920px)", expandedStart);
  const expanded = stylesSource.slice(expandedStart, expandedEnd);
  assert.match(expanded, /\.duel-hud-line \{[^}]*justify-content: flex-start;/);
  assert.match(expanded, /\.view-toggles \{ justify-content: flex-start; \}/);
});

test("RI and RC canvases devote their overlays and scale margins to plot data", () => {
  assert.doesNotMatch(clientSource, /drawProjectionLegend/);
  assert.doesNotMatch(clientSource, /REFERENCE ORBIT · HCW COAST/);
  assert.doesNotMatch(clientSource, /PAIR · SATELLITES/);
  assert.doesNotMatch(clientSource, /span\.toFixed\([^\n]+km/);
  assert.match(clientSource, /const pad = 0;/);
});

test("landing header uses one wordmark scale and links back to the level selector", () => {
  assert.match(indexSource, /class="brand-product">RPO DUEL<\/strong>/);
  assert.match(stylesSource, /\.brand strong \{ color: var\(--text\); font-weight: 700; \}/);
  assert.match(stylesSource, /\.brand-product \{ font-size: inherit; \}/);
  assert.match(indexSource, /id="level-selector-link"[^>]+>Level Selector<\/a>/);
  assert.match(indexSource, /id="frame-convention-button"[^>]+>Frame: OEL<\/button>/);
  assert.match(indexSource, /name="oel-level-selector-url" content="https:\/\/orbital-engagement-lab\.vercel\.app\/"/);
  assert.match(clientSource, /localHost \? "\/trainer\/" : HOSTED_LEVEL_SELECTOR_URL/);
  assert.match(clientSource, /frameConvention: frameConventionFromSearch\(location\.search\)/);
  assert.match(clientSource, /window\.history\.replaceState\(window\.history\.state, "", currentUrl\.href\)/);
  assert.match(clientSource, /spatialSign \* inTrackDisplaySign/);
  assert.match(clientSource, /binding\[1\] \* inputSign/);
  assert.match(clientSource, /frame-convention-button"\]\.addEventListener\("click", toggleFrameConvention\)/);
  assert.doesNotMatch(indexSource, /id="connection-pill"/);
  assert.doesNotMatch(clientSource, /setConnectionPill/);
});

test("match-complete overlay offers a server-owned rematch and a clean lobby return", () => {
  assert.match(indexSource, /id="play-again"[^>]*>Play Again<\/button>/);
  assert.match(indexSource, /id="return-lobby"[^>]*>Return to Lobby<\/button>/);
  assert.match(clientSource, /JSON\.stringify\(\{ type: "rematch" \}\)/);
  assert.match(clientSource, /sessionStorage\.removeItem\("rpo-duel-session"\)/);
  assert.match(clientSource, /url\.searchParams\.delete\("room"\)/);
  assert.match(roomSource, /allRematchPlayersReady/);
  assert.match(roomSource, /deterministicSeed\(this\.matchSeed, this\.rematchIndex, 0x52454d54\)/);
});
