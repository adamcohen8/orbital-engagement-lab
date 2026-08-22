import assert from "node:assert/strict";
import { once } from "node:events";
import { test } from "node:test";

import WebSocket from "ws";

import { createPrototypeServer, DuelRoom } from "../server/server.mjs";
import { DUEL_PROTOTYPE_RULES } from "../src/shared/duel-engine.js";
import {
  DUEL_COMPUTER_CADENCE,
  DuelRoomCore,
  materiallyImprovesComputerOutcome,
} from "../src/shared/duel-room.js";

const shortRules = {
  ...DUEL_PROTOTYPE_RULES,
  round_duration_s: 60,
  capture_range_km: 0.000001,
};

test("disconnect immediately neutralizes the player while the authoritative round continues", () => {
  const now = 1_000_000;
  const room = new DuelRoom({ code: "COAST1", regulationRounds: 2, matchSeed: 91, now, rules: shortRules });
  const first = room.addPlayer("Alpha", now);
  const second = room.addPlayer("Bravo", now);
  const firstSocket = fakeSocket();
  const secondSocket = fakeSocket();
  room.connect(first.token, firstSocket, now);
  room.connect(second.token, secondSocket, now);
  assert.equal(room.phase, "countdown");

  room.tick(now + 3001);
  assert.equal(room.phase, "active");
  room.receive(first.player.id, JSON.stringify({ type: "input", sequence: 1, controls: { r: 1, i: 0, c: 0 } }), now + 3002);
  const role = roleFor(room, first.player.id);
  assert.equal(room.series.snapshot().round.controls[role].r, 1);

  room.disconnect(first.player.id, firstSocket, now + 3010);
  assert.equal(room.series.snapshot().round.controls[role].r, 0);
  assert.equal(room.players.get(first.player.id).connected, false);
  const tickBefore = room.series.snapshot().round.tick;
  room.tick(now + 3110);
  assert.ok(room.series.snapshot().round.tick > tickBefore, "the round should not pause for a disconnected player");
});

test("200x coast retains a bounded backlog and catches up after a delayed tick", () => {
  const now = 1_500_000;
  const room = new DuelRoom({
    code: "FAST20",
    regulationRounds: 2,
    matchSeed: 200,
    now,
    rules: { ...DUEL_PROTOTYPE_RULES, round_duration_s: 1000, capture_range_km: 1.0e-9 },
  });
  const first = room.addPlayer("Alpha", now);
  const second = room.addPlayer("Bravo", now);
  room.connect(first.token, fakeSocket(), now);
  room.connect(second.token, fakeSocket(), now);

  room.tick(now + 2900);
  room.tick(now + 3000);
  assert.equal(room.phase, "active");
  const initialTick = room.series.snapshot().round.tick;

  room.tick(now + 3100);
  assert.equal(room.series.snapshot().round.tick - initialTick, 20);
  assert.equal(room.stepAccumulatorS, 0);
  assert.deepEqual(room.speedState, { speed_multiple: 200, reason: "coasting" });

  const beforeDelay = room.series.snapshot().round.tick;
  room.tick(now + 3350);
  assert.equal(room.series.snapshot().round.tick - beforeDelay, 30);
  assert.equal(room.stepAccumulatorS, 20);

  room.tick(now + 3450);
  assert.equal(room.stepAccumulatorS, 10);
  room.tick(now + 3550);
  assert.equal(room.series.snapshot().round.tick - beforeDelay, 90);
  assert.equal(room.stepAccumulatorS, 0);
});

test("newer input sequences cannot be overwritten by delayed messages", () => {
  const now = 2_000_000;
  const room = new DuelRoom({ code: "ORDER1", regulationRounds: 2, matchSeed: 92, now, rules: shortRules });
  const first = room.addPlayer("Alpha", now);
  const second = room.addPlayer("Bravo", now);
  room.connect(first.token, fakeSocket(), now);
  room.connect(second.token, fakeSocket(), now);
  room.tick(now + 3001);
  const role = roleFor(room, first.player.id);
  room.receive(first.player.id, JSON.stringify({ type: "input", sequence: 8, controls: { r: 0, i: 1, c: 0 } }), now + 3002);
  room.receive(first.player.id, JSON.stringify({ type: "input", sequence: 7, controls: { r: -1, i: 0, c: 0 } }), now + 3003);
  assert.deepEqual(room.series.snapshot().round.controls[role], { r: 0, i: 1, c: 0 });
});

test("a reconnected client may restart sequences while the replaced socket is ignored", () => {
  const now = 2_500_000;
  const room = new DuelRoom({ code: "REJOIN1", regulationRounds: 2, matchSeed: 93, now, rules: shortRules });
  const first = room.addPlayer("Alpha", now);
  const second = room.addPlayer("Bravo", now);
  const oldSocket = fakeSocket();
  room.connect(first.token, oldSocket, now);
  room.connect(second.token, fakeSocket(), now);
  room.tick(now + 3001);
  const role = roleFor(room, first.player.id);
  room.receive(first.player.id, JSON.stringify({ type: "input", sequence: 12, controls: { r: 1, i: 0, c: 0 } }), now + 3002, oldSocket);

  const newSocket = fakeSocket();
  room.connect(first.token, newSocket, now + 3003);
  assert.deepEqual(room.series.snapshot().round.controls[role], { r: 0, i: 0, c: 0 });
  room.receive(first.player.id, JSON.stringify({ type: "input", sequence: 1, controls: { r: 0, i: 1, c: 0 } }), now + 3004, newSocket);
  room.receive(first.player.id, JSON.stringify({ type: "input", sequence: 13, controls: { r: -1, i: 0, c: 0 } }), now + 3005, oldSocket);
  assert.deepEqual(room.series.snapshot().round.controls[role], { r: 0, i: 1, c: 0 });
});

test("computer rooms start with one human and a server-owned policy opponent", () => {
  const now = 2_800_000;
  const room = new DuelRoom({
    code: "BOT001",
    regulationRounds: 2,
    matchSeed: 95,
    now,
    matchMode: "computer",
    rules: { ...DUEL_PROTOTYPE_RULES, round_duration_s: 180 },
  });
  const human = room.addPlayer("Pilot", now);
  const computer = room.addComputerOpponent(now);
  assert.equal(computer.kind, "computer");
  assert.equal(room.publicSummary().joinable, false);
  assert.equal(room.hasConnectedPlayers(), false, "the logical computer connection must not keep a room alive");

  room.connect(human.token, fakeSocket(), now);
  assert.equal(room.phase, "countdown");
  room.tick(now + 3001);
  const snapshot = room.snapshotFor(human.player.id, now + 3001);
  const computerRole = snapshot.series.roles.chaser === computer.id ? "chaser" : "target";
  const event = room.series.result().current_round.input_events.find((item) => item.player_id === computer.id);
  assert.equal(event.source, "computer_policy");
  assert.equal(event.role, computerRole);
  assert.match(event.policy_phase, computerRole === "chaser" ? /intercept/ : /evasion/);
  assert.equal(snapshot.computer_opponent.policy_version, "rpo-duel.computer-policy.v1");
  assert.equal(snapshot.computer_opponent.cadence_version, "rpo-duel.computer-cadence.v2");
});

test("computer benefit gate accepts decisive changes and rejects marginal burns", () => {
  const coast = (capture, range) => ({
    acceleration_ric_m_s2: [0, 0, 0],
    predicted_capture_time_s: capture,
    predicted_closest_range_m: range,
    predicted_closest_time_s: 600,
    phase: "coast",
  });
  const burn = (capture, range) => ({
    acceleration_ric_m_s2: [0.01, 0, 0],
    predicted_capture_time_s: capture,
    predicted_closest_range_m: range,
    predicted_closest_time_s: 500,
    phase: "burn",
  });

  assert.equal(materiallyImprovesComputerOutcome("chaser", burn(700, 80), coast(null, 500)).phase, "burn");
  assert.equal(materiallyImprovesComputerOutcome("chaser", burn(null, 350), coast(null, 500)).phase, "burn");
  assert.equal(
    materiallyImprovesComputerOutcome("chaser", burn(null, 450), coast(null, 500)).phase,
    "intercept_benefit_gate_coast",
  );
  assert.equal(materiallyImprovesComputerOutcome("target", burn(null, 300), coast(800, 80)).phase, "burn");
  assert.equal(materiallyImprovesComputerOutcome("target", burn(null, 250), coast(null, 100)).phase, "burn");
  assert.equal(
    materiallyImprovesComputerOutcome("target", burn(null, 1500), coast(null, 1000)).phase,
    "predictive_evasion_benefit_gate_coast",
  );
  assert.equal(
    materiallyImprovesComputerOutcome("target", burn(null, 150), coast(null, 100)).phase,
    "predictive_evasion_benefit_gate_coast",
  );
});

test("computer Chaser burns for material progress, then coasts before a responsive replan", () => {
  const { room, computer, now } = createComputerCadenceRoom(1);
  assert.equal(roleFor(room, computer.id), "chaser");

  room.tick(now + 3001);
  assert.equal(room.computerController.next_plan_time_s, DUEL_COMPUTER_CADENCE.decision_interval_s);
  assert.equal(room.computerController.action_until_time_s, DUEL_COMPUTER_CADENCE.pulse_duration_s);
  const firstEvent = room.series.result().current_round.input_events.find((item) => (
    item.player_id === computer.id && item.source === "computer_policy"
  ));
  assert.ok(firstEvent, "the Chaser should record its material opening burn");
  assert.ok(
    Math.abs(firstEvent.controls.r) + Math.abs(firstEvent.controls.i) + Math.abs(firstEvent.controls.c) > 0,
    "the fixture must exercise a burn before the coast window",
  );

  const speedSamples = [];
  for (let elapsedMs = 3101; elapsedMs <= 7001; elapsedMs += 100) {
    room.tick(now + elapsedMs);
    speedSamples.push(room.speedState.speed_multiple);
  }
  assert.ok(speedSamples.filter((speed) => speed === 200).length >= 2);
  assert.equal(room.computerController.sequence, 2, "only the opening burn and neutralization should be recorded");

  let elapsedMs = 7101;
  while (
    room.phase === "active"
    && room.computerController.phase !== "intercept_benefit_gate_coast"
    && room.series.snapshot().round.time_s < 1300
  ) {
    room.tick(now + elapsedMs);
    elapsedMs += 100;
  }
  assert.equal(room.computerController.phase, "intercept_benefit_gate_coast");
  assert.ok(room.series.snapshot().round.time_s < 1200, "the Chaser should stop burning once progress becomes marginal");
  assert.equal(room.speedState.speed_multiple, 200);
});

test("computer Target coasts when the passive prediction is safely outside its guard range", () => {
  const { room, computer, now } = createComputerCadenceRoom(2);
  assert.equal(roleFor(room, computer.id), "target");

  room.tick(now + 3001);
  assert.equal(room.speedState.speed_multiple, 200);
  assert.equal(room.computerController.phase, "predictive_evasion_benefit_gate_coast");
  assert.equal(room.computerController.next_plan_time_s, DUEL_COMPUTER_CADENCE.decision_interval_s);
  assert.equal(room.computerController.action_until_time_s, DUEL_COMPUTER_CADENCE.decision_interval_s);
  assert.equal(room.series.result().current_round.input_events.length, 0);

  for (let elapsedMs = 3101; elapsedMs <= 3401; elapsedMs += 100) room.tick(now + elapsedMs);
  assert.equal(room.speedState.speed_multiple, 200);
  assert.equal(room.computerController.phase, "predictive_evasion_benefit_gate_coast");
  assert.equal(room.computerController.next_plan_time_s, 2 * DUEL_COMPUTER_CADENCE.decision_interval_s);
  assert.equal(room.series.result().current_round.input_events.length, 0);
});

test("computer policy timing and role survive room persistence", () => {
  const now = 2_900_000;
  const room = new DuelRoom({
    code: "BOT002",
    regulationRounds: 2,
    matchSeed: 96,
    now,
    matchMode: "computer",
    rules: { ...DUEL_PROTOTYPE_RULES, round_duration_s: 180 },
  });
  const human = room.addPlayer("Pilot", now);
  room.addComputerOpponent(now);
  room.connect(human.token, fakeSocket(), now);
  room.tick(now + 3001);
  room.tick(now + 3101);

  const restored = DuelRoomCore.restore(room.serialize(), {
    rules: room.rules,
    tokenFactory: () => "unused-token",
  });
  assert.equal(restored.matchMode, "computer");
  assert.equal(restored.players.get(restored.computerPlayerId).connected, true);
  assert.deepEqual(restored.computerController, room.computerController);
  assert.deepEqual(restored.series.snapshot(), room.series.snapshot());
  assert.deepEqual(
    restored.series.result().current_round.input_events,
    room.series.result().current_round.input_events,
  );
  assert.equal(restored.hasConnectedPlayers(), false);
});

test("the computer alternates between pursuit and evasion roles", () => {
  const now = 2_950_000;
  const room = new DuelRoom({
    code: "BOT003",
    regulationRounds: 2,
    matchSeed: 97,
    now,
    matchMode: "computer",
    rules: { ...DUEL_PROTOTYPE_RULES, round_duration_s: 1 },
  });
  const human = room.addPlayer("Pilot", now);
  const computer = room.addComputerOpponent(now);
  room.connect(human.token, fakeSocket(), now);
  const firstRole = roleFor(room, computer.id);
  room.tick(now + 3001);
  assert.equal(room.phase, "round_complete");
  room.tick(room.phaseEndsAt + 1);
  assert.equal(room.phase, "countdown");
  assert.notEqual(roleFor(room, computer.id), firstRole);
});

test("human rematches wait for both players and survive room persistence", () => {
  const now = 3_000_000;
  const room = new DuelRoom({ code: "AGAIN1", regulationRounds: 2, matchSeed: 101, now, rules: shortRules });
  const first = room.addPlayer("Alpha", now);
  const second = room.addPlayer("Bravo", now);
  room.connect(first.token, fakeSocket(), now);
  room.connect(second.token, fakeSocket(), now);
  const completedAt = finishMatch(room, now);
  const previousSeed = room.series.snapshot().match_seed;

  assert.equal(
    room.receive(first.player.id, JSON.stringify({ type: "rematch" }), completedAt + 1),
    true,
  );
  assert.equal(room.phase, "complete");
  assert.equal(room.snapshotFor(first.player.id).rematch.your_ready, true);
  assert.deepEqual(room.snapshotFor(second.player.id).rematch.ready_player_ids, [first.player.id]);

  const restored = DuelRoomCore.restore(room.serialize(), { rules: room.rules, tokenFactory: () => "unused-token" });
  assert.equal(restored.rematchIndex, 0);
  assert.deepEqual([...restored.rematchReady], [first.player.id]);
  assert.equal(restored.phase, "complete");

  room.receive(second.player.id, JSON.stringify({ type: "rematch" }), completedAt + 2);
  assert.equal(room.phase, "countdown");
  assert.equal(room.rematchIndex, 1);
  assert.equal(room.rematchReady.size, 0);
  assert.notEqual(room.series.snapshot().match_seed, previousSeed);
  assert.deepEqual(room.series.snapshot().score, { [first.player.id]: 0, [second.player.id]: 0 });
});

test("computer rematches start as soon as the human is ready", () => {
  const now = 3_100_000;
  const room = new DuelRoom({
    code: "AGAIN2",
    regulationRounds: 2,
    matchSeed: 102,
    now,
    matchMode: "computer",
    rules: { ...shortRules, capture_range_km: 0.1 },
  });
  const human = room.addPlayer("Pilot", now);
  room.addComputerOpponent(now);
  room.connect(human.token, fakeSocket(), now);
  const completedAt = finishMatch(room, now);

  room.receive(human.player.id, JSON.stringify({ type: "rematch" }), completedAt + 1);
  assert.equal(room.phase, "countdown");
  assert.equal(room.rematchIndex, 1);
  assert.equal(room.computerController.round_index, null);
  assert.equal(room.snapshotFor(human.player.id).rematch.required_human_players, 1);
});

test("HTTP room creation and two authenticated WebSocket clients share one room", async (t) => {
  const { httpServer, rooms } = createPrototypeServer({ roomRules: shortRules });
  httpServer.listen(0, "127.0.0.1");
  await once(httpServer, "listening");
  const address = httpServer.address();
  const origin = `http://127.0.0.1:${address.port}`;
  const sockets = [];
  t.after(async () => {
    for (const socket of sockets) socket.close();
    await new Promise((resolve) => httpServer.close(resolve));
  });

  const page = await fetch(origin);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /RPO Duel Beta/);

  const music = await fetch(`${origin}/assets/39_perigee_afterburner_demo.wav`);
  assert.equal(music.status, 200);
  assert.equal(music.headers.get("content-type"), "audio/wav");
  await music.body.cancel();

  const trainerPage = await fetch(`${origin}/trainer/`);
  assert.equal(trainerPage.status, 200);
  assert.match(await trainerPage.text(), /data-level-option="rpoDuel"/);
  const duelAlias = await fetch(`${origin}/rpo-duel-prototype/`);
  assert.equal(duelAlias.status, 200);
  assert.match(await duelAlias.text(), /RPO Duel Beta/);

  const created = await postJson(`${origin}/api/rooms`, { name: "Laptop", regulation_rounds: 2 });
  assert.match(created.room_code, /^[A-Z2-9]{6}$/);
  assert.equal(created.player.name, "Laptop");
  const joined = await postJson(`${origin}/api/rooms/${created.room_code}/join`, { name: "Phone" });
  assert.equal(joined.players.length, 2);
  assert.equal(rooms.get(created.room_code).phase, "waiting");

  const firstSocket = openSocket(origin, created.room_code, created.reconnect_token);
  const secondSocket = openSocket(origin, created.room_code, joined.reconnect_token);
  sockets.push(firstSocket, secondSocket);
  const [firstSnapshot, secondSnapshot] = await Promise.all([nextSnapshot(firstSocket), nextSnapshot(secondSocket)]);
  assert.equal(firstSnapshot.room_code, created.room_code);
  assert.equal(secondSnapshot.room_code, created.room_code);
  assert.notEqual(firstSnapshot.you.id, secondSnapshot.you.id);
  assert.equal(rooms.get(created.room_code).phase, "countdown");

  const solo = await postJson(`${origin}/api/rooms`, {
    name: "Solo",
    regulation_rounds: 2,
    opponent: "computer",
  });
  assert.equal(solo.match_mode, "computer");
  assert.equal(solo.players.length, 2);
  assert.equal(solo.players[1].kind, "computer");
  assert.equal(solo.joinable, false);
  const soloSocket = openSocket(origin, solo.room_code, solo.reconnect_token);
  sockets.push(soloSocket);
  const soloSnapshot = await nextSnapshot(soloSocket);
  assert.equal(soloSnapshot.computer_opponent.player_id, solo.players[1].id);
  assert.equal(rooms.get(solo.room_code).phase, "countdown");
});

function roleFor(room, playerId) {
  const roles = room.series.snapshot().roles;
  return roles.chaser === playerId ? "chaser" : "target";
}

function createComputerCadenceRoom(matchSeed) {
  const now = 2_850_000 + matchSeed * 10_000;
  const room = new DuelRoom({
    code: `PACE0${matchSeed}`,
    regulationRounds: 2,
    matchSeed,
    now,
    matchMode: "computer",
    rules: { ...DUEL_PROTOTYPE_RULES, round_duration_s: 2000, capture_range_km: 0.021 },
  });
  const human = room.addPlayer("Pilot", now);
  const computer = room.addComputerOpponent(now);
  room.connect(human.token, fakeSocket(), now);
  return { room, computer, now };
}

function finishMatch(room, now) {
  let current = now + room.timing.countdown_ms + 1;
  room.tick(current);
  while (room.phase !== "complete") {
    if (room.phase === "active") {
      room.series.step(Math.ceil(room.rules.round_duration_s / room.rules.dt_s));
      current += 1;
      room.tick(current);
    } else if (room.phase === "round_complete" || room.phase === "countdown") {
      current = room.phaseEndsAt + 1;
      room.tick(current);
    } else {
      throw new Error(`Unexpected room phase while completing match: ${room.phase}`);
    }
  }
  return current;
}

function fakeSocket() {
  return { readyState: 1, messages: [], send(value) { this.messages.push(value); }, close() { this.readyState = 3; } };
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const value = await response.json();
  assert.equal(response.ok, true, value.error);
  return value;
}

function openSocket(origin, roomCode, token) {
  const wsOrigin = origin.replace(/^http/, "ws");
  return new WebSocket(
    `${wsOrigin}/ws?room=${encodeURIComponent(roomCode)}`,
    ["oel-rpo-duel-v1", `oel-token.${token}`],
  );
}

async function nextSnapshot(socket) {
  if (socket.readyState !== WebSocket.OPEN) await once(socket, "open");
  return await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Timed out waiting for snapshot.")), 3000);
    socket.on("message", function onMessage(raw) {
      const message = JSON.parse(String(raw));
      if (message.type !== "snapshot") return;
      clearTimeout(timeout);
      socket.off("message", onMessage);
      resolve(message);
    });
  });
}
