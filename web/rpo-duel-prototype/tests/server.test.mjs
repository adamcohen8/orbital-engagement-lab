import assert from "node:assert/strict";
import { once } from "node:events";
import { test } from "node:test";

import WebSocket from "ws";

import { createPrototypeServer, DuelRoom } from "../server/server.mjs";
import { DUEL_PROTOTYPE_RULES } from "../src/shared/duel-engine.js";

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
});

function roleFor(room, playerId) {
  const roles = room.series.snapshot().roles;
  return roles.chaser === playerId ? "chaser" : "target";
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
  return new WebSocket(`${wsOrigin}/ws?room=${encodeURIComponent(roomCode)}&token=${encodeURIComponent(token)}`);
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
