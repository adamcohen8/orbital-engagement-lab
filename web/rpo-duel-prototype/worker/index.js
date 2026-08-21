import { DUEL_PROTOTYPE_RULES } from "../src/shared/duel-engine.js";
import {
  DuelRoomCore,
  normalizeRoomCode,
  RoomError,
} from "../src/shared/duel-room.js";

const MAX_BODY_BYTES = 8192;
const ROOM_TICK_MS = 100;
const PERSIST_INTERVAL_MS = 5000;
const SECURITY_HEADERS = Object.freeze({
  "Content-Security-Policy": "default-src 'self'; connect-src 'self' wss:; img-src 'self' data:; style-src 'self'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
});

export default {
  async fetch(request, env) {
    try {
      const url = new URL(request.url);
      if (url.pathname.startsWith("/api/") || url.pathname === "/ws") {
        assertSameOrigin(request);
        return await routeRoomRequest(request, env);
      }
      return withSecurityHeaders(await env.ASSETS.fetch(request));
    } catch (error) {
      return errorResponse(error);
    }
  },
};

export class RpoDuelRoom {
  constructor(state) {
    this.state = state;
    this.room = null;
    this.timer = null;
    this.lastPersistAt = 0;
    this.state.blockConcurrencyWhile(async () => {
      const saved = await this.state.storage.get("room");
      if (saved) this.room = DuelRoomCore.restore(saved, { tokenFactory: createReconnectToken });
    });
  }

  async fetch(request) {
    try {
      const url = new URL(request.url);
      if (request.method === "POST" && url.pathname === "/api/rooms") {
        if (this.room) throw new RoomError(409, "Room code is already in use.");
        const body = await readJson(request);
        this.room = new DuelRoomCore({
          code: request.headers.get("x-oel-room-code"),
          regulationRounds: body.regulation_rounds,
          matchSeed: body.match_seed,
          tokenFactory: createReconnectToken,
        });
        const joined = this.room.addPlayer(body.name);
        await this.persist();
        return jsonResponse(201, {
          ...this.room.publicSummary(),
          player: joined.player,
          reconnect_token: joined.token,
        });
      }

      if (!this.room) throw new RoomError(404, "Room not found.");
      const route = url.pathname.match(/^\/api\/rooms\/([A-Za-z0-9]+)(?:\/(join))?$/);
      if (route) {
        if (normalizeRoomCode(route[1]) !== this.room.code) throw new RoomError(404, "Room not found.");
        if (request.method === "GET" && !route[2]) return jsonResponse(200, this.room.publicSummary());
        if (request.method === "POST" && route[2] === "join") {
          const body = await readJson(request);
          const joined = this.room.addPlayer(body.name);
          await this.persist();
          return jsonResponse(200, {
            ...this.room.publicSummary(),
            player: joined.player,
            reconnect_token: joined.token,
          });
        }
      }
      if (request.method === "GET" && url.pathname === "/ws") return await this.acceptSocket(request, url);
      throw new RoomError(404, "Room route not found.");
    } catch (error) {
      return errorResponse(error);
    }
  }

  async acceptSocket(request, url) {
    if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
      throw new RoomError(426, "WebSocket upgrade required.");
    }
    const roomCode = normalizeRoomCode(url.searchParams.get("room"));
    if (roomCode !== this.room.code) throw new RoomError(404, "Room not found.");
    const pair = new WebSocketPair();
    const client = pair[0];
    const server = pair[1];
    server.accept();
    const player = this.room.connect(url.searchParams.get("token"), server);
    server.addEventListener("message", (event) => {
      try {
        this.room?.receive(player.id, event.data, Date.now(), server);
        this.room?.tick(Date.now());
        this.state.waitUntil(this.persist());
      } catch (error) {
        server.send(JSON.stringify({ type: "error", error: error.message }));
      }
    });
    const disconnect = () => {
      this.room?.disconnect(player.id, server);
      this.stopTimerIfIdle();
      this.state.waitUntil(this.persist());
    };
    server.addEventListener("close", disconnect);
    server.addEventListener("error", disconnect);
    this.startTimer();
    await this.persist();
    return new Response(null, { status: 101, webSocket: client });
  }

  startTimer() {
    if (this.timer !== null) return;
    this.timer = setInterval(() => {
      if (!this.room) return;
      const now = Date.now();
      this.room.tick(now);
      if (now - this.lastPersistAt >= PERSIST_INTERVAL_MS) this.state.waitUntil(this.persist(now));
      this.stopTimerIfIdle();
    }, ROOM_TICK_MS);
  }

  stopTimerIfIdle() {
    if (this.timer === null || this.room?.hasConnectedPlayers()) return;
    clearInterval(this.timer);
    this.timer = null;
  }

  async persist(now = Date.now()) {
    if (!this.room) return;
    this.lastPersistAt = now;
    await this.state.storage.put("room", this.room.serialize());
    await this.state.storage.setAlarm(this.room.expiresAt(now));
  }

  async alarm() {
    if (this.room?.hasConnectedPlayers()) {
      await this.persist();
      return;
    }
    this.stopTimerIfIdle();
    this.room = null;
    await this.state.storage.deleteAll();
  }
}

async function routeRoomRequest(request, env) {
  const url = new URL(request.url);
  if (request.method === "POST" && url.pathname === "/api/rooms") {
    const body = await readJson(request);
    for (let attempt = 0; attempt < 12; attempt += 1) {
      const code = createRoomCode();
      const stub = env.DUEL_ROOMS.getByName(code);
      const headers = new Headers(request.headers);
      headers.set("Content-Type", "application/json");
      headers.set("x-oel-room-code", code);
      const forwarded = new Request(request.url, {
        method: "POST",
        headers,
        body: JSON.stringify({ ...body, match_seed: randomUint32() }),
      });
      const response = await stub.fetch(forwarded);
      if (response.status !== 409) return withSecurityHeaders(response);
    }
    throw new RoomError(503, "Unable to allocate a room code.");
  }

  const route = url.pathname.match(/^\/api\/rooms\/([A-Za-z0-9]+)(?:\/join)?$/);
  const code = normalizeRoomCode(route?.[1] || url.searchParams.get("room"));
  if (!code) throw new RoomError(400, "Room code is required.");
  const stub = env.DUEL_ROOMS.getByName(code);
  return withSecurityHeaders(await stub.fetch(request));
}

function assertSameOrigin(request) {
  const origin = request.headers.get("Origin");
  if (origin && origin !== new URL(request.url).origin) throw new RoomError(403, "Cross-origin room access is not allowed.");
}

async function readJson(request) {
  const text = await request.text();
  if (new TextEncoder().encode(text).byteLength > MAX_BODY_BYTES) {
    throw new RoomError(413, "Request body is too large.");
  }
  try {
    return JSON.parse(text || "{}");
  } catch {
    throw new RoomError(400, "Request body must be valid JSON.");
  }
}

function createRoomCode() {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  const values = new Uint8Array(6);
  crypto.getRandomValues(values);
  return [...values].map((value) => alphabet[value % alphabet.length]).join("");
}

function createReconnectToken() {
  const values = new Uint8Array(24);
  crypto.getRandomValues(values);
  return btoa(String.fromCharCode(...values)).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

function randomUint32() {
  const value = new Uint32Array(1);
  crypto.getRandomValues(value);
  return value[0];
}

function withSecurityHeaders(response) {
  const headers = new Headers(response.headers);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) headers.set(name, value);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
    webSocket: response.webSocket,
  });
}

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
  });
}

function errorResponse(error) {
  const status = error instanceof RoomError ? error.status : 500;
  return withSecurityHeaders(jsonResponse(status, {
    error: error instanceof Error ? error.message : String(error),
  }));
}

export { DUEL_PROTOTYPE_RULES };
