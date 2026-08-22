import { DUEL_PROTOTYPE_RULES } from "../src/shared/duel-engine.js";
import {
  DuelRoomCore,
  normalizeRoomCode,
  RoomError,
} from "../src/shared/duel-room.js";

const MAX_BODY_BYTES = 8192;
const ROOM_TICK_MS = 100;
const PERSIST_INTERVAL_MS = 5000;
const DISCONNECTED_ALARM_MS = 5000;
const MAX_ALARM_CATCHUP_TICKS = 200;
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
  constructor(state, env) {
    this.state = state;
    this.env = env;
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
          matchMode: body.opponent,
          tokenFactory: createReconnectToken,
        });
        const joined = this.room.addPlayer(body.name);
        if (this.room.matchMode === "computer") this.room.addComputerOpponent();
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
    const token = reconnectTokenFromProtocols(request.headers.get("Sec-WebSocket-Protocol"));
    const player = this.room.connect(token, server);
    server.addEventListener("message", (event) => {
      try {
        const changed = this.room?.receive(player.id, event.data, Date.now(), server);
        this.room?.tick(Date.now());
        if (changed) this.state.waitUntil(this.persist());
      } catch (error) {
        server.send(JSON.stringify({ type: "error", error: error.message }));
        if (error instanceof RoomError && error.status === 429) server.close(4008, "Message rate limit exceeded.");
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
    return new Response(null, {
      status: 101,
      webSocket: client,
      headers: { "Sec-WebSocket-Protocol": "oel-rpo-duel-v1" },
    });
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
    if (this.timer === null || this.room?.shouldKeepTicking()) return;
    clearInterval(this.timer);
    this.timer = null;
  }

  async persist(now = Date.now()) {
    if (!this.room) return;
    this.lastPersistAt = now;
    await this.state.storage.put("room", this.room.serialize());
    const expiry = this.room.expiresAt(now);
    const nextAlarm = !this.room.hasConnectedPlayers() && this.room.shouldKeepTicking()
      ? Math.min(expiry, now + DISCONNECTED_ALARM_MS)
      : expiry;
    await this.state.storage.setAlarm(nextAlarm);
    if (this.env?.DUEL_ADMISSION) {
      const admission = this.env.DUEL_ADMISSION.getByName("global");
      await admission.fetch("https://oel.internal/renew", {
        method: "POST",
        body: JSON.stringify({ room_code: this.room.code, expires_at_ms: expiry }),
      });
    }
  }

  async alarm() {
    if (!this.room) return;
    const now = Date.now();
    if (this.room.expiresAt(now) <= now) {
      const roomCode = this.room.code;
      if (this.timer !== null) clearInterval(this.timer);
      this.timer = null;
      this.room = null;
      await this.state.storage.deleteAll();
      await this.releaseAdmission(roomCode);
      return;
    }
    if (!this.room.hasConnectedPlayers() && this.room.shouldKeepTicking()) {
      let catchupTicks = 0;
      while (this.room.lastTickAt < now && this.room.shouldKeepTicking() && catchupTicks < MAX_ALARM_CATCHUP_TICKS) {
        this.room.tick(Math.min(this.room.lastTickAt + this.room.timing.max_wall_delta_ms, now));
        catchupTicks += 1;
      }
    }
    await this.persist(now);
  }

  async releaseAdmission(roomCode) {
    if (!this.env?.DUEL_ADMISSION || !roomCode) return;
    const admission = this.env.DUEL_ADMISSION.getByName("global");
    await admission.fetch("https://oel.internal/release", {
      method: "POST",
      body: JSON.stringify({ room_code: roomCode }),
    });
  }
}

export class RpoDuelAdmission {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    const url = new URL(request.url);
    const body = await readJson(request);
    const roomCode = normalizeRoomCode(body.room_code);
    if (!roomCode) return jsonResponse(400, { error: "room_code is required" });
    if (url.pathname === "/release") {
      await this.state.storage.delete(`room:${roomCode}`);
      return jsonResponse(200, { status: "released" });
    }
    const expiresAt = Math.max(Number(body.expires_at_ms) || 0, Date.now() + 1000);
    if (url.pathname === "/renew") {
      await this.state.storage.put(`room:${roomCode}`, { expires_at_ms: expiresAt });
      return jsonResponse(200, { status: "renewed" });
    }
    if (url.pathname !== "/admit") return jsonResponse(404, { error: "admission route not found" });

    const now = Date.now();
    const rooms = await this.state.storage.list({ prefix: "room:" });
    for (const [key, value] of rooms) {
      if (Number(value?.expires_at_ms || 0) <= now) {
        await this.state.storage.delete(key);
        rooms.delete(key);
      }
    }
    const maximum = Math.max(1, Number(this.env.DUEL_MAX_ACTIVE_ROOMS || 25));
    if (!rooms.has(`room:${roomCode}`) && rooms.size >= maximum) {
      return jsonResponse(503, { error: "Duel room capacity is currently full." });
    }

    const clientKey = String(body.client_key || "");
    if (!clientKey) return jsonResponse(403, { error: "Client admission identity is required." });
    const rateKey = `rate:${clientKey}`;
    const windowMs = Math.max(1000, Number(this.env.DUEL_CREATE_RATE_WINDOW_MS || 600000));
    const limit = Math.max(1, Number(this.env.DUEL_CREATE_RATE_LIMIT || 5));
    const current = await this.state.storage.get(rateKey);
    const rate = !current || now - Number(current.started_at_ms || 0) >= windowMs
      ? { started_at_ms: now, count: 0 }
      : current;
    rate.count += 1;
    if (rate.count > limit) return jsonResponse(429, { error: "Duel room creation rate limit exceeded." });
    await this.state.storage.put(rateKey, rate, { expirationTtl: Math.ceil(windowMs / 1000) });
    await this.state.storage.put(`room:${roomCode}`, { expires_at_ms: expiresAt });
    return jsonResponse(200, { status: "admitted", active_rooms: rooms.size + 1, maximum });
  }
}

async function routeRoomRequest(request, env) {
  const url = new URL(request.url);
  if (request.method === "POST" && url.pathname === "/api/rooms") {
    if (String(env.DUEL_CREATE_ENABLED || "").toLowerCase() !== "true") {
      throw new RoomError(503, "Duel room creation is disabled by the operator.");
    }
    const body = await readJson(request);
    const clientAddress = request.headers.get("CF-Connecting-IP");
    if (!clientAddress) throw new RoomError(403, "Cloudflare client identity is required for room creation.");
    const clientKey = await sha256Text(clientAddress);
    for (let attempt = 0; attempt < 12; attempt += 1) {
      const code = createRoomCode();
      const admission = env.DUEL_ADMISSION.getByName("global");
      const admitted = await admission.fetch("https://oel.internal/admit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          room_code: code,
          client_key: clientKey,
          expires_at_ms: Date.now() + 30 * 60 * 1000,
        }),
      });
      if (!admitted.ok) return withSecurityHeaders(admitted);
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
      await admission.fetch("https://oel.internal/release", {
        method: "POST",
        body: JSON.stringify({ room_code: code }),
      });
    }
    throw new RoomError(503, "Unable to allocate a room code.");
  }

  const route = url.pathname.match(/^\/api\/rooms\/([A-Za-z0-9]+)(?:\/join)?$/);
  const code = normalizeRoomCode(route?.[1] || url.searchParams.get("room"));
  if (!code) throw new RoomError(400, "Room code is required.");
  const stub = env.DUEL_ROOMS.getByName(code);
  return withSecurityHeaders(await stub.fetch(request));
}

async function sha256Text(value) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(String(value)));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function assertSameOrigin(request) {
  const origin = request.headers.get("Origin");
  if (origin && origin !== new URL(request.url).origin) throw new RoomError(403, "Cross-origin room access is not allowed.");
}

async function readJson(request) {
  const declaredLength = Number(request.headers.get("Content-Length") || 0);
  if (Number.isFinite(declaredLength) && declaredLength > MAX_BODY_BYTES) {
    throw new RoomError(413, "Request body is too large.");
  }
  const reader = request.body?.getReader();
  const chunks = [];
  let size = 0;
  if (reader) {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY_BYTES) {
        await reader.cancel("Request body is too large.");
        throw new RoomError(413, "Request body is too large.");
      }
      chunks.push(value);
    }
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  const text = new TextDecoder().decode(bytes);
  try {
    return JSON.parse(text || "{}");
  } catch {
    throw new RoomError(400, "Request body must be valid JSON.");
  }
}

function reconnectTokenFromProtocols(value) {
  const protocols = String(value || "").split(",").map((item) => item.trim());
  if (!protocols.includes("oel-rpo-duel-v1")) throw new RoomError(400, "Duel WebSocket protocol is required.");
  const encoded = protocols.find((item) => item.startsWith("oel-token."));
  const token = encoded?.slice("oel-token.".length) || "";
  if (!/^[A-Za-z0-9_-]{24,128}$/.test(token)) throw new RoomError(401, "Reconnect token is required.");
  return token;
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
