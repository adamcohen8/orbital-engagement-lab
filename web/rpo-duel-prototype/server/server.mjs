import { randomBytes, randomInt, randomUUID } from "node:crypto";
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { WebSocketServer } from "ws";

import { DUEL_PROTOTYPE_RULES } from "../src/shared/duel-engine.js";
import {
  DuelRoomCore,
  normalizeRoomCode,
  RoomError,
} from "../src/shared/duel-room.js";

const MODULE_DIR = fileURLToPath(new URL(".", import.meta.url));
const STATIC_ROOT = resolve(MODULE_DIR, "../public");
const TRAINER_ROOT = resolve(MODULE_DIR, "../../rpo-trainer-preview");
const TICK_INTERVAL_MS = 20;
const MAX_BODY_BYTES = 8192;
const MAX_MESSAGE_BYTES = 2048;

export class DuelRoom extends DuelRoomCore {
  constructor(options = {}) {
    super({
      ...options,
      code: options.code || createRoomCode(),
      matchSeed: options.matchSeed ?? randomInt(0, 0x100000000),
      tokenFactory: () => randomBytes(24).toString("base64url"),
      limits: { max_message_bytes: MAX_MESSAGE_BYTES },
    });
  }
}

export function createPrototypeServer({ staticRoot = STATIC_ROOT, roomRules = DUEL_PROTOTYPE_RULES } = {}) {
  const rooms = new Map();
  const httpServer = createServer(async (request, response) => {
    try {
      if (request.url?.startsWith("/api/")) {
        await handleApi(request, response, rooms, roomRules);
        return;
      }
      const url = new URL(request.url || "/", "http://oel.local");
      if (url.pathname === "/trainer" || url.pathname.startsWith("/trainer/")) {
        serveStatic(request, response, TRAINER_ROOT, "/trainer");
      } else if (url.pathname === "/rpo-duel-prototype" || url.pathname.startsWith("/rpo-duel-prototype/")) {
        serveStatic(request, response, staticRoot, "/rpo-duel-prototype");
      } else {
        serveStatic(request, response, staticRoot);
      }
    } catch (error) {
      const status = error instanceof RoomError ? error.status : 500;
      sendJson(response, status, { error: error instanceof Error ? error.message : String(error) });
    }
  });
  const webSocketServer = new WebSocketServer({
    noServer: true,
    maxPayload: MAX_MESSAGE_BYTES,
    handleProtocols: (protocols) => protocols.has("oel-rpo-duel-v1") ? "oel-rpo-duel-v1" : false,
  });

  httpServer.on("upgrade", (request, socket, head) => {
    try {
      const url = new URL(request.url || "/", "http://oel.local");
      if (url.pathname !== "/ws") throw new RoomError(404, "Unknown WebSocket path.");
      const room = rooms.get(normalizeRoomCode(url.searchParams.get("room")));
      if (!room) throw new RoomError(404, "Room not found.");
      const token = reconnectTokenFromProtocols(request.headers["sec-websocket-protocol"]);
      webSocketServer.handleUpgrade(request, socket, head, (webSocket) => {
        try {
          const player = room.connect(token, webSocket);
          webSocket.on("message", (raw) => {
            try {
              room.receive(player.id, raw, Date.now(), webSocket);
            } catch (error) {
              webSocket.send(JSON.stringify({ type: "error", error: error.message }));
              if (error instanceof RoomError && error.status === 429) {
                webSocket.close(4008, "Message rate limit exceeded.");
              }
            }
          });
          webSocket.on("close", () => room.disconnect(player.id, webSocket));
          webSocket.on("error", () => room.disconnect(player.id, webSocket));
        } catch (error) {
          webSocket.close(4003, error.message);
        }
      });
    } catch (error) {
      socket.write(`HTTP/1.1 ${error.status || 400} Error\r\nConnection: close\r\n\r\n`);
      socket.destroy();
    }
  });

  const timer = setInterval(() => {
    const now = Date.now();
    for (const [code, room] of rooms) {
      room.tick(now);
      if (!room.hasConnectedPlayers() && room.expiresAt(now) <= now) rooms.delete(code);
    }
  }, TICK_INTERVAL_MS);
  timer.unref();

  httpServer.on("close", () => {
    clearInterval(timer);
    for (const room of rooms.values()) {
      for (const player of room.players.values()) player.connection?.close(1001, "Server shutting down.");
    }
    webSocketServer.close();
  });

  return { httpServer, rooms };
}

async function handleApi(request, response, rooms, roomRules) {
  const url = new URL(request.url || "/", "http://oel.local");
  if (request.method === "POST" && url.pathname === "/api/rooms") {
    const body = await readJson(request);
    const code = uniqueRoomCode(rooms);
    const room = new DuelRoom({
      code,
      regulationRounds: body.regulation_rounds,
      matchSeed: randomInt(0, 0x100000000),
      rules: roomRules,
    });
    rooms.set(code, room);
    const joined = room.addPlayer(body.name);
    sendJson(response, 201, { ...room.publicSummary(), player: joined.player, reconnect_token: joined.token });
    return;
  }
  const match = url.pathname.match(/^\/api\/rooms\/([A-Za-z0-9]+)(?:\/(join))?$/);
  if (match) {
    const room = rooms.get(normalizeRoomCode(match[1]));
    if (!room) throw new RoomError(404, "Room not found.");
    if (request.method === "GET" && !match[2]) {
      sendJson(response, 200, room.publicSummary());
      return;
    }
    if (request.method === "POST" && match[2] === "join") {
      const body = await readJson(request);
      const joined = room.addPlayer(body.name);
      sendJson(response, 200, { ...room.publicSummary(), player: joined.player, reconnect_token: joined.token });
      return;
    }
  }
  throw new RoomError(404, "API route not found.");
}

function serveStatic(request, response, staticRoot, mountPath = "") {
  const url = new URL(request.url || "/", "http://oel.local");
  let pathname = decodeURIComponent(url.pathname);
  if (mountPath && (pathname === mountPath || pathname.startsWith(`${mountPath}/`))) {
    pathname = pathname.slice(mountPath.length) || "/";
  }
  if (pathname === "/") pathname = "/index.html";
  const safePath = normalize(pathname).replace(/^(\.\.(\/|\\|$))+/, "");
  const fullPath = resolve(join(staticRoot, safePath));
  if (!fullPath.startsWith(resolve(staticRoot)) || !existsSync(fullPath) || !statSync(fullPath).isFile()) {
    sendJson(response, 404, { error: "Not found." });
    return;
  }
  response.writeHead(200, {
    "Content-Type": contentType(extname(fullPath)),
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'self'; connect-src 'self' ws: wss:; img-src 'self' data:; style-src 'self'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'",
  });
  createReadStream(fullPath).pipe(response);
}

async function readJson(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw new RoomError(413, "Request body is too large.");
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
  } catch {
    throw new RoomError(400, "Request body must be valid JSON.");
  }
}

function sendJson(response, status, body) {
  const payload = JSON.stringify(body);
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(payload),
    "Cache-Control": "no-store",
  });
  response.end(payload);
}

function uniqueRoomCode(rooms) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const code = createRoomCode();
    if (!rooms.has(code)) return code;
  }
  return randomUUID().replaceAll("-", "").slice(0, 8).toUpperCase();
}

function createRoomCode() {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let code = "";
  for (let index = 0; index < 6; index += 1) code += alphabet[randomInt(0, alphabet.length)];
  return code;
}

function reconnectTokenFromProtocols(value) {
  const protocols = String(value || "").split(",").map((item) => item.trim());
  if (!protocols.includes("oel-rpo-duel-v1")) throw new RoomError(400, "Duel WebSocket protocol is required.");
  const encoded = protocols.find((item) => item.startsWith("oel-token."));
  const token = encoded?.slice("oel-token.".length) || "";
  if (!/^[A-Za-z0-9_-]{24,128}$/.test(token)) throw new RoomError(401, "Reconnect token is required.");
  return token;
}

function contentType(extension) {
  return {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
  }[extension] || "application/octet-stream";
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  const port = Math.max(1, Number(process.env.PORT || 8787));
  const host = process.env.HOST || "0.0.0.0";
  const { httpServer } = createPrototypeServer();
  httpServer.listen(port, host, () => {
    console.log(`RPO Duel prototype listening on http://localhost:${port}`);
    console.log("Use your computer's LAN address with the same port for phone testing.");
  });
}
