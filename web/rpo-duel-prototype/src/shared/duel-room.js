import {
  automaticSpeedState,
  createDuelSeries,
  DUEL_PROTOTYPE_RULES,
  DUEL_ROUND_COUNTS,
  restoreDuelSeries,
} from "./duel-engine.js";

export const DUEL_ROOM_TIMING = Object.freeze({
  countdown_ms: 3000,
  round_transition_ms: 4000,
  max_wall_delta_ms: 250,
  max_steps_per_tick: 30,
  broadcast_interval_ms: 100,
});

export const DUEL_ROOM_LIMITS = Object.freeze({
  max_message_bytes: 2048,
  waiting_ttl_ms: 30 * 60 * 1000,
  disconnected_ttl_ms: 30 * 60 * 1000,
  complete_ttl_ms: 10 * 60 * 1000,
});

export class DuelRoomCore {
  constructor({
    code,
    regulationRounds,
    matchSeed,
    now = Date.now(),
    rules = DUEL_PROTOTYPE_RULES,
    tokenFactory = defaultTokenFactory,
    timing = DUEL_ROOM_TIMING,
    limits = DUEL_ROOM_LIMITS,
  } = {}) {
    this.code = normalizeRoomCode(code);
    this.regulationRounds = normalizeRoundCount(regulationRounds);
    this.matchSeed = Number(matchSeed) >>> 0;
    this.rules = rules;
    this.tokenFactory = tokenFactory;
    this.timing = { ...DUEL_ROOM_TIMING, ...timing };
    this.limits = { ...DUEL_ROOM_LIMITS, ...limits };
    this.createdAt = now;
    this.updatedAt = now;
    this.players = new Map();
    this.playerTokens = new Map();
    this.tokenPlayers = new Map();
    this.series = null;
    this.phase = "waiting";
    this.phaseEndsAt = null;
    this.lastTickAt = now;
    this.lastBroadcastAt = 0;
    this.lastManeuverAt = Number.NEGATIVE_INFINITY;
    this.stepAccumulatorS = 0;
    this.speedState = { speed_multiple: rules.coast_speed_multiple, reason: "coasting" };
  }

  static restore(payload, options = {}) {
    if (!payload || payload.schema_version !== "rpo-duel.room.v1") {
      throw new Error("Unsupported serialized duel room.");
    }
    const room = new DuelRoomCore({
      code: payload.code,
      regulationRounds: payload.regulation_rounds,
      matchSeed: payload.match_seed,
      now: payload.created_at_ms,
      ...options,
    });
    room.updatedAt = payload.updated_at_ms;
    room.phase = payload.phase;
    room.phaseEndsAt = payload.phase_ends_at_ms;
    room.lastTickAt = payload.last_tick_at_ms;
    room.lastBroadcastAt = 0;
    room.lastManeuverAt = Number.isFinite(payload.last_maneuver_at_ms)
      ? payload.last_maneuver_at_ms
      : Number.NEGATIVE_INFINITY;
    room.stepAccumulatorS = payload.step_accumulator_s;
    room.speedState = payload.speed;
    for (const saved of payload.players || []) {
      const player = {
        id: saved.id,
        name: saved.name,
        connected: false,
        connection: null,
        lastSequence: saved.last_sequence,
        joinedAt: saved.joined_at_ms,
        disconnectedAt: saved.disconnected_at_ms,
      };
      room.players.set(player.id, player);
      room.playerTokens.set(player.id, saved.reconnect_token);
      room.tokenPlayers.set(saved.reconnect_token, player.id);
    }
    if (payload.series_result) {
      room.series = restoreDuelSeries({ result: payload.series_result, rules: room.rules });
    }
    return room;
  }

  addPlayer(name, now = Date.now()) {
    if (this.players.size >= 2) throw new RoomError(409, "Room is full.");
    if (this.phase === "complete") throw new RoomError(409, "Match has ended.");
    const id = `player-${this.players.size + 1}`;
    const token = String(this.tokenFactory());
    if (!token || this.tokenPlayers.has(token)) throw new Error("Reconnect token factory returned an invalid token.");
    const player = {
      id,
      name: normalizePlayerName(name, this.players.size + 1),
      connected: false,
      connection: null,
      lastSequence: -1,
      joinedAt: now,
      disconnectedAt: null,
    };
    this.players.set(id, player);
    this.playerTokens.set(id, token);
    this.tokenPlayers.set(token, id);
    this.updatedAt = now;
    if (this.players.size === 2) this.createSeries();
    return { player: publicPlayer(player), token };
  }

  createSeries() {
    if (this.series || this.players.size !== 2) return;
    this.series = createDuelSeries({
      playerIds: [...this.players.keys()],
      regulationRounds: this.regulationRounds,
      matchSeed: this.matchSeed,
      rules: this.rules,
    });
  }

  playerForToken(token) {
    const id = this.tokenPlayers.get(String(token || ""));
    return id ? this.players.get(id) || null : null;
  }

  connect(token, socket, now = Date.now()) {
    if (this.phase === "complete") throw new RoomError(409, "Match has ended; rejoin is closed.");
    const player = this.playerForToken(token);
    if (!player) throw new RoomError(401, "Invalid reconnect token.");
    if (this.series && this.phase === "active") {
      this.series.neutralizePlayer(player.id, { sequence: player.lastSequence + 1 });
    }
    if (player.connection && player.connection !== socket) player.connection.close(4001, "Replaced by a newer connection.");
    player.connection = socket;
    player.lastSequence = -1;
    player.connected = true;
    player.disconnectedAt = null;
    this.updatedAt = now;
    if (this.series && this.allPlayersConnected() && this.phase === "waiting") this.beginCountdown(now);
    this.sendSnapshotTo(player, now);
    return player;
  }

  disconnect(playerId, socket, now = Date.now()) {
    const player = this.players.get(playerId);
    if (!player || player.connection !== socket) return;
    player.connection = null;
    player.connected = false;
    player.disconnectedAt = now;
    this.updatedAt = now;
    if (this.series && this.phase === "active") {
      this.series.neutralizePlayer(player.id, { sequence: player.lastSequence + 1 });
      player.lastSequence += 1;
    }
    this.broadcast(now, true);
  }

  receive(playerId, raw, now = Date.now(), socket = null) {
    const player = this.players.get(playerId);
    if (!player) throw new RoomError(401, "Unknown player.");
    if (socket && player.connection !== socket) return;
    if (utf8ByteLength(String(raw)) > this.limits.max_message_bytes) {
      throw new RoomError(413, "Message is too large.");
    }
    let message;
    try {
      message = JSON.parse(String(raw));
    } catch {
      throw new RoomError(400, "Message must be valid JSON.");
    }
    if (message.type === "ping") {
      player.connection?.send(JSON.stringify({ type: "pong", now_ms: now }));
      return;
    }
    if (message.type !== "input") throw new RoomError(400, "Unsupported message type.");
    if (this.phase !== "active" || !this.series) return;
    const sequence = Math.floor(Number(message.sequence));
    if (!Number.isInteger(sequence) || sequence <= player.lastSequence) return;
    player.lastSequence = sequence;
    this.series.setPlayerControls(player.id, message.controls, { sequence });
    if (this.series.hasActiveManeuver()) this.lastManeuverAt = now;
    this.updatedAt = now;
  }

  tick(now = Date.now()) {
    const wallDeltaMs = Math.max(0, Math.min(now - this.lastTickAt, this.timing.max_wall_delta_ms));
    this.lastTickAt = now;
    if (this.phase === "countdown" && now >= this.phaseEndsAt) {
      this.phase = "active";
      this.phaseEndsAt = null;
      this.stepAccumulatorS = 0;
      this.lastManeuverAt = Number.NEGATIVE_INFINITY;
      this.updatedAt = now;
    }
    if (this.phase === "round_complete" && now >= this.phaseEndsAt) {
      this.series.advanceRound();
      this.beginCountdown(now);
    }
    if (this.phase === "active" && this.series) {
      const maneuvering = this.series.hasActiveManeuver();
      if (maneuvering) this.lastManeuverAt = now;
      this.speedState = automaticSpeedState({
        maneuvering,
        nowMs: now,
        lastManeuverMs: this.lastManeuverAt,
        rules: this.rules,
      });
      this.stepAccumulatorS += (wallDeltaMs / 1000) * this.speedState.speed_multiple;
      const requestedSteps = Math.floor(this.stepAccumulatorS / this.rules.dt_s);
      const steps = Math.min(requestedSteps, this.timing.max_steps_per_tick);
      if (steps > 0) {
        this.series.step(steps);
        this.stepAccumulatorS -= steps * this.rules.dt_s;
      }
      if (requestedSteps > this.timing.max_steps_per_tick) this.stepAccumulatorS = 0;
      const snapshot = this.series.snapshot();
      if (snapshot.match_terminal) {
        this.phase = "complete";
        this.phaseEndsAt = null;
      } else if (snapshot.round_complete) {
        this.phase = "round_complete";
        this.phaseEndsAt = now + this.timing.round_transition_ms;
      }
      this.updatedAt = now;
    }
    this.broadcast(now);
  }

  beginCountdown(now) {
    this.phase = "countdown";
    this.phaseEndsAt = now + this.timing.countdown_ms;
    this.lastTickAt = now;
    this.stepAccumulatorS = 0;
    this.updatedAt = now;
    this.broadcast(now, true);
  }

  allPlayersConnected() {
    return this.players.size === 2 && [...this.players.values()].every((player) => player.connected);
  }

  hasConnectedPlayers() {
    return [...this.players.values()].some((player) => player.connected);
  }

  snapshotFor(playerId, now = Date.now()) {
    return {
      type: "snapshot",
      protocol_version: "rpo-duel.ws.v1",
      room_code: this.code,
      phase: this.phase,
      phase_remaining_ms: this.phaseEndsAt === null ? 0 : Math.max(this.phaseEndsAt - now, 0),
      speed: this.speedState,
      players: [...this.players.values()].map(publicPlayer),
      you: this.players.has(playerId) ? publicPlayer(this.players.get(playerId)) : null,
      series: this.series?.snapshot() || null,
      invite_path: `/?room=${encodeURIComponent(this.code)}`,
    };
  }

  sendSnapshotTo(player, now = Date.now()) {
    if (!player?.connection || player.connection.readyState !== 1) return;
    player.connection.send(JSON.stringify(this.snapshotFor(player.id, now)));
  }

  broadcast(now = Date.now(), force = false) {
    if (!force && now - this.lastBroadcastAt < this.timing.broadcast_interval_ms) return;
    this.lastBroadcastAt = now;
    for (const player of this.players.values()) this.sendSnapshotTo(player, now);
  }

  publicSummary() {
    return {
      room_code: this.code,
      phase: this.phase,
      regulation_rounds: this.regulationRounds,
      players: [...this.players.values()].map(publicPlayer),
      joinable: this.players.size < 2 && this.phase !== "complete",
    };
  }

  expiresAt(now = Date.now()) {
    if (this.phase === "complete") return now + this.limits.complete_ttl_ms;
    if (!this.hasConnectedPlayers()) return now + this.limits.disconnected_ttl_ms;
    return now + this.limits.waiting_ttl_ms;
  }

  serialize() {
    const seriesResult = this.series?.result() || null;
    if (seriesResult?.current_round) seriesResult.current_round.history = [];
    return {
      schema_version: "rpo-duel.room.v1",
      code: this.code,
      regulation_rounds: this.regulationRounds,
      match_seed: this.matchSeed,
      created_at_ms: this.createdAt,
      updated_at_ms: this.updatedAt,
      phase: this.phase,
      phase_ends_at_ms: this.phaseEndsAt,
      last_tick_at_ms: this.lastTickAt,
      last_maneuver_at_ms: Number.isFinite(this.lastManeuverAt) ? this.lastManeuverAt : null,
      step_accumulator_s: this.stepAccumulatorS,
      speed: this.speedState,
      players: [...this.players.values()].map((player) => ({
        id: player.id,
        name: player.name,
        last_sequence: player.lastSequence,
        joined_at_ms: player.joinedAt,
        disconnected_at_ms: player.disconnectedAt,
        reconnect_token: this.playerTokens.get(player.id),
      })),
      series_result: seriesResult,
    };
  }
}

export class RoomError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export function normalizeRoomCode(value) {
  return String(value || "").trim().toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8);
}

export function normalizeRoundCount(value) {
  const count = Math.floor(Number(value) || 0);
  if (!DUEL_ROUND_COUNTS.includes(count)) throw new RoomError(400, "regulation_rounds must be 2, 4, or 6.");
  return count;
}

function defaultTokenFactory() {
  if (typeof crypto?.randomUUID !== "function") throw new Error("A reconnect token factory is required.");
  return crypto.randomUUID().replaceAll("-", "");
}

function publicPlayer(player) {
  return { id: player.id, name: player.name, connected: player.connected };
}

function normalizePlayerName(value, fallbackIndex) {
  const name = String(value || "").trim().replace(/[^A-Za-z0-9 _-]/g, "").slice(0, 24);
  return name || `PLAYER ${fallbackIndex}`;
}

function utf8ByteLength(value) {
  return new TextEncoder().encode(value).byteLength;
}
