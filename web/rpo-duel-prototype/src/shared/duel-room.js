import {
  automaticSpeedState,
  createDuelSeries,
  DUEL_PROTOTYPE_RULES,
  DUEL_ROUND_COUNTS,
  restoreDuelSeries,
} from "./duel-engine.js";
import {
  PREDICTIVE_ENGAGEMENT_POLICY,
  selectEvasionAction,
  selectInterceptAction,
} from "./predictive-engagement.js";
import { deterministicSeed } from "../../../rpo-trainer-preview/src/competition/arcade-engine.js";

export const DUEL_ROOM_MODES = Object.freeze({ HUMAN: "human", COMPUTER: "computer" });

export const DUEL_COMPUTER_CADENCE = Object.freeze({
  schema_version: "rpo-duel.computer-cadence.v2",
  pulse_duration_s: 30,
  decision_interval_s: 120,
  minimum_range_improvement_m: 100,
  minimum_capture_time_improvement_s: 60,
  target_guard_range_m: 600,
});

export const DUEL_ROOM_TIMING = Object.freeze({
  countdown_ms: 3000,
  round_transition_ms: 4000,
  max_wall_delta_ms: 250,
  max_steps_per_tick: 30,
  broadcast_interval_ms: 100,
});

export const DUEL_ROOM_LIMITS = Object.freeze({
  max_message_bytes: 2048,
  max_messages_per_window: 80,
  message_rate_window_ms: 10000,
  heartbeat_timeout_ms: 45000,
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
    matchMode = DUEL_ROOM_MODES.HUMAN,
  } = {}) {
    this.code = normalizeRoomCode(code);
    this.regulationRounds = normalizeRoundCount(regulationRounds);
    this.matchSeed = Number(matchSeed) >>> 0;
    this.rules = rules;
    this.tokenFactory = tokenFactory;
    this.timing = { ...DUEL_ROOM_TIMING, ...timing };
    this.limits = { ...DUEL_ROOM_LIMITS, ...limits };
    this.matchMode = normalizeMatchMode(matchMode);
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
    this.computerPlayerId = null;
    this.computerController = defaultComputerController();
    this.rematchIndex = 0;
    this.rematchReady = new Set();
  }

  static restore(payload, options = {}) {
    if (!payload || !["rpo-duel.room.v1", "rpo-duel.room.v2"].includes(payload.schema_version)) {
      throw new Error("Unsupported serialized duel room.");
    }
    const persistedRules = payload.schema_version === "rpo-duel.room.v2" ? payload.rules : undefined;
    if (payload.schema_version === "rpo-duel.room.v2") {
      if (!persistedRules || payload.engine_identity?.schema_version !== persistedRules.schema_version ||
          payload.engine_identity?.physics_version !== persistedRules.physics_version) {
        throw new Error("Serialized duel room has invalid engine/rules identity.");
      }
    }
    const room = new DuelRoomCore({
      code: payload.code,
      regulationRounds: payload.regulation_rounds,
      matchSeed: payload.match_seed,
      now: payload.created_at_ms,
      rules: persistedRules,
      matchMode: payload.match_mode || DUEL_ROOM_MODES.HUMAN,
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
        connected: saved.kind === "computer",
        connection: null,
        lastSequence: saved.last_sequence,
        joinedAt: saved.joined_at_ms,
        disconnectedAt: saved.disconnected_at_ms,
        lastSeenAt: saved.last_seen_at_ms ?? saved.disconnected_at_ms ?? payload.updated_at_ms,
        messageWindowStartedAt: payload.updated_at_ms,
        messageCount: 0,
        kind: saved.kind || "human",
      };
      room.players.set(player.id, player);
      if (saved.reconnect_token) {
        room.playerTokens.set(player.id, saved.reconnect_token);
        room.tokenPlayers.set(saved.reconnect_token, player.id);
      }
    }
    room.computerPlayerId = payload.computer_player_id || null;
    room.computerController = {
      ...defaultComputerController(room.computerPlayerId),
      ...(payload.computer_controller || {}),
    };
    room.rematchIndex = Math.max(0, Math.floor(Number(payload.rematch_index) || 0));
    room.rematchReady = new Set(
      (payload.rematch_ready_player_ids || []).filter((id) => room.players.get(id)?.kind === "human"),
    );
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
      lastSeenAt: now,
      messageWindowStartedAt: now,
      messageCount: 0,
      kind: "human",
    };
    this.players.set(id, player);
    this.playerTokens.set(id, token);
    this.tokenPlayers.set(token, id);
    this.updatedAt = now;
    if (this.players.size === 2) this.createSeries();
    return { player: publicPlayer(player), token };
  }

  addComputerOpponent(now = Date.now()) {
    if (this.matchMode !== DUEL_ROOM_MODES.COMPUTER) {
      throw new RoomError(409, "This room is configured for a human opponent.");
    }
    if (this.computerPlayerId) return publicPlayer(this.players.get(this.computerPlayerId));
    if (this.players.size !== 1) throw new RoomError(409, "Add the human player before the computer opponent.");
    const id = "player-2";
    const player = {
      id,
      name: "OEL COMPUTER",
      connected: true,
      connection: null,
      lastSequence: -1,
      joinedAt: now,
      disconnectedAt: null,
      kind: "computer",
    };
    this.players.set(id, player);
    this.computerPlayerId = id;
    this.computerController = defaultComputerController(id);
    this.updatedAt = now;
    this.createSeries();
    return publicPlayer(player);
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
    player.lastSeenAt = now;
    player.messageWindowStartedAt = now;
    player.messageCount = 0;
    this.updatedAt = now;
    if (this.series && this.allPlayersConnected() && this.phase === "waiting") this.beginCountdown(now);
    if (this.phase === "complete" && this.allPlayersConnected() && this.allRematchPlayersReady()) {
      this.startRematch(now);
    }
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
    if (player.kind === "computer") throw new RoomError(403, "Computer controls are server-owned.");
    if (socket && player.connection !== socket) return;
    player.lastSeenAt = now;
    if (now - player.messageWindowStartedAt >= this.limits.message_rate_window_ms) {
      player.messageWindowStartedAt = now;
      player.messageCount = 0;
    }
    player.messageCount += 1;
    if (player.messageCount > this.limits.max_messages_per_window) {
      throw new RoomError(429, "WebSocket message rate limit exceeded.");
    }
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
      return false;
    }
    if (message.type === "rematch") {
      this.requestRematch(player.id, now);
      return true;
    }
    if (message.type !== "input") throw new RoomError(400, "Unsupported message type.");
    if (this.phase !== "active" || !this.series) return false;
    const sequence = Math.floor(Number(message.sequence));
    if (!Number.isInteger(sequence) || sequence <= player.lastSequence) return false;
    player.lastSequence = sequence;
    this.series.setPlayerControls(player.id, message.controls, { sequence });
    if (this.series.hasActiveManeuver()) this.lastManeuverAt = now;
    this.updatedAt = now;
    return true;
  }

  tick(now = Date.now()) {
    this.expireStaleConnections(now);
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
      this.applyComputerControl();
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
        let completedSteps = 0;
        while (completedSteps < steps && !this.series.snapshot().round_complete) {
          const changed = this.applyComputerControl();
          if (
            changed
            && this.series.hasActiveManeuver()
            && this.speedState.speed_multiple !== this.rules.maneuver_speed_multiple
          ) {
            this.stepAccumulatorS = 0;
            break;
          }
          this.series.step(1);
          completedSteps += 1;
        }
        const remainingBacklogS = Math.max(0, this.stepAccumulatorS - completedSteps * this.rules.dt_s);
        this.stepAccumulatorS = Math.min(
          remainingBacklogS,
          this.timing.max_steps_per_tick * this.rules.dt_s,
        );
      }
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

  applyComputerControl() {
    if (!this.computerPlayerId || !this.series || this.phase !== "active") return false;
    const snapshot = this.series.snapshot();
    if (snapshot.round_complete) return false;
    const controller = this.computerController;
    if (controller.round_index !== snapshot.round_index) {
      controller.round_index = snapshot.round_index;
      controller.next_plan_time_s = 0;
      controller.action_until_time_s = 0;
      controller.phase = "standby";
    }
    const timeS = snapshot.round.time_s;
    const remaining = snapshot.round.delta_v_remaining_m_s;
    const role = snapshot.roles.chaser === this.computerPlayerId ? "chaser" : "target";
    if (remaining[role] <= 1.0e-12) {
      controller.phase = "delta_v_exhausted_coast";
      controller.next_plan_time_s = this.rules.round_duration_s + 1;
      controller.action_until_time_s = timeS;
      return this.setComputerControls({ r: 0, i: 0, c: 0 }, controller.phase);
    }
    if (timeS + 1.0e-9 >= controller.next_plan_time_s) {
      const guidance = this.series.guidanceStateForPlayer(this.computerPlayerId);
      const maximum = role === "chaser"
        ? this.rules.chaser_max_accel_km_s2 * 1000
        : this.rules.target_max_accel_km_s2 * 1000;
      const common = {
        mean_motion_rad_s: guidance.mean_motion_rad_s,
        max_acceleration_m_s2: maximum,
        horizon_s: PREDICTIVE_ENGAGEMENT_POLICY.horizon_s,
        step_s: PREDICTIVE_ENGAGEMENT_POLICY.step_s,
        pulse_duration_s: DUEL_COMPUTER_CADENCE.pulse_duration_s,
        capture_radius_m: this.rules.capture_range_km * 1000,
        capture_margin_m: PREDICTIVE_ENGAGEMENT_POLICY.capture_margin_m,
        acceleration_fractions: PREDICTIVE_ENGAGEMENT_POLICY.acceleration_fractions,
      };
      const proposedDecision = role === "chaser"
        ? selectInterceptAction(guidance.state_ric_si, common)
        : selectEvasionAction(guidance.state_ric_si, {
            ...common,
            opponent_max_acceleration_m_s2: this.rules.chaser_max_accel_km_s2 * 1000,
          });
      const passiveOptions = { ...common, max_acceleration_m_s2: 0 };
      const passiveDecision = role === "chaser"
        ? selectInterceptAction(guidance.state_ric_si, passiveOptions)
        : selectEvasionAction(guidance.state_ric_si, {
            ...passiveOptions,
            opponent_max_acceleration_m_s2: this.rules.chaser_max_accel_km_s2 * 1000,
          });
      const decision = materiallyImprovesComputerOutcome(role, proposedDecision, passiveDecision);
      controller.phase = decision.phase;
      controller.next_plan_time_s = timeS + DUEL_COMPUTER_CADENCE.decision_interval_s;
      const maneuvering = vectorMagnitude(decision.acceleration_ric_m_s2) > 1.0e-12;
      controller.action_until_time_s = maneuvering
        ? timeS + DUEL_COMPUTER_CADENCE.pulse_duration_s
        : controller.next_plan_time_s;
      const [r, i, c] = guidance.action_basis_to_game_ric.map((row) => (
        row.reduce((total, value, index) => (
          total + value * decision.acceleration_ric_m_s2[index]
        ), 0) / maximum
      ));
      return this.setComputerControls({ r, i, c }, decision.phase);
    }
    if (timeS + 1.0e-9 >= controller.action_until_time_s) {
      controller.phase = role === "chaser"
        ? "intercept_replan_coast"
        : "predictive_evasion_replan_coast";
      return this.setComputerControls({ r: 0, i: 0, c: 0 }, controller.phase);
    }
    return false;
  }

  setComputerControls(controls, policyPhase) {
    this.computerController.sequence += 1;
    const changed = this.series.setPlayerControls(this.computerPlayerId, controls, {
      sequence: this.computerController.sequence,
      source: "computer_policy",
      policyPhase,
    });
    if (!changed) this.computerController.sequence -= 1;
    return changed;
  }

  beginCountdown(now) {
    this.phase = "countdown";
    this.phaseEndsAt = now + this.timing.countdown_ms;
    this.lastTickAt = now;
    this.stepAccumulatorS = 0;
    this.updatedAt = now;
    this.broadcast(now, true);
  }

  requestRematch(playerId, now = Date.now()) {
    const player = this.players.get(playerId);
    if (!player || player.kind !== "human") throw new RoomError(403, "Only human players may request a rematch.");
    if (this.phase !== "complete" || !this.series?.snapshot().match_terminal) {
      throw new RoomError(409, "A rematch is available only after the match ends.");
    }
    this.rematchReady.add(playerId);
    this.updatedAt = now;
    if (this.allPlayersConnected() && this.allRematchPlayersReady()) this.startRematch(now);
    else this.broadcast(now, true);
  }

  allRematchPlayersReady() {
    const humans = [...this.players.values()].filter((player) => player.kind === "human");
    return humans.length > 0 && humans.every((player) => this.rematchReady.has(player.id));
  }

  startRematch(now = Date.now()) {
    this.rematchIndex += 1;
    const rematchSeed = deterministicSeed(this.matchSeed, this.rematchIndex, 0x52454d54);
    this.series = createDuelSeries({
      playerIds: [...this.players.keys()],
      regulationRounds: this.regulationRounds,
      matchSeed: rematchSeed,
      rules: this.rules,
    });
    this.rematchReady.clear();
    this.computerController = defaultComputerController(this.computerPlayerId);
    this.speedState = { speed_multiple: this.rules.coast_speed_multiple, reason: "coasting" };
    this.lastManeuverAt = Number.NEGATIVE_INFINITY;
    this.beginCountdown(now);
  }

  allPlayersConnected() {
    return this.players.size === 2 && [...this.players.values()].every((player) => player.connected);
  }

  hasConnectedPlayers() {
    return [...this.players.values()].some((player) => player.kind !== "computer" && player.connected);
  }

  shouldKeepTicking() {
    return this.hasConnectedPlayers() || ["countdown", "active", "round_complete"].includes(this.phase);
  }

  expireStaleConnections(now = Date.now()) {
    for (const player of this.players.values()) {
      if (
        player.kind === "computer"
        || !player.connected
        || now - player.lastSeenAt <= this.limits.heartbeat_timeout_ms
      ) continue;
      const socket = player.connection;
      socket?.close(4002, "Heartbeat timeout.");
      this.disconnect(player.id, socket, now);
    }
  }

  snapshotFor(playerId, now = Date.now()) {
    return {
      type: "snapshot",
      protocol_version: "rpo-duel.ws.v1",
      room_code: this.code,
      phase: this.phase,
      phase_remaining_ms: this.phaseEndsAt === null ? 0 : Math.max(this.phaseEndsAt - now, 0),
      speed: this.speedState,
      match_mode: this.matchMode,
      computer_opponent: this.computerPlayerId ? {
        player_id: this.computerPlayerId,
        policy_version: PREDICTIVE_ENGAGEMENT_POLICY.schema_version,
        cadence_version: DUEL_COMPUTER_CADENCE.schema_version,
        phase: this.computerController.phase,
      } : null,
      rematch: {
        index: this.rematchIndex,
        ready_player_ids: [...this.rematchReady],
        your_ready: this.rematchReady.has(playerId),
        required_human_players: [...this.players.values()].filter((player) => player.kind === "human").length,
      },
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
      match_mode: this.matchMode,
      players: [...this.players.values()].map(publicPlayer),
      joinable: this.players.size < 2 && this.phase !== "complete",
    };
  }

  expiresAt(now = Date.now()) {
    if (this.phase === "complete") return this.updatedAt + this.limits.complete_ttl_ms;
    if (!this.hasConnectedPlayers()) {
      const disconnectedAt = Math.max(
        this.createdAt,
        ...[...this.players.values()].map((player) => player.disconnectedAt ?? this.createdAt),
      );
      const ttl = this.phase === "waiting" ? this.limits.waiting_ttl_ms : this.limits.disconnected_ttl_ms;
      return disconnectedAt + ttl;
    }
    return now + this.limits.waiting_ttl_ms;
  }

  serialize() {
    const seriesResult = this.series?.result() || null;
    if (seriesResult?.current_round) seriesResult.current_round.history = [];
    return {
      schema_version: "rpo-duel.room.v2",
      engine_identity: {
        schema_version: this.rules.schema_version,
        physics_version: this.rules.physics_version,
      },
      rules: JSON.parse(JSON.stringify(this.rules)),
      code: this.code,
      regulation_rounds: this.regulationRounds,
      match_mode: this.matchMode,
      computer_player_id: this.computerPlayerId,
      computer_controller: { ...this.computerController },
      rematch_index: this.rematchIndex,
      rematch_ready_player_ids: [...this.rematchReady],
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
        last_seen_at_ms: player.lastSeenAt,
        kind: player.kind,
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

export function normalizeMatchMode(value) {
  const mode = String(value || DUEL_ROOM_MODES.HUMAN).trim().toLowerCase();
  if (!Object.values(DUEL_ROOM_MODES).includes(mode)) {
    throw new RoomError(400, "opponent must be human or computer.");
  }
  return mode;
}

export function materiallyImprovesComputerOutcome(role, proposed, passive) {
  if (vectorMagnitude(proposed.acceleration_ric_m_s2) <= 1.0e-12) return proposed;
  const proposedCapture = proposed.predicted_capture_time_s;
  const passiveCapture = passive.predicted_capture_time_s;
  let material = false;
  if (role === "chaser") {
    material = (
      (proposedCapture !== null && passiveCapture === null)
      || (
        proposedCapture !== null
        && passiveCapture !== null
        && passiveCapture - proposedCapture >= DUEL_COMPUTER_CADENCE.minimum_capture_time_improvement_s
      )
      || (
        passive.predicted_closest_range_m - proposed.predicted_closest_range_m
        >= DUEL_COMPUTER_CADENCE.minimum_range_improvement_m
      )
    );
  } else if (role === "target") {
    material = (
      (proposedCapture === null && passiveCapture !== null)
      || (
        proposedCapture !== null
        && passiveCapture !== null
        && proposedCapture - passiveCapture >= DUEL_COMPUTER_CADENCE.minimum_capture_time_improvement_s
      )
      || (
        passive.predicted_closest_range_m <= DUEL_COMPUTER_CADENCE.target_guard_range_m
        && proposed.predicted_closest_range_m - passive.predicted_closest_range_m
          >= DUEL_COMPUTER_CADENCE.minimum_range_improvement_m
      )
    );
  } else {
    throw new Error(`Unsupported computer role: ${role}`);
  }
  if (material) return proposed;
  return {
    ...passive,
    acceleration_ric_m_s2: [0, 0, 0],
    phase: role === "chaser"
      ? "intercept_benefit_gate_coast"
      : "predictive_evasion_benefit_gate_coast",
  };
}

function defaultTokenFactory() {
  if (typeof crypto?.randomUUID !== "function") throw new Error("A reconnect token factory is required.");
  return crypto.randomUUID().replaceAll("-", "");
}

function publicPlayer(player) {
  return { id: player.id, name: player.name, connected: player.connected, kind: player.kind || "human" };
}

function defaultComputerController(playerId = null) {
  return {
    player_id: playerId,
    round_index: null,
    next_plan_time_s: 0,
    action_until_time_s: 0,
    sequence: 0,
    phase: "standby",
  };
}

function normalizePlayerName(value, fallbackIndex) {
  const name = String(value || "").trim().replace(/[^A-Za-z0-9 _-]/g, "").slice(0, 24);
  return name || `PLAYER ${fallbackIndex}`;
}

function utf8ByteLength(value) {
  return new TextEncoder().encode(value).byteLength;
}

function vectorMagnitude(values) {
  return Math.hypot(...values);
}
