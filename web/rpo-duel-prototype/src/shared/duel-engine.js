import {
  DEFAULT_PURSUIT_CHALLENGE,
  deterministicSeed,
  keplerianToEci,
  relativeRicState,
  samplePursuitInitialRic,
  stateFromRelativeRic,
  stepControlledTwoBodyPair,
  stepTwoBodyState,
} from "../../../rpo-trainer-preview/src/competition/arcade-engine.js";

export const DUEL_ROLES = Object.freeze({ CHASER: "chaser", TARGET: "target" });
export const DUEL_ROUND_COUNTS = Object.freeze([2, 4, 6]);

export const DUEL_PROTOTYPE_RULES = deepFreeze({
  schema_version: "rpo-duel.prototype.v1",
  physics_version: "web-two-body-duel-v1",
  dt_s: 1,
  round_duration_s: 18000,
  capture_range_km: 0.1,
  capture_relative_speed_limit_km_s: null,
  chaser_delta_v_budget_m_s: 15,
  target_delta_v_budget_m_s: 5,
  chaser_max_accel_km_s2: 1.5e-5,
  target_max_accel_km_s2: 7.5e-6,
  coast_speed_multiple: 100,
  maneuver_speed_multiple: 10,
  neutral_cooldown_ms: 1000,
  target_coes: { ...DEFAULT_PURSUIT_CHALLENGE.target_coes },
  random_initial_state: { ...DEFAULT_PURSUIT_CHALLENGE.arcade.random_initial_state },
  history_stride_ticks: 10,
  max_history_samples: 900,
});

export function createDuelRound({ pairSeed, rules = DUEL_PROTOTYPE_RULES } = {}) {
  const cfg = normalizeRules(rules);
  const seed = uint32(pairSeed ?? 1);
  const geometryConfig = pursuitGeometryConfig(cfg);
  const target = keplerianToEci(cfg.target_coes, cfg.mu_km3_s2);
  const initialRelativeRic = samplePursuitInitialRic(geometryConfig, seed);
  const chaser = stateFromRelativeRic(target, initialRelativeRic);
  let sim = {
    tick: 0,
    time_s: 0,
    target_reference: cloneEci(target),
    target: cloneEci(target),
    chaser: cloneEci(chaser),
    delta_v_m_s: { chaser: 0, target: 0 },
  };
  const controls = { chaser: neutralControls(), target: neutralControls() };
  const inputEvents = [];
  const history = [];
  let terminal = false;
  let winnerRole = null;
  let terminalReason = "";
  let closestRangeKm = Infinity;

  const recordSample = (force = false) => {
    const rel = relativeRicState(sim.target, sim.chaser);
    const rangeKm = norm([rel.r_km, rel.i_km, rel.c_km]);
    closestRangeKm = Math.min(closestRangeKm, rangeKm);
    if (force || sim.tick % cfg.history_stride_ticks === 0) {
      history.push({
        tick: sim.tick,
        time_s: metric(sim.time_s),
        relative_ric: roundedRelative(rel),
        range_km: metric(rangeKm),
      });
      if (history.length > cfg.max_history_samples) history.splice(0, history.length - cfg.max_history_samples);
    }
    return { rel, rangeKm };
  };

  const evaluateTerminal = () => {
    const { rangeKm } = recordSample();
    if (!terminal && rangeKm <= cfg.capture_range_km + 1.0e-12) {
      terminal = true;
      winnerRole = DUEL_ROLES.CHASER;
      terminalReason = "Chaser entered the capture region.";
      recordSample(true);
    }
    if (!terminal && sim.time_s >= cfg.round_duration_s - 1.0e-9) {
      terminal = true;
      winnerRole = DUEL_ROLES.TARGET;
      terminalReason = "Target survived until time expired.";
      recordSample(true);
    }
  };

  recordSample(true);
  evaluateTerminal();

  return {
    rules: cfg,
    pairSeed: seed,
    initialGeometry: {
      pair_seed: seed,
      target_coes: clone(cfg.target_coes),
      chaser_initial_ric: clone(initialRelativeRic),
    },
    setControls(role, nextControls, { playerId = null, sequence = null } = {}) {
      assertRole(role);
      const normalized = normalizeControls(nextControls);
      if (sameControls(controls[role], normalized)) return false;
      controls[role] = normalized;
      inputEvents.push({
        tick: sim.tick,
        time_s: metric(sim.time_s),
        role,
        player_id: playerId,
        sequence,
        controls: clone(normalized),
      });
      return true;
    },
    neutralize(role, options = {}) {
      return this.setControls(role, neutralControls(), options);
    },
    hasActiveManeuver(role = null) {
      if (role !== null) {
        assertRole(role);
        return controlMagnitude(controls[role]) > 0 && remainingDeltaV(sim, cfg, role) > 1.0e-12;
      }
      return this.hasActiveManeuver(DUEL_ROLES.CHASER) || this.hasActiveManeuver(DUEL_ROLES.TARGET);
    },
    step(count = 1) {
      const steps = Math.max(0, Math.floor(Number(count) || 0));
      for (let index = 0; index < steps && !terminal; index += 1) {
        const chaserAccel = appliedAcceleration(controls.chaser, cfg.chaser_max_accel_km_s2, remainingDeltaV(sim, cfg, "chaser"), cfg.dt_s);
        const targetAccel = appliedAcceleration(controls.target, cfg.target_max_accel_km_s2, remainingDeltaV(sim, cfg, "target"), cfg.dt_s);
        const next = stepControlledTwoBodyPair({
          target: sim.target,
          chaser: sim.chaser,
          mu_km3_s2: cfg.mu_km3_s2,
          chaser_accel_ric_km_s2: chaserAccel,
          target_accel_ric_km_s2: targetAccel,
          dt_s: cfg.dt_s,
        });
        sim = {
          tick: sim.tick + 1,
          time_s: sim.time_s + cfg.dt_s,
          target_reference: stepTwoBodyState(sim.target_reference, cfg.mu_km3_s2, cfg.dt_s),
          target: next.target,
          chaser: next.chaser,
          delta_v_m_s: {
            chaser: Math.min(cfg.chaser_delta_v_budget_m_s, sim.delta_v_m_s.chaser + norm(chaserAccel) * cfg.dt_s * 1000),
            target: Math.min(cfg.target_delta_v_budget_m_s, sim.delta_v_m_s.target + norm(targetAccel) * cfg.dt_s * 1000),
          },
        };
        evaluateTerminal();
      }
      return this.snapshot();
    },
    snapshot() {
      const rel = relativeRicState(sim.target, sim.chaser);
      const targetReferenceRel = relativeRicState(sim.target_reference, sim.target);
      const chaserReferenceRel = relativeRicState(sim.target_reference, sim.chaser);
      const rangeKm = norm([rel.r_km, rel.i_km, rel.c_km]);
      return {
        schema_version: cfg.schema_version,
        pair_seed: seed,
        tick: sim.tick,
        time_s: metric(sim.time_s),
        time_remaining_s: metric(Math.max(cfg.round_duration_s - sim.time_s, 0)),
        relative_ric: roundedRelative(rel),
        target_reference_ric: roundedRelative(targetReferenceRel),
        chaser_reference_ric: roundedRelative(chaserReferenceRel),
        reference_mean_motion_rad_s: metric(Math.sqrt(cfg.mu_km3_s2 / cfg.target_coes.a_km ** 3)),
        capture_range_km: metric(cfg.capture_range_km),
        range_km: metric(rangeKm),
        relative_speed_km_s: metric(norm([rel.rd_km_s, rel.id_km_s, rel.cd_km_s])),
        delta_v_m_s: roundedMap(sim.delta_v_m_s),
        delta_v_remaining_m_s: {
          chaser: metric(remainingDeltaV(sim, cfg, "chaser")),
          target: metric(remainingDeltaV(sim, cfg, "target")),
        },
        controls: clone(controls),
        terminal,
        winner_role: winnerRole,
        terminal_reason: terminalReason,
        closest_range_km: metric(closestRangeKm),
      };
    },
    result() {
      return {
        ...this.snapshot(),
        initial_geometry: clone(this.initialGeometry),
        input_events: clone(inputEvents),
        history: clone(history),
      };
    },
  };
}

export function createDuelSeries({
  playerIds = ["player-1", "player-2"],
  regulationRounds = 2,
  matchSeed = 1,
  rules = DUEL_PROTOTYPE_RULES,
} = {}) {
  if (!Array.isArray(playerIds) || playerIds.length !== 2 || playerIds[0] === playerIds[1]) {
    throw new Error("Duel series requires two distinct player ids.");
  }
  const rounds = normalizeRoundCount(regulationRounds);
  const seed = uint32(matchSeed);
  const players = playerIds.map(String);
  const firstChaserIndex = deterministicSeed(seed, 0, 0x4455454c) % 2;
  let roundIndex = 1;
  let round = startRound();
  let summaries = [];
  let score = Object.fromEntries(players.map((id) => [id, 0]));
  let matchTerminal = false;
  let matchWinnerPlayerId = null;
  let matchDraw = false;

  function roleAssignments(index = roundIndex) {
    const chaserIndex = (firstChaserIndex + index - 1) % 2;
    return {
      chaser: players[chaserIndex],
      target: players[1 - chaserIndex],
    };
  }

  function pairIndexForRound(index = roundIndex) {
    return Math.floor((index - 1) / 2);
  }

  function pairSeedForRound(index = roundIndex) {
    return deterministicSeed(seed, pairIndexForRound(index) + 1, 0x50414952);
  }

  function startRound() {
    return createDuelRound({ pairSeed: pairSeedForRound(roundIndex), rules });
  }

  function finishRoundIfNeeded() {
    const result = round.result();
    if (!result.terminal || summaries.some((summary) => summary.round_index === roundIndex)) return;
    const assignments = roleAssignments();
    const winnerPlayerId = assignments[result.winner_role];
    score[winnerPlayerId] += 1;
    summaries.push({
      round_index: roundIndex,
      pair_index: pairIndexForRound() + 1,
      pair_seed: pairSeedForRound(),
      roles: assignments,
      winner_role: result.winner_role,
      winner_player_id: winnerPlayerId,
      terminal_reason: result.terminal_reason,
      tick: result.tick,
      time_s: result.time_s,
      range_km: result.range_km,
      delta_v_m_s: result.delta_v_m_s,
      initial_geometry: result.initial_geometry,
      input_events: result.input_events,
    });
    if (roundIndex >= rounds) {
      matchTerminal = true;
      const [first, second] = players;
      matchDraw = score[first] === score[second];
      matchWinnerPlayerId = matchDraw ? null : score[first] > score[second] ? first : second;
    }
  }

  return {
    rules: normalizeRules(rules),
    matchSeed: seed,
    regulationRounds: rounds,
    players: players.slice(),
    setPlayerControls(playerId, controls, options = {}) {
      const assignments = roleAssignments();
      const role = assignments.chaser === playerId ? "chaser" : assignments.target === playerId ? "target" : null;
      if (!role) throw new Error(`Player ${playerId} is not part of this match.`);
      return round.setControls(role, controls, { ...options, playerId });
    },
    neutralizePlayer(playerId, options = {}) {
      return this.setPlayerControls(playerId, neutralControls(), options);
    },
    hasActiveManeuver() {
      return round.hasActiveManeuver();
    },
    step(count = 1) {
      if (matchTerminal) return this.snapshot();
      round.step(count);
      finishRoundIfNeeded();
      return this.snapshot();
    },
    advanceRound() {
      if (matchTerminal) return false;
      finishRoundIfNeeded();
      if (!round.snapshot().terminal) throw new Error("Cannot advance an active round.");
      roundIndex += 1;
      round = startRound();
      return true;
    },
    snapshot() {
      finishRoundIfNeeded();
      return {
        schema_version: "rpo-duel.series.v1",
        players: players.slice(),
        match_seed: seed,
        regulation_rounds: rounds,
        round_index: roundIndex,
        pair_index: pairIndexForRound() + 1,
        pair_seed: pairSeedForRound(),
        roles: roleAssignments(),
        score: clone(score),
        round: round.snapshot(),
        round_complete: round.snapshot().terminal,
        match_terminal: matchTerminal,
        match_winner_player_id: matchWinnerPlayerId,
        match_draw: matchDraw,
        round_summaries: clone(summaries),
      };
    },
    result() {
      return {
        ...this.snapshot(),
        current_round: round.result(),
      };
    },
  };
}

export function replayDuelRound({ pairSeed, rules = DUEL_PROTOTYPE_RULES, inputEvents = [], finalTick = null } = {}) {
  const round = createDuelRound({ pairSeed, rules });
  const grouped = new Map();
  for (const event of inputEvents) {
    const tick = Math.max(0, Math.floor(Number(event.tick) || 0));
    if (!grouped.has(tick)) grouped.set(tick, []);
    grouped.get(tick).push(event);
  }
  const maxTick = finalTick === null
    ? Math.ceil(round.rules.round_duration_s / round.rules.dt_s)
    : Math.max(0, Math.floor(Number(finalTick) || 0));
  for (let tick = 0; tick <= maxTick && !round.snapshot().terminal; tick += 1) {
    for (const event of grouped.get(tick) || []) round.setControls(event.role, event.controls, event);
    if (tick < maxTick) round.step(1);
  }
  return round.result();
}

export function restoreDuelSeries({ result, rules = DUEL_PROTOTYPE_RULES } = {}) {
  if (!result || !Array.isArray(result.players) || result.players.length !== 2) {
    throw new Error("A serialized duel-series result with two players is required.");
  }
  const series = createDuelSeries({
    playerIds: result.players,
    regulationRounds: result.regulation_rounds,
    matchSeed: result.match_seed,
    rules,
  });
  const targetRoundIndex = Math.max(1, Math.floor(Number(result.round_index) || 1));
  const summaries = new Map(
    (result.round_summaries || []).map((summary) => [Number(summary.round_index), summary]),
  );

  for (let roundIndex = 1; roundIndex <= targetRoundIndex; roundIndex += 1) {
    const record = summaries.get(roundIndex) || (roundIndex === targetRoundIndex ? result.current_round : null);
    if (!record) throw new Error(`Serialized duel series is missing round ${roundIndex}.`);
    replaySeriesRound(series, record.input_events || [], record.tick);
    if (roundIndex < targetRoundIndex) {
      if (!series.snapshot().round_complete) {
        throw new Error(`Serialized duel round ${roundIndex} is not complete.`);
      }
      series.advanceRound();
    }
  }
  return series;
}

export function automaticSpeedState({
  maneuvering,
  nowMs,
  lastManeuverMs,
  rules = DUEL_PROTOTYPE_RULES,
} = {}) {
  const cfg = normalizeRules(rules);
  const active = Boolean(maneuvering);
  const now = Number(nowMs) || 0;
  const last = Number(lastManeuverMs);
  const coolingDown = Number.isFinite(last) && now - last < cfg.neutral_cooldown_ms;
  return {
    speed_multiple: active || coolingDown ? cfg.maneuver_speed_multiple : cfg.coast_speed_multiple,
    reason: active ? "maneuvering" : coolingDown ? "neutral_cooldown" : "coasting",
  };
}

function pursuitGeometryConfig(rules) {
  return {
    ...DEFAULT_PURSUIT_CHALLENGE,
    mu_km3_s2: rules.mu_km3_s2,
    target_coes: clone(rules.target_coes),
    max_time_s: rules.round_duration_s,
    goal_range_km: rules.capture_range_km,
    goal_speed_km_s: rules.capture_relative_speed_limit_km_s,
    max_delta_v_m_s: rules.chaser_delta_v_budget_m_s,
    max_target_delta_v_m_s: rules.target_delta_v_budget_m_s,
    target_defense: {
      ...DEFAULT_PURSUIT_CHALLENGE.target_defense,
      enabled: false,
      max_delta_v_m_s: rules.target_delta_v_budget_m_s,
    },
    arcade: {
      ...DEFAULT_PURSUIT_CHALLENGE.arcade,
      random_initial_state: clone(rules.random_initial_state),
    },
  };
}

function replaySeriesRound(series, inputEvents, finalTick) {
  const grouped = new Map();
  for (const event of inputEvents) {
    const tick = Math.max(0, Math.floor(Number(event.tick) || 0));
    if (!grouped.has(tick)) grouped.set(tick, []);
    grouped.get(tick).push(event);
  }
  const maxTick = Math.max(0, Math.floor(Number(finalTick) || 0));
  for (let tick = 0; tick <= maxTick && !series.snapshot().round_complete; tick += 1) {
    for (const event of grouped.get(tick) || []) {
      series.setPlayerControls(event.player_id, event.controls, { sequence: event.sequence });
    }
    if (tick < maxTick) series.step(1);
  }
}

function normalizeRules(value) {
  const rules = { ...clone(DUEL_PROTOTYPE_RULES), ...clone(value || {}) };
  rules.target_coes = { ...DUEL_PROTOTYPE_RULES.target_coes, ...(value?.target_coes || {}) };
  rules.random_initial_state = {
    ...DUEL_PROTOTYPE_RULES.random_initial_state,
    ...(value?.random_initial_state || {}),
  };
  rules.mu_km3_s2 = positive(rules.mu_km3_s2 ?? DEFAULT_PURSUIT_CHALLENGE.mu_km3_s2, "mu_km3_s2");
  for (const name of [
    "dt_s", "round_duration_s", "capture_range_km", "chaser_delta_v_budget_m_s",
    "target_delta_v_budget_m_s", "chaser_max_accel_km_s2", "target_max_accel_km_s2",
    "coast_speed_multiple", "maneuver_speed_multiple", "neutral_cooldown_ms",
    "history_stride_ticks", "max_history_samples",
  ]) rules[name] = positive(rules[name], name);
  return deepFreeze(rules);
}

function appliedAcceleration(controls, maxAccel, remainingMps, dtS) {
  const direction = [controls.r, controls.i, controls.c];
  const magnitude = norm(direction);
  if (magnitude <= 0 || remainingMps <= 0) return [0, 0, 0];
  const allowed = Math.min(maxAccel, remainingMps / 1000 / dtS);
  return direction.map((value) => (value / magnitude) * allowed);
}

function remainingDeltaV(sim, rules, role) {
  const budget = role === "chaser" ? rules.chaser_delta_v_budget_m_s : rules.target_delta_v_budget_m_s;
  return Math.max(budget - sim.delta_v_m_s[role], 0);
}

function normalizeControls(value = {}) {
  const controls = { r: axis(value.r), i: axis(value.i), c: axis(value.c) };
  const magnitude = controlMagnitude(controls);
  if (magnitude <= 1) return controls;
  return { r: controls.r / magnitude, i: controls.i / magnitude, c: controls.c / magnitude };
}

function neutralControls() {
  return { r: 0, i: 0, c: 0 };
}

function sameControls(first, second) {
  return first.r === second.r && first.i === second.i && first.c === second.c;
}

function controlMagnitude(controls) {
  return Math.hypot(controls.r, controls.i, controls.c);
}

function axis(value) {
  const number = Number(value) || 0;
  return Math.max(-1, Math.min(number, 1));
}

function assertRole(role) {
  if (!Object.values(DUEL_ROLES).includes(role)) throw new Error(`Unknown duel role: ${role}.`);
}

function normalizeRoundCount(value) {
  const count = Math.floor(Number(value) || 0);
  if (!DUEL_ROUND_COUNTS.includes(count)) throw new Error("regulationRounds must be 2, 4, or 6.");
  return count;
}

function roundedRelative(value) {
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, metric(item)]));
}

function roundedMap(value) {
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, metric(item)]));
}

function cloneEci(value) {
  return { r: value.r.map(Number), v: value.v.map(Number) };
}

function metric(value) {
  return Number(Number(value).toPrecision(12));
}

function positive(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) throw new Error(`${name} must be positive.`);
  return number;
}

function uint32(value) {
  return Number(value) >>> 0;
}

function norm(values) {
  return Math.hypot(...values);
}

function clone(value) {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.freeze(value);
  Object.values(value).forEach(deepFreeze);
  return value;
}
