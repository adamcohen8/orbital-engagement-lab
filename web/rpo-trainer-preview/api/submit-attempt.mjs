import {
  buildChallengeRecord,
  DEFAULT_PURSUIT_CHALLENGE,
  trajectoryPlotSvg,
  validateAttemptPacket,
} from "../src/competition/arcade-engine.js";
import {
  createVerificationToken,
  sendScoreVerificationEmail,
  verificationExpiryIso,
  verificationUrl,
} from "./_email.mjs";
import {
  normalizeEmail,
  normalizedUsernameKey,
  normalizeUsername,
  readBody,
  sendJson,
  supabaseRest,
} from "./_supabase.mjs";

const CHALLENGE_RECORD = buildChallengeRecord(DEFAULT_PURSUIT_CHALLENGE);
const VALID_LEADERBOARD_STATUSES = new Set(["valid", "suspicious"]);

export default async function handler(req, res) {
  if (req.method === "OPTIONS") {
    sendJson(res, 204, {});
    return;
  }
  if (req.method !== "POST") {
    sendJson(res, 405, { error: "Use POST for attempt submission." });
    return;
  }

  try {
    const body = await readBody(req);
    const attempt = body.attempt || body;
    const username = normalizeUsername(body.username || attempt.username);
    const email = normalizeEmail(body.email || attempt.email);
    const validation = validateAttemptPacket({ ...attempt, username, email }, CHALLENGE_RECORD, {
      sample_stride_ticks: 10,
    });

    await upsertChallenge(CHALLENGE_RECORD);
    const player = await findOrCreatePlayer(username, email);
    const attemptRow = await insertAttempt(player.id, { ...attempt, username, email }, validation);
    const accepted = VALID_LEADERBOARD_STATUSES.has(validation.status);
    let leaderboardUpdated = false;
    if (accepted) {
      leaderboardUpdated = await upsertLeaderboard(player.id, attemptRow.id, validation);
    }
    const emailResult = await maybeSendVerificationEmail({
      req,
      player,
      attemptRow,
      email,
      username,
      validation,
      accepted,
    });

    sendJson(res, accepted ? 200 : 422, {
      status: validation.status,
      errors: validation.errors,
      warnings: validation.warnings,
      score: validation.canonical_score ?? 0,
      metrics: validation.canonical_metrics ?? {},
      attempt_id: attemptRow.id,
      leaderboard_updated: leaderboardUpdated,
      email_status: emailResult.status,
      email_error: emailResult.error,
    });
  } catch (error) {
    sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
  }
}

async function upsertChallenge(record) {
  await supabaseRest("challenges?on_conflict=id", {
    method: "POST",
    headers: { Prefer: "resolution=merge-duplicates" },
    body: JSON.stringify([
      {
        id: record.challenge_id,
        title: record.config.title || "Pursuit Arcade",
        physics_version: record.physics_version,
        scoring_version: record.scoring_version,
        config_hash: record.config_hash,
        config: record.config,
        active: true,
      },
    ]),
  });
}

async function findOrCreatePlayer(username, email) {
  const key = normalizedUsernameKey(username);
  const selectQuery = new URLSearchParams({
    username_normalized: `eq.${key}`,
    select: "id,username,email",
    limit: "1",
  });
  const existing = await supabaseRest(`players?${selectQuery.toString()}`);
  if (existing?.[0]) {
    if (email && existing[0].email !== email) {
      await supabaseRest(`players?id=eq.${existing[0].id}`, {
        method: "PATCH",
        body: JSON.stringify({ email }),
      });
    }
    return existing[0];
  }

  try {
    const inserted = await supabaseRest("players", {
      method: "POST",
      headers: { Prefer: "return=representation" },
      body: JSON.stringify([{ username, email: email || null }]),
    });
    return inserted[0];
  } catch (error) {
    const retry = await supabaseRest(`players?${selectQuery.toString()}`);
    if (retry?.[0]) return retry[0];
    throw error;
  }
}

async function insertAttempt(playerId, attempt, validation) {
  const replaySummary = validation.replay
    ? {
        score: validation.replay.score,
        metrics: validation.replay.metrics,
        round_summaries: validation.replay.round_summaries || [],
        round_attempts: validation.replay.round_attempts || [],
      }
    : null;
  const accepted = VALID_LEADERBOARD_STATUSES.has(validation.status);
  const inserted = await supabaseRest("attempts", {
    method: "POST",
    headers: { Prefer: "return=representation" },
    body: JSON.stringify([
      {
        player_id: playerId,
        challenge_id: CHALLENGE_RECORD.challenge_id,
        status: validation.status,
        score: validation.canonical_score ?? 0,
        metrics: validation.canonical_metrics ?? {},
        replay: { attempt, canonical: replaySummary },
        config_hash: CHALLENGE_RECORD.config_hash,
        physics_version: CHALLENGE_RECORD.physics_version,
        scoring_version: CHALLENGE_RECORD.scoring_version,
        validation_errors: validation.errors || [],
        validation_warnings: validation.warnings || [],
        ri_plot_svg: accepted && validation.replay ? trajectoryPlotSvg(validation.replay, "RI") : null,
        rc_plot_svg: accepted && validation.replay ? trajectoryPlotSvg(validation.replay, "RC") : null,
        validated_at: new Date().toISOString(),
      },
    ]),
  });
  return inserted[0];
}

async function upsertLeaderboard(playerId, attemptId, validation) {
  const currentQuery = new URLSearchParams({
    challenge_id: `eq.${CHALLENGE_RECORD.challenge_id}`,
    player_id: `eq.${playerId}`,
    select: "score",
    limit: "1",
  });
  const current = await supabaseRest(`leaderboard_entries?${currentQuery.toString()}`);
  if (current?.[0] && Number(current[0].score || 0) >= Number(validation.canonical_score || 0)) return false;

  await supabaseRest("leaderboard_entries?on_conflict=challenge_id,player_id", {
    method: "POST",
    headers: { Prefer: "resolution=merge-duplicates" },
    body: JSON.stringify([
      {
        challenge_id: CHALLENGE_RECORD.challenge_id,
        player_id: playerId,
        attempt_id: attemptId,
        score: validation.canonical_score ?? 0,
        metrics: validation.canonical_metrics ?? {},
        updated_at: new Date().toISOString(),
      },
    ]),
  });
  return true;
}

async function maybeSendVerificationEmail({ req, player, attemptRow, email, username, validation, accepted }) {
  if (!accepted || !email) return { status: "skipped" };
  const tokenRecord = createVerificationToken();
  await supabaseRest("email_verifications", {
    method: "POST",
    body: JSON.stringify([
      {
        player_id: player.id,
        attempt_id: attemptRow.id,
        email,
        token_hash: tokenRecord.token_hash,
        expires_at: verificationExpiryIso(),
      },
    ]),
  });

  try {
    return await sendScoreVerificationEmail({
      email,
      username,
      score: validation.canonical_score ?? 0,
      roundsCleared: validation.canonical_metrics?.rounds_cleared || validation.replay?.rounds_cleared || 0,
      attemptId: attemptRow.id,
      verifyUrl: verificationUrl(req, tokenRecord.token),
    });
  } catch (error) {
    return { status: "failed", error: error instanceof Error ? error.message : String(error) };
  }
}
