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
import { isLeaderboardEligibleStatus, upsertLeaderboardIfBetter } from "./_leaderboard.mjs";
import { decideOwnership } from "./_ownership.mjs";
import {
  normalizeEmail,
  normalizedUsernameKey,
  normalizeUsername,
  readBody,
  sendJson,
  supabaseRest,
} from "./_supabase.mjs";

const CHALLENGE_RECORD = buildChallengeRecord(DEFAULT_PURSUIT_CHALLENGE);

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
    const player = await findOrCreatePlayer(username);
    const ownership = decideOwnership({ player, email });
    const attemptRow = await insertAttempt(player.id, { ...attempt, username, email }, validation);
    const accepted = isLeaderboardEligibleStatus(validation.status);
    let leaderboardUpdated = false;
    if (accepted && ownership.leaderboard_allowed) {
      leaderboardUpdated = await upsertLeaderboardIfBetter({
        challengeId: CHALLENGE_RECORD.challenge_id,
        playerId: player.id,
        attemptId: attemptRow.id,
        score: validation.canonical_score ?? 0,
        metrics: validation.canonical_metrics ?? {},
      });
    }
    const emailResult = await maybeSendVerificationEmail({
      req,
      player,
      attemptRow,
      email,
      username,
      validation,
      accepted,
      ownership,
    });

    sendJson(res, accepted ? 200 : 422, {
      status: validation.status,
      errors: validation.errors,
      warnings: validation.warnings,
      score: validation.canonical_score ?? 0,
      metrics: validation.canonical_metrics ?? {},
      attempt_id: attemptRow.id,
      ownership_status: ownership.status,
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

async function findOrCreatePlayer(username) {
  const key = normalizedUsernameKey(username);
  const selectQuery = new URLSearchParams({
    username_normalized: `eq.${key}`,
    select: "id,username,email,email_verified_at,username_locked_at",
    limit: "1",
  });
  const existing = await supabaseRest(`players?${selectQuery.toString()}`);
  if (existing?.[0]) {
    return existing[0];
  }

  try {
    const inserted = await supabaseRest("players", {
      method: "POST",
      headers: { Prefer: "return=representation" },
      body: JSON.stringify([{ username }]),
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
  const accepted = isLeaderboardEligibleStatus(validation.status);
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

async function maybeSendVerificationEmail({ req, player, attemptRow, email, username, validation, accepted, ownership }) {
  if (!accepted || !email || !ownership.verification_allowed) return { status: "skipped" };
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
