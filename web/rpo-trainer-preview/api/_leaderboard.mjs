import { supabaseRest } from "./_supabase.mjs";

export const VALID_LEADERBOARD_STATUSES = new Set(["valid", "suspicious"]);

export function isLeaderboardEligibleStatus(status) {
  return VALID_LEADERBOARD_STATUSES.has(String(status || ""));
}

export function shouldReplaceLeaderboardScore(currentScore, candidateScore) {
  if (currentScore == null) return true;
  return Number(candidateScore || 0) > Number(currentScore || 0);
}

export async function upsertLeaderboardIfBetter({
  challengeId,
  playerId,
  attemptId,
  score,
  metrics,
  username,
  submittedAt,
  emailVerified,
}) {
  const candidateScore = Number(score ?? 0);
  const privateRow = {
    challenge_id: challengeId,
    player_id: playerId,
    attempt_id: attemptId,
    score: candidateScore,
    metrics: metrics ?? {},
  };
  const updatedAt = new Date().toISOString();
  const inserted = await supabaseRest("leaderboard_entries?on_conflict=challenge_id,player_id", {
    method: "POST",
    headers: { Prefer: "resolution=ignore-duplicates,return=representation" },
    body: JSON.stringify([{ ...privateRow, updated_at: updatedAt }]),
  });
  let privateChanged = Array.isArray(inserted) && inserted.length > 0;
  if (!privateChanged) {
    const updateQuery = new URLSearchParams({
      challenge_id: `eq.${challengeId}`,
      player_id: `eq.${playerId}`,
      score: `lt.${candidateScore}`,
    });
    const updated = await supabaseRest(`leaderboard_entries?${updateQuery.toString()}`, {
      method: "PATCH",
      headers: { Prefer: "return=representation" },
      body: JSON.stringify({ ...privateRow, updated_at: updatedAt }),
    });
    privateChanged = Array.isArray(updated) && updated.length > 0;
  }

  // Re-read the private winner after the conditional write.  A retry creates a
  // new attempt row, so the winner's attempt_id may differ from this request's
  // attemptId even when it is the same packet and score.  Mirroring the stored
  // winner makes public repair independent of request retry identity and also
  // lets a lower-scoring concurrent request repair a stale public row.
  const currentQuery = new URLSearchParams({
    challenge_id: `eq.${challengeId}`,
    player_id: `eq.${playerId}`,
    select: "score,attempt_id,metrics",
    limit: "1",
  });
  const current = await supabaseRest(`leaderboard_entries?${currentQuery.toString()}`);
  const winner = current?.[0];
  if (winner?.attempt_id) {
    const publicRow = await publicLeaderboardRow({
      challengeId,
      playerId,
      attemptId: winner.attempt_id,
      score: winner.score,
      metrics: winner.metrics,
      username,
      emailVerified,
    });
    await upsertPublicLeaderboardIfBetter({ publicRow, updatedAt, repairEqual: true });
  }
  return privateChanged;
}

async function upsertPublicLeaderboardIfBetter({ publicRow, updatedAt, repairEqual }) {
  const inserted = await supabaseRest("public_leaderboard?on_conflict=challenge_id,username", {
    method: "POST",
    headers: { Prefer: "resolution=ignore-duplicates,return=representation" },
    body: JSON.stringify([{ ...publicRow, updated_at: updatedAt }]),
  });
  if (Array.isArray(inserted) && inserted.length > 0) return true;

  const candidateScore = Number(publicRow.score ?? 0);
  const lowerQuery = new URLSearchParams({
    challenge_id: `eq.${publicRow.challenge_id}`,
    username: `eq.${publicRow.username}`,
    score: `lt.${candidateScore}`,
  });
  const updated = await supabaseRest(`public_leaderboard?${lowerQuery.toString()}`, {
    method: "PATCH",
    headers: { Prefer: "return=representation" },
    body: JSON.stringify({ ...publicRow, updated_at: updatedAt }),
  });
  if (Array.isArray(updated) && updated.length > 0) return true;
  if (!repairEqual) return false;

  const equalQuery = new URLSearchParams({
    challenge_id: `eq.${publicRow.challenge_id}`,
    username: `eq.${publicRow.username}`,
    score: `eq.${candidateScore}`,
  });
  const repaired = await supabaseRest(`public_leaderboard?${equalQuery.toString()}`, {
    method: "PATCH",
    headers: { Prefer: "return=representation" },
    body: JSON.stringify({ ...publicRow, updated_at: updatedAt }),
  });
  return Array.isArray(repaired) && repaired.length > 0;
}

async function publicLeaderboardRow({
  challengeId,
  playerId,
  attemptId,
  score,
  metrics,
  username,
  submittedAt,
  emailVerified,
}) {
  let publicUsername = username;
  let publicEmailVerified = emailVerified;
  if (!publicUsername || publicEmailVerified == null) {
    const playerQuery = new URLSearchParams({
      id: `eq.${playerId}`,
      select: "username,email_verified_at",
      limit: "1",
    });
    const players = await supabaseRest(`players?${playerQuery.toString()}`);
    publicUsername = publicUsername || players?.[0]?.username;
    publicEmailVerified = publicEmailVerified ?? Boolean(players?.[0]?.email_verified_at);
  }

  let publicSubmittedAt = submittedAt;
  if (!publicSubmittedAt) {
    const attemptQuery = new URLSearchParams({
      id: `eq.${attemptId}`,
      select: "submitted_at",
      limit: "1",
    });
    const attempts = await supabaseRest(`attempts?${attemptQuery.toString()}`);
    publicSubmittedAt = attempts?.[0]?.submitted_at;
  }

  if (!publicUsername || !publicSubmittedAt) {
    throw new Error("Cannot publish leaderboard row without username and submitted_at.");
  }

  return {
    challenge_id: challengeId,
    username: publicUsername,
    score: score ?? 0,
    metrics: metrics ?? {},
    attempt_id: attemptId,
    submitted_at: publicSubmittedAt,
    email_verified: Boolean(publicEmailVerified),
  };
}

export async function publishUnclaimedAttempt(attemptId) {
  const result = await supabaseRest("rpc/publish_unclaimed_arcade_attempt", {
    method: "POST",
    body: JSON.stringify({ p_attempt_id: attemptId }),
  });
  return result === true;
}
