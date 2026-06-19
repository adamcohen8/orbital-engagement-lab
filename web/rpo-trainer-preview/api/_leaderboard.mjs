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
  const currentQuery = new URLSearchParams({
    challenge_id: `eq.${challengeId}`,
    player_id: `eq.${playerId}`,
    select: "score",
    limit: "1",
  });
  const current = await supabaseRest(`leaderboard_entries?${currentQuery.toString()}`);
  if (!shouldReplaceLeaderboardScore(current?.[0]?.score, score)) return false;

  const publicRow = await publicLeaderboardRow({
    challengeId,
    playerId,
    attemptId,
    score,
    metrics,
    username,
    submittedAt,
    emailVerified,
  });
  const updatedAt = new Date().toISOString();
  await supabaseRest("leaderboard_entries?on_conflict=challenge_id,player_id", {
    method: "POST",
    headers: { Prefer: "resolution=merge-duplicates" },
    body: JSON.stringify([
      {
        challenge_id: challengeId,
        player_id: playerId,
        attempt_id: attemptId,
        score: score ?? 0,
        metrics: metrics ?? {},
        updated_at: updatedAt,
      },
    ]),
  });
  await supabaseRest("public_leaderboard?on_conflict=challenge_id,username", {
    method: "POST",
    headers: { Prefer: "resolution=merge-duplicates" },
    body: JSON.stringify([{ ...publicRow, updated_at: updatedAt }]),
  });
  return true;
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
