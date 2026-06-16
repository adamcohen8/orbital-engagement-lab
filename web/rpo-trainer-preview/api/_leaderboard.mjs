import { supabaseRest } from "./_supabase.mjs";

export const VALID_LEADERBOARD_STATUSES = new Set(["valid", "suspicious"]);

export function isLeaderboardEligibleStatus(status) {
  return VALID_LEADERBOARD_STATUSES.has(String(status || ""));
}

export function shouldReplaceLeaderboardScore(currentScore, candidateScore) {
  if (currentScore == null) return true;
  return Number(candidateScore || 0) > Number(currentScore || 0);
}

export async function upsertLeaderboardIfBetter({ challengeId, playerId, attemptId, score, metrics }) {
  const currentQuery = new URLSearchParams({
    challenge_id: `eq.${challengeId}`,
    player_id: `eq.${playerId}`,
    select: "score",
    limit: "1",
  });
  const current = await supabaseRest(`leaderboard_entries?${currentQuery.toString()}`);
  if (!shouldReplaceLeaderboardScore(current?.[0]?.score, score)) return false;

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
        updated_at: new Date().toISOString(),
      },
    ]),
  });
  return true;
}
