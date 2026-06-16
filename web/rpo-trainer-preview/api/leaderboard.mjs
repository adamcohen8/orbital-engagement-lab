import { sendJson, supabaseRest } from "./_supabase.mjs";
import { buildChallengeRecord, DEFAULT_PURSUIT_CHALLENGE } from "../src/competition/arcade-engine.js";

const DEFAULT_RECORD = buildChallengeRecord(DEFAULT_PURSUIT_CHALLENGE);

export default async function handler(req, res) {
  if (req.method === "OPTIONS") {
    sendJson(res, 204, {});
    return;
  }
  if (req.method !== "GET") {
    sendJson(res, 405, { error: "Use GET for leaderboard reads." });
    return;
  }

  try {
    const url = new URL(req.url || "/api/leaderboard", "https://oel.local");
    const challengeId = url.searchParams.get("challenge") || DEFAULT_RECORD.challenge_id;
    const limit = Math.min(Math.max(Number(url.searchParams.get("limit") || 25), 1), 100);
    const query = new URLSearchParams({
      challenge_id: `eq.${challengeId}`,
      select: "challenge_id,username,score,metrics,attempt_id,submitted_at,email_verified",
      order: "score.desc,submitted_at.asc",
      limit: String(limit),
    });
    const rows = await supabaseRest(`public_leaderboard?${query.toString()}`);
    sendJson(res, 200, { status: "ok", challenge_id: challengeId, entries: rows || [] });
  } catch (error) {
    sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
  }
}
