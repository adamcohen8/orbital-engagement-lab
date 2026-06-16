import { escapeHtml, hashVerificationToken, publicOrigin } from "./_email.mjs";
import { isLeaderboardEligibleStatus, upsertLeaderboardIfBetter } from "./_leaderboard.mjs";
import { canVerifyUsernameForEmail } from "./_ownership.mjs";
import { supabaseRest } from "./_supabase.mjs";

export default async function handler(req, res) {
  if (req.method !== "GET") {
    sendHtml(res, 405, "Use GET for email verification.");
    return;
  }

  try {
    const url = new URL(req.url || "/api/verify-email", publicOrigin(req));
    const token = url.searchParams.get("token") || "";
    if (!token) {
      sendHtml(res, 400, "The verification link is missing its token.");
      return;
    }

    const query = new URLSearchParams({
      token_hash: `eq.${hashVerificationToken(token)}`,
      select: "id,player_id,attempt_id,email,expires_at,verified_at",
      limit: "1",
    });
    const rows = await supabaseRest(`email_verifications?${query.toString()}`);
    const verification = rows?.[0];
    if (!verification) {
      sendHtml(res, 404, "This verification link was not found.");
      return;
    }
    if (new Date(verification.expires_at).getTime() < Date.now()) {
      sendHtml(res, 410, "This verification link has expired.");
      return;
    }

    const playerQuery = new URLSearchParams({
      id: `eq.${verification.player_id}`,
      select: "id,email,email_verified_at,username_locked_at",
      limit: "1",
    });
    const players = await supabaseRest(`players?${playerQuery.toString()}`);
    const player = players?.[0];
    if (!player) {
      sendHtml(res, 404, "The username for this verification link was not found.");
      return;
    }
    if (!canVerifyUsernameForEmail({ player, email: verification.email })) {
      sendHtml(res, 409, "This username is already reserved to a different verified email address.");
      return;
    }

    const verifiedAt = verification.verified_at || new Date().toISOString();
    if (!verification.verified_at) {
      await supabaseRest(`email_verifications?id=eq.${verification.id}`, {
        method: "PATCH",
        body: JSON.stringify({ verified_at: verifiedAt }),
      });
    }
    await supabaseRest(`players?id=eq.${verification.player_id}`, {
      method: "PATCH",
      body: JSON.stringify({
        email: verification.email,
        email_verified_at: verifiedAt,
        username_locked_at: player.username_locked_at || verifiedAt,
      }),
    });
    const promoted = await promoteVerifiedAttempt(verification);
    sendSuccess(
      res,
      promoted
        ? "Email verified. Your username is now reserved and your best linked score is on the leaderboard."
        : "Email verified. Your username is now reserved.",
    );
  } catch (error) {
    sendHtml(res, 500, error instanceof Error ? error.message : String(error));
  }
}

async function promoteVerifiedAttempt(verification) {
  if (!verification.attempt_id) return false;
  const attemptQuery = new URLSearchParams({
    id: `eq.${verification.attempt_id}`,
    select: "id,player_id,challenge_id,status,score,metrics",
    limit: "1",
  });
  const attempts = await supabaseRest(`attempts?${attemptQuery.toString()}`);
  const attempt = attempts?.[0];
  if (!attempt || attempt.player_id !== verification.player_id || !isLeaderboardEligibleStatus(attempt.status)) {
    return false;
  }
  return await upsertLeaderboardIfBetter({
    challengeId: attempt.challenge_id,
    playerId: attempt.player_id,
    attemptId: attempt.id,
    score: attempt.score,
    metrics: attempt.metrics || {},
  });
}

function sendSuccess(res, message) {
  sendHtml(res, 200, message, true);
}

function sendHtml(res, statusCode, message, ok = false) {
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.status(statusCode).send(`<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OEL Email Verification</title>
    <style>
      body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0d141f; color: #e8eef8; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
      main { max-width: 560px; padding: 32px; border: 2px solid #53657e; background: #111a26; }
      h1 { margin: 0 0 16px; font-size: 24px; }
      p { margin: 0 0 24px; color: #b7c4d7; }
      a { color: #8fd3ff; }
    </style>
  </head>
  <body>
    <main>
      <h1>${ok ? "Verified" : "Verification issue"}</h1>
      <p>${escapeHtml(message)}</p>
      <a href="/">Return to Pursuit Arcade</a>
    </main>
  </body>
</html>`);
}
