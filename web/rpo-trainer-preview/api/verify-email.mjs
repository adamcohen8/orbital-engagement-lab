import { escapeHtml, hashVerificationToken } from "./_email.mjs";
import { supabaseRest } from "./_supabase.mjs";

export default async function handler(req, res) {
  if (req.method !== "GET" && req.method !== "POST") {
    sendHtml(res, 405, "Use the verification page and its confirmation button.");
    return;
  }

  try {
    const token = req.method === "GET" ? tokenFromUrl(req) : tokenFromBody(req);
    if (!token || token.length > 128) {
      sendHtml(res, 400, req.method === "GET"
        ? "The verification link is missing a usable token."
        : "The verification token is missing or invalid.");
      return;
    }

    if (req.method === "GET") {
      sendConfirmation(res, token);
      return;
    }

    // Generated verification tokens are 32 random bytes encoded as base64url.
    // Do not call the claim RPC for malformed or oversized form submissions.
    if (!/^[A-Za-z0-9_-]{43}$/.test(token)) {
      sendHtml(res, 400, "The verification token is missing or invalid.");
      return;
    }

    // One database transaction owns the token and player locks, first-claim
    // transition, and exact-attempt promotion. A missing migration fails closed.
    const result = await supabaseRest("rpc/claim_arcade_attempt", {
      method: "POST",
      body: JSON.stringify({ p_token_hash: hashVerificationToken(token) }),
    });
    switch (result?.status) {
      case "verified":
        sendHtml(res, 200, result.promoted
          ? "Email verified. Your username is now reserved and your linked score is on the leaderboard."
          : "Email verified. Your username is now reserved.", true);
        return;
      case "already_verified":
        sendHtml(res, 200, "This link was already used to verify your username.", true);
        return;
      case "expired":
        sendHtml(res, 410, "This verification link has expired.");
        return;
      case "locked":
        sendHtml(res, 409, "This username is already reserved to a different verified email address.");
        return;
      case "invalid_attempt":
        sendHtml(res, 409, "The linked attempt is no longer eligible for verification.");
        return;
      default:
        sendHtml(res, 404, "This verification link was not found.");
    }
  } catch (error) {
    sendHtml(res, 500, error instanceof Error ? error.message : String(error));
  }
}

function tokenFromUrl(req) {
  try {
    const url = new URL(req.url || "/api/verify-email", "https://localhost.invalid");
    return url.searchParams.get("token") || "";
  } catch {
    return "";
  }
}

function tokenFromBody(req) {
  const body = req.body;
  if (typeof body === "string") {
    const contentType = String(req.headers?.["content-type"] || req.headers?.["Content-Type"] || "");
    if (contentType.toLowerCase().includes("application/x-www-form-urlencoded")) {
      return new URLSearchParams(body).get("token") || "";
    }
    return "";
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) return "";
  return typeof body.token === "string" ? body.token : "";
}

function sendConfirmation(res, token) {
  sendHtml(res, 200, "Opening this page has not verified your email or reserved your username. Press the button below to confirm.", {
    title: "Confirm email verification",
    formToken: token,
  });
}

function sendHtml(res, statusCode, message, options = {}) {
  const view = typeof options === "boolean" ? { ok: options } : options;
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.status(statusCode).send(`<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${escapeHtml(view.title || "OEL Email Verification")}</title>
    <style>
      body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0d141f; color: #e8eef8; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
      main { max-width: 560px; padding: 32px; border: 2px solid #53657e; background: #111a26; }
      h1 { margin: 0 0 16px; font-size: 24px; }
      p { margin: 0 0 24px; color: #b7c4d7; }
      a { color: #8fd3ff; }
      form { margin: 0 0 24px; }
      button { padding: 12px 18px; border: 0; background: #8fd3ff; color: #0d141f; font: inherit; font-weight: 700; cursor: pointer; }
    </style>
  </head>
  <body>
    <main>
      <h1>${view.formToken !== undefined ? "Confirm email" : view.ok ? "Verified" : "Verification issue"}</h1>
      <p>${escapeHtml(message)}</p>
      ${view.formToken !== undefined ? `<form method="post" action="/api/verify-email"><input type="hidden" name="token" value="${escapeHtml(view.formToken)}" /><button type="submit">Confirm and reserve username</button></form>` : ""}
      <a href="/">Return to Pursuit Arcade</a>
    </main>
  </body>
</html>`);
}
