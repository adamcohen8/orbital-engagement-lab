import { createHash, randomBytes } from "node:crypto";

const EMAIL_TOKEN_BYTES = 32;
const EMAIL_EXPIRY_DAYS = 7;

export function emailConfigured() {
  return Boolean(process.env.RESEND_API_KEY && process.env.OEL_ARCADE_EMAIL_FROM);
}

export function createVerificationToken() {
  const token = randomBytes(EMAIL_TOKEN_BYTES).toString("base64url");
  return {
    token,
    token_hash: hashVerificationToken(token),
  };
}

export function hashVerificationToken(token) {
  return createHash("sha256").update(String(token || ""), "utf8").digest("hex");
}

export function verificationExpiryIso(now = new Date()) {
  return new Date(now.getTime() + EMAIL_EXPIRY_DAYS * 24 * 60 * 60 * 1000).toISOString();
}

export function publicOrigin(req) {
  const configured = String(process.env.OEL_ARCADE_PUBLIC_ORIGIN || "").replace(/\/+$/, "");
  if (configured) return configured;
  const allowed = String(process.env.OEL_ARCADE_ALLOWED_ORIGIN || "").replace(/\/+$/, "");
  if (allowed && allowed !== "*") return allowed;
  const host = req?.headers?.host;
  const proto = req?.headers?.["x-forwarded-proto"] || "https";
  return host ? `${proto}://${host}` : "https://orbital-engagement-lab.vercel.app";
}

export function verificationUrl(req, token) {
  return `${publicOrigin(req)}/api/verify-email?token=${encodeURIComponent(token)}`;
}

export async function sendScoreVerificationEmail({ email, username, score, roundsCleared, attemptId, verifyUrl }) {
  if (!emailConfigured()) {
    return { status: "not_configured" };
  }

  const from = String(process.env.OEL_ARCADE_EMAIL_FROM || "");
  const subject = `Verify your Pursuit Arcade score: ${Number(score || 0).toLocaleString()}`;
  const text = [
    `Nice flying, ${username}.`,
    "",
    `Your Pursuit Arcade score was ${Number(score || 0).toLocaleString()}.`,
    `Rounds cleared: ${Number(roundsCleared || 0)}`,
    `Attempt ID: ${attemptId}`,
    "",
    "Verify this email address so you can prove ownership of the score:",
    verifyUrl,
    "",
    "This link expires in 7 days.",
  ].join("\n");
  const html = `
    <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #0f172a; line-height: 1.5;">
      <h1 style="font-size: 22px;">Pursuit Arcade score received</h1>
      <p>Nice flying, <strong>${escapeHtml(username)}</strong>.</p>
      <p>Your score was <strong>${Number(score || 0).toLocaleString()}</strong>.</p>
      <p>Rounds cleared: <strong>${Number(roundsCleared || 0)}</strong></p>
      <p>Attempt ID: <code>${escapeHtml(attemptId)}</code></p>
      <p><a href="${escapeHtml(verifyUrl)}">Verify this email address</a> so you can prove ownership of the score.</p>
      <p style="color: #475569; font-size: 13px;">This link expires in 7 days.</p>
    </div>
  `;

  const response = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from,
      to: [email],
      subject,
      html,
      text,
      tags: [
        { name: "app", value: "oel_arcade" },
        { name: "event", value: "score_verification" },
      ],
    }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload?.message || `Email send failed with HTTP ${response.status}`);
  }
  return { status: "sent", id: payload?.id || null };
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
