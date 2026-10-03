import assert from "node:assert/strict";
import { test } from "node:test";

import {
  maybeSendVerificationEmail,
  reserveVerificationSend,
} from "../api/submit-attempt.mjs";
import { upsertLeaderboardIfBetter } from "../api/_leaderboard.mjs";

const SUPABASE_URL = "https://supabase.example";

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function withSupabaseFetch(handler) {
  const originalFetch = globalThis.fetch;
  const originalUrl = process.env.SUPABASE_URL;
  const originalKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  process.env.SUPABASE_URL = SUPABASE_URL;
  process.env.SUPABASE_SERVICE_ROLE_KEY = "test-service-key";
  return {
    restore() {
      globalThis.fetch = originalFetch;
      if (originalUrl === undefined) delete process.env.SUPABASE_URL;
      else process.env.SUPABASE_URL = originalUrl;
      if (originalKey === undefined) delete process.env.SUPABASE_SERVICE_ROLE_KEY;
      else process.env.SUPABASE_SERVICE_ROLE_KEY = originalKey;
    },
    install() {
      globalThis.fetch = handler;
    },
  };
}

test("verification reservations are recipient-scoped and atomically reject a concurrent claimant", async () => {
  const calls = [];
  let attempt = 0;
  const fixture = withSupabaseFetch(async (url, options = {}) => {
    calls.push({ url: String(url), options });
    const method = options.method || "GET";
    if (method === "POST" && String(url).includes("verification_send_locks")) {
      attempt += 1;
      return jsonResponse(attempt === 1 ? [{ email: "pilot@example.com" }] : []);
    }
    if (method === "PATCH" && String(url).includes("verification_send_locks")) {
      return jsonResponse([]);
    }
    throw new Error(`unexpected request: ${method} ${url}`);
  });
  fixture.install();
  try {
    const first = await reserveVerificationSend(" Pilot@Example.COM ", new Date("2026-09-25T00:00:00Z"));
    const second = await reserveVerificationSend("pilot@example.com", new Date("2026-09-25T00:00:01Z"));
    assert.equal(first.acquired, true);
    assert.equal(second.acquired, false);
    assert.match(calls[0].options.body, /pilot@example\.com/);
    assert.match(calls[2].url, /reserved_until=lt/);
  } finally {
    fixture.restore();
  }
});

test("failed verification delivery releases the reservation and unsent token", async () => {
  const calls = [];
  const originalApiKey = process.env.RESEND_API_KEY;
  const originalFrom = process.env.OEL_ARCADE_EMAIL_FROM;
  const originalOrigin = process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  process.env.OEL_ARCADE_PUBLIC_ORIGIN = "https://arcade.example";
  process.env.RESEND_API_KEY = "resend-test-key";
  process.env.OEL_ARCADE_EMAIL_FROM = "arcade@example.com";
  const fixture = withSupabaseFetch(async (url, options = {}) => {
    calls.push({ url: String(url), options });
    const method = options.method || "GET";
    if (String(url).startsWith(SUPABASE_URL)) {
      if (method === "POST" && String(url).includes("verification_send_locks")) return jsonResponse([{ id: "lock" }]);
      if (method === "GET" && String(url).includes("email_verifications")) return jsonResponse([]);
      if (method === "POST" && String(url).endsWith("/email_verifications")) return jsonResponse([{ id: "token" }]);
      if (method === "DELETE" && String(url).includes("email_verifications")) return jsonResponse(null);
      if (method === "DELETE" && String(url).includes("verification_send_locks")) return jsonResponse(null);
    }
    if (String(url) === "https://api.resend.com/emails") return jsonResponse({ message: "provider down" }, 503);
    throw new Error(`unexpected request: ${method} ${url}`);
  });
  fixture.install();
  try {
    const result = await maybeSendVerificationEmail({
      req: { headers: { host: "arcade.example" } },
      player: { id: "player-1", email_verified_at: null },
      attemptRow: { id: "attempt-1" },
      email: "pilot@example.com",
      username: "pilot",
      validation: { canonical_score: 12, canonical_metrics: {}, replay: { rounds_cleared: 1 } },
      accepted: true,
      ownership: { verification_allowed: true },
    });
    assert.equal(result.status, "failed");
    assert.equal(calls.filter(({ options }) => options.method === "DELETE").length, 2);
    assert.equal(calls.some(({ url }) => url.includes("reservation_id")), true);
  } finally {
    fixture.restore();
    if (originalApiKey === undefined) delete process.env.RESEND_API_KEY;
    else process.env.RESEND_API_KEY = originalApiKey;
    if (originalFrom === undefined) delete process.env.OEL_ARCADE_EMAIL_FROM;
    else process.env.OEL_ARCADE_EMAIL_FROM = originalFrom;
    if (originalOrigin === undefined) delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    else process.env.OEL_ARCADE_PUBLIC_ORIGIN = originalOrigin;
  }
});

test("ambiguous provider transport keeps the reservation to prevent duplicate mail", async () => {
  const calls = [];
  const originalApiKey = process.env.RESEND_API_KEY;
  const originalFrom = process.env.OEL_ARCADE_EMAIL_FROM;
  const originalOrigin = process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  process.env.OEL_ARCADE_PUBLIC_ORIGIN = "https://arcade.example";
  process.env.RESEND_API_KEY = "resend-test-key";
  process.env.OEL_ARCADE_EMAIL_FROM = "arcade@example.com";
  const fixture = withSupabaseFetch(async (url, options = {}) => {
    calls.push({ url: String(url), options });
    const method = options.method || "GET";
    if (String(url).startsWith(SUPABASE_URL)) {
      if (method === "POST" && String(url).includes("verification_send_locks")) return jsonResponse([{ id: "lock" }]);
      if (method === "GET" && String(url).includes("email_verifications")) return jsonResponse([]);
      if (method === "POST" && String(url).endsWith("/email_verifications")) return jsonResponse([{ id: "token" }]);
    }
    if (String(url) === "https://api.resend.com/emails") throw new Error("request timed out");
    throw new Error(`unexpected request: ${method} ${url}`);
  });
  fixture.install();
  try {
    const result = await maybeSendVerificationEmail({
      req: { headers: { host: "arcade.example" } },
      player: { id: "player-1", email_verified_at: null },
      attemptRow: { id: "attempt-1" },
      email: "pilot@example.com",
      username: "pilot",
      validation: { canonical_score: 12, canonical_metrics: {}, replay: { rounds_cleared: 1 } },
      accepted: true,
      ownership: { verification_allowed: true },
    });
    assert.equal(result.status, "delivery_ambiguous");
    assert.equal(calls.some(({ options }) => options.method === "DELETE"), false);
  } finally {
    fixture.restore();
    if (originalApiKey === undefined) delete process.env.RESEND_API_KEY;
    else process.env.RESEND_API_KEY = originalApiKey;
    if (originalFrom === undefined) delete process.env.OEL_ARCADE_EMAIL_FROM;
    else process.env.OEL_ARCADE_EMAIL_FROM = originalFrom;
    if (originalOrigin === undefined) delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    else process.env.OEL_ARCADE_PUBLIC_ORIGIN = originalOrigin;
  }
});

test("leaderboard retry mirrors the stored private winner after a new attempt id", async () => {
  const calls = [];
  let privatePostCount = 0;
  let publicPostCount = 0;
  const fixture = withSupabaseFetch(async (url, options = {}) => {
    const request = { url: String(url), options };
    calls.push(request);
    const method = options.method || "GET";
    const path = String(url);
    if (method === "POST" && path.includes("leaderboard_entries")) {
      privatePostCount += 1;
      return jsonResponse(privatePostCount === 1 ? [{ attempt_id: "winner-attempt", score: 12 }] : []);
    }
    if (method === "PATCH" && path.includes("leaderboard_entries")) return jsonResponse([]);
    if (method === "GET" && path.includes("leaderboard_entries")) {
      return jsonResponse([{ attempt_id: "winner-attempt", score: 12, metrics: { rounds_cleared: 1 } }]);
    }
    if (method === "GET" && path.includes("attempts")) return jsonResponse([{ submitted_at: "2026-09-25T00:00:00Z" }]);
    if (method === "POST" && path.includes("public_leaderboard")) {
      publicPostCount += 1;
      if (publicPostCount === 1) return jsonResponse({ message: "temporary outage" }, 503);
      return jsonResponse([{ attempt_id: "winner-attempt" }]);
    }
    throw new Error(`unexpected request: ${method} ${url}`);
  });
  fixture.install();
  try {
    await assert.rejects(
      upsertLeaderboardIfBetter({
        challengeId: "challenge",
        playerId: "player",
        attemptId: "first-attempt",
        score: 12,
        metrics: { rounds_cleared: 1 },
        username: "pilot",
        submittedAt: "2026-09-25T00:00:00Z",
        emailVerified: false,
      }),
    );
    const retried = await upsertLeaderboardIfBetter({
      challengeId: "challenge",
      playerId: "player",
      attemptId: "second-attempt",
      score: 12,
      metrics: { rounds_cleared: 1 },
      username: "pilot",
      submittedAt: "2026-09-25T00:00:01Z",
      emailVerified: false,
    });
    assert.equal(retried, false);
    const publicBodies = calls
      .filter(({ options, url }) => options.method === "POST" && url.includes("public_leaderboard"))
      .map(({ options }) => JSON.parse(options.body));
    assert.equal(publicBodies.at(-1)[0].attempt_id, "winner-attempt");
  } finally {
    fixture.restore();
  }
});
