import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { publishUnclaimedAttempt } from "../api/_leaderboard.mjs";
import { hashVerificationToken, verificationUrl } from "../api/_email.mjs";
import { decideOwnership, OWNERSHIP_STATUS } from "../api/_ownership.mjs";
import submitAttempt, { maybeSendVerificationEmail } from "../api/submit-attempt.mjs";
import verifyEmail from "../api/verify-email.mjs";

async function withBackend(responder, action) {
  const oldFetch = globalThis.fetch;
  const oldUrl = process.env.SUPABASE_URL;
  const oldKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  process.env.SUPABASE_URL = "https://supabase.example";
  process.env.SUPABASE_SERVICE_ROLE_KEY = "test-service-key";
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url: String(url), options });
    const answer = await responder(String(url), options);
    return {
      ok: answer.status < 400,
      status: answer.status,
      text: async () => JSON.stringify(answer.body),
      json: async () => answer.body,
    };
  };
  try {
    return await action(calls);
  } finally {
    globalThis.fetch = oldFetch;
    if (oldUrl === undefined) delete process.env.SUPABASE_URL;
    else process.env.SUPABASE_URL = oldUrl;
    if (oldKey === undefined) delete process.env.SUPABASE_SERVICE_ROLE_KEY;
    else process.env.SUPABASE_SERVICE_ROLE_KEY = oldKey;
  }
}

function response() {
  const headers = {};
  return {
    code: null,
    html: null,
    headers,
    setHeader(name, value) { headers[name.toLowerCase()] = value; },
    status(code) { this.code = code; return this; },
    send(html) { this.html = html; return this; },
  };
}

test("knowing a locked username's email does not grant score publication", () => {
  const player = {
    email: "ace@example.edu",
    email_verified_at: "2026-09-25T00:00:00Z",
    username_locked_at: "2026-09-25T00:00:00Z",
  };
  assert.deepEqual(decideOwnership({ player, email: "ACE@example.edu" }), {
    status: OWNERSHIP_STATUS.VERIFIED_OWNER,
    leaderboard_allowed: false,
    verification_allowed: true,
  });
  assert.deepEqual(decideOwnership({ player, email: "other@example.edu" }), {
    status: OWNERSHIP_STATUS.LOCKED,
    leaderboard_allowed: false,
    verification_allowed: false,
  });
});

test("verification links reject missing or unsafe public origin despite hostile Host headers", () => {
  const previous = process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  const hostile = { headers: { host: "attacker.invalid", "x-forwarded-proto": "http" } };
  try {
    delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    assert.throws(() => verificationUrl(hostile, "secret-token"), /fixed HTTPS origin/);
    process.env.OEL_ARCADE_PUBLIC_ORIGIN = "https://example.test/unsafe/path";
    assert.throws(() => verificationUrl(hostile, "secret-token"), /fixed HTTPS origin/);
    process.env.OEL_ARCADE_PUBLIC_ORIGIN = "https://example.test";
    assert.equal(verificationUrl(hostile, "secret-token"),
      "https://example.test/api/verify-email?token=secret-token");
  } finally {
    if (previous === undefined) delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    else process.env.OEL_ARCADE_PUBLIC_ORIGIN = previous;
  }
});

test("missing fixed origin creates no token and sends no email", async () => {
  const oldOrigin = process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  const oldKey = process.env.RESEND_API_KEY;
  const oldFrom = process.env.OEL_ARCADE_EMAIL_FROM;
  const oldFetch = globalThis.fetch;
  const calls = [];
  delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  process.env.RESEND_API_KEY = "test-key";
  process.env.OEL_ARCADE_EMAIL_FROM = "arcade@example.test";
  globalThis.fetch = async (...args) => { calls.push(args); throw new Error("unexpected network call"); };
  try {
    const result = await maybeSendVerificationEmail({
      req: { headers: { host: "attacker.invalid", "x-forwarded-proto": "http" } },
      player: { id: "player-1" },
      attemptRow: { id: "attempt-1" },
      email: "pilot@example.test",
      username: "pilot",
      validation: { canonical_score: 1, canonical_metrics: {} },
      accepted: true,
      ownership: { verification_allowed: true },
    });
    assert.equal(result.status, "not_configured");
    assert.equal(calls.length, 0);
  } finally {
    globalThis.fetch = oldFetch;
    if (oldOrigin === undefined) delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    else process.env.OEL_ARCADE_PUBLIC_ORIGIN = oldOrigin;
    if (oldKey === undefined) delete process.env.RESEND_API_KEY;
    else process.env.RESEND_API_KEY = oldKey;
    if (oldFrom === undefined) delete process.env.OEL_ARCADE_EMAIL_FROM;
    else process.env.OEL_ARCADE_EMAIL_FROM = oldFrom;
  }
});

test("locked owner can request a token for a new exact attempt, with one send per recipient", async () => {
  const oldOrigin = process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  const oldKey = process.env.RESEND_API_KEY;
  const oldFrom = process.env.OEL_ARCADE_EMAIL_FROM;
  process.env.OEL_ARCADE_PUBLIC_ORIGIN = "https://arcade.example";
  process.env.RESEND_API_KEY = "test-key";
  process.env.OEL_ARCADE_EMAIL_FROM = "arcade@example.test";
  let reservationPosts = 0;
  try {
    await withBackend(async (url, options) => {
      const method = options.method || "GET";
      if (url.includes("verification_send_locks") && method === "POST") {
        reservationPosts += 1;
        return { status: 200, body: reservationPosts === 1 ? [{ email: "pilot@example.test" }] : [] };
      }
      if (url.includes("verification_send_locks") && method === "PATCH") return { status: 200, body: [] };
      if (url.includes("email_verifications") && method === "GET") return { status: 200, body: [] };
      if (url.endsWith("/email_verifications") && method === "POST") return { status: 200, body: [{ id: "token-row" }] };
      if (url === "https://api.resend.com/emails") return { status: 200, body: { id: "mail-row" } };
      throw new Error(`unexpected request: ${options.method} ${url}`);
    }, async (calls) => {
      const owner = { id: "player-1", email: "pilot@example.test", email_verified_at: "2026-09-25T00:00:00Z" };
      const ownership = decideOwnership({ player: owner, email: "PILOT@example.test" });
      const payload = {
        req: { headers: { host: "attacker.invalid" } }, player: owner,
        attemptRow: { id: "new-attempt" }, email: "pilot@example.test",
        username: "pilot", accepted: true, ownership,
        validation: { canonical_score: 25, canonical_metrics: { rounds_cleared: 2 } },
      };
      assert.equal(ownership.leaderboard_allowed, false);
      assert.equal((await maybeSendVerificationEmail(payload)).status, "sent");
      assert.equal((await maybeSendVerificationEmail(payload)).status, "already_pending");
      const providerCalls = calls.filter(({ url }) => url === "https://api.resend.com/emails");
      assert.equal(providerCalls.length, 1);
      assert.match(providerCalls[0].options.body, /https:\/\/arcade\.example\/api\/verify-email/);
      const tokens = calls.filter(({ url, options }) => url.endsWith("/email_verifications") && options.method === "POST");
      assert.equal(tokens.length, 1);
      assert.equal(JSON.parse(tokens[0].options.body)[0].attempt_id, "new-attempt");
    });
  } finally {
    if (oldOrigin === undefined) delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    else process.env.OEL_ARCADE_PUBLIC_ORIGIN = oldOrigin;
    if (oldKey === undefined) delete process.env.RESEND_API_KEY;
    else process.env.RESEND_API_KEY = oldKey;
    if (oldFrom === undefined) delete process.env.OEL_ARCADE_EMAIL_FROM;
    else process.env.OEL_ARCADE_EMAIL_FROM = oldFrom;
  }
});

test("consuming an owner's token frees only its matching recipient reservation", async () => {
  const oldOrigin = process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  const oldKey = process.env.RESEND_API_KEY;
  const oldFrom = process.env.OEL_ARCADE_EMAIL_FROM;
  process.env.OEL_ARCADE_PUBLIC_ORIGIN = "https://arcade.example";
  process.env.RESEND_API_KEY = "test-key";
  process.env.OEL_ARCADE_EMAIL_FROM = "arcade@example.test";
  let lock = null;
  let latestLink = null;
  const tokens = [];
  try {
    await withBackend(async (url, options) => {
      const method = options.method || "GET";
      if (url.includes("verification_send_locks") && method === "POST") {
        if (lock) return { status: 200, body: [] };
        lock = JSON.parse(options.body)[0];
        return { status: 200, body: [lock] };
      }
      if (url.includes("verification_send_locks") && method === "PATCH") return { status: 200, body: [] };
      if (url.includes("email_verifications") && method === "GET") {
        return { status: 200, body: tokens.filter((row) => !row.verified_at).map((row) => ({ id: row.id })) };
      }
      if (url.endsWith("/email_verifications") && method === "POST") {
        const token = { ...JSON.parse(options.body)[0], id: `token-${tokens.length + 1}` };
        tokens.push(token);
        return { status: 200, body: [token] };
      }
      if (url.endsWith("/rpc/claim_arcade_attempt") && method === "POST") {
        const { p_token_hash: digest } = JSON.parse(options.body);
        const token = tokens.find((row) => row.token_hash === digest);
        if (!token) return { status: 200, body: { status: "not_found" } };
        if (token.verified_at) return { status: 200, body: { status: "already_verified" } };
        token.verified_at = new Date().toISOString();
        if (lock?.reservation_id === token.reservation_id) lock = null;
        return { status: 200, body: { status: "verified", promoted: true } };
      }
      if (url === "https://api.resend.com/emails") {
        latestLink = JSON.parse(options.body).text.match(/https:\/\/arcade\.example\/api\/verify-email\?token=\S+/)?.[0];
        return { status: 200, body: { id: `mail-${tokens.length}` } };
      }
      throw new Error(`unexpected request: ${method} ${url}`);
    }, async (calls) => {
      const player = { id: "player-1", email: "pilot@example.test", email_verified_at: "2026-09-25T00:00:00Z" };
      const payload = {
        req: {}, player, email: "pilot@example.test", username: "pilot",
        validation: { canonical_score: 25 }, accepted: true,
        ownership: decideOwnership({ player, email: "pilot@example.test" }),
      };
      assert.equal((await maybeSendVerificationEmail({ ...payload, attemptRow: { id: "attempt-1" } })).status, "sent");
      assert.equal(tokens[0].reservation_id, lock.reservation_id);
      assert.equal((await maybeSendVerificationEmail({ ...payload, attemptRow: { id: "attempt-2" } })).status, "already_pending");
      const firstReservation = lock.reservation_id;
      const firstLink = latestLink;
      const verified = response();
      await verifyEmail({ method: "POST", body: { token: new URL(latestLink).searchParams.get("token") } }, verified);
      assert.equal(verified.code, 200);
      assert.equal(lock, null);
      assert.equal((await maybeSendVerificationEmail({ ...payload, attemptRow: { id: "attempt-2" } })).status, "sent");
      assert.notEqual(tokens[1].reservation_id, firstReservation);
      assert.equal(tokens[1].attempt_id, "attempt-2");
      const replay = response();
      await verifyEmail({ method: "POST", body: { token: new URL(firstLink).searchParams.get("token") } }, replay);
      assert.equal(replay.code, 200);
      assert.equal(lock.reservation_id, tokens[1].reservation_id);
      assert.equal(calls.filter(({ url }) => url === "https://api.resend.com/emails").length, 2);
    });
  } finally {
    if (oldOrigin === undefined) delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    else process.env.OEL_ARCADE_PUBLIC_ORIGIN = oldOrigin;
    if (oldKey === undefined) delete process.env.RESEND_API_KEY;
    else process.env.RESEND_API_KEY = oldKey;
    if (oldFrom === undefined) delete process.env.OEL_ARCADE_EMAIL_FROM;
    else process.env.OEL_ARCADE_EMAIL_FROM = oldFrom;
  }
});

test("different email cannot trigger a locked owner's token", async () => {
  const owner = { id: "player-1", email: "pilot@example.test", email_verified_at: "2026-09-25T00:00:00Z" };
  const ownership = decideOwnership({ player: owner, email: "other@example.test" });
  assert.equal(ownership.verification_allowed, false);
  const result = await maybeSendVerificationEmail({
    req: {}, player: owner, attemptRow: { id: "attempt-1" },
    email: "other@example.test", username: "pilot", accepted: true,
    ownership, validation: { canonical_score: 2 },
  });
  assert.equal(result.status, "skipped");
});

test("disallowed browser Origin is rejected before validation, storage, or mail", async () => {
  const oldOrigin = process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  const oldFetch = globalThis.fetch;
  process.env.OEL_ARCADE_PUBLIC_ORIGIN = "https://arcade.example";
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; throw new Error("unexpected network call"); };
  const res = {
    code: null, body: null, setHeader() {},
    status(code) { this.code = code; return this; },
    json(body) { this.body = body; return this; },
  };
  try {
    await submitAttempt({ method: "POST", headers: { origin: "https://attacker.invalid" }, body: {} }, res);
    assert.equal(res.code, 403);
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = oldFetch;
    if (oldOrigin === undefined) delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    else process.env.OEL_ARCADE_PUBLIC_ORIGIN = oldOrigin;
  }
});

test("foreign username can delay recipient once, then owner can retry after cooldown", async () => {
  const oldOrigin = process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  const oldKey = process.env.RESEND_API_KEY;
  const oldFrom = process.env.OEL_ARCADE_EMAIL_FROM;
  process.env.OEL_ARCADE_PUBLIC_ORIGIN = "https://arcade.example";
  process.env.RESEND_API_KEY = "test-key";
  process.env.OEL_ARCADE_EMAIL_FROM = "arcade@example.test";
  let lock = null;
  let sends = 0;
  let tokens = 0;
  let expired = false;
  try {
    await withBackend(async (url, options) => {
      const method = options.method || "GET";
      if (url.includes("verification_send_locks") && method === "POST") {
        if (lock) return { status: 200, body: [] };
        lock = JSON.parse(options.body)[0];
        return { status: 200, body: [lock] };
      }
      if (url.includes("verification_send_locks") && method === "PATCH") {
        if (!expired) return { status: 200, body: [] };
        const replacement = JSON.parse(options.body);
        lock = { ...lock, ...replacement };
        return { status: 200, body: [lock] };
      }
      if (url.endsWith("/email_verifications") && method === "POST") {
        tokens += 1;
        return { status: 200, body: [{ id: `token-${tokens}` }] };
      }
      if (url === "https://api.resend.com/emails") {
        sends += 1;
        return { status: 200, body: { id: `mail-${sends}` } };
      }
      throw new Error(`unexpected request: ${method} ${url}`);
    }, async () => {
      const common = {
        req: {}, email: "victim@example.test", accepted: true,
        validation: { canonical_score: 2 }, ownership: { verification_allowed: true },
      };
      assert.equal((await maybeSendVerificationEmail({
        ...common, player: { id: "foreign" }, username: "foreign",
        attemptRow: { id: "foreign-attempt" },
      })).status, "sent");
      assert.equal((await maybeSendVerificationEmail({
        ...common, player: { id: "owner" }, username: "owner",
        attemptRow: { id: "owner-attempt" },
      })).status, "already_pending");
      const firstReservation = lock.reservation_id;
      expired = true;
      assert.equal((await maybeSendVerificationEmail({
        ...common, player: { id: "owner" }, username: "owner",
        attemptRow: { id: "owner-attempt" },
      })).status, "sent");
      assert.notEqual(lock.reservation_id, firstReservation);
      assert.equal(sends, 2);
      assert.equal(tokens, 2);
    });
  } finally {
    if (oldOrigin === undefined) delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    else process.env.OEL_ARCADE_PUBLIC_ORIGIN = oldOrigin;
    if (oldKey === undefined) delete process.env.RESEND_API_KEY;
    else process.env.RESEND_API_KEY = oldKey;
    if (oldFrom === undefined) delete process.env.OEL_ARCADE_EMAIL_FROM;
    else process.env.OEL_ARCADE_EMAIL_FROM = oldFrom;
  }
});

test("ambiguous mail transport retains token and recipient reservation", async () => {
  const oldOrigin = process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  const oldKey = process.env.RESEND_API_KEY;
  const oldFrom = process.env.OEL_ARCADE_EMAIL_FROM;
  process.env.OEL_ARCADE_PUBLIC_ORIGIN = "https://arcade.example";
  process.env.RESEND_API_KEY = "test-key";
  process.env.OEL_ARCADE_EMAIL_FROM = "arcade@example.test";
  try {
    await withBackend(async (url, options) => {
      const method = options.method || "GET";
      if (url.includes("verification_send_locks") && method === "POST") return { status: 200, body: [{ email: "pilot@example.test" }] };
      if (url.includes("email_verifications") && method === "GET") return { status: 200, body: [] };
      if (url.endsWith("/email_verifications") && method === "POST") return { status: 200, body: [{ id: "token-row" }] };
      if (url === "https://api.resend.com/emails") throw new Error("timeout after send");
      throw new Error(`unexpected request: ${options.method} ${url}`);
    }, async (calls) => {
      const result = await maybeSendVerificationEmail({
        req: {}, player: { id: "player-1" }, attemptRow: { id: "attempt-1" },
        email: "pilot@example.test", username: "pilot", accepted: true,
        ownership: { verification_allowed: true }, validation: { canonical_score: 2 },
      });
      assert.equal(result.status, "delivery_ambiguous");
      assert.equal(calls.some(({ options }) => options.method === "DELETE"), false);
    });
  } finally {
    if (oldOrigin === undefined) delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    else process.env.OEL_ARCADE_PUBLIC_ORIGIN = oldOrigin;
    if (oldKey === undefined) delete process.env.RESEND_API_KEY;
    else process.env.RESEND_API_KEY = oldKey;
    if (oldFrom === undefined) delete process.env.OEL_ARCADE_EMAIL_FROM;
    else process.env.OEL_ARCADE_EMAIL_FROM = oldFrom;
  }
});

test("definitive provider rejection removes unsent token and reservation", async () => {
  const oldOrigin = process.env.OEL_ARCADE_PUBLIC_ORIGIN;
  const oldKey = process.env.RESEND_API_KEY;
  const oldFrom = process.env.OEL_ARCADE_EMAIL_FROM;
  process.env.OEL_ARCADE_PUBLIC_ORIGIN = "https://arcade.example";
  process.env.RESEND_API_KEY = "test-key";
  process.env.OEL_ARCADE_EMAIL_FROM = "arcade@example.test";
  try {
    await withBackend(async (url, options) => {
      const method = options.method || "GET";
      if (url.includes("verification_send_locks") && method === "POST") return { status: 200, body: [{ email: "pilot@example.test" }] };
      if (url.endsWith("/email_verifications") && method === "POST") return { status: 200, body: [{ id: "token-row" }] };
      if (url.includes("email_verifications") && method === "DELETE") return { status: 200, body: null };
      if (url.includes("verification_send_locks") && method === "DELETE") return { status: 200, body: null };
      if (url === "https://api.resend.com/emails") return { status: 503, body: { message: "provider rejected" } };
      throw new Error(`unexpected request: ${method} ${url}`);
    }, async (calls) => {
      const result = await maybeSendVerificationEmail({
        req: {}, player: { id: "player-1" }, attemptRow: { id: "attempt-1" },
        email: "pilot@example.test", username: "pilot", accepted: true,
        ownership: { verification_allowed: true }, validation: { canonical_score: 2 },
      });
      assert.equal(result.status, "failed");
      assert.equal(calls.filter(({ options }) => options.method === "DELETE").length, 2);
    });
  } finally {
    if (oldOrigin === undefined) delete process.env.OEL_ARCADE_PUBLIC_ORIGIN;
    else process.env.OEL_ARCADE_PUBLIC_ORIGIN = oldOrigin;
    if (oldKey === undefined) delete process.env.RESEND_API_KEY;
    else process.env.RESEND_API_KEY = oldKey;
    if (oldFrom === undefined) delete process.env.OEL_ARCADE_EMAIL_FROM;
    else process.env.OEL_ARCADE_EMAIL_FROM = oldFrom;
  }
});

test("migration rebuilds verified winners only from consumed current-email tokens", () => {
  const schema = readFileSync(new URL("../supabase/schema.sql", import.meta.url), "utf8");
  const reset = schema.indexOf("delete from leaderboard_entries le using players p");
  const verified = schema.indexOf("and ev.verified_at is not null", reset);
  const boundAttempt = schema.indexOf("join attempts a on a.id = ev.attempt_id and a.player_id = p.id", reset);
  const currentEmail = schema.indexOf("lower(trim(p.email)) = lower(trim(ev.email))", reset);
  const publicReset = schema.indexOf("delete from public_leaderboard;", reset);
  const backfill = schema.indexOf("insert into public_leaderboard (", publicReset);
  assert.ok(reset >= 0 && boundAttempt > reset && currentEmail > reset && verified > reset);
  assert.ok(publicReset > verified && backfill > publicReset);
  assert.match(schema, /update email_verifications set email = lower\(trim\(email\)\)/);
  assert.match(schema, /created_at > now\(\) - interval '1 hour'/);
  assert.match(schema, /v_promoted := found and v_attempt\.status in \('valid', 'suspicious'\)/);
  assert.ok(schema.indexOf("update players set", schema.indexOf("v_promoted := found"))
    < schema.indexOf("if v_promoted then", schema.indexOf("v_promoted := found")));
  const firstClaim = schema.indexOf("if v_first_claim then", schema.indexOf("v_promoted := found"));
  const clearPublic = schema.indexOf("delete from public_leaderboard where username = v_player.username", firstClaim);
  const clearPrivate = schema.indexOf("delete from leaderboard_entries where player_id = v_player.id", firstClaim);
  const tokenPromotion = schema.indexOf("if v_promoted then", firstClaim);
  assert.ok(firstClaim >= 0 && clearPublic > firstClaim && clearPrivate > clearPublic
    && tokenPromotion > clearPrivate);
});

test("unclaimed publication uses a single ownership-checked database transaction", async () => {
  await withBackend(async () => ({ status: 200, body: true }), async (calls) => {
    assert.equal(await publishUnclaimedAttempt("attempt-1"), true);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "https://supabase.example/rest/v1/rpc/publish_unclaimed_arcade_attempt");
    assert.deepEqual(JSON.parse(calls[0].options.body), { p_attempt_id: "attempt-1" });
  });
  await withBackend(async () => ({ status: 200, body: false }), async () => {
    assert.equal(await publishUnclaimedAttempt("attempt-1"), false);
  });
});

test("verification GET renders an escaped confirmation form without claiming", async () => {
  await withBackend(async () => { throw new Error("GET must not call the backend"); }, async (calls) => {
    const res = response();
    const token = `\"><script>alert(1)</script>`;
    await verifyEmail({ method: "GET", url: `/api/verify-email?token=${encodeURIComponent(token)}` }, res);
    assert.equal(res.code, 200);
    assert.equal(calls.length, 0);
    assert.match(res.html, /Opening this page has not verified/);
    assert.match(res.html, /<form method="post" action="\/api\/verify-email">/);
    assert.match(res.html, /type="hidden" name="token" value="&quot;&gt;&lt;script&gt;alert\(1\)&lt;\/script&gt;"/);
    assert.match(res.html, /<button type="submit">Confirm and reserve username<\/button>/);
    assert.equal(res.headers["cache-control"], "no-store");
    assert.equal(res.headers["referrer-policy"], "no-referrer");
  });
});

test("verification POST sends only a token digest to the atomic claim endpoint", async () => {
  const token = "A".repeat(43);
  await withBackend(async () => ({ status: 200, body: { status: "verified", promoted: true } }), async (calls) => {
    const requests = [
      { method: "POST", body: { token } },
      {
        method: "POST",
        headers: { "content-type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ token }).toString(),
      },
    ];
    for (const req of requests) {
      const res = response();
      await verifyEmail(req, res);
      assert.equal(res.code, 200);
      assert.match(res.html, /linked score is on the leaderboard/);
    }
    assert.equal(calls.length, requests.length);
    for (const call of calls) {
      assert.equal(call.url, "https://supabase.example/rest/v1/rpc/claim_arcade_attempt");
      assert.equal(call.options.method, "POST");
      const sent = JSON.parse(call.options.body);
      assert.equal(sent.p_token_hash, hashVerificationToken(token));
      assert.equal(JSON.stringify(sent).includes(token), false);
      assert.equal(JSON.stringify(sent).includes("attempt_id"), false);
    }
  });
});

test("invalid verification POST is rejected before RPC; replay and missing migration preserve statuses", async () => {
  await withBackend(async () => { throw new Error("invalid tokens must not call the backend"); }, async (calls) => {
    const res = response();
    await verifyEmail({ method: "POST", body: { token: "short-token" } }, res);
    assert.equal(res.code, 400);
    assert.equal(calls.length, 0);
  });
  await withBackend(async () => ({ status: 200, body: { status: "already_verified" } }), async (calls) => {
    const res = response();
    await verifyEmail({ method: "POST", body: { token: "B".repeat(43) } }, res);
    assert.equal(res.code, 200);
    assert.equal(calls.length, 1);
    assert.match(res.html, /already used/);
  });
  await withBackend(async () => ({ status: 404, body: { message: "function missing" } }), async (calls) => {
    const res = response();
    await verifyEmail({ method: "POST", body: { token: "C".repeat(43) } }, res);
    assert.equal(res.code, 500);
    assert.equal(calls.length, 1);
  });
});
