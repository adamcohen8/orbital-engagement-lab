import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";

import worker, { RpoDuelAdmission } from "../worker/index.js";

function request(body) {
  return new Request("https://duel.example/api/rooms", {
    method: "POST",
    headers: {
      "CF-Connecting-IP": "192.0.2.10",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

test("invalid room configuration is rejected before global admission", async () => {
  let admissionCalls = 0;
  const env = {
    DUEL_CREATE_ENABLED: "true",
    DUEL_ADMISSION: {
      getByName() {
        return {
          fetch: async () => {
            admissionCalls += 1;
            return new Response(null, { status: 200 });
          },
        };
      },
    },
    DUEL_ROOMS: { getByName() { throw new Error("room Durable Object must not be reached"); } },
  };

  const response = await worker.fetch(request({ name: "pilot", regulation_rounds: 3, opponent: "human" }), env);

  assert.equal(response.status, 400);
  assert.equal(admissionCalls, 0);
});

test("room creation releases an admitted slot for any non-created response", async () => {
  const admissionPaths = [];
  const env = {
    DUEL_CREATE_ENABLED: "true",
    DUEL_ADMISSION: {
      getByName() {
        return {
          fetch: async (url) => {
            admissionPaths.push(new URL(url).pathname);
            return new Response(JSON.stringify({ status: "admitted" }), { status: 200 });
          },
        };
      },
    },
    DUEL_ROOMS: {
      getByName() {
        return { fetch: async () => new Response(JSON.stringify({ error: "rejected" }), { status: 400 }) };
      },
    },
  };

  const response = await worker.fetch(request({ name: "pilot", regulation_rounds: 4, opponent: "human" }), env);

  assert.equal(response.status, 400);
  assert.deepEqual(admissionPaths, ["/admit", "/release"]);
});

test("room creation releases an admitted slot when the room request throws", async () => {
  const admissionPaths = [];
  const env = {
    DUEL_CREATE_ENABLED: "true",
    DUEL_ADMISSION: {
      getByName() {
        return {
          fetch: async (url) => {
            admissionPaths.push(new URL(url).pathname);
            return new Response(JSON.stringify({ status: "admitted" }), { status: 200 });
          },
        };
      },
    },
    DUEL_ROOMS: {
      getByName() {
        return { fetch: async () => { throw new Error("room unavailable"); } };
      },
    },
  };

  const response = await worker.fetch(request({ name: "pilot", regulation_rounds: 4, opponent: "human" }), env);

  assert.equal(response.status, 500);
  assert.deepEqual(admissionPaths, ["/admit", "/release"]);
});

test("room creation releases an admitted slot when room lookup throws", async () => {
  const admissionPaths = [];
  const env = {
    DUEL_CREATE_ENABLED: "true",
    DUEL_ADMISSION: {
      getByName() {
        return {
          fetch: async (url) => {
            admissionPaths.push(new URL(url).pathname);
            return new Response(JSON.stringify({ status: "admitted" }), { status: 200 });
          },
        };
      },
    },
    DUEL_ROOMS: {
      getByName() {
        throw new Error("room namespace unavailable");
      },
    },
  };

  const response = await worker.fetch(request({ name: "pilot", regulation_rounds: 4, opponent: "human" }), env);

  assert.equal(response.status, 500);
  assert.deepEqual(admissionPaths, ["/admit", "/release"]);
});

function rendezvousSnapshot(timeoutMs = 250) {
  let waiting = null;
  return (snapshot) => new Promise((resolve) => {
    if (waiting) {
      const first = waiting;
      waiting = null;
      clearTimeout(first.timer);
      first.resolve(first.snapshot);
      resolve(snapshot);
      return;
    }
    const entry = { snapshot, resolve, timer: null };
    entry.timer = setTimeout(() => {
      if (waiting === entry) waiting = null;
      resolve(snapshot);
    }, timeoutMs);
    waiting = entry;
  });
}

function admissionStorageMock({ operationDelayMs = 0 } = {}) {
  const rows = new Map();
  const listSnapshots = rendezvousSnapshot();
  const getSnapshots = rendezvousSnapshot();
  let activeOperations = 0;
  let maxConcurrentOperations = 0;

  async function tracked(operation) {
    activeOperations += 1;
    maxConcurrentOperations = Math.max(maxConcurrentOperations, activeOperations);
    try {
      if (operationDelayMs) await new Promise((resolve) => setTimeout(resolve, operationDelayMs));
      return await operation();
    } finally {
      activeOperations -= 1;
    }
  }

  return {
    rows,
    get maxConcurrentOperations() { return maxConcurrentOperations; },
    storage: {
      async list({ prefix }) {
        // Capture before waiting: concurrent callers both observe the same
        // stale room snapshot unless the Durable Object serializes requests.
        const snapshot = new Map([...rows]
          .filter(([key]) => key.startsWith(prefix))
          .map(([key, value]) => [key, structuredClone(value)]));
        return tracked(() => listSnapshots(snapshot));
      },
      async get(key) {
        const snapshot = rows.has(key) ? structuredClone(rows.get(key)) : null;
        return tracked(() => getSnapshots(snapshot));
      },
      async put(key, value) {
        return tracked(async () => rows.set(key, structuredClone(value)));
      },
      async delete(key) {
        return tracked(async () => rows.delete(key));
      },
    },
  };
}

function admissionRequest(pathname, body) {
  return new Request(`https://oel.internal${pathname}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function clientKey(value) {
  return createHash("sha256").update(value).digest("hex");
}

function admissionInstance(store, overrides = {}) {
  return new RpoDuelAdmission({ storage: store.storage }, {
    DUEL_MAX_ACTIVE_ROOMS: "25",
    DUEL_CREATE_RATE_LIMIT: "5",
    DUEL_CREATE_RATE_WINDOW_MS: "600000",
    ...overrides,
  });
}

test("concurrent admissions cannot exceed the active-room cap after stale room reads", async () => {
  const store = admissionStorageMock();
  const admission = admissionInstance(store, { DUEL_MAX_ACTIVE_ROOMS: "1" });
  const expiresAt = Date.now() + 60_000;
  const responses = await Promise.all(["ABC234", "DEF567"].map((room_code, index) =>
    admission.fetch(admissionRequest("/admit", {
      room_code,
      client_key: clientKey(`client-${index}`),
      expires_at_ms: expiresAt,
    })),
  ));

  assert.deepEqual(responses.map((response) => response.status).sort(), [200, 503]);
  assert.equal([...store.rows.keys()].filter((key) => key.startsWith("room:")).length, 1);
  assert.equal(store.maxConcurrentOperations, 1);
});

test("concurrent admissions cannot lose same-client rate increments after stale reads", async () => {
  const store = admissionStorageMock();
  const admission = admissionInstance(store, { DUEL_MAX_ACTIVE_ROOMS: "5", DUEL_CREATE_RATE_LIMIT: "1" });
  const expiresAt = Date.now() + 60_000;
  const sameClientKey = clientKey("same-client");
  const responses = await Promise.all(["ABC234", "DEF567"].map((room_code) =>
    admission.fetch(admissionRequest("/admit", {
      room_code,
      client_key: sameClientKey,
      expires_at_ms: expiresAt,
    })),
  ));

  assert.deepEqual(responses.map((response) => response.status).sort(), [200, 429]);
  assert.equal(store.rows.get(`rate:${sameClientKey}`).count, 1);
  assert.equal([...store.rows.keys()].filter((key) => key.startsWith("room:")).length, 1);
  assert.equal(store.maxConcurrentOperations, 1);
});

test("admit, renew, and release mutations share one critical section", async () => {
  const store = admissionStorageMock({ operationDelayMs: 10 });
  const admission = admissionInstance(store);
  const expiresAt = Date.now() + 60_000;
  const responses = await Promise.all([
    admission.fetch(admissionRequest("/renew", { room_code: "ABC234", expires_at_ms: expiresAt })),
    admission.fetch(admissionRequest("/release", { room_code: "ABC234" })),
    admission.fetch(admissionRequest("/admit", { room_code: "DEF567", client_key: clientKey("client-a"), expires_at_ms: expiresAt })),
  ]);

  assert.deepEqual(responses.map((response) => response.status), [200, 200, 200]);
  assert.equal(store.maxConcurrentOperations, 1);
});

test("malformed admission limits fail closed before changing stored admission state", async () => {
  const invalid = [
    ["DUEL_MAX_ACTIVE_ROOMS", "NaN"],
    ["DUEL_CREATE_RATE_LIMIT", "Infinity"],
    ["DUEL_CREATE_RATE_WINDOW_MS", "1e100"],
    ["DUEL_MAX_ACTIVE_ROOMS", "1001"],
    ["DUEL_CREATE_RATE_LIMIT", "0"],
    ["DUEL_CREATE_RATE_WINDOW_MS", "999"],
  ];
  for (const [name, value] of invalid) {
    const store = admissionStorageMock();
    const admission = admissionInstance(store, { [name]: value });
    const response = await admission.fetch(admissionRequest("/admit", {
      room_code: "ABC234",
      client_key: clientKey("config-check"),
      expires_at_ms: Date.now() + 60_000,
    }));
    assert.equal(response.status, 503, `${name}=${value} must reject admission`);
    assert.equal(store.rows.size, 0, `${name}=${value} must not persist room or rate state`);
  }
});

test("admission rejects unhashed or oversized client identities before storage access", async () => {
  for (const invalidClientKey of ["client-label", "a".repeat(65), "F".repeat(64)]) {
    const store = admissionStorageMock();
    const admission = admissionInstance(store);
    const response = await admission.fetch(admissionRequest("/admit", {
      room_code: "ABC234",
      client_key: invalidClientKey,
      expires_at_ms: Date.now() + 60_000,
    }));
    assert.equal(response.status, 403);
    assert.equal(store.rows.size, 0);
  }
});

test("missing admission limit settings use bounded defaults", async () => {
  const store = admissionStorageMock();
  const admission = new RpoDuelAdmission({ storage: store.storage }, {});
  const response = await admission.fetch(admissionRequest("/admit", {
    room_code: "ABC234",
    client_key: clientKey("defaults"),
    expires_at_ms: Date.now() + 60_000,
  }));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { status: "admitted", active_rooms: 1, maximum: 25 });
});
