const JSON_HEADERS = {
  "Content-Type": "application/json",
};

export function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": process.env.OEL_ARCADE_ALLOWED_ORIGIN || "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

export function sendJson(res, statusCode, payload) {
  Object.entries({ ...corsHeaders(), ...JSON_HEADERS }).forEach(([key, value]) => res.setHeader(key, value));
  res.status(statusCode).json(payload);
}

export function requireSupabaseEnv() {
  const url = String(process.env.SUPABASE_URL || "").replace(/\/+$/, "");
  const serviceKey = String(process.env.SUPABASE_SERVICE_ROLE_KEY || "");
  if (!url || !serviceKey) {
    throw new Error("Supabase environment is not configured.");
  }
  return { url, serviceKey };
}

export async function supabaseRest(path, options = {}) {
  const { url, serviceKey } = requireSupabaseEnv();
  const response = await fetch(`${url}/rest/v1/${path}`, {
    ...options,
    headers: {
      apikey: serviceKey,
      Authorization: `Bearer ${serviceKey}`,
      ...JSON_HEADERS,
      ...(options.headers || {}),
    },
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(payload?.message || payload?.hint || `Supabase request failed with HTTP ${response.status}.`);
  }
  return payload;
}

export async function readBody(req) {
  if (req.body && typeof req.body === "object") return req.body;
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const text = Buffer.concat(chunks).toString("utf8");
  return text ? JSON.parse(text) : {};
}

export function normalizeUsername(username) {
  const cleaned = String(username || "anonymous").trim().slice(0, 24);
  return cleaned || "anonymous";
}

export function normalizeEmail(email) {
  const cleaned = String(email || "").trim().slice(0, 254);
  return cleaned && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleaned) ? cleaned : "";
}

export function normalizedUsernameKey(username) {
  return normalizeUsername(username).toLowerCase().replace(/\s+/g, "");
}
