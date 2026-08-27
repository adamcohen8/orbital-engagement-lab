export const DEFAULT_SANDBOX_SETUP = Object.freeze({
  target_a_km: 7000.0,
  target_ecc: 0.0,
  target_inc_deg: 45.0,
  target_raan_deg: 0.0,
  target_argp_deg: 0.0,
  target_true_anomaly_deg: 0.0,
  radial_km: 0.0,
  in_track_km: -3.0,
  cross_track_km: 0.0,
  radial_rate_m_s: 0.0,
  in_track_rate_m_s: 0.0,
  cross_track_rate_m_s: 0.0,
});

export const SANDBOX_TARGET_FIELDS = Object.freeze([
  Object.freeze({ key: "target_a_km", label: "Semimajor Axis", unit: "km" }),
  Object.freeze({ key: "target_ecc", label: "Eccentricity", unit: "" }),
  Object.freeze({ key: "target_inc_deg", label: "Inclination", unit: "deg" }),
  Object.freeze({ key: "target_raan_deg", label: "RAAN", unit: "deg" }),
  Object.freeze({ key: "target_argp_deg", label: "Argument of Periapsis", unit: "deg" }),
  Object.freeze({ key: "target_true_anomaly_deg", label: "True Anomaly", unit: "deg" }),
]);

export const SANDBOX_CHASER_FIELDS = Object.freeze([
  Object.freeze({ key: "radial_km", label: "Radial R", unit: "km" }),
  Object.freeze({ key: "in_track_km", label: "In-Track I", unit: "km" }),
  Object.freeze({ key: "cross_track_km", label: "Cross-Track C", unit: "km" }),
  Object.freeze({ key: "radial_rate_m_s", label: "Radial Rate dR", unit: "m/s" }),
  Object.freeze({ key: "in_track_rate_m_s", label: "In-Track Rate dI", unit: "m/s" }),
  Object.freeze({ key: "cross_track_rate_m_s", label: "Cross-Track Rate dC", unit: "m/s" }),
]);

export const SANDBOX_SETUP_FIELDS = Object.freeze([...SANDBOX_TARGET_FIELDS, ...SANDBOX_CHASER_FIELDS]);

export function validateSandboxSetup(rawValues = {}) {
  const parsed = {};
  for (const field of SANDBOX_SETUP_FIELDS) {
    const raw = rawValues[field.key];
    const text = typeof raw === "number" ? String(raw) : String(raw ?? "").trim();
    const value = text ? Number(text) : Number.NaN;
    if (!Number.isFinite(value)) {
      const suffix = field.unit ? ` (${field.unit})` : "";
      return { value: null, error: `Enter a numeric value for ${field.label}${suffix}.` };
    }
    parsed[field.key] = value;
  }
  if (parsed.target_a_km <= 0) {
    return { value: null, error: "Target Semimajor Axis must be positive." };
  }
  if (parsed.target_ecc < 0 || parsed.target_ecc >= 1) {
    return { value: null, error: "Target Eccentricity must satisfy 0 <= e < 1." };
  }
  if (parsed.target_inc_deg < 0 || parsed.target_inc_deg > 180) {
    return { value: null, error: "Target Inclination must satisfy 0 <= i <= 180 degrees." };
  }
  return { value: parsed, error: "" };
}

export function sandboxSetupInputValues(setup = DEFAULT_SANDBOX_SETUP) {
  return Object.fromEntries(
    SANDBOX_SETUP_FIELDS.map((field) => [field.key, Number(setup[field.key]).toPrecision(7).replace(/\.?0+$/, "")]),
  );
}

export function sandboxRelativeSeed(setup = DEFAULT_SANDBOX_SETUP) {
  return {
    r: Number(setup.radial_km),
    i: Number(setup.in_track_km),
    c: Number(setup.cross_track_km),
    rd: Number(setup.radial_rate_m_s) / 1000,
    id: Number(setup.in_track_rate_m_s) / 1000,
    cd: Number(setup.cross_track_rate_m_s) / 1000,
  };
}

export function sandboxTargetCoes(setup = DEFAULT_SANDBOX_SETUP) {
  return {
    a_km: Number(setup.target_a_km),
    ecc: Number(setup.target_ecc),
    inc_deg: Number(setup.target_inc_deg),
    raan_deg: Number(setup.target_raan_deg),
    argp_deg: Number(setup.target_argp_deg),
    true_anomaly_deg: Number(setup.target_true_anomaly_deg),
  };
}

export function sandboxOrbitPeriodS(setup = DEFAULT_SANDBOX_SETUP, muKm3S2 = 398600.4418) {
  return 2 * Math.PI * Math.sqrt(Number(setup.target_a_km) ** 3 / Number(muKm3S2));
}

export function sandboxProjectionModel(setup = DEFAULT_SANDBOX_SETUP) {
  return Number(setup.target_ecc) === 0 ? "two_body" : "tschauner_hempel";
}
