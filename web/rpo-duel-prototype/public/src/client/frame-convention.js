export const FRAME_CONVENTIONS = Object.freeze({
  OEL_DEFAULT: "oel_default",
  SPACE_FORCE: "space_force",
});

export function normalizeFrameConvention(value) {
  const key = String(value || "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
  return ["space_force", "spaceforce", "sf"].includes(key)
    ? FRAME_CONVENTIONS.SPACE_FORCE
    : FRAME_CONVENTIONS.OEL_DEFAULT;
}

export function frameConventionFromSearch(search = "") {
  return normalizeFrameConvention(new URLSearchParams(search).get("frame_convention"));
}

export function frameConventionLabel(convention) {
  return normalizeFrameConvention(convention) === FRAME_CONVENTIONS.SPACE_FORCE ? "Space Force" : "OEL";
}

export function nextFrameConvention(convention) {
  return normalizeFrameConvention(convention) === FRAME_CONVENTIONS.SPACE_FORCE
    ? FRAME_CONVENTIONS.OEL_DEFAULT
    : FRAME_CONVENTIONS.SPACE_FORCE;
}

export function frameConventionDisplayAxisSign(convention, axis) {
  const key = String(axis || "").trim().toLowerCase();
  const inTrackAxis = key === "i" || key === "i_km" || key === "id_km_s";
  return normalizeFrameConvention(convention) === FRAME_CONVENTIONS.SPACE_FORCE && inTrackAxis ? -1 : 1;
}

export function frameConventionDisplayValue(convention, axis, value) {
  return frameConventionDisplayAxisSign(convention, axis) * (Number(value) || 0);
}

export function urlWithFrameConvention(rawUrl, convention, baseUrl) {
  const url = new URL(rawUrl, baseUrl);
  url.searchParams.set("frame_convention", normalizeFrameConvention(convention));
  return url;
}
