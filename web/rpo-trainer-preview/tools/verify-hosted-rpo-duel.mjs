const DEFAULT_SELECTOR_URL = "https://orbital-engagement-lab.vercel.app/";
const DEFAULT_DUEL_URL = "https://oel-rpo-duel.oel-rpo-duel.workers.dev";

function parseArguments(argv) {
  const options = {
    selectorUrl: DEFAULT_SELECTOR_URL,
    duelUrl: DEFAULT_DUEL_URL,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const value = argv[index + 1];
    if (argument === "--selector-url" && value) {
      options.selectorUrl = value;
      index += 1;
    } else if (argument === "--duel-url" && value) {
      options.duelUrl = value;
      index += 1;
    } else {
      throw new Error(`Unknown or incomplete argument: ${argument}`);
    }
  }
  options.selectorUrl = normalizeUrl(options.selectorUrl, true);
  options.duelUrl = normalizeUrl(options.duelUrl, false);
  return options;
}

function normalizeUrl(value, trailingSlash) {
  const url = new URL(value);
  if (url.protocol !== "https:") throw new Error(`Hosted URL must use HTTPS: ${value}`);
  url.hash = "";
  url.search = "";
  url.pathname = trailingSlash ? `${url.pathname.replace(/\/+$/, "")}/` : url.pathname.replace(/\/+$/, "");
  return url.toString().replace(/\/$/, trailingSlash ? "/" : "");
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function withCacheBust(value) {
  const url = new URL(value);
  url.searchParams.set("oel_release_check", String(Date.now()));
  return url;
}

async function fetchText(value) {
  const url = withCacheBust(value);
  const response = await fetch(url, {
    headers: { "Cache-Control": "no-cache" },
    redirect: "follow",
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok) throw new Error(`${url.origin}${url.pathname} returned HTTP ${response.status}`);
  return response.text();
}

function requireMatch(content, pattern, message) {
  if (!pattern.test(content)) throw new Error(message);
}

async function verifyHostedRoundTrip({ selectorUrl, duelUrl }) {
  const selectorAppUrl = new URL("src/app.js", selectorUrl).toString();
  const duelAppUrl = new URL("src/client/app.js", `${duelUrl}/`).toString();
  const duelFrameConventionUrl = new URL("src/client/frame-convention.js", `${duelUrl}/`).toString();
  const [selectorHtml, selectorApp, duelHtml, duelApp, duelFrameConvention] = await Promise.all([
    fetchText(selectorUrl),
    fetchText(selectorAppUrl),
    fetchText(`${duelUrl}/`),
    fetchText(duelAppUrl),
    fetchText(duelFrameConventionUrl),
  ]);

  requireMatch(selectorHtml, /data-level-option="rpoDuel"/, "Production selector does not list RPO Duel.");
  requireMatch(selectorHtml, /level-beta-badge">Beta</, "Production selector does not label RPO Duel as Beta.");
  requireMatch(
    selectorHtml,
    new RegExp(`name="oel-rpo-duel-url"\\s+content="${escapeRegExp(duelUrl)}"`),
    `Production selector does not point to ${duelUrl}.`,
  );
  requireMatch(selectorApp, /id: "rpoDuel"/, "Production selector client does not define the RPO Duel option.");
  requireMatch(selectorApp, /destination\.searchParams\.set\("frame_convention", state\.frameConvention\)/, "Production selector client does not carry the selected frame convention into RPO Duel.");
  requireMatch(selectorApp, /window\.location\.assign\(destination\.href\)/, "Production selector client does not launch its external destination.");

  requireMatch(duelHtml, /id="level-selector-link"/, "Production RPO Duel page has no Level Selector action.");
  requireMatch(
    duelHtml,
    new RegExp(`name="oel-level-selector-url"\\s+content="${escapeRegExp(selectorUrl)}"`),
    `Production RPO Duel page does not return to ${selectorUrl}.`,
  );
  requireMatch(duelApp, /localHost \? "\/trainer\/" : HOSTED_LEVEL_SELECTOR_URL/, "Production RPO Duel client does not use its hosted selector URL.");
  requireMatch(duelHtml, /id="frame-convention-label"/, "Production RPO Duel page does not display its active frame convention.");
  requireMatch(duelApp, /frameConventionFromSearch\(location\.search\)/, "Production RPO Duel client does not read the selector frame convention.");
  requireMatch(duelApp, /urlWithFrameConvention\(window\.location\.origin, state\.frameConvention\)/, "Production RPO Duel invite links do not retain the frame convention.");
  requireMatch(duelFrameConvention, /frame_convention/, "Production RPO Duel frame module does not define the URL contract.");
  requireMatch(duelFrameConvention, /inTrackAxis[^\n]+i_km/, "Production RPO Duel frame module does not map the in-track display axis.");
}

try {
  const options = parseArguments(process.argv.slice(2));
  await verifyHostedRoundTrip(options);
  console.log(`Hosted RPO Duel round trip passed: ${options.selectorUrl} <-> ${options.duelUrl}`);
} catch (error) {
  console.error(`Hosted RPO Duel round trip failed: ${error.message}`);
  process.exitCode = 1;
}
