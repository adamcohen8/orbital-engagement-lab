# RPO Duel Beta

This directory contains the public experimental two-player RPO Duel Beta. The
room server is authoritative for fixed-step physics, automatic time control,
delta-v budgets, scoring, disconnect coast, and rejoin. The same browser client
provides keyboard and touch controls.

## Run locally

Requirements: Node.js 22 or newer.

```bash
cd web/rpo-duel-prototype
npm install
npm test
npm run dev
```

Open `http://localhost:8787` in two browser tabs. Create a room in one tab and
join it from the other using the displayed code or invite URL.

The local Node server also exposes the regular RPO Trainer web preview at
`http://localhost:8787/trainer/` for side-by-side testing. RPO Duel remains a
standalone experience at `http://localhost:8787/`; the preview's RPO Duel Beta
selector entry resolves to that root while using this combined local server.

## Run the deployed-shaped Worker locally

```bash
cd web/rpo-duel-prototype
npm install
npm run dev:worker -- --local
```

Wrangler serves only `public/` as static assets. The authoritative Worker,
shared room/physics source, tests, package metadata, and local Node server are
outside that asset directory.

## Deploy the Beta

Deployment requires an authenticated Cloudflare Workers Free account:

```bash
cd web/rpo-duel-prototype
npm install
npx wrangler deploy --dry-run
npx wrangler deploy
```

The v0.27.0 Beta is deployed at
`https://oel-rpo-duel.oel-rpo-duel.workers.dev`.

Do not upgrade to Workers Paid for this Beta without explicit owner approval.
On Workers Free, SQLite-backed Durable Objects are supported and provider
limits fail closed when exhausted. After deployment, set the exact HTTPS URL
in the `oel-rpo-duel-url` meta element in
`web/rpo-trainer-preview/index.html`, rerun both web test suites, and verify two
remote clients before release.

## Play from a phone

Run the server on the laptop and keep both devices on the same Wi-Fi network.
Find the laptop's LAN IP address, then open this address on the phone:

```text
http://LAPTOP_LAN_IP:8787
```

For example, a laptop at `192.168.1.25` is reached at
`http://192.168.1.25:8787`. If macOS asks whether Node may accept incoming
connections, allow it for this local test. This LAN route is for prototype
testing only; it does not expose the game outside the local network.

## Controls

| Axis | Negative | Positive |
| --- | --- | --- |
| R | `S` or `-R` | `W` or `+R` |
| I | `A` or `-I` | `D` or `+I` |
| C | `Left Arrow` or `-C` | `Right Arrow` or `+C` |

Press `C` or use the camera button to switch between the default full-trajectory
view centered on the propagated Target reference orbit and a satellites-only
view centered on the current Target/Chaser midpoint. In the reference view,
solid red/yellow lines are recorded Target/Chaser trails and dashed red/blue
lines are their current HCW coast projections.

The six on-screen controls work with mouse, pen, or touch. Players cannot set
simulation speed. The server selects 100x while coasting and 10x while either
spacecraft maneuvers or during the one-second neutral cooldown.

## Beta boundary

There are no accounts, matchmaking, chat, rankings, or durable public result
storage. The Cloudflare adapter uses one SQLite-backed Durable Object per room
and persists the replay inputs needed to restore authoritative match state.
Rooms expire after inactivity. RPO Duel remains an experimental browser-native
two-body game and must not be described as the downloadable trainer's complete
OEL engine.
