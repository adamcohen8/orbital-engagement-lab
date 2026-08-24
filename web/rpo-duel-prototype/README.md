# RPO Duel Beta

This directory contains the public experimental RPO Duel Beta for two human
players or one human playing the deterministic OEL computer. The
room server is authoritative for fixed-step physics, automatic time control,
delta-v budgets, scoring, disconnect coast, and rejoin. The same browser client
provides keyboard and touch controls.

The match uses the original procedural `39_perigee_afterburner_demo.wav` cue
as its looping level music. Browsers start it only after a player gesture;
press `M` or use the HUD button to turn it off or back on. Lean public exports
created with `--without-game-music` omit the optional WAV and the duel remains
fully playable without audio.

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

For a one-browser match, choose `Play computer`. The server assigns the
computer each alternating role: coast-aware predictive pursuit as Chaser and
bounded predictive evasion as Target. HCW is the computer's prediction model
only; both spacecraft still use the same authoritative fixed-step two-body
game propagation and normal delta-v enforcement.

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

The v0.27.1 Beta is deployed at
`https://oel-rpo-duel.oel-rpo-duel.workers.dev`.

Do not upgrade to Workers Paid for this Beta without explicit owner approval.
On Workers Free, SQLite-backed Durable Objects are supported and provider
limits fail closed when exhausted. After deployment, set the exact HTTPS URL
in the `oel-rpo-duel-url` meta element in
`web/rpo-trainer-preview/index.html`, rerun both web test suites, and verify two
remote clients before release. Deploy the updated trainer preview to its
production Vercel project only after the Worker is live, then run
`npm run verify:hosted-duel` from `web/rpo-trainer-preview`. A release that
ships RPO Duel is not complete until the live selector-to-Duel and
Duel-to-selector paths both pass.

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

Press `C` or use the camera button to cycle among the default full-trajectory
view tightly framing the origin, both satellites, recorded trajectories, and
both HCW projections, a pair view centered
on the current Target/Chaser midpoint, and a projection view that tightly frames
the two satellites plus both one-orbital-period HCW coast projections with a
small margin without keeping the origin in frame. The pair view suppresses
recorded trails and sizes the frame only around the two current satellites, so
the projections may continue beyond it. The reference view includes only the
margin needed around all of its required content. Solid red/yellow lines are recorded
Target/Chaser trails; in all three views, dashed red/blue lines are the current HCW
coast projections.

Press `M` or use the music button to toggle the looping Perigee Afterburner
level cue.

The six on-screen controls work with mouse, pen, or touch. Players cannot set
simulation speed. The server selects 200x while coasting and 10x while either
spacecraft maneuvers or during the one-second neutral cooldown.

At 200x, the client renders the authoritative snapshots through a bounded
120-millisecond interpolation buffer and eases the pair and projection camera
centers and spans.
This is visual-only: the server's fixed-step physics, inputs, timing, and
broadcast cadence remain authoritative. The buffer and camera easing reset on
maneuvers, speed changes, round changes, reconnect gaps, and camera toggles so
interactive state changes remain immediate.

The computer replans every 120 simulated seconds over a 1,800-second HCW
horizon and applies at most a 30-second policy pulse. Before burning, a
Duel-specific supervisor compares the proposed maneuver with coasting and
requires a material capture, timing, or closest-range improvement. This keeps
the computer responsive while making 200x coasting a policy decision rather
than a forced wait. In particular, the Target conserves delta-v when the
passive worst-case pass remains outside its 600-meter guard range. Its
normalized RIC commands are recorded in the same replay event stream as human
inputs.

After a match, `Play Again` keeps the same room and players. Human-versus-human
rooms wait until both players are ready; computer rooms restart as soon as the
human requests the rematch. Each rematch receives a new deterministic match
seed and begins with the normal countdown. `Return to Lobby` closes the current
socket, clears the reconnect session and room URL, and restores the setup
screen while retaining the player's callsign.

## Beta boundary

There are no accounts, matchmaking, chat, rankings, or durable public result
storage. The Cloudflare adapter uses one SQLite-backed Durable Object per room
and persists the replay inputs needed to restore authoritative match state.
Rooms expire after inactivity. RPO Duel remains an experimental browser-native
two-body game and must not be described as the downloadable trainer's complete
OEL engine.
