# Orbital Engagement Lab RPO Trainer Preview

This is the browser-native OEL RPO Trainer Preview. It is a lightweight web app
intended for social-media clickthroughs, quick demos, and small Pursuit Arcade
competitions.

Open locally:

```bash
python -m http.server 8765 --directory web/rpo-trainer-preview
```

Then visit:

```text
http://localhost:8765
```

## Included

- Unified level selector for Tutorial, Sandbox, and Pursuit Arcade.
- Automatic mobile/computer layout detection with manual view switching.
- Tutorial mode based on the Level 0 RIC-control lesson.
- Curated sandbox mode with preset starts, range, drift, reset, and randomize.
- Pursuit Arcade multi-round browser gameplay with deterministic replay
  validation.
- Mobile-friendly portrait and landscape controls with compact speed-multiple
  buttons, explicit camera toggling, and long-press selection suppression.
- RI and RC canvas plots with HCW-style projection for tutorial/sandbox and
  browser-native arcade projections for Pursuit Arcade.
- Keyboard controls for computer users and touch controls for mobile users.
- Browser-started music matched to each mode.
- Hosted leaderboard submission hooks with optional email ownership
  verification.
- Debrief and repository links for follow-up.

## Lightweight Analytics

The hosted preview can send privacy-focused Plausible and Vercel Web Analytics
events for product-funnel questions: preview views, tutorial starts, primer
completion, tutorial completion, sandbox starts, download clicks, music
toggles, and returns to the level selector.

Analytics are disabled for `file://`, `localhost`, and `127.0.0.1` runs. The
static page reads its analytics configuration from:

```html
<meta name="oel-analytics-provider" content="plausible,vercel" />
<meta name="oel-analytics-domain" content="adamcohen8.github.io" />
<meta name="oel-vercel-analytics-script" content="/_vercel/insights/script.js" />
<meta name="oel-vercel-analytics-hosts" content=".vercel.app,orbital-engagement-lab.vercel.app" />
```

Completion events use coarse buckets for time, delta-v, and closest range.
Browser analytics does not send raw trajectories, per-frame controls, names,
emails, or a player identifier. Vercel Analytics is only loaded on configured
Vercel-hosted domains so GitHub Pages and local runs do not request the Vercel
insights script. Pursuit Arcade leaderboard submissions are a separate explicit
form submit that sends the username, optional email, and attempt packet to the
hosted validation API.

## Not Included

This preview intentionally does not include the full OEL Python simulator,
scenario YAML support, all downloadable trainer levels, downloadable-game
recordings, or full debrief reports. Pursuit Arcade leaderboard attempts are
validated by replaying the browser-native deterministic arcade engine, not by
trusting client-submitted scores. See `docs/physics-contract.md` for the model
boundary.
