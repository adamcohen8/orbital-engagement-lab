# RPO Duel

Status: public Beta deployed; v0.27.0 live two-client acceptance passed

Support posture: experimental browser multiplayer. The deterministic room
server is authoritative for this game, but it is not the full downloadable OEL
physics engine and does not establish an operational mission-analysis claim.

## Prototype Implementation Snapshot

The prototype is implemented in `web/rpo-duel-prototype`. It runs one
authoritative room server and serves one responsive browser client for both
laptop and phone players. The v0.27.0 Beta is deployed on Cloudflare Workers
Free at `https://oel-rpo-duel.oel-rpo-duel.workers.dev`; no paid service or
plan upgrade was connected.

Implemented now:

- a standalone local Duel create/join screen for desktop and mobile testing;
- create and join by six-character room code or invite URL;
- two authenticated player sockets with reconnect tokens scoped to the match;
- deterministic, server-authoritative two-body duel propagation using the
  existing Pursuit Arcade engine;
- server-seeded first roles, alternating roles, 2/4/6 regulation rounds, and
  mirrored randomized geometry for every two-round pair;
- the frozen `rpo-duel.prototype.v1` duration, capture, delta-v, and no-safety
  rules;
- 100x shared coasting time, immediate 10x maneuver time, and the one-second
  neutral cooldown;
- immediate control neutralization and continued authoritative coast on
  disconnect, with token-based rejoin while the match remains active;
- ordered normalized RIC inputs shared by keyboard and touch controls;
- responsive R/I and R/C trajectory plots using the regular RPO Trainer
  dashboard composition: a default full-trajectory frame centered on the
  propagated Target reference orbit, visible one-orbital-period Target and Chaser HCW coast
  projections, and a `C`/touch camera toggle to a satellites-only midpoint
  view; twin plots and a bottom HUD on laptops, compact stacked plots in phone
  portrait, and the existing side-by-side plots with the lower HUD and
  right-side three-by-two touch-control group in phone
  landscape; role/score/time/delta-v telemetry, connection state, countdowns,
  round results, and draw-capable match results; and
- deterministic engine tests plus HTTP, WebSocket, ordering, disconnect, and
  reconnect tests;
- a Cloudflare Worker with one SQLite-backed Durable Object per room, a
  dedicated static upload surface, bounded request/message payloads,
  same-origin room access, reconnect-state persistence, room expiry alarms,
  and security headers; and
- an `RPO Duel` entry in the RPO Trainer web selector with a visible `Beta`
  label and a separately configured hosted destination.

The prototype uses the existing Arcade maximum acceleration as an explicitly
provisional playtesting parameter: 0.015 m/s squared for the Chaser and 0.0075
m/s squared for the Target. The accepted hard delta-v budgets remain 15 m/s and
5 m/s respectively.

Release acceptance completed:

- two independent remote browser clients created and joined the same room,
  received opposite roles, shared authoritative snapshots, rendered both
  trajectory planes, and propagated a maneuver/time-control transition without
  an error overlay;

Post-release Beta validation:

- expand remote-network playtesting across multiple laptop/phone combinations;

Post-Beta follow-on work:

- snapshot interpolation beyond the current lightweight plotted snapshot
  trail and client-rendered HCW coast projections;
- persisted final replay/result downloads and rematch UX;
- provider usage monitoring and a lower application-level admission threshold
  if Beta traffic approaches the Workers Free daily limits.

## Product Concept

RPO Duel is a hosted two-player version of the Orbital Engagement Lab RPO
Trainer. One player flies the chaser and the other flies the target. The first
version should be a focused pursuit/evasion game that makes relative orbital
motion understandable through direct competition.

RPO Duel should be hosted in the web version of the trainer. It should retain
the educational RIC control language and deterministic browser physics boundary
rather than presenting itself as the full Python OEL simulator or as an
operational mission-design tool.

The current browser Pursuit Arcade engine is the intended starting point. It
already provides deterministic two-body ECI propagation, target-RIC control
mapping, fixed one-second physics ticks, target and chaser delta-v accounting,
and replayable input events. The multiplayer version can replace the scripted
defensive target with a second human player instead of creating a new physics
model from scratch.

## Initial Player Experience

The proposed v0.1 flow is:

1. A player creates a room and receives an invite link or short room code.
2. A second player joins the room.
3. At launch, the players select 2, 4, or 6 regulation rounds.
4. The players receive their initial Target and Chaser roles.
5. A short synchronized countdown begins the first round.
6. Both players command their spacecraft with RIC translation controls.
7. After each round, the players exchange roles for the next round.
8. After all regulation rounds, a tied prototype match ends as a draw.
9. The result screen shows the complete series and offers a rematch.

The first version should use one curated scenario. It should not initially add
public matchmaking, accounts, chat, spectators, rankings, multiple levels, or
the full downloadable trainer scenario catalog.

Cross-device play is a v0.1 requirement. A laptop player and a phone player
must be able to join the same match in either role. Laptop-versus-laptop and
phone-versus-phone matches should use the same room and protocol rather than
separate game implementations.

## Prototype Rules Contract

The first runnable multiplayer ruleset is `rpo-duel.prototype.v1`.

Each regulation round uses:

- 18,000 seconds of simulated time;
- a 0.1 km (100 meter) Chaser capture distance;
- no relative-speed requirement for capture;
- a 15 m/s Chaser delta-v budget;
- a 5 m/s Target delta-v budget; and
- no keepout, collision-speed, forbidden-region, or other safety-failure
  constraints.

The delta-v budgets are hard maneuver caps, not automatic round losses. When a
spacecraft reaches its budget, the server rejects further thrust commands for
that spacecraft and it continues to coast. A budget-exhausted Chaser may still
capture through its resulting passive trajectory.

The authoritative server randomly assigns the first Target and Chaser roles
from the recorded match seed. Roles alternate after every round. Tied
regulation matches end as draws in the prototype; the finite tiebreaker remains
a post-prototype design decision.

## Match Format And Role Fairness

The default match should use a tennis-style series rather than expecting one
Target-versus-Chaser setup to give both roles an exactly equal chance of
winning.

At match launch, the players select one of three regulation lengths:

- 2 rounds;
- 4 rounds; or
- 6 rounds.

The selected value is the number of regulation rounds, not a best-of limit.
The players alternate Target and Chaser roles after every round and play the
entire selected regulation series. Because every supported series length is
even, both players receive the same number of rounds in each role.

Each round awards one match point to its winner. After an even number of
rounds, any non-tied score differs by at least two points, satisfying the
match's win-by-two rule. A tied prototype regulation score ends as a draw.

### Future Finite Tiebreaker Requirement

The prototype does not implement a tiebreaker. If a tiebreaker is added after
prototype playtesting, it must not be another repeating pair of
Target-versus-Chaser rounds. Repeating mirrored pairs could allow a match to
continue indefinitely.

The selected tiebreaker must:

- be one bounded event rather than an indefinitely repeatable sequence;
- avoid giving either player a privileged Target or Chaser assignment;
- have a fixed maximum simulated and wall-clock duration;
- produce a deterministic winner even if neither player completes its primary
  objective; and
- record the tiebreak result separately from regulation round points.

The exact tiebreak mechanic remains unresolved and should be selected through
design review and playtesting before a post-prototype ruleset is frozen.

One candidate is a simultaneous rendezvous shootout: both players fly as
Chaser in separate but identical deterministic lanes against passive targets.
They receive the same initial geometry, budgets, and time limit. A valid capture
wins; if both or neither capture, a declared deterministic comparison such as
capture time, safe closest approach, and remaining delta-v resolves the result.
This is a candidate for evaluation, not an accepted rule.

### Paired Scenario Conditions

Every two-round role pair should begin with a new random initial geometry
generated using the Pursuit Arcade-style randomization approach. Both rounds
inside that pair must then use the exact same scenario definition, initial
geometry, target orbit, budgets, and random seed. Only the player-role
assignment changes.

For a six-round regulation match, the sequence is therefore:

- rounds 1 and 2 use random geometry A;
- rounds 3 and 4 use new random geometry B; and
- rounds 5 and 6 use new random geometry C.

The same rule applies to 2- and 4-round regulation matches. Tiebreak initial
conditions will depend on the selected finite tiebreak mechanic and must have a
separate versioned fairness contract.

This makes each pair a direct mirrored comparison rather than giving one player
a random advantage or disadvantage in the Chaser role. It also ensures that
each player's successive Chaser rounds present different initial conditions.

The authoritative match server should generate a match seed and derive each
pair seed deterministically from the match seed and pair index. The pair seed,
generated initial state, randomization-contract version, and resulting geometry
should be included in the replayable match record. Clients must not select or
submit authoritative random geometry.

The randomization envelope should follow the existing arcade principles,
including bounded RIC position/rate ranges, a minimum initial separation, and
orbital-energy-consistent initialization. Exact ranges should be versioned match
configuration rather than hidden constants. Unlike the current arcade flow's
special fixed first-round start, every RPO Duel pair, including the first pair,
should receive a random initial geometry.

The match record should distinguish:

- individual round winners and terminal reasons;
- regulation rounds and the separate tiebreak event, when present;
- each player's results as Target and Chaser;
- the regulation score;
- the final match score; and
- whether the match ended in regulation or a tiebreaker.

## Roles And Victory Conditions

### Chaser

In each round, the chaser attempts to enter a defined capture region around the
target. Under `rpo-duel.prototype.v1`, capture occurs at or inside 0.1 km
regardless of relative speed. The Chaser has a hard 15 m/s delta-v cap.

### Target

In each round, the target attempts to survive until the round timer expires
without being captured. Under `rpo-duel.prototype.v1`, the round timer is
18,000 seconds of simulated time and the Target has a hard 5 m/s delta-v cap.

The prototype has no relative-speed capture gate and no safety-failure
conditions. Capture distance, round duration, delta-v caps, and the absence of
safety constraints remain versioned balancing parameters and should be tuned
through two-player playtesting rather than silently changed in place.

## Shared Automatic Time Control

Players should not manually control the simulation speed. The match server
owns one shared time multiplier for both players.

Initial proposed policy:

- When both players are neutral, the simulation runs at 100x.
- When either player commands a maneuver, the simulation immediately drops to
  10x.
- The simulation remains at 10x until both players have been neutral for a
  short cooldown, initially proposed as one second of wall-clock time.
- After the cooldown, the simulation returns automatically to 100x.
- If either player disconnects, that player's controls are immediately forced
  to neutral and their spacecraft continues to coast. The match does not pause.
- The HUD always displays the current automatic speed and its cause, such as
  `AUTO TIME 10x - TARGET MANEUVERING`.

The cooldown prevents rapid key releases or alternating inputs from making the
game oscillate visibly between 10x and 100x.

The speed multiplier must change wall-clock pacing, not the authoritative
physics timestep. The engine should continue to integrate canonical fixed
one-second simulation ticks and execute more of those ticks per wall-clock
second during high-speed coast.

### Latency Risk

Network latency is magnified at high time compression. At 100x, 100
milliseconds of latency corresponds to 10 seconds of simulation time. The
initial implementation should therefore include:

- immediate input-intent messages on key-down and key-up;
- small server-side simulation batches rather than large coast jumps;
- client-side visual prediction for responsive controls;
- reconciliation to authoritative server snapshots;
- sequence numbers so delayed or duplicated inputs cannot supersede newer
  commands; and
- playtesting of 50x as a possible coast speed if 100x feels too unforgiving.

The server remains authoritative even when the client predicts immediate visual
feedback.

## Cross-Device Support

Both laptop and phone clients should run the same web application and connect
to the same authoritative match service. Device-specific controls must produce
the same normalized RIC input messages:

- laptop players use keyboard controls;
- phone players use large touch controls for +R, -R, +I, -I, +C, and -C; and
- neither input surface may own or alter the simulation speed.

The responsive interface should support phone portrait and landscape layouts
without hiding essential trajectory, objective, time-control, delta-v, or
connection information. Touch controls must not cover the active trajectory
plots or require precise gestures during a maneuver.

The networking and rendering path should remain lightweight enough for mobile
connections. Clients should receive bounded authoritative snapshots and render
smooth motion through interpolation rather than requiring a physics message for
every display frame.

Mobile browsers may suspend background tabs or briefly disconnect when the
device locks, changes orientation, or changes networks. A lost connection
should be treated as loss of spacecraft command rather than a reason to freeze
the match:

- the server immediately clears every held or queued maneuver input for the
  disconnected player so a lost key-up message cannot create a stuck burn;
- the disconnected spacecraft continues its authoritative passive coast;
- the connected player remains free to coast or maneuver;
- the disconnected player is considered neutral by the automatic time-control
  policy;
- the HUD identifies the disconnected spacecraft, for example
  `TARGET DISCONNECTED - COASTING`;
- the match timer, physics, scoring, and victory conditions continue normally;
  and
- the player may reconnect and reclaim the same role at any time while the
  match is still active.

Reconnection does not rewind or restore a prior state. The returning client
receives the current authoritative snapshot and resumes from wherever the
spacecraft coasted. If the match reaches a terminal result while a player is
disconnected, that result remains final.

## Authoritative Multiplayer Model

Each match should have one authoritative room process. Both clients send input
intent, not spacecraft state. The room process owns:

- player roles and reconnect credentials;
- the canonical deterministic simulation state;
- accepted input state and input sequence numbers;
- the shared automatic time multiplier;
- target and chaser delta-v accounting;
- round phase, timer, and victory evaluation;
- role alternation, regulation length, series score, and tiebreak state;
- periodic authoritative snapshots; and
- the final replay and result packet.

The server should broadcast snapshots at a bounded wall-clock rate while the
clients interpolate rendering between them. Physics ticks and input events
should remain deterministic and replayable independently of rendering cadence.

```text
Target browser --\
                  >-- WebSockets -- Authoritative match room
Chaser browser --/                    |-- deterministic duel engine
                                      |-- shared automatic clock
                                      `-- replay/result packet
```

## Hosting Direction

The recommended multiplayer host is Cloudflare Workers with one Durable Object
per match. A Durable Object provides one stateful authority for the two player
connections, room state, deterministic simulation timeline, reconnects, and
match results.

Relevant platform documentation:

- [Cloudflare Durable Objects overview](https://developers.cloudflare.com/durable-objects/)
- [Durable Objects WebSocket guidance](https://developers.cloudflare.com/durable-objects/best-practices/websockets/)
- [Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/)

The existing web preview uses static hosting plus Vercel serverless endpoints
for leaderboard and email-verification workflows. Vercel Functions can host
WebSockets, but connections are limited by function duration and durable room
coordination requires an external store such as Redis. That remains a possible
prototype path, but it adds more coordination components for this stateful
match-server use case.

- [Vercel WebSocket guidance](https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections)

Two deployment arrangements remain viable:

1. Keep the existing static frontend and leaderboard endpoints on Vercel while
   Cloudflare hosts authoritative multiplayer rooms.
2. Move the overhauled frontend and multiplayer service together to Cloudflare,
   while retaining Supabase or another bounded store only where durable public
   results are needed.

The second arrangement is the current preference for the eventual replacement,
but the existing preview should remain live until the new version passes its
acceptance checks. No paid service should be provisioned without explicit
approval.

## Cost And Budget Posture

RPO Duel should begin on free-tier infrastructure. The expected hosting cost
for development, a private prototype, and a modest invitation-only beta is
`$0`, subject to measured usage and the providers' current terms.

The initial free deployment should use:

- Cloudflare Pages for the web application;
- Cloudflare Workers and SQLite-backed Durable Objects for authoritative match
  rooms and bounded match state;
- the provider-supplied `pages.dev` or `workers.dev` address unless an existing
  OEL domain is appropriate;
- no required account system, email service, public matchmaking, or external
  realtime database; and
- no paid plan, paid add-on, or usage overage without explicit approval.

As checked on 2026-08-20, Cloudflare documents a Workers/Durable Objects free
allowance that includes 100,000 requests and 13,000 GB-s of Durable Object
duration per day. Cloudflare Pages documents up to 500 builds per month on its
free plan. These figures are planning inputs, not permanent guarantees, and
must be rechecked against the official pricing and limits pages before the
first deployment and before any public launch:

- [Cloudflare Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/)
- [Cloudflare Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/)
- [Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/)

The free tier is not an unlimited availability promise. If usage approaches a
free limit, the service should fail closed by preventing new room creation
while allowing active matches to finish when the platform permits. It should
not silently authorize a paid upgrade.

The private-beta deployment should include:

- an invitation-only room flow;
- a hard cap on concurrent active rooms;
- a fixed maximum regulation length and bounded tiebreak duration;
- automatic cleanup of abandoned rooms and bounded replay data;
- compact server snapshots and input messages;
- provider usage monitoring and threshold notifications; and
- a documented shutdown switch for new match creation.

Before expanding beyond a small beta, measure match duration, Durable Object
active time, incoming WebSocket message volume, storage growth, and bandwidth
per completed match. Use those measurements to calculate capacity and cost
rather than assuming that the prototype's free-tier behavior will scale.

Possible later costs include a paid Workers plan, a new custom domain, larger
or durable result storage, email delivery, accounts, public matchmaking, and
expanded analytics. Supabase can remain optional for leaderboard or account
features; the multiplayer v0.1 match loop must not depend on it.

## Web Trainer Overhaul

The current preview should not absorb multiplayer by continuing to grow its
single large application script. The replacement should be developed alongside
the existing preview and should preserve working modes until feature and visual
acceptance is complete.

Proposed structure:

- a TypeScript and Vite web application;
- separate launcher, room, game, and debrief screens;
- a reusable Canvas renderer;
- a pure deterministic duel engine shared by server execution and replay tests;
- a server networking and room-protocol layer;
- a client prediction, interpolation, and reconciliation layer;
- first-class keyboard and touch input surfaces using one normalized protocol;
- responsive laptop, phone portrait, and phone landscape layouts;
- preserved Tutorial, Sandbox, and Pursuit Arcade modes; and
- RPO Duel as a distinct multiplayer mode.

The first multiplayer release should remain browser-native. It should not try
to host the Python OEL engine, port the entire Pygame trainer, or silently claim
physics parity that has not been demonstrated.

## Safety, Integrity, And Abuse Boundaries

- Clients may submit controls but never authoritative position, velocity,
  score, time, or delta-v state.
- Room identifiers and reconnect tokens should be unguessable and scoped to one
  match.
- Reconnect credentials should remain valid until the match reaches a terminal
  result, and should restore only the originally assigned role.
- Both roles must have explicit input ownership; a client cannot command the
  other spacecraft.
- Inputs, snapshots, and results should be schema-versioned.
- Server-side rate and message-size limits should bound abusive clients.
- Match logs and replay packets should be bounded rather than retaining every
  render frame.
- A disconnected role must be forced to neutral while its spacecraft continues
  under authoritative passive dynamics.
- Deployments must preserve the public/private OEL boundary and must not expose
  private scenarios, source, or generated evidence.

## Verification Priorities

The first acceptance suite should cover:

- deterministic replay from the same initial state and input sequence;
- exact role-based input ownership;
- automatic 100x-to-10x transition when either player maneuvers;
- neutral cooldown and return to coast speed;
- no speed oscillation during alternating inputs;
- fixed physics timestep across both speed modes;
- two-browser synchronization;
- laptop-versus-laptop, laptop-versus-phone, and phone-versus-phone sessions;
- either device type playing either role;
- phone portrait/landscape changes without losing the room or obscuring
  required controls and match information;
- delayed, duplicated, reordered, and dropped input messages;
- client prediction followed by authoritative reconciliation;
- immediate input neutralization on disconnect, including loss of a key-up
  message;
- authoritative coast while one or both players are disconnected;
- continued match timer, scoring, and terminal-condition evaluation during a
  disconnect;
- mobile backgrounding and reconnection to the current state at any point
  before the match ends;
- rejection of role reclamation after a terminal result;
- delta-v budget enforcement for both players;
- exact `rpo-duel.prototype.v1` round rules: 18,000 simulated seconds, 0.1 km
  capture distance, no relative-speed gate, 15 m/s Chaser delta-v, 5 m/s Target
  delta-v, and no safety-failure constraints;
- hard delta-v caps that force coast without creating an automatic loss;
- capture and survival victory conditions;
- complete 2-, 4-, and 6-round regulation series;
- exact Target/Chaser role alternation and equal regulation role counts;
- a new deterministic random geometry for every two-round pair;
- identical initial conditions and pair seed within each mirrored role pair;
- different pair seeds and initial geometries across successive pairs;
- server-only geometry generation from a recorded match seed, pair index, and
  versioned randomization contract;
- orbital-energy and minimum-separation constraints on generated geometry;
- correct separation of round wins from the overall match result;
- tied prototype regulation series ending as recorded draws;
- enforcement of concurrent-room, message-size, match-duration, and replay
  retention limits;
- rejection of new room creation when the configured free-tier safety threshold
  is reached;
- bounded, replayable final result packets.

## Proposed Implementation Sequence

1. Freeze a versioned multiplayer rules, physics, and network contract.
2. Extract the current browser competition engine behind a stable pure-module
   interface and preserve its existing replay tests.
3. Add human-commanded target inputs and run a local two-player match in one
   browser or deterministic test harness.
4. Implement the authoritative room service and verify two browser clients
   against it locally.
5. Build the new responsive web shell, multiplayer waiting-room experience,
   and shared keyboard/touch input protocol.
6. Deploy a private, invitation-only prototype on free-tier infrastructure with
   hard room, duration, storage, and usage safeguards.
7. Playtest shared time control, latency feel, capture rules, delta-v balance,
   and laptop/phone combinations in both roles.
8. Measure per-match compute, messages, storage, and bandwidth before making a
   public capacity or cost claim.
9. Promote the new web trainer only after deterministic, networking, visual,
   mobile/desktop, cost-safety, and deployment acceptance.

## Open Design Questions

- Is 100x coast comfortable under real network latency, or should v0.1 begin at
  50x?
- How long should the neutral cooldown be before returning to high-speed coast?
- Should a player see the opponent's active thrust vector immediately?
- Should both players see identical plots, or should role-specific information
  be limited?
- After prototype playtesting, should draws remain valid results or should a
  finite, role-neutral tiebreak format decide tied regulation series?

These questions should be resolved through a small deterministic prototype and
playtesting rather than by expanding the first implementation prematurely.
