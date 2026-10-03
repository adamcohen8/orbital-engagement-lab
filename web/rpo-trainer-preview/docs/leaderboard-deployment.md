# Pursuit Arcade Leaderboard Deployment

This is the small-hosting path for a Pursuit Arcade leaderboard with roughly
100 users per week.

## Recommended Stack

- Vercel static hosting for `web/rpo-trainer-preview`.
- Vercel serverless functions in `api/`.
- Supabase Postgres using `supabase/schema.sql`.
- Resend for optional score receipt and email verification messages.

The browser never writes directly to Supabase. Attempts go through
`api/submit-attempt.mjs`, which runs the deterministic validator before
inserting rows. Public leaderboard reads use a denormalized
`public_leaderboard` table containing only public fields; private player,
attempt, and verification tables stay service-role only.

## Supabase Setup

1. Create a Supabase project.
2. Open the SQL editor.
3. Run `web/rpo-trainer-preview/supabase/schema.sql`.
   - Existing projects should run it again after this update; the script
     idempotently adds `players.username_locked_at`, recipient send locks, and
     the atomic publication and email-claim functions. Apply the schema before
     deploying the API.
   - Stop submissions through the old API while applying the migration. Its
     matching-email score writes can recreate unbound winners after the
     cleanup if it remains live.
   - The migration discards legacy verified winners without a consumed token
     for an eligible exact attempt and the player's current email, then rebuilds
     the public table. Review any removed scores with affected users before
     claiming that earlier matching-email submissions proved ownership.
   - In a disposable database with the schema installed, run
     `tests/ownership-db-regression.sql`. Before production rollout, exercise
     concurrent anonymous publication and competing token claims against the
     real database; the local JavaScript tests use mocked RPC responses and
     cannot verify Postgres row locks or migration behavior.
4. Copy these values from Project Settings:
   - Project URL
   - Service role key

Do not expose the service role key in browser JavaScript or commit it to git.
It belongs only in serverless function environment variables.

## Vercel Setup

1. Create a Vercel project with `web/rpo-trainer-preview` as the project root.
2. No build command is required for the static preview.
3. Add environment variables:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `OEL_ARCADE_ALLOWED_ORIGIN`
   - `OEL_ARCADE_PUBLIC_ORIGIN`
   - `RESEND_API_KEY`
   - `OEL_ARCADE_EMAIL_FROM`
   See `deployment-env-template.txt` for the expected names.
4. Deploy.

`RESEND_API_KEY` and `OEL_ARCADE_EMAIL_FROM` are optional for leaderboard
storage, but required for score receipt and ownership verification emails. If
they are missing, score submission still succeeds and the API returns
`email_status: "not_configured"`.
`OEL_ARCADE_PUBLIC_ORIGIN` is also required for verification email and must be
the exact HTTPS origin of this deployment, with no path, credentials, query, or
fragment. Request Host and forwarded headers never determine the emailed link.
An unclaimed username can continue to publish a provisional score. Once
verified, a direct submission cannot change its leaderboard row merely by
providing the matching email address. A matching address receives a fresh,
single-use token tied to that exact attempt; only a click on that link can
promote a higher score. Verification mail is limited to one pending send per
normalized recipient for one hour. A successful token claim releases only its
matching reservation, allowing the owner to submit another score. An older
unconsumed link remains valid for its seven-day token lifetime but cannot
extend or clear a newer send lock. Ambiguous provider timeouts retain the
reservation until a delivered token is claimed or the one-hour cooldown
expires, because the provider may already have accepted the message.
Browser POSTs with an Origin header are admitted only from the configured
public origin. This does not authenticate nonbrowser clients; production
hosting still needs provider or IP rate limits against repeated requests.

### Required mail-abuse gate before production

Do not enable `RESEND_API_KEY` until both controls below are configured and
verified for the production project:

1. Add a Vercel edge rate rule for `POST /api/submit-attempt`, keyed by the
   platform-observed client IP, so excess requests are rejected before the
   serverless function can write attempts or contact Resend. Choose and record
   the threshold/window for expected launch traffic and the deployed Vercel
   plan. Do not derive the key from a client-supplied forwarding header in
   application code.
2. Configure an operator-controlled, account-wide send ceiling or budget that
   can stop mail across all recipients and source IPs, with monitoring. Select
   the bound for the active Resend account and confirm how operators will
   detect and stop unexpected volume; alerts alone do not bound sends. If the
   account has no suitable hard cap or stop control, leave email sending
   disabled.

The database reservation is a separate per-recipient control: it allows at
most one pending send per normalized address during its one-hour reservation,
but distinct addresses can each acquire a reservation. The allowed-Origin
check only constrains browser requests that provide an Origin header. Neither
control is a source or global rate limit. Local tests use mocked fetch calls;
they can verify API behavior but cannot prove that Vercel's edge rule or the
provider account budget is active. Record the configured rule, quota, and a
hosted rejection/quota check as deployment evidence before enabling the mail
key. If a rate-limit rejection reaches the API function or produces an email,
keep the mail key unset until the edge control is corrected.

## API Contract

Submit a validated leaderboard attempt. Do not hand-author the nested attempt;
generate it from the active challenge and recorded replay so its identifiers,
rounds, claims, and input events remain content-consistent:

```js
import {
  buildChallengeRecord,
  createPursuitArcadeSession,
  validateArcadeAttemptPacket,
} from "../src/competition/arcade-engine.js";

const challenge = buildChallengeRecord();
const session = createPursuitArcadeSession(challenge.config, { seed: 4242 });
session.step(1); // record a contiguous first-round replay
const attempt = session.attemptPacket({
  challengeRecord: challenge,
  username: "ORBITACE",
  email: "optional@example.edu",
  client_build_hash: "deployed-build-id",
});
if (validateArcadeAttemptPacket(attempt, challenge).status === "invalid") {
  throw new Error("Refusing to submit an invalid arcade attempt");
}
await fetch("/api/submit-attempt", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ username: attempt.username, email: attempt.email, attempt }),
});
```

The HTTP wrapper accepted by the API is:

```http
POST /api/submit-attempt
Content-Type: application/json
```

```json
{
  "username": "ORBITACE",
  "email": "optional@example.edu",
  "attempt": "the generated makeArcadeAttemptPacket-shaped object"
}
```

The generated object includes the active challenge/physics/scoring/config
identifiers, seed, claimed score and metrics, and at least one contiguous round
with `final_tick` and `input_events`. An empty `round_attempts` array is invalid
and returns HTTP 422.

Read public leaderboard rows:

```http
GET /api/leaderboard?challenge=rpo_arcade_pursuit&limit=25
```

Accepted submissions store the canonical score, metrics, validation warnings,
the submitted attempt packet, and server-generated RI/RC plot SVGs. When an
attempt improves an unclaimed player's best eligible score, the service-role
API updates both the private `leaderboard_entries` bookkeeping table and the
public `public_leaderboard` table. A verified player's score changes only
after the bound email token is consumed.

If an accepted submission includes an email address, the API stores a hashed
verification token in Supabase and sends a verification link. Opening the link
shows a confirmation page without changing database state; only pressing its
confirmation button submits the token to the atomic claim endpoint. That claim
updates `players.email`, `players.email_verified_at`, and
`players.username_locked_at`; public leaderboard reads expose only the
denormalized boolean `email_verified`, never the email address. Keeping GET
read-only prevents email-security scanners and link previews from claiming a
username before its owner confirms.

## Username Ownership Policy

Emails are optional. Anonymous usernames can still submit and score, but a
username becomes reserved after a player verifies an email link for that
username.

| Username state | Submission email | Attempt saved | Leaderboard update |
| --- | --- | --- | --- |
| Unclaimed | none | yes | yes |
| Unclaimed | email provided | yes | yes, verification pending |
| Verified owner | same email | yes | after fresh token claim |
| Verified owner | none | yes | no |
| Verified owner | different email | yes | no |

Typed emails are not trusted as ownership by themselves. A permitted email
submission creates an `email_verifications` row, and `players.email` becomes
authoritative only after the owner submits the confirmation form. The database
claims the username and consumes the token in one transaction. For a first claim,
only the exact linked eligible attempt replaces any higher provisional score.
If that attempt was deleted or became ineligible, the token still reserves the
username but promotes no score. Later owner tokens promote only a higher
exact linked eligible attempt.

This is still a lightweight bragging-rights system, not a full account system.
If a player loses email access or a username dispute arises, resolve it
manually in Supabase.

## First Production Check

After deployment:

1. Play Pursuit Arcade.
2. Submit an attempt through the hosted page.
3. Confirm `/api/leaderboard` returns the row.
4. Confirm the Supabase `attempts` row has status `valid` or `suspicious`.
5. If email is configured, confirm the submit response includes
   `email_status: "sent"`. Open the verification link and confirm it shows a
   confirmation page without changing `email_verified`; press its button, then
   confirm the claim updates `email_verified` on `/api/leaderboard`.
6. Submit the verified username with no email or a different email and confirm
   the API returns `ownership_status: "locked"` with no leaderboard update.
7. Submit the verified username with the same email and confirm no direct
   leaderboard update occurs. Open the newly emailed, attempt-bound link,
   submit its confirmation form, and confirm the score updates if it improves.
   Confirm a second owner attempt can request a fresh link after the first
   link is consumed.
8. Try changing `claimed_score` in a copied packet and confirm the endpoint
   returns `invalid`.
