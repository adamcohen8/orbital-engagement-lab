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
inserting rows.

## Supabase Setup

1. Create a Supabase project.
2. Open the SQL editor.
3. Run `web/rpo-trainer-preview/supabase/schema.sql`.
   - Existing projects should run it again after this update; the script
     idempotently adds `players.username_locked_at`.
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

## API Contract

Submit a validated leaderboard attempt:

```http
POST /api/submit-attempt
Content-Type: application/json
```

```json
{
  "username": "ORBITACE",
  "email": "optional@example.edu",
  "attempt": {
    "schema_version": 2,
    "attempt_type": "arcade_run",
    "round_attempts": []
  }
}
```

Read leaderboard rows:

```http
GET /api/leaderboard?challenge=rpo_arcade_pursuit&limit=25
```

Accepted submissions store the canonical score, metrics, validation warnings,
the submitted attempt packet, and server-generated RI/RC plot SVGs.

If an accepted submission includes an email address, the API stores a hashed
verification token in Supabase and sends a verification link. Visiting the link
updates `players.email`, `players.email_verified_at`, and
`players.username_locked_at`; public leaderboard reads expose only the boolean
`email_verified`, never the email address.

## Username Ownership Policy

Emails are optional. Anonymous usernames can still submit and score, but a
username becomes reserved after a player verifies an email link for that
username.

| Username state | Submission email | Attempt saved | Leaderboard update |
| --- | --- | --- | --- |
| Unclaimed | none | yes | yes |
| Unclaimed | email provided | yes | yes, verification pending |
| Verified owner | same email | yes | yes |
| Verified owner | none | yes | no |
| Verified owner | different email | yes | no |

Typed emails are not trusted as ownership by themselves. A typed email creates
an `email_verifications` row, and `players.email` becomes authoritative only
after the verification link is opened. If the linked attempt beats the current
leaderboard score, the verification endpoint promotes it with the canonical
score and metrics.

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
   `email_status: "sent"` and the verification link updates
   `email_verified` on `/api/leaderboard`.
6. Submit the verified username with no email or a different email and confirm
   the API returns `ownership_status: "locked"` with no leaderboard update.
7. Submit the verified username with the same email and confirm the leaderboard
   can update if the score improves.
8. Try changing `claimed_score` in a copied packet and confirm the endpoint
   returns `invalid`.
