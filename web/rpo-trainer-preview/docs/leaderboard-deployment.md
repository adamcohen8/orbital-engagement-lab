# Pursuit Arcade Leaderboard Deployment

This is the small-hosting path for a beta leaderboard with roughly 100 users per
week.

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
updates `players.email_verified_at`; public leaderboard reads expose only the
boolean `email_verified`, never the email address.

## First Production Check

After deployment:

1. Play Pursuit Arcade.
2. Submit an attempt through the hosted page.
3. Confirm `/api/leaderboard` returns the row.
4. Confirm the Supabase `attempts` row has status `valid` or `suspicious`.
5. If email is configured, confirm the submit response includes
   `email_status: "sent"` and the verification link updates
   `email_verified` on `/api/leaderboard`.
6. Try changing `claimed_score` in a copied packet and confirm the endpoint
   returns `invalid`.
