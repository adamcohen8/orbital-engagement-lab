-- Browser Pursuit Arcade leaderboard schema.
-- Intended for Supabase/Postgres. Keep private emails out of public views.

create table if not exists players (
  id uuid primary key default gen_random_uuid(),
  username text not null,
  username_normalized text generated always as (lower(regexp_replace(trim(username), '\s+', '', 'g'))) stored,
  email text,
  email_verified_at timestamptz,
  username_locked_at timestamptz,
  created_at timestamptz not null default now(),
  unique (username_normalized)
);

alter table if exists players
  add column if not exists username_locked_at timestamptz;

create table if not exists challenges (
  id text primary key,
  title text not null,
  starts_at timestamptz,
  ends_at timestamptz,
  physics_version text not null,
  scoring_version text not null,
  config_hash text not null,
  config jsonb not null,
  active boolean not null default false,
  created_at timestamptz not null default now()
);

create table if not exists attempts (
  id uuid primary key default gen_random_uuid(),
  player_id uuid not null references players(id) on delete cascade,
  challenge_id text not null references challenges(id) on delete cascade,
  status text not null check (status in ('pending', 'valid', 'invalid', 'suspicious', 'hidden')),
  score integer not null default 0,
  metrics jsonb not null default '{}'::jsonb,
  replay jsonb not null,
  config_hash text not null,
  physics_version text not null,
  scoring_version text not null,
  validator_version text not null default 'web-two-body-v1',
  validation_errors text[] not null default '{}',
  validation_warnings text[] not null default '{}',
  ri_plot_svg text,
  rc_plot_svg text,
  submitted_at timestamptz not null default now(),
  validated_at timestamptz
);

create index if not exists attempts_challenge_score_idx
  on attempts(challenge_id, status, score desc, submitted_at asc);

create table if not exists leaderboard_entries (
  challenge_id text not null references challenges(id) on delete cascade,
  player_id uuid not null references players(id) on delete cascade,
  attempt_id uuid not null references attempts(id) on delete cascade,
  score integer not null,
  metrics jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now(),
  primary key (challenge_id, player_id)
);

create index if not exists leaderboard_entries_rank_idx
  on leaderboard_entries(challenge_id, score desc, updated_at asc);

create table if not exists email_verifications (
  id uuid primary key default gen_random_uuid(),
  player_id uuid not null references players(id) on delete cascade,
  attempt_id uuid references attempts(id) on delete set null,
  email text not null,
  token_hash text not null,
  expires_at timestamptz not null,
  verified_at timestamptz,
  created_at timestamptz not null default now()
);

create table if not exists validator_runs (
  id uuid primary key default gen_random_uuid(),
  attempt_id uuid references attempts(id) on delete cascade,
  challenge_id text not null,
  status text not null,
  validator_version text not null,
  runtime_ms integer,
  errors text[] not null default '{}',
  warnings text[] not null default '{}',
  created_at timestamptz not null default now()
);

create or replace view public_leaderboard as
select
  le.challenge_id,
  p.username,
  le.score,
  le.metrics,
  a.id as attempt_id,
  a.submitted_at,
  p.email_verified_at is not null as email_verified
from leaderboard_entries le
join players p on p.id = le.player_id
join attempts a on a.id = le.attempt_id
where a.status in ('valid', 'suspicious');

-- Suggested RLS posture once Supabase auth/API keys are wired:
-- 1. Public read access only to public_leaderboard and active challenges.
-- 2. Attempt inserts go through a service-role API endpoint, not direct browser writes.
-- 3. Email fields are never exposed through public policies or views.
