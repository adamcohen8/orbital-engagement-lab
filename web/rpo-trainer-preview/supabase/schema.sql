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
  validator_version text not null default 'web-two-body-v2',
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

do $$
begin
  if exists (
    select 1
    from pg_class c
    join pg_namespace n on n.oid = c.relnamespace
    where n.nspname = 'public'
      and c.relname = 'public_leaderboard'
      and c.relkind = 'v'
  ) then
    drop view public.public_leaderboard;
  end if;
end $$;

create table if not exists public_leaderboard (
  challenge_id text not null references challenges(id) on delete cascade,
  username text not null,
  score integer not null default 0,
  metrics jsonb not null default '{}'::jsonb,
  attempt_id uuid not null references attempts(id) on delete cascade,
  submitted_at timestamptz not null,
  email_verified boolean not null default false,
  updated_at timestamptz not null default now(),
  primary key (challenge_id, username)
);

create index if not exists public_leaderboard_rank_idx
  on public_leaderboard(challenge_id, score desc, submitted_at asc);

insert into public_leaderboard (
  challenge_id,
  username,
  score,
  metrics,
  attempt_id,
  submitted_at,
  email_verified,
  updated_at
)
select
  le.challenge_id,
  p.username,
  le.score,
  le.metrics,
  a.id as attempt_id,
  a.submitted_at,
  p.email_verified_at is not null as email_verified,
  le.updated_at
from leaderboard_entries le
join players p on p.id = le.player_id
join attempts a on a.id = le.attempt_id
where a.status in ('valid', 'suspicious')
on conflict (challenge_id, username) do update set
  score = excluded.score,
  metrics = excluded.metrics,
  attempt_id = excluded.attempt_id,
  submitted_at = excluded.submitted_at,
  email_verified = excluded.email_verified,
  updated_at = excluded.updated_at;

alter table public_leaderboard enable row level security;
alter table players enable row level security;
alter table challenges enable row level security;
alter table attempts enable row level security;
alter table leaderboard_entries enable row level security;
alter table email_verifications enable row level security;
alter table validator_runs enable row level security;

drop policy if exists "Public leaderboard is readable." on public_leaderboard;
create policy "Public leaderboard is readable."
  on public_leaderboard for select
  to anon, authenticated
  using (true);

drop policy if exists "Active challenges are readable." on challenges;
create policy "Active challenges are readable."
  on challenges for select
  to anon, authenticated
  using (active);

grant select on public_leaderboard to anon, authenticated;
grant select on challenges to anon, authenticated;

revoke all on players from anon, authenticated;
revoke all on attempts from anon, authenticated;
revoke all on leaderboard_entries from anon, authenticated;
revoke all on email_verifications from anon, authenticated;
revoke all on validator_runs from anon, authenticated;

-- Suggested RLS posture once Supabase auth/API keys are wired:
-- 1. Public read access is limited to denormalized public_leaderboard rows.
-- 2. Attempt inserts and leaderboard promotion go through a service-role API
--    endpoint, not direct browser writes.
-- 3. Email fields are never exposed through public policies or public tables.
