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
  reservation_id uuid,
  expires_at timestamptz not null,
  verified_at timestamptz,
  created_at timestamptz not null default now()
);

alter table if exists email_verifications
  add column if not exists reservation_id uuid;

create unique index if not exists email_verifications_token_hash_idx
  on email_verifications(token_hash);

-- Recipient-wide mail budget. The primary key plus conditional expiry PATCH
-- serializes sends across concurrent usernames and serverless instances.
create table if not exists verification_send_locks (
  email text primary key,
  reservation_id uuid not null,
  reserved_until timestamptz not null,
  created_at timestamptz not null default now()
);

begin;
-- Pre-lock recently sent legacy links during rollout. Older mixed-case
-- recipient values are normalized before the new API's lowercase keying.
update email_verifications set email = lower(trim(email))
  where email <> lower(trim(email));

with latest as (
  select distinct on (email) id
  from email_verifications
  where verified_at is null and expires_at > now()
    and created_at > now() - interval '1 hour'
  order by email, created_at desc, id desc
)
update email_verifications ev
  set reservation_id = coalesce(ev.reservation_id, gen_random_uuid())
  from latest where ev.id = latest.id;

insert into verification_send_locks (email, reservation_id, reserved_until)
select ev.email, ev.reservation_id, ev.created_at + interval '1 hour'
from email_verifications ev
where ev.id in (
  select distinct on (email) id
  from email_verifications
  where verified_at is null and expires_at > now()
    and created_at > now() - interval '1 hour'
  order by email, created_at desc, id desc
)
on conflict (email) do nothing;

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

-- Legacy API accepted a matching email string as authority for a locked
-- username. Rebuild both score tables from consumed tokens for the player's
-- currently verified email; an unbound legacy winner is never grandfathered
-- into a verified badge or allowed to block a lower genuine token claim.
delete from leaderboard_entries le using players p
  where le.player_id = p.id and p.email_verified_at is not null;

insert into leaderboard_entries
  (challenge_id, player_id, attempt_id, score, metrics, updated_at)
select distinct on (a.challenge_id, a.player_id)
  a.challenge_id, a.player_id, a.id, a.score, a.metrics, ev.verified_at
from email_verifications ev
join players p on p.id = ev.player_id
join attempts a on a.id = ev.attempt_id and a.player_id = p.id
where p.email_verified_at is not null
  and lower(trim(p.email)) = lower(trim(ev.email))
  and ev.verified_at is not null
  and a.status in ('valid', 'suspicious')
order by a.challenge_id, a.player_id, a.score desc, a.submitted_at asc, a.id asc;

-- Public rows are derived. Rebuild all of them so stale verified badges and
-- rows orphaned by a partial earlier write do not survive the migration.
delete from public_leaderboard;
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
commit;

alter table public_leaderboard enable row level security;
alter table players enable row level security;
alter table challenges enable row level security;
alter table attempts enable row level security;
alter table leaderboard_entries enable row level security;
alter table email_verifications enable row level security;
alter table verification_send_locks enable row level security;
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
revoke all on verification_send_locks from anon, authenticated;
revoke all on validator_runs from anon, authenticated;

-- These functions are the only leaderboard write paths used by the hosted API.
-- Both lock the player row, so a pending unclaimed submission cannot race a
-- token claim and subsequently overwrite a verified player's public score.
create or replace function publish_unclaimed_arcade_attempt(p_attempt_id uuid)
returns boolean
language plpgsql
security invoker
set search_path = public
as $$
declare
  v_player players%rowtype;
  v_attempt attempts%rowtype;
  v_winner record;
  v_changed boolean := false;
begin
  select p.* into v_player
    from players p join attempts a on a.player_id = p.id
    where a.id = p_attempt_id
    for update of p;
  if not found or v_player.email_verified_at is not null
      or v_player.username_locked_at is not null then
    return false;
  end if;
  select * into v_attempt from attempts where id = p_attempt_id;
  if not found or v_attempt.status not in ('valid', 'suspicious') then
    return false;
  end if;

  insert into leaderboard_entries
    (challenge_id, player_id, attempt_id, score, metrics)
  values
    (v_attempt.challenge_id, v_player.id, v_attempt.id, v_attempt.score, v_attempt.metrics)
  on conflict (challenge_id, player_id) do update set
    attempt_id = excluded.attempt_id,
    score = excluded.score,
    metrics = excluded.metrics,
    updated_at = now()
  where leaderboard_entries.score < excluded.score;
  v_changed := found;

  select le.score, le.metrics, a.id as attempt_id, a.submitted_at
    into v_winner
    from leaderboard_entries le join attempts a on a.id = le.attempt_id
    where le.challenge_id = v_attempt.challenge_id and le.player_id = v_player.id;
  if not found then
    return false;
  end if;
  insert into public_leaderboard
    (challenge_id, username, score, metrics, attempt_id, submitted_at, email_verified)
  values
    (v_attempt.challenge_id, v_player.username, v_winner.score, v_winner.metrics,
     v_winner.attempt_id, v_winner.submitted_at, false)
  on conflict (challenge_id, username) do update set
    score = excluded.score,
    metrics = excluded.metrics,
    attempt_id = excluded.attempt_id,
    submitted_at = excluded.submitted_at,
    email_verified = false,
    updated_at = now();
  return v_changed;
end;
$$;

create or replace function claim_arcade_attempt(p_token_hash text)
returns jsonb
language plpgsql
security invoker
set search_path = public
as $$
declare
  v_ver email_verifications%rowtype;
  v_player players%rowtype;
  v_attempt attempts%rowtype;
  v_winner record;
  v_first_claim boolean := false;
  v_promoted boolean := false;
  v_now timestamptz := now();
begin
  if p_token_hash !~ '^[0-9a-f]{64}$' then
    return jsonb_build_object('status', 'not_found');
  end if;
  select * into v_ver from email_verifications
    where token_hash = p_token_hash for update;
  if not found then
    return jsonb_build_object('status', 'not_found');
  end if;
  if v_ver.verified_at is not null then
    return jsonb_build_object('status', 'already_verified');
  end if;
  if v_ver.expires_at <= v_now then
    return jsonb_build_object('status', 'expired');
  end if;
  select * into v_player from players where id = v_ver.player_id for update;
  if not found then
    return jsonb_build_object('status', 'not_found');
  end if;
  if v_player.email_verified_at is null then
    if v_player.username_locked_at is not null then
      return jsonb_build_object('status', 'locked');
    end if;
    v_first_claim := true;
  elsif lower(coalesce(v_player.email, '')) <> lower(v_ver.email) then
    return jsonb_build_object('status', 'locked');
  end if;
  select * into v_attempt from attempts
    where id = v_ver.attempt_id and player_id = v_ver.player_id;
  v_promoted := found and v_attempt.status in ('valid', 'suspicious');

  if v_first_claim then
    update players set
      email = lower(v_ver.email),
      email_verified_at = v_now,
      username_locked_at = v_now
      where id = v_player.id;
    -- Every pre-claim score was anonymous and could have been submitted by
    -- someone else. Clear all challenges, including a username-only claim
    -- whose linked attempt was deleted, before any verified score is added.
    delete from public_leaderboard where username = v_player.username;
    delete from leaderboard_entries where player_id = v_player.id;
  end if;
  update email_verifications set verified_at = v_now where id = v_ver.id;
  if v_promoted then
    insert into leaderboard_entries
    (challenge_id, player_id, attempt_id, score, metrics)
  values
    (v_attempt.challenge_id, v_player.id, v_attempt.id, v_attempt.score, v_attempt.metrics)
  on conflict (challenge_id, player_id) do update set
    attempt_id = excluded.attempt_id,
    score = excluded.score,
    metrics = excluded.metrics,
    updated_at = now()
  where v_first_claim or leaderboard_entries.score < excluded.score;
    v_promoted := found;

  -- The first claim replaces any higher provisional score. Otherwise a
  -- lower-scoring token cannot badge another attempt as its own.
    select le.score, le.metrics, a.id as attempt_id, a.submitted_at
    into v_winner
    from leaderboard_entries le join attempts a on a.id = le.attempt_id
    where le.challenge_id = v_attempt.challenge_id and le.player_id = v_player.id;
    insert into public_leaderboard
    (challenge_id, username, score, metrics, attempt_id, submitted_at, email_verified)
  values
    (v_attempt.challenge_id, v_player.username, v_winner.score, v_winner.metrics,
     v_winner.attempt_id, v_winner.submitted_at, true)
  on conflict (challenge_id, username) do update set
    score = excluded.score,
    metrics = excluded.metrics,
    attempt_id = excluded.attempt_id,
    submitted_at = excluded.submitted_at,
    email_verified = true,
    updated_at = now();
  end if;
  -- Only the token that acquired this recipient reservation may clear it.
  -- An old link cannot erase a newer send lock after an expiry takeover.
  if v_ver.reservation_id is not null then
    delete from verification_send_locks
      where email = lower(v_ver.email)
        and reservation_id = v_ver.reservation_id;
  end if;
  return jsonb_build_object('status', 'verified', 'promoted', v_promoted);
end;
$$;

revoke all on function publish_unclaimed_arcade_attempt(uuid) from public, anon, authenticated;
revoke all on function claim_arcade_attempt(text) from public, anon, authenticated;
grant execute on function publish_unclaimed_arcade_attempt(uuid) to service_role;
grant execute on function claim_arcade_attempt(text) to service_role;

-- Suggested RLS posture once Supabase auth/API keys are wired:
-- 1. Public read access is limited to denormalized public_leaderboard rows.
-- 2. Attempt inserts and leaderboard promotion go through a service-role API
--    endpoint, not direct browser writes.
-- 3. Email fields are never exposed through public policies or public tables.
