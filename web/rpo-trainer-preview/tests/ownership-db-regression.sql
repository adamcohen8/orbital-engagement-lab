-- Run after supabase/schema.sql in a disposable Postgres/Supabase database.
-- The transaction rolls back all fixtures. This exercises the real claim RPC;
-- the JavaScript mock tests cannot verify its row-lock and score semantics.
begin;

do $$
declare
  v_player uuid;
  v_attempt uuid;
  v_challenge text := 'qa-' || gen_random_uuid()::text;
  v_username text := 'qa' || replace(gen_random_uuid()::text, '-', '');
  v_email text := 'qa+' || gen_random_uuid()::text || '@example.invalid';
  v_first_token text := md5(gen_random_uuid()::text) || md5(gen_random_uuid()::text);
  v_second_token text := md5(gen_random_uuid()::text) || md5(gen_random_uuid()::text);
  v_result jsonb;
begin
  insert into challenges (id, title, physics_version, scoring_version, config_hash, config)
    values (v_challenge, 'Claim regression', 'test', 'test', 'test', '{}'::jsonb);
  insert into players (username) values (v_username) returning id into v_player;
  insert into attempts
    (player_id, challenge_id, status, score, replay, config_hash,
     physics_version, scoring_version)
    values (v_player, v_challenge, 'valid', 999, '{}'::jsonb,
            'test', 'test', 'test')
    returning id into v_attempt;
  insert into leaderboard_entries (challenge_id, player_id, attempt_id, score)
    values (v_challenge, v_player, v_attempt, 999);
  insert into public_leaderboard
    (challenge_id, username, score, attempt_id, submitted_at)
    values (v_challenge, v_username, 999, v_attempt, now());

  -- Legacy email tokens can lose their attempt through ON DELETE SET NULL.
  -- Claiming such a token reserves the username but must discard every
  -- anonymous score, so a later genuine lower score cannot badge the old one.
  insert into email_verifications
    (player_id, attempt_id, email, token_hash, expires_at)
    values (v_player, null, v_email, v_first_token, now() + interval '1 day');
  v_result := claim_arcade_attempt(v_first_token);
  if v_result->>'status' is distinct from 'verified'
     or (v_result->>'promoted')::boolean is distinct from false then
    raise exception 'username-only first claim returned %', v_result;
  end if;
  if exists (select 1 from leaderboard_entries where player_id = v_player)
     or exists (select 1 from public_leaderboard where username = v_username) then
    raise exception 'anonymous provisional winner survived username-only claim';
  end if;

  insert into attempts
    (player_id, challenge_id, status, score, replay, config_hash,
     physics_version, scoring_version)
    values (v_player, v_challenge, 'valid', 10, '{}'::jsonb,
            'test', 'test', 'test')
    returning id into v_attempt;
  insert into email_verifications
    (player_id, attempt_id, email, token_hash, expires_at)
    values (v_player, v_attempt, v_email, v_second_token, now() + interval '1 day');
  v_result := claim_arcade_attempt(v_second_token);
  if v_result->>'status' is distinct from 'verified'
     or (v_result->>'promoted')::boolean is distinct from true then
    raise exception 'exact second claim returned %', v_result;
  end if;
  if not exists (
    select 1 from leaderboard_entries
    where player_id = v_player and attempt_id = v_attempt and score = 10
  ) or not exists (
    select 1 from public_leaderboard
    where username = v_username and attempt_id = v_attempt
      and score = 10 and email_verified
  ) then
    raise exception 'verified public winner is not the exact lower token attempt';
  end if;
end $$;

rollback;
