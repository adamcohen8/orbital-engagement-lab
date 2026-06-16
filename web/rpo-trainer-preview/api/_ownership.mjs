import { normalizeEmail } from "./_supabase.mjs";

export const OWNERSHIP_STATUS = {
  UNCLAIMED: "unclaimed",
  PENDING_VERIFICATION: "pending_verification",
  VERIFIED_OWNER: "verified_owner",
  LOCKED: "locked",
};

export function usernameIsLocked(player) {
  const storedEmail = ownershipEmail(player?.email);
  return Boolean(player?.username_locked_at || (storedEmail && player?.email_verified_at));
}

export function decideOwnership({ player, email }) {
  const providedEmail = ownershipEmail(email);
  const storedEmail = ownershipEmail(player?.email);
  const locked = usernameIsLocked(player);

  if (!locked && !providedEmail) {
    return {
      status: OWNERSHIP_STATUS.UNCLAIMED,
      leaderboard_allowed: true,
      verification_allowed: false,
    };
  }

  if (!locked && providedEmail) {
    return {
      status: OWNERSHIP_STATUS.PENDING_VERIFICATION,
      leaderboard_allowed: true,
      verification_allowed: true,
    };
  }

  if (locked && storedEmail && providedEmail === storedEmail) {
    return {
      status: OWNERSHIP_STATUS.VERIFIED_OWNER,
      leaderboard_allowed: true,
      verification_allowed: true,
    };
  }

  return {
    status: OWNERSHIP_STATUS.LOCKED,
    leaderboard_allowed: false,
    verification_allowed: false,
  };
}

export function canVerifyUsernameForEmail({ player, email }) {
  const providedEmail = ownershipEmail(email);
  const storedEmail = ownershipEmail(player?.email);
  if (!providedEmail) return false;
  if (!usernameIsLocked(player)) return true;
  return Boolean(storedEmail && providedEmail === storedEmail);
}

function ownershipEmail(email) {
  return normalizeEmail(email).toLowerCase();
}
