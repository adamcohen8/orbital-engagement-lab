#!/bin/sh
set -eu

DEFAULT_BASE_URL="__OEL_DEFAULT_BASE_URL__"
DEFAULT_CHANNEL_URL="__OEL_DEFAULT_CHANNEL_URL__"
BOOTSTRAP_SHA256="__OEL_BOOTSTRAP_SHA256__"
INSTALLER_RENDERED="__OEL_INSTALLER_RENDERED__"
BASE_URL="${OEL_INSTALL_BASE_URL:-$DEFAULT_BASE_URL}"
CHANNEL_URL="${OEL_UPDATE_CHANNEL_URL:-$DEFAULT_CHANNEL_URL}"
PROFILE="${OEL_INSTALL_PROFILE:-core}"

if [ "$INSTALLER_RENDERED" != "true" ] && { [ -z "${OEL_INSTALL_BASE_URL:-}" ] || [ -z "${OEL_UPDATE_CHANNEL_URL:-}" ]; }; then
  echo "This installer template has not been rendered for a release. Set OEL_INSTALL_BASE_URL and OEL_UPDATE_CHANNEL_URL, or use a released installer." >&2
  exit 2
fi

PYTHON=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,15) else 1)' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  fi
done
if [ -z "$PYTHON" ]; then
  echo "OEL requires CPython >=3.10,<3.15. Install a supported Python and rerun this command." >&2
  exit 2
fi

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/oel-install.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT HUP INT TERM
BOOTSTRAP="$TEMP_DIR/bootstrap_install.py"
curl --proto '=https' --tlsv1.2 -fsSL "$BASE_URL/bootstrap_install.py" -o "$BOOTSTRAP"

if [ "$INSTALLER_RENDERED" != "true" ]; then
  echo "Rendered bootstrap digest is missing." >&2
  exit 2
fi
"$PYTHON" -c 'import hashlib, pathlib, sys; p=pathlib.Path(sys.argv[1]); raise SystemExit(0 if hashlib.sha256(p.read_bytes()).hexdigest() == sys.argv[2] else 1)' "$BOOTSTRAP" "$BOOTSTRAP_SHA256" || {
  echo "OEL bootstrap SHA-256 verification failed." >&2
  exit 2
}

"$PYTHON" "$BOOTSTRAP" --manifest-url "$BASE_URL/release-manifest.json" --channel-url "$CHANNEL_URL" --profile "$PROFILE" "$@"
