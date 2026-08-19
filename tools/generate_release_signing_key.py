"""Generate a dedicated RSA key pair for OEL release metadata signing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.installation.signing import generate_rsa_private_key, write_key_files  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an OEL release-signing key pair.")
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=3072)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    key = generate_rsa_private_key(args.key_id, bits=args.bits)
    write_key_files(
        key,
        private_output=args.private_output,
        public_output=args.public_output,
        force=args.force,
    )
    print(f"Wrote private release-signing key: {args.private_output}")
    print(f"Wrote trusted release-key registry: {args.public_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
