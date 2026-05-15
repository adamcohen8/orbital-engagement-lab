#!/usr/bin/env python
from __future__ import annotations

import argparse

from sim.config.help import format_config_help, format_config_help_list, load_config_help_context


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show valid options and descriptions for Orbital Engagement Lab YAML config fields."
    )
    parser.add_argument("query", nargs="*", help='Field/topic to look up, for example: "ephemeris model".')
    parser.add_argument("--list", action="store_true", help="List known config help topics.")
    parser.add_argument(
        "--config",
        help="Optional YAML config to inspect for the matched field's current value. The file is parsed as data only.",
    )
    parser.add_argument(
        "--scope",
        choices=("public", "pro", "all"),
        default=None,
        help="Limit topics by distribution surface. Defaults to all in the private tree and public in public exports.",
    )
    args = parser.parse_args()

    kwargs = {} if args.scope is None else {"scope": args.scope}
    if args.list:
        print(format_config_help_list(**kwargs))
        return 0

    query = " ".join(args.query).strip()
    if not query:
        parser.error('provide a field/topic query or use --list, for example: python config_help.py "ephemeris model"')

    config_data = load_config_help_context(args.config) if args.config else None
    print(format_config_help(query, config_data=config_data, config_path=args.config, **kwargs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
