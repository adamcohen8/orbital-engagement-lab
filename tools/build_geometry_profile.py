#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.dynamics.spacecraft_geometry import GeometryAreaProfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an OEL attitude-dependent projected-area profile from a body-frame STL mesh. "
            "The generated JSON can be referenced by satellite specs.geometry.profile_path."
        )
    )
    parser.add_argument("stl", help="Input body-frame STL mesh. Vertex units are interpreted as meters.")
    parser.add_argument("-o", "--output", required=True, help="Output geometry area profile JSON path.")
    parser.add_argument(
        "--samples",
        type=int,
        default=642,
        help="Number of Fibonacci-sphere sample directions before adding body axes. Default: 642.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Optional profile display name stored in metadata.",
    )
    parser.add_argument(
        "--no-body-axes",
        action="store_true",
        help="Do not force +/- body axes into the sample directions.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact JSON summary after writing the profile.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = {}
    if args.name.strip():
        metadata["name"] = args.name.strip()
    profile = GeometryAreaProfile.from_stl(
        args.stl,
        sample_count=int(args.samples),
        include_body_axes=not bool(args.no_body_axes),
        metadata=metadata,
    )
    out = profile.save(args.output)
    if args.summary:
        summary = {
            "output": str(Path(out).resolve()),
            "directions": int(profile.directions_body.shape[0]),
            "area_min_m2": float(profile.projected_area_m2.min()),
            "area_max_m2": float(profile.projected_area_m2.max()),
            "area_mean_m2": float(profile.projected_area_m2.mean()),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
