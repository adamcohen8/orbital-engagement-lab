"""Stable public facade and CLI for canonical frame and time-scale transformations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.eop import (
    EOP_CONTRACT,
    EopError,
    EopRecord,
    EopSeries,
    audit_eop_series,
    load_iers_eop,
)
from sim.dynamics.orbit.frame_time import (
    FRAME_TIME_CONTRACT,
    FRAME_TRANSFORM_MODEL,
    FRAME_TRANSFORM_MODEL_IAU2006,
    CanonicalFrame,
    EarthOrientation,
    Epoch,
    FrameTimeError,
    FrameTransformContext,
    TimeScale,
    epoch_conversion_receipt,
    epoch_julian_date,
    format_epoch,
    frame_transform_receipt,
    leap_second_table_receipt,
    normalize_canonical_frame,
    parse_epoch,
    state_transform_matrix,
    tai_minus_utc,
    transform_cartesian_state,
    transform_covariance,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.frame_time",
        description="Convert bounded epochs, Cartesian states, and covariances without executing a scenario.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    epoch_parser = commands.add_parser("convert-epoch", help="Convert one epoch between UTC, TAI, TT, and UT1.")
    epoch_parser.add_argument("epoch")
    epoch_parser.add_argument("--from-scale", required=True)
    epoch_parser.add_argument("--to-scale", required=True)
    epoch_parser.add_argument("--dut1-s", type=float)
    epoch_parser.add_argument("--json", action="store_true")

    eop_parser = commands.add_parser("inspect-eop", help="Inspect and freshness-audit an IERS EOP source.")
    eop_parser.add_argument("path", type=Path)
    eop_parser.add_argument("--source-format", choices=("auto", "finals2000a", "c04_csv"), default="auto")
    eop_parser.add_argument("--as-of", help="UTC timestamp for freshness evaluation; defaults to now.")
    eop_parser.add_argument("--max-observed-age-days", type=float, default=45.0)
    eop_parser.add_argument("--json", action="store_true")

    state_parser = commands.add_parser("transform-state", help="Transform one Cartesian position/velocity state.")
    _add_frame_arguments(state_parser)
    state_parser.add_argument("--position-km", nargs=3, type=float, required=True)
    state_parser.add_argument("--velocity-km-s", nargs=3, type=float, required=True)

    covariance_parser = commands.add_parser("transform-covariance", help="Transform one Cartesian 6x6 covariance.")
    _add_frame_arguments(covariance_parser)
    covariance_parser.add_argument("--covariance-json", required=True, type=Path)
    return parser


def _add_frame_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--time-scale", default="UTC")
    parser.add_argument("--source-frame", required=True)
    parser.add_argument("--target-frame", required=True)
    parser.add_argument("--eop", type=Path)
    parser.add_argument("--eop-format", choices=("auto", "finals2000a", "c04_csv"), default="auto")
    parser.add_argument("--json", action="store_true")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "convert-epoch":
            epoch = parse_epoch(args.epoch, args.from_scale, dut1_s=args.dut1_s)
            payload = epoch_conversion_receipt(epoch, args.to_scale, dut1_s=args.dut1_s)
        elif args.command == "inspect-eop":
            series = load_iers_eop(args.path, source_format=args.source_format)
            as_of = _as_of(args.as_of)
            payload = audit_eop_series(
                series,
                as_of=as_of,
                max_observed_age_days=args.max_observed_age_days,
            )
        else:
            context, eop_receipt = _cli_context(args)
            if args.command == "transform-state":
                position, velocity = transform_cartesian_state(
                    args.position_km,
                    args.velocity_km_s,
                    args.source_frame,
                    args.target_frame,
                    context=context,
                )
                result: dict[str, Any] = {
                    "position_km": position.tolist(),
                    "velocity_km_s": velocity.tolist(),
                }
            else:
                covariance = np.asarray(json.loads(args.covariance_json.read_text(encoding="utf-8")), dtype=float)
                result = {
                    "covariance": transform_covariance(
                        covariance,
                        args.source_frame,
                        args.target_frame,
                        context=context,
                    ).tolist()
                }
            payload = {
                "schema": "oel.frame-time-cli-result.v1",
                "status": "converted",
                "result": result,
                "transform": frame_transform_receipt(
                    args.source_frame,
                    args.target_frame,
                    context=context,
                ),
                "eop_source": eop_receipt,
                "execution_occurred": False,
            }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_summary(payload)
        return 0 if payload.get("status") not in {"fail"} else 2
    except (OSError, ValueError, json.JSONDecodeError, EopError, FrameTimeError) as exc:
        print(f"frame/time command failed: {exc}", file=sys.stderr)
        return 2


def _cli_context(args: argparse.Namespace) -> tuple[FrameTransformContext, dict[str, Any] | None]:
    time_scale = TimeScale(str(args.time_scale).upper())
    series = None if args.eop is None else load_iers_eop(args.eop, source_format=args.eop_format)
    if time_scale is TimeScale.UT1 and series is None:
        raise FrameTimeError("UT1 input requires --eop so DUT1 is epoch-matched.")
    if time_scale is TimeScale.UT1:
        provisional = parse_epoch(args.epoch, time_scale, dut1_s=0.0)
        provisional_eop = series.sample(provisional)
        epoch = parse_epoch(args.epoch, time_scale, dut1_s=provisional_eop.dut1_s)
    else:
        epoch = parse_epoch(args.epoch, time_scale)
    eop = None if series is None else series.sample(epoch)
    return FrameTransformContext(epoch=epoch, earth_orientation=eop), None if series is None else series.receipt()


def _as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _print_summary(payload: dict[str, Any]) -> None:
    if "output" in payload:
        print(f"{payload['output']['text']} {payload['output']['scale']}")
    elif "result" in payload:
        print(json.dumps(payload["result"], indent=2, sort_keys=True))
    else:
        freshness = payload.get("eop", {}).get("freshness", {})
        print(f"status: {payload.get('status')}")
        if freshness:
            print(f"freshness: {freshness.get('status')}")

__all__ = [
    "FRAME_TIME_CONTRACT",
    "FRAME_TRANSFORM_MODEL",
    "FRAME_TRANSFORM_MODEL_IAU2006",
    "EOP_CONTRACT",
    "CanonicalFrame",
    "EarthOrientation",
    "EopError",
    "EopRecord",
    "EopSeries",
    "Epoch",
    "FrameTimeError",
    "FrameTransformContext",
    "TimeScale",
    "epoch_conversion_receipt",
    "epoch_julian_date",
    "format_epoch",
    "frame_transform_receipt",
    "leap_second_table_receipt",
    "normalize_canonical_frame",
    "parse_epoch",
    "state_transform_matrix",
    "tai_minus_utc",
    "transform_cartesian_state",
    "transform_covariance",
    "build_parser",
    "audit_eop_series",
    "load_iers_eop",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
