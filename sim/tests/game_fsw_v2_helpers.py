from __future__ import annotations

from dataclasses import replace

from sim.flight_software import (
    FrameId,
    GameAerodynamicEffectorBinding,
    GamePilotInputProfile,
    GamePilotMode,
    GamePilotReferenceStackConfig,
)
from sim.gnc.attitude_v2 import AttitudeAllocatorConfig, AttitudeAllocatorKind
from sim.gnc.orbit_v2 import TranslationAllocatorConfig, TranslationAllocatorKind
from sim.tests.fsw_v2_helpers import ACTUATOR_FRAME, BODY_FRAME, INERTIAL_FRAME, SATELLITE_ID
from sim.tests.fsw_v2_orbit_helpers import RELATIVE_FRAME

TRANSLATION_FRAME = FrameId("OEL/ACTUATOR/sat/translation", "frames-v1")
DEPLOYMENT_FRAME = FrameId("OEL/ACTUATOR/sat/deployment", "frames-v1")
BANK_FRAME = FrameId("OEL/ACTUATOR/sat/bank", "frames-v1")


def game_stack_config(mode: GamePilotMode) -> GamePilotReferenceStackConfig:
    profile = GamePilotInputProfile("game-profile-v1", mode)
    effectors = (
        GameAerodynamicEffectorBinding(
            "deployment", "deployment", "deployment", DEPLOYMENT_FRAME, "fraction", 0.0, 1.0, 0.5
        ),
        GameAerodynamicEffectorBinding("bank", "bank", "bank", BANK_FRAME, "rad", -1.0, 1.0, 0.0),
    )
    return GamePilotReferenceStackConfig(
        SATELLITE_ID,
        BODY_FRAME,
        INERTIAL_FRAME,
        RELATIVE_FRAME,
        profile,
        TranslationAllocatorConfig(
            SATELLITE_ID,
            TranslationAllocatorKind.IDEAL_WRENCH,
            "translation",
            TRANSLATION_FRAME,
            2.0,
        ),
        100.0,
        0.01,
        attitude_allocator=(
            AttitudeAllocatorConfig(
                SATELLITE_ID,
                AttitudeAllocatorKind.IDEAL_WRENCH,
                "attitude",
                ACTUATOR_FRAME,
            )
            if mode is GamePilotMode.ATTITUDE_THRUST
            else None
        ),
        effectors=effectors if mode is GamePilotMode.AERODYNAMIC else (),
    )


def with_profile(config: GamePilotReferenceStackConfig, profile: GamePilotInputProfile) -> GamePilotReferenceStackConfig:
    return replace(config, profile=profile)
