# Canonical Frames, Time Scales, And Covariance Transforms

OEL exposes a bounded public frame/time contract through `sim.frame_time`.
This is the supported interoperability surface for converting an epoch, a
Cartesian state, or its complete 6x6 covariance without guessing what `ECI`,
`J2000`, or `ECEF` means.

The contract identifier is `oel.frame-time.v1`. EME2000/TEME/ITRF retains the
existing OEL IAU-76/FK5 + IAU-80 path. GCRF/ITRF uses ERFA's IAU 2006/2000A
CIO chain with explicit IERS Earth-orientation values; GCRF is not silently
treated as EME2000.

## Supported Contract

Time scales:

- UTC, including positive leap-second notation such as
  `2016-12-31T23:59:60`;
- TAI;
- TT, with `TT - TAI = 32.184 s`; and
- sampled UT1, only when the caller supplies epoch-matched `DUT1`.

The packaged IERS Bulletin C table covers 1972-01-01 through the end of
2026. UTC input outside that interval fails closed so a stale table cannot
silently produce a plausible epoch.

Frames:

- EME2000 and its exact OEL alias `OEL/ECI/J2000`;
- TEME under OEL's Vallado IAU-80 state-vector contract;
- ITRF and its exact OEL alias `OEL/ECEF/IAU76_80_EOP`; and
- GCRF, including frame-bias conversion with EME2000 and IAU 2006/2000A
  conversion with ITRF.

Generic `ECI`, `ECEF`, `J2000`, and `ITRF2000` labels are rejected.

## Epoch Conversion

```python
from sim.frame_time import format_epoch, parse_epoch

epoch = parse_epoch("2016-12-31T23:59:60", "UTC")
assert format_epoch(epoch, "TAI") == "2017-01-01T00:00:36"
assert format_epoch(epoch, "TT") == "2017-01-01T00:01:08.184"
```

UT1 conversion requires an explicit sampled correction:

```python
epoch = parse_epoch("2024-01-01T00:00:00", "UTC")
ut1 = format_epoch(epoch, "UT1", dut1_s=0.0087572)
```

`load_iers_eop()` reads local IERS finals2000A fixed-column data or a bounded
C04-style CSV, records its SHA-256 identity, linearly interpolates within
coverage, and refuses extrapolation. `audit_eop_series()` reports observed
coverage, prediction presence, and freshness/expiry. OEL does not download or
predict EOP values; the caller still owns source selection and refresh policy.

CLI/JSON examples:

```bash
python -m sim.frame_time convert-epoch 2024-01-01T00:00:00 \
  --from-scale UTC --to-scale TAI --json
python -m sim.frame_time inspect-eop path/to/finals2000A.all \
  --source-format finals2000a --json
python -m sim.frame_time transform-state \
  --epoch 2024-01-01T00:00:00 --source-frame GCRF --target-frame ITRF \
  --position-km 7000 120 30 --velocity-km-s -0.2 7.45 1.1 \
  --eop path/to/finals2000A.all --json
```

## State And Covariance Conversion

```python
import numpy as np

from sim.frame_time import (
    EarthOrientation,
    FrameTransformContext,
    parse_epoch,
    transform_cartesian_state,
    transform_covariance,
)

context = FrameTransformContext(
    epoch=parse_epoch("2024-01-01T00:00:00", "UTC"),
    earth_orientation=EarthOrientation(
        dut1_s=0.0087572,
        xp_arcsec=0.136928,
        yp_arcsec=0.202199,
        ddpsi_rad=-5.0e-8,
        ddeps_rad=3.0e-8,
        source="epoch-matched IERS sample",
        source_sha256="0" * 64,
    ),
)

position_itrf_km, velocity_itrf_km_s = transform_cartesian_state(
    [7000.0, 120.0, 30.0],
    [-0.2, 7.45, 1.1],
    "EME2000",
    "ITRF",
    context=context,
)

covariance_itrf = transform_covariance(
    np.diag([4e-4, 9e-4, 1.6e-3, 4e-10, 9e-10, 1.6e-9]),
    "EME2000",
    "ITRF",
    context=context,
)
```

The state Jacobian includes the Earth-rotation position-to-velocity coupling.
Covariance follows `P_target = J P_source J^T`. Input must be a finite,
symmetric, positive-semidefinite 6x6 matrix ordered as
`[x, y, z, vx, vy, vz]`.

`epoch_conversion_receipt()` and `frame_transform_receipt()` produce portable
JSON-ready provenance that binds the contract, epoch, leap-second table,
frames, model, and supplied EOP source.

## Validation

Routine replay is fully offline:

```bash
python -m pytest -q sim/tests/test_frame_time.py
```

The retained manifest at
`sim/dynamics/orbit/data/frame_time_validation_manifest.json` binds the IERS
leap-second and EOP sources, current frame/time implementation modules,
validator, pinned Orekit 13.1.7 runtime identity, and an independent
state/Jacobian/covariance comparison. Routine tests recompute the retained GCRF
residuals from the current implementation rather than trusting only the saved
pass/fail booleans. The frozen comparison
envelopes are 2 m in position, 0.003 m/s in velocity, `2e-7` for Jacobian
elements, and `2e-6` for normalized covariance residual on the legacy path.
The GCRF/ITRF envelope is 0.25 m, 0.0005 m/s, and `5e-8` for Jacobian
elements; the retained 2024 sample is well inside those limits.

Maintainers with the pinned open-source reference bundle may refresh it:

```bash
python -m tools.validate_frame_time \
  --output sim/dynamics/orbit/data/frame_time_orekit_13_1_7_acceptance.json \
  --orekit-root path/to/pinned/orekit
```

No STK, ODTK, or paid service is required. Agreement with Orekit is
independent-implementation evidence for the one retained epoch, state, EOP
inputs, and tolerance envelope. It is not arbitrary-epoch or arbitrary-EOP
equivalence, flight qualification, EOP prediction, or covariance calibration.
