# CCSDS TDM Tracking Orbit Determination

OEL provides a bounded public workflow that turns one CCSDS Tracking Data
Message (TDM) into normalized measurements, a batch orbit solution, and
separate fit and holdout evidence. It is an inspectable engineering workflow,
not a general TDM implementation or operational orbit-determination system.

## Quickstart

Inspect the supplied TDM before estimation:

```bash
python -m sim.tracking_od inspect-tdm \
  examples/tracking_od/public_reduced_geometric_azel_range.tdm
```

Exercise deterministic semantic parsing and canonical serialization:

```bash
python -m sim.tracking_od roundtrip-tdm \
  examples/tracking_od/public_reduced_geometric_azel_range.tdm \
  /tmp/oel-canonical.tdm
```

Fit the declared arc and predict the retained holdout:

```bash
python -m sim.tracking_od fit \
  examples/tracking_od/public_reduced_geometric_azel_range.tdm \
  examples/tracking_od/public_tdm_fit_holdout_problem.json \
  --output-dir outputs/public_tdm_fit_holdout
```

The command prints `tracking_od_evidence.json`. That packet binds the problem,
source TDM, normalized dataset, estimator result, holdout prediction ledger,
and generated artifacts by digest.

## Supported Public Profile

The `oel.ccsds-tdm-kvn.v0.1` parser accepts a deliberately narrow subset of
CCSDS 503.0-B-2 Technical Corrigendum 1:

- TDM version 2.0 KVN with printable ASCII and bounded input size;
- UTC epochs and sequential `PATH = 2,1` segments;
- one station as `PARTICIPANT_1` and one tracked object as `PARTICIPANT_2`;
- `ANGLE_TYPE = AZEL`, where `ANGLE_1` is azimuth and `ANGLE_2` is elevation
  in degrees; and
- unambiguous one-way `RANGE` with `RANGE_UNITS = km` and
  `RANGE_MODULUS = 0`.

The reduced-geometric profile accepts only receive-referenced timetags when
`TIMETAG_REF` is present, only `DATA_QUALITY = VALIDATED` when data quality is
declared, and only `CORRECTIONS_APPLIED = YES` when correction disposition is
declared. `CORRECTION_*`, transmit-referenced timetags, raw/degraded quality,
extra participants, duplicate keyword/timetag pairs, and non-chronological
data records fail closed.

The workflow requires the problem file to declare
`measurement_semantics = "reduced_geometric"`. This is an analyst assertion
that range and angles are already compatible with OEL's instantaneous
geometric measurement model. Parsing does not silently treat raw radiometric
observables as geometric values.

TDM XML, Doppler, frequency, phase, RA/Dec and other angle conventions,
multi-way or ambiguous range, light-time and media corrections, clock or
transponder calibration, association, and custody fail closed or remain out of
scope. Unsupported metadata and observable keywords are rejected rather than
discarded.

## Problem Contract

`oel.tracking_od_problem.v1` requires:

- one tracked `object_id` matching TDM `PARTICIPANT_2`;
- one or more WGS84 station locations matching TDM `PARTICIPANT_1` values;
- a six-component ECI Cartesian initial state in km and km/s;
- an explicit UTC initial-state epoch that exactly matches the first retained
  measurement, `initial_state_frame = "ECI"`, and
  `frame_model = "simple_gmst"`;
- positive angle and range standard deviations;
- explicit, positive fit and holdout durations; and
- bounded two-body propagation, with optional J2, plus declared least-squares
  settings.

The problem schema is closed and type-strict: unknown fields, string-valued
booleans, and other implicit JSON coercions are rejected. The public workflow
limits a problem to seven days and 200 estimator function evaluations. It uses
the existing OEL ground batch least-squares estimator; the TDM layer is an
interchange and evidence boundary, not a second estimator.

## Evidence And Interpretation

The output directory contains:

- `canonical_input.tdm`, preserving supported semantics in canonical KVN;
- `normalized_tracking_dataset.json`, with source keywords, station metadata,
  component uncertainties, and content identity;
- the estimator JSON and Markdown reports, residual CSV and plot, fitted state
  packet, and read-only review database; and
- `tracking_od_evidence.json`, with artifact receipts, solver convergence,
  covariance, quality gates, fit metrics, and holdout metrics.

The holdout ledger is produced by fresh OEL dynamics propagation of the fitted
state at withheld epochs. Holdout measurement residuals are the primary
predictive evidence. They are not state-error truth unless an independent
truth trajectory is also available, and the returned local covariance is not
a calibrated predicted-orbit-accuracy claim.

Calendar and ordinal CCSDS epochs use the shared OEL frame/time parser. The
normalizer retains a TAI-second epoch identity before scalar Julian-date
conversion so distinct microsecond records remain distinct. Fit uses the exact
declared boundary (`time_s <= fit_duration_s`); any later record is holdout.
Evidence binds the state epoch, ECI label, resolved frame model, and frame
provenance alongside the covariance.

## Validation Without Paid Services

The public acceptance suite does not require STK, ODTK, or another paid
service. It validates each layer separately:

1. Parser fixtures exercise the CCSDS KVN structure, units, epoch bounds,
   comments, canonical round-trip, source digest, malformed records, and
   fail-closed unsupported observables.
2. Independent geometry checks use ERFA GMST plus separately implemented
   WGS84/ENU equations and an analytic circular orbit. These checks do not call
   the estimator's measurement-prediction path.
3. A deterministic synthetic TDM starts from a perturbed prior and must recover
   the known state while keeping fit and untouched holdout rows distinct.
4. Boundary tests prove that microsecond epoch identity is retained and that a
   record after the exact fit boundary cannot enter the fitted objective.
5. Existing estimator tests cover residual ledgers, observability, covariance,
   rejection behavior, and non-convergent or invalid inputs.
6. The generated public export reruns the workflow so omitted private modules
   or accidental boundary dependencies fail before release.

These checks establish the bounded parser, measurement geometry, estimator
integration, evidence provenance, and exported installation. They do not
validate raw radiometric reduction, operational calibration, or general TDM
coverage.

## Public And Pro Boundary

Public includes the strict single-dataset KVN parser, normalized reduced-
geometric AZEL/range dataset, bounded batch estimator, explicit fit/holdout
prediction, evidence packet, tests, and synthetic example.

Pro remains the home for bulk or live ingestion, customer and operational data
governance, raw radiometric reduction, calibrated biases and media/light-time
models, broader observables, association and custody, multi-dataset campaigns,
covariance calibration, predicted-accuracy qualification, scheduling trades,
and review-ready customer packaging.
