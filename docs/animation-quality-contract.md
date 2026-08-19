# OEL Animation Quality Contract

OEL animation artifacts should remain readable, numerically stable, and
inspectable throughout playback. A movie is not presentation-ready merely
because an encoder produced a non-empty file.

This contract governs presentation, temporal behavior, and encoding only. It
does not change review queries, simulation samples, deterministic physics, or
the meaning of plotted evidence.

Support posture: version 1 is supported for agent-native rectangular-RIC
review animations and the saved single- and multi-object ground-track and
multi-object RIC pilot renderers. Other legacy single-run, 3D, attitude,
dashboard, Pro, validation, interactive, and game-recording families remain
separate compatibility migrations.

## Contract Layers

The animation-quality workflow has five ordered layers:

1. Validate the review evidence and content-bound animation specification.
2. Select a bounded, deterministic frame plan.
3. Choose one numeric format and one declared camera policy for the sequence.
4. Inspect rendered artist geometry and temporal invariants across every frame.
5. Encode and decode-check the movie, write a receipt and contact sheet, and
   require agent visual review.

## Stable Numeric Presentation

- A numeric axis uses one formatter, resolution, and shared engineering
  exponent for the entire movie.
- Visible labels suppress negative zero and preserve meaningful trailing zeros.
- Time annotations use one resolution and a fixed-width field.
- Units and series identities do not change between frames.
- Formatting is chosen from the sequence-wide display envelope, not from one
  transient frame.

## Camera Policies

Every supported animation declares one camera policy:

- `fixed`: symmetric, origin-centered limits remain unchanged.
- `fit_history`: padded limits fit all selected evidence and remain unchanged.
- `follow`: the camera center may move, but each axis span, scale, formatter,
  aspect, and unit remain stable.

Undeclared autoscaling and frame-by-frame zoom changes fail the strict policy.

## Frame And Temporal Invariants

Every rendered frame is checked for the static plot-quality invariants:
minimum font size, text inside the canvas, text overlap, ambiguous numeric tick
labels, and legend obstruction. Sequence-level checks additionally require:

- finite, monotonic frame times;
- stable numeric formatter signatures;
- camera behavior consistent with the declared policy;
- deterministic first/final frame inclusion and bounded stride;
- a maximum of 600 encoded frames and 30 seconds under policy version 1;
- no silently truncated review query.

Presentation repairs apply globally. OEL may move a legend, reduce tick
density, or reserve more layout space, but it may not repair individual frames
differently, drop a series, change evidence, or alter physics.

## Encoding And Quality Evidence

Version 1 supports MP4 and GIF. A successful render produces:

- the encoded movie;
- a `.quality.json` receipt;
- a `.contact-sheet.png` containing deterministic first, final, stratified,
  extrema, and event frames when available.

The receipt records the policy, source review-store identity, recipe and query,
frame plan, formatting and camera decisions, per-frame failures, repairs,
artifact hash, dimensions, frame count, frame rate, duration, and decode result.

Automated success leaves `visual_qa_status: pending_agent_review`. The agent
handing off the artifact must inspect both the contact sheet and the movie for
flicker, apparent jitter, pacing, transitions, semantic visibility, and overall
professional presentation.

## Agent-Native Workflow

Connected agents should read `oel://review/animation-recipes/v1`, call
`oel.plan_review_animation.v1`, and then pass the unchanged specification and
content-bound plan ID to `oel.render_review_animation.v1` with an
operator-configured write approval.

The initial supported recipe is `relative_position_ric_2d`. It reads recorded
rectangular-RIC review columns and never reconstructs missing motion or invokes
the physics engine.

## Adoption

Animation families migrate through their focused owners. Byte-identical movie
goldens are not portable across encoders; tests should assert frame plans,
formatting and camera decisions, receipts, decode behavior, and representative
contact-sheet visuals instead.
