# OEL Plot Quality Contract

OEL plot artifacts should be readable, numerically honest, and reproducible
across the CLI, Python API, MCP, and agent workflows. A figure is not considered
presentation-ready merely because Matplotlib saved a non-empty file.

This contract governs display formatting and layout only. It never changes the
review-store query, source evidence, simulation results, plotted samples, or
deterministic physics.

Support posture: version 1 is supported for static plots produced through the
review-store Python, CLI, and MCP render paths. Saved agent-native animations
use the separate [OEL Animation Quality Contract](animation-quality-contract.md).
Applying the static policy by default to legacy single-run, Pro, validation, or
other animation families remains a separate compatibility migration.

## Contract Layers

The plot-quality workflow has four ordered layers:

1. Validate the evidence query and plot specification.
2. Apply one deterministic numeric-format decision per numeric axis.
3. Render, inspect artist geometry, and apply bounded layout repairs.
4. Save the artifact, run raster/file checks, and require agent visual review.

Automated checks are supporting evidence. They do not replace visual
inspection by the agent handing off the artifact.

## Numeric Formatting

One formatter applies to every major tick on an axis. OEL does not independently
round each tick or annotation.

- Major ticks use a bounded set of engineering intervals: 1, 2, 2.5, 5, and
  10 times a power of ten.
- Fixed-point decimal places come from that common tick interval. Trailing
  zeros are retained when they communicate the chosen display resolution.
- Very large or small values use one shared engineering exponent for the axis.
  Per-tick mixtures of fixed-point and scientific notation are not allowed.
- Values that round to zero at the selected axis resolution display as positive
  zero rather than `-0` or `-0.00`.
- Distinct visible tick values must not collapse to the same formatted label.
- Aspect and projection constraints must resolve before the formatter is chosen.
- Named domain recipes should pin display units when the professional convention
  is known. Unit conversion must be declared and recorded; it must not alter the
  underlying evidence.
- Annotations and tabular values derived from the axis should reuse its format
  decision. They must not imply more resolution than the plot can show or the
  evidence supports.

The strict public implementation is `sim.plotting.quality` and the initial
policy identifier is `oel.agent_strict`, version 1.

## Layout Invariants

After every title, label, legend, annotation, and artifact footer is present,
the renderer must check:

- minimum readable font size, with a separate compact provenance-footer floor;
- text entirely inside the intended figure canvas;
- overlapping titles, labels, ticks, annotations, legend text, and footers;
- legends that obscure plotted data;
- excessive inside-legend density.

`tight_layout`, constrained layout, and `bbox_inches="tight"` are layout tools,
not proof that these invariants hold. OEL checks the rendered artist bounding
boxes in display coordinates.

## Bounded Repairs

The strict policy may apply only presentation-level repairs that preserve the
data and requested meaning:

1. Reduce numeric tick density.
2. Rotate categorical tick labels through declared initial and maximum angles.
3. Move an obstructive or overcrowded legend outside the plotting area.
4. Re-run a bounded layout pass with reserved title, footer, and legend space.

If defects remain, automated status is `failed`. The renderer must report the
unresolved checks instead of silently shrinking text below the minimum,
dropping labels, clipping content, changing the query, removing series, or
changing axis limits to make the figure fit.

## Quality Evidence

Agent-native plot artifacts should record this structured evidence alongside
query and renderer provenance:

- `policy_id` and `policy_version`;
- `automated_status`;
- checks performed and unresolved issues;
- repairs applied;
- the formatter, tick interval, decimal places, and shared exponent per axis;
- `visual_qa_status` and `visual_review_required`.

The initial automated result uses
`visual_qa_status: pending_agent_review`. A successful automatic result does
not authorize an agent to skip opening the rendered artifact.

## Adoption

The review-store and MCP plotting path should adopt the strict policy first.
Existing OEL plot families should migrate through focused owners and regression
fixtures rather than receiving an unreviewed global formatting change. Shared
save helpers may enforce the contract by default only after representative
single-run, animation, validation, and Pro artifact families demonstrate
compatible output.

Tests should emphasize structural geometry and formatting decisions rather
than exact image bytes. Include adversarial fixtures for long category labels,
dense legends, titles and footers, tiny and large numeric ranges, negative
zero, and multi-panel figures. Golden images remain useful for a small number
of representative visual acceptance cases.
