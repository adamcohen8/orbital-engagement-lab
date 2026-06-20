# Spacecraft Twin Packages

Spacecraft twin packages are local, versioned bundles that collect object specs,
geometry profiles, mass properties, evidence, assumptions, and validation output
for reuse in OEL scenarios.

The v0 package format is intentionally file-based and reviewable:

```text
assets/twins/my_sat/
  twin.yaml
  object.yaml
  geometry_area_profile.json
  mass_properties.yaml
  source_evidence.yaml
  assumptions.yaml
  validation_report.md
```

`twin.yaml` is the package manifest:

```yaml
schema: oel.spacecraft_twin.v0
object_id: chaser
display_name: Example Satellite
version: 0.1.0

object:
  path: object.yaml

geometry:
  area_profile_path: geometry_area_profile.json
  source_mesh_path: source/spacecraft.stl
  confidence: medium

mass_properties:
  path: mass_properties.yaml
  confidence: high

source_evidence:
  path: source_evidence.yaml

assumptions:
  path: assumptions.yaml

validation:
  report_path: validation_report.md
```

Build, validate, and assemble a package with:

```bash
.venv/bin/python tools/build_spacecraft_twin.py examples/twins/demo_sat/twin.yaml \
  --validate \
  --report \
  --emit-object-yaml outputs/demo_sat_object.yaml \
  --print-summary
```

The emitted object YAML is a scenario-ready `objects.<object_id>` block. The
validation report includes artifact inventory, mass-property audit results,
geometry profile summary, missing input categories, and suggested next steps.

The v0 package builder does not perform AI document extraction. It provides the
stable container that future CAD/document agents can write into with traceable
source evidence and explicit assumptions.
