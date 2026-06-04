# Orbital Engagement Lab RPO Trainer Preview

This is a browser-native version 1 of the OEL RPO Trainer Preview. It is a
static web app intended for social-media clickthroughs and quick demos.

Open locally:

```bash
python -m http.server 8765 --directory web/rpo-trainer-preview
```

Then visit:

```text
http://localhost:8765
```

## Included

- Tutorial mode based on the Level 0 RIC-control lesson.
- Curated sandbox mode with preset starts, range, drift, reset, and randomize.
- RI and RC canvas plots.
- Keyboard and touch controls.
- Browser-started tutorial and sandbox music.
- Simple debrief after tutorial success.
- Download CTA for the full OEL trainer.

## Not Included

This preview intentionally does not include the full OEL Python simulator,
scenario YAML support, all trainer levels, recordings, or full debrief reports.
See `docs/physics-contract.md` for the model boundary.
