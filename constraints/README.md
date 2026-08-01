# Approved Python Constraint Sets

OEL's source dependency ranges live in `pyproject.toml`. The files in this
directory select the approved dependency graph for each supported CPython
minor:

```bash
python -m pip install -c constraints/py311.txt ".[cross-platform]"
```

These constraints pin the direct and transitive packages exercised by the
cross-platform profile. They are not complete cross-platform lockfiles:

- they do not contain wheel hashes;
- wheel tags and source URLs vary by operating system and architecture;
- separately qualified profiles such as `ml` and external integrations may
  require an overlay or release lock; and
- a release row must retain its pip installation report, wheel inventory,
  `pip check`, freeze, SBOM, audit result, and this file's SHA-256 digest.

The approved graphs were selected with binary-wheel-only resolver probes.
Python 3.10 resolves older NumPy, SciPy, Matplotlib, and ContourPy releases
because later releases have raised their interpreter floor. Python 3.14 uses
`pygame-ce`, which provides the `pygame` import surface and publishes a CPython
3.14 wheel; Python 3.10 through 3.13 retain upstream `pygame`. Python 3.10
also pins `tomli` as the standard-library-compatible TOML reader fallback.
Every development graph pins Setuptools because clean virtual environments do
not guarantee that the build backend is importable at test runtime.

The aggregate profile conditionally omits Numba on Intel macOS because the
approved Numba/llvmlite releases do not publish wheels for that architecture.
The constraint files still record those versions for architectures where the
acceleration profile is qualified.

Do not update a constraint file merely because a newer release exists. Update
the source range and applicable constraint sets together, then rerun resolver,
runtime, physics, validation, trainer, audit, SBOM, and public-export gates.
