# OEL MCP SDK v2.0.0 Supply-Chain Evidence

Status: reviewed local evidence for the exact M2 SDK profile. This evidence is
platform-specific and does not promote the optional SDK path beyond the gates
in the adoption checklist.

Review date: 2026-07-31

## Reviewed Environment

- Host: macOS 15.7.7 (24G720), Apple arm64
- Python: CPython 3.11.14
- Runtime selection: `mcp==2.0.0`, including `mcp-types==2.0.0`
- Vulnerability scanner: `pip-audit 2.10.1`
- SBOM generator: `cyclonedx-bom 7.3.1`
- SBOM format: CycloneDX JSON 1.6

The exact dependency graph is recorded in
[`mcp-v2.0.0-macos-arm64-py311-freeze.txt`](mcp-v2.0.0-macos-arm64-py311-freeze.txt).
The wheel hashes are recorded in
[`mcp-v2.0.0-macos-arm64-py311-wheelhouse.sha256`](mcp-v2.0.0-macos-arm64-py311-wheelhouse.sha256).
The reviewed source archive `mcp-2.0.0.tar.gz` matched the published SHA-256
`0f440e735c13ece8bb19bc62cf0b86f4313448432fbb77d35e14034f4e050728`.

## Offline Installation

Thirty wheels were downloaded through the authorized dependency path and then
installed into a fresh virtual environment with index access disabled:

```bash
python3.11 -m venv offline-venv
PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
  offline-venv/bin/python -m pip install --no-index \
  --find-links wheelhouse pip==26.2 setuptools==83.0.0
PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
  offline-venv/bin/python -m pip install --no-index \
  --find-links wheelhouse mcp==2.0.0
offline-venv/bin/python -m pip check
```

The install completed without contacting an index and `pip check` reported no
broken requirements. The wheelhouse occupied 11,152 KiB. The complete virtual
environment occupied 61,532 KiB, of which 61,468 KiB was `site-packages`.
These figures include `pip` and `setuptools` and are local filesystem
measurements, not distribution guarantees.

## Vulnerability Disposition

The first scan found no findings in MCP or its runtime dependencies, but it did
find four unique advisories represented by six scanner records in the fresh
environment's bootstrap tools:

| Package | Initial version | Advisories | Disposition |
| --- | --- | --- | --- |
| `pip` | `26.0` | `PYSEC-2026-196`, `PYSEC-2026-2875`, `PYSEC-2026-2876` | Replaced from the offline wheelhouse with `pip==26.2` |
| `setuptools` | `80.10.2` | `PYSEC-2026-3447` | Replaced from the offline wheelhouse with `setuptools==83.0.0` |

The exact final environment was rescanned. The saved
[`pip-audit` result](mcp-v2.0.0-macos-arm64-py311-pip-audit.json) contains 30
dependencies, no vulnerability findings, and no outstanding fixes. This is a
point-in-time result and must be regenerated for a later release or build.

## Licenses And Dependency Surfaces

The declared license and bundled license-file presence for every wheel are
recorded in
[`mcp-v2.0.0-macos-arm64-py311-licenses.csv`](mcp-v2.0.0-macos-arm64-py311-licenses.csv).
All reviewed expressions are permissive: MIT-family, BSD-3-Clause,
Apache-2.0, PSF-2.0, or the `cryptography` Apache/BSD choice.

The graph includes web and ASGI packages (`httpx2`, `httpcore2`, `starlette`,
`sse-starlette`, `uvicorn`, `python-multipart`), authentication and crypto
packages (`PyJWT`, `cryptography`, `cffi`), schema packages (`pydantic`,
`jsonschema` and their dependencies), and async-runtime packages (`anyio`,
`h11`). These packages remain installed even though OEL's first SDK slice
exposes stdio only. No HTTP, SSE, OAuth, or remote transport is enabled by OEL.

The SDK requires `opentelemetry-api`, but the reviewed profile did not contain
`opentelemetry-sdk` or an OTLP exporter. The active tracer provider was the
API's no-export `ProxyTracerProvider`. A socket-denial smoke test imported MCP
and constructed its low-level `Server` without attempting a connection or an
update check.

## Startup Measurements

Ten fresh subprocess runs on the reviewed host produced these local medians:

| Operation | Median | Observed range |
| --- | ---: | ---: |
| Bare Python startup | 14.38 ms | 14.29–14.75 ms |
| `import mcp` | 386.83 ms | 384.77–410.47 ms |
| Import and construct low-level `Server` | 386.35 ms | 384.32–388.40 ms |
| OEL legacy stdio process through EOF | 309.56 ms | 292.05–333.49 ms |
| OEL SDK stdio process through EOF | 661.77 ms | 644.51–758.99 ms |

The startup numbers are diagnostic measurements, not budgets or cross-platform
performance claims.

## Machine-Readable Records

- [CycloneDX SBOM](mcp-v2.0.0-macos-arm64-py311.cdx.json)
- [Exact freeze](mcp-v2.0.0-macos-arm64-py311-freeze.txt)
- [Wheel SHA-256 manifest](mcp-v2.0.0-macos-arm64-py311-wheelhouse.sha256)
- [License inventory](mcp-v2.0.0-macos-arm64-py311-licenses.csv)
- [Final vulnerability audit](mcp-v2.0.0-macos-arm64-py311-pip-audit.json)
- [SDK, Inspector, Codex, and Claude interoperability](interop-2026-07-31.json)
- [M3 package, resource, lifecycle, and host interoperability](interop-m3-2026-07-31.json)
- [M4 approval, execution, cancellation, evidence, and host interoperability](interop-m4-2026-07-31.json)
- [M5.1 offline SDK and complete public-workflow acceptance](m5-1-2026-07-31.json)
- [Five-round public-export dogfood and feedback repairs](dogfood-5x-2026-07-31.md)
- [Pinned Inspector vulnerability audit](inspector-2.0.0-audit-2026-07-31.json)

The Inspector is test-only tooling and is not an OEL runtime or packaged
dependency. Its exact `2.0.0` npm graph contained 266 dependencies including
optional and peer dependencies; `npm audit` reported zero findings on the
review date.

The M5.1 record is retained as historical evidence for its exact cited commit.
It is explicitly superseded for release use after subsequent dogfood and
v0.23.0 review fixes; regenerate it against the final clean v0.23.0 commit
before release rather than relabeling the historical run.
