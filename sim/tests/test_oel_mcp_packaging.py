from __future__ import annotations

import os
import site
import subprocess
import venv
import zipfile
from pathlib import Path

import pytest
from setuptools.build_meta import build_wheel

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.external
def test_m5_wheel_packages_supported_mcp_profiles_and_keeps_dependency_optional(
    tmp_path: Path, monkeypatch
) -> None:
    import anyio
    from mcp import Client, StdioServerParameters, stdio_client

    monkeypatch.chdir(ROOT)
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    wheel_name = build_wheel(str(wheel_dir))

    with zipfile.ZipFile(wheel_dir / wheel_name) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        entry_points_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        wheel_metadata_name = next(name for name in names if name.endswith(".dist-info/WHEEL"))
        metadata = archive.read(metadata_name).decode("utf-8")
        entry_points = archive.read(entry_points_name).decode("utf-8")
        wheel_metadata = archive.read(wheel_metadata_name).decode("utf-8")

    assert wheel_name.endswith("-py3-none-any.whl")
    assert "Root-Is-Purelib: true" in wheel_metadata
    assert "Tag: py3-none-any" in wheel_metadata
    assert "License-Expression: Apache-2.0" in metadata
    assert "License-File: LICENSE.txt" in metadata
    assert any(name.startswith("sim/") for name in names)
    assert "integrations/oel_mcp/acceptance.py" in names
    assert "integrations/oel_mcp/diagnostics.py" in names
    assert "integrations/oel_mcp/public_server.py" in names
    assert "integrations/oel_mcp/reporting.py" in names
    assert "integrations/oel_mcp/sdk_protocol.py" in names
    assert "integrations/oel_mcp/execution.py" in names
    assert "integrations/oel_mcp/resources.py" in names
    assert "integrations/oel_mcp/resource_data/operator-guide.md" in names
    assert "sim/installation/cli.py" in names
    assert "sim/installation/manager.py" in names
    assert "sim/installation/schemas/channel-config.schema.json" in names
    assert "sim/installation/schemas/release-manifest.schema.json" in names
    assert "sim/schema_versions.py" in names
    assert any(name.endswith(".data/data/share/oel/configs/quickstart_5min.yaml") for name in names)
    pro_modules = {
        "integrations/oel_mcp/pro_handlers.py",
        "integrations/oel_mcp/pro_acceptance.py",
        "integrations/oel_mcp/pro_diagnostics.py",
        "integrations/oel_mcp/pro_registry.py",
        "integrations/oel_mcp/server.py",
    }
    if (ROOT / "integrations" / "oel_mcp" / "pro_handlers.py").is_file():
        assert pro_modules <= names
    else:
        assert not pro_modules & names
    mcp_requirements = [line for line in metadata.splitlines() if line.startswith("Requires-Dist: mcp")]
    assert mcp_requirements == [
        'Requires-Dist: mcp<3,>=2.0.0; extra == "mcp"',
        'Requires-Dist: mcp<3,>=2.0.0; extra == "full"',
    ]
    assert "oel-mcp = integrations.oel_mcp.public_server:main" in entry_points
    assert "oel = sim.installation.cli:main" in entry_points
    environment = tmp_path / "installed"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    entrypoint = scripts / ("oel-mcp.exe" if os.name == "nt" else "oel-mcp")
    installed_site = Path(
        subprocess.run(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    (installed_site / "oel-mcp-test-parent-venv.pth").write_text(
        site.getsitepackages()[0] + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheel_dir / wheel_name),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    async def exercise() -> tuple[str, tuple[str, ...], str]:
        parameters = StdioServerParameters(
            command=str(entrypoint),
            cwd=tmp_path,
            env={**os.environ, "OEL_MCP_READ_ROOTS": str(tmp_path)},
        )
        async with Client(stdio_client(parameters), mode="auto", cache=None) as client:
            resources = await client.list_resources(cache_mode="reload")
            guide = await client.read_resource("oel://docs/operator-guide/v1", cache_mode="reload")
            return (
                client.protocol_version,
                tuple(resource.uri for resource in resources.resources),
                str(guide.contents[0].text),
            )

    protocol, resource_uris, guide = anyio.run(exercise)

    assert protocol == "2026-07-28"
    assert resource_uris == (
        "oel://capabilities/tools/v1",
        "oel://review/saved-queries/v1",
        "oel://agent/tasks/v1",
        "oel://docs/operator-guide/v1",
        "oel://handoff/product-kinds/v1",
        "oel://review/plot-recipes/v1",
        "oel://review/animation-recipes/v1",
    )
    assert "supported local stdio OEL MCP surface" in guide
