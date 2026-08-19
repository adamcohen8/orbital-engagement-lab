# Installing OEL On Windows, macOS, And Linux

This is the authoritative managed- and source-installation guide for Orbital
Engagement Lab. OEL supports CPython 3.10 through 3.14 on its declared Windows,
macOS, and Linux compatibility targets. Python 3.14 is recommended.

## Managed Installation

An official public release publishes a small `install.sh`, `install.ps1`,
signed `release-manifest.json`, source artifact, and offline bundle as immutable
assets on the public OEL GitHub release. The stable convenience URL below
resolves to the installer asset on the latest promoted public release. Never
pipe an uninspected URL from a third party. Managed OEL Pro installation is a
separate, deferred distribution decision and does not use the public feed.

POSIX:

```bash
curl --proto '=https' --tlsv1.2 -fsSLo /tmp/oel-install.sh \
  https://github.com/adamcohen8/orbital-engagement-lab/releases/latest/download/install.sh
less /tmp/oel-install.sh
sh /tmp/oel-install.sh
```

After verifying the release host and script through the documented channel,
the equivalent convenience form is:

```bash
curl --proto '=https' --tlsv1.2 -fsSL \
  https://github.com/adamcohen8/orbital-engagement-lab/releases/latest/download/install.sh | sh
```

The download-inspect-execute form remains preferred for first use because it
makes the small trust bootstrap visible before execution.

Windows PowerShell:

```powershell
Invoke-WebRequest https://github.com/adamcohen8/orbital-engagement-lab/releases/latest/download/install.ps1 -OutFile $env:TEMP\oel-install.ps1
Get-Content $env:TEMP\oel-install.ps1
& $env:TEMP\oel-install.ps1
```

The bootstrap requires an existing supported CPython. It verifies the embedded
bootstrap digest, signed release manifest, artifact size and SHA-256, and safe
archive shape before importing release code. It installs immutable versions
side by side under platform-native application data and writes a stable
launcher. The rendered installer also records the official signed-channel URL
beside OEL's trusted release keys, so future update checks do not require users
to copy URLs or key paths. The channel URL locates signed metadata; it is not a
replacement trust root. The bootstrap does not edit a workspace.

After installation:

```text
oel update status --full
oel update check
oel doctor
oel workspace init path/to/my-oel-workspace
oel --workspace path/to/my-oel-workspace sim --quickstart --validate-only
oel --workspace path/to/my-oel-workspace sim --quickstart
oel --workspace path/to/my-oel-workspace review outputs/quickstart_5min --saved-query run_metadata
```

See [Updating OEL](updating.md), [OEL Workspaces](workspaces.md), and
[Offline Installation](offline-installation.md). Managed installation and
workspace adoption are separate operations by design.

## Developer Source Installation

### Get The Source

Clone the public repository, or start in the root of an existing OEL checkout:

```text
https://github.com/adamcohen8/orbital-engagement-lab.git
```

The checkout directory may contain spaces. Run the commands below from the
directory containing `pyproject.toml` and `run_simulation.py`.

## Windows PowerShell

List the Python installations known to the Windows Python launcher:

```powershell
py --list
```

Older launcher versions use `py -0p` for the same inventory. Select a supported
minor, then create, install, diagnose, and run OEL with that same interpreter:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install ".[dev]"
.\.venv\Scripts\python.exe run_simulation.py --doctor
.\.venv\Scripts\python.exe run_simulation.py --quickstart
```

These commands do not require virtual-environment activation. This avoids
PowerShell execution-policy problems and guarantees that installation and
execution use the same interpreter.

## macOS Or Linux (POSIX Shell)

Select a supported interpreter installed on the host:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ".[dev]"
.venv/bin/python run_simulation.py --doctor
.venv/bin/python run_simulation.py --quickstart
```

Use these commands from Bash, Zsh, or another POSIX-compatible shell. Do not use
them unchanged in PowerShell; the virtual-environment interpreter path is
different.

## Choose Another Supported Python Minor

Replace `3.14` consistently with `3.10`, `3.11`, `3.12`, or `3.13`. On Windows,
for example:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
```

On macOS or Linux:

```bash
python3.11 -m venv .venv
.venv/bin/python --version
```

OEL maintains one approved constraints file per supported minor:

| Python | Constraints file |
| --- | --- |
| 3.10 | `constraints/py310.txt` |
| 3.11 | `constraints/py311.txt` |
| 3.12 | `constraints/py312.txt` |
| 3.13 | `constraints/py313.txt` |
| 3.14 | `constraints/py314.txt` |

Use the matching file when you need the approved cross-platform dependency
graph or release-compatible evidence.

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: `
  -c constraints/py314.txt ".[cross-platform]"
.\.venv\Scripts\python.exe -m pip check
```

POSIX:

```bash
.venv/bin/python -m pip install --only-binary=:all: \
  -c constraints/py314.txt ".[cross-platform]"
.venv/bin/python -m pip check
```

Do not use a constraints file for a different Python minor. Constraints are
approved reproducibility inputs, not universal lockfiles for every optional
external integration.

## Portable Command Convention

Onboarding, installation, classroom, and troubleshooting material shows
explicit PowerShell and POSIX commands. Other OEL documentation uses `python`
after the virtual environment has been activated, or links back to this guide.

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

Activate it in a POSIX shell:

```bash
source .venv/bin/activate
python --version
```

Activation is optional. If local policy blocks `Activate.ps1`, keep using
`.\.venv\Scripts\python.exe` directly. After activation, the following commands
are portable across PowerShell, macOS, and Linux:

```text
python run_simulation.py --doctor
python run_simulation.py --quickstart --validate-only
python run_simulation.py --quickstart
python -m sim.review outputs/quickstart_5min --saved-query run_metadata
```

When copying a command from a general OEL document, first activate the
environment or replace its leading `python` with the explicit interpreter path
for the current platform.

## Install Profiles

Choose only the profile required by the workflow:

| Command | Purpose |
| --- | --- |
| `python -m pip install .` | Core CLI, YAML/API runtime, plotting, and review store |
| `python -m pip install ".[dev]"` | Core plus tests and Ruff |
| `python -m pip install ".[game]"` | RPO trainer and media dependencies |
| `python -m pip install ".[accel]"` | Separately qualified Numba acceleration |
| `python -m pip install ".[validation]"` | OEL-native validation dependencies |
| `python -m pip install ".[cross-platform]"` | Aggregate compatibility-acceptance profile |
| `python -m pip install ".[ml]"` | Separately qualified ML dependencies |
| `python -m pip install ".[full]"` | Convenience union; not a universal support claim |

See [Compatibility And Install Profiles](compatibility.md) before using
acceleration, ML, `full`, or an external integration as support evidence.

## Classroom Or Restricted Environment Check

Only execute scenario YAML from a trusted source. For an unfamiliar file, use
safe validation first.

PowerShell:

```powershell
.\.venv\Scripts\python.exe run_simulation.py --config <path> --safe-validate
.\.venv\Scripts\python.exe run_simulation.py --config <path> --sealed-mode --validate-only
```

POSIX:

```bash
.venv/bin/python run_simulation.py --config <path> --safe-validate
.venv/bin/python run_simulation.py --config <path> --sealed-mode --validate-only
```

Safe validation is an inspection boundary, not permission to execute an
untrusted config. Sealed mode restricts plugins, external paths, hosted AI,
networked integrations, and high-detail outputs unless explicitly allowed.

## Troubleshooting A Failed Installation

Run the commands for the platform where the failure occurred.

PowerShell:

```powershell
py --list
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe run_simulation.py --doctor
```

POSIX:

```bash
.venv/bin/python --version
.venv/bin/python -m pip --version
.venv/bin/python -m pip check
.venv/bin/python run_simulation.py --doctor
```

Common recovery rules:

- If `py` is not recognized on Windows, install a supported CPython from
  python.org with the Python launcher, reopen PowerShell, and run `py --list`.
- If `python3.14` is not found on macOS or Linux, install a supported Python
  minor and use that minor consistently.
- If `.venv` was created by another interpreter, remove it through your normal
  file-management workflow and create a new one; do not reuse it across Python
  minors or operating systems.
- If activation is blocked, use the explicit interpreter path. Activation is
  never required for OEL.
- If binary dependency installation fails, confirm the OS, architecture,
  Python minor, matching constraints file, and install profile shown by
  `--doctor`.
- Do not post secrets, customer inputs, controlled data, or private report
  packets in a public bug report.

For the guided simulation walkthrough, continue to [Quickstart](quickstart.md).
For support boundaries and evidence requirements, read
[Compatibility And Install Profiles](compatibility.md).
