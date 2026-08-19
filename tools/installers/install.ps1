$ErrorActionPreference = "Stop"

$DefaultBaseUrl = "__OEL_DEFAULT_BASE_URL__"
$DefaultChannelUrl = "__OEL_DEFAULT_CHANNEL_URL__"
$BootstrapSha256 = "__OEL_BOOTSTRAP_SHA256__"
$InstallerRendered = "__OEL_INSTALLER_RENDERED__"
$BaseUrl = if ($env:OEL_INSTALL_BASE_URL) { $env:OEL_INSTALL_BASE_URL } else { $DefaultBaseUrl }
$ChannelUrl = if ($env:OEL_UPDATE_CHANNEL_URL) { $env:OEL_UPDATE_CHANNEL_URL } else { $DefaultChannelUrl }
$Profile = if ($env:OEL_INSTALL_PROFILE) { $env:OEL_INSTALL_PROFILE } else { "core" }

if ($InstallerRendered -ne "true" -and (-not $env:OEL_INSTALL_BASE_URL -or -not $env:OEL_UPDATE_CHANNEL_URL)) {
    throw "This installer template has not been rendered for a release. Set OEL_INSTALL_BASE_URL and OEL_UPDATE_CHANNEL_URL, or use a released installer."
}

$PythonArgs = $null
foreach ($Version in @("3.14", "3.13", "3.12", "3.11", "3.10")) {
    try {
        & py "-$Version" -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,15) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PythonArgs = @("py", "-$Version")
            break
        }
    } catch {}
}
if (-not $PythonArgs) {
    throw "OEL requires CPython >=3.10,<3.15 with the Windows Python launcher."
}

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("oel-install-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $TempDir | Out-Null
try {
    $Bootstrap = Join-Path $TempDir "bootstrap_install.py"
    Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/bootstrap_install.py" -OutFile $Bootstrap
    if ($InstallerRendered -ne "true") {
        throw "Rendered bootstrap digest is missing."
    }
    $Actual = (Get-FileHash -Algorithm SHA256 $Bootstrap).Hash.ToLowerInvariant()
    if ($Actual -ne $BootstrapSha256) {
        throw "OEL bootstrap SHA-256 verification failed."
    }
    & $PythonArgs[0] $PythonArgs[1] $Bootstrap --manifest-url "$BaseUrl/release-manifest.json" --channel-url $ChannelUrl --profile $Profile @args
    if ($LASTEXITCODE -ne 0) {
        throw "OEL installation failed with exit code $LASTEXITCODE."
    }
} finally {
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}
