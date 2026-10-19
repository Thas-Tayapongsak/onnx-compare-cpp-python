$ErrorActionPreference = 'Stop'

try {
    Push-Location (Join-Path $PSScriptRoot "..")

    uv run --project python/compare compare @args
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
