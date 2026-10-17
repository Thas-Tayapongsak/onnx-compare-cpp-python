$ErrorActionPreference = 'Stop'

try {
    Push-Location (Join-Path $PSScriptRoot "../cpp")

    cmake --preset default
    if ($LASTEXITCODE) { exit $LASTEXITCODE }

    cmake --build --preset default
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}

