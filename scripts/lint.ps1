$ErrorActionPreference = 'Stop'

function Invoke-Uv {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] 
        $Arguments
    )
    uv @Arguments
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}

# Python
Write-Output "== Ruff Lint =="
Invoke-Uv --directory python/pipeline run ruff check --fix
Invoke-Uv --directory python/compare run ruff check --fix
Write-Output ""

Write-Output "== Ruff Format =="
Invoke-Uv --directory python/pipeline run ruff format
Invoke-Uv --directory python/compare run ruff format
Write-Output ""

Write-Output "== Pyrefly =="
Invoke-Uv --directory python/pipeline run pyrefly check
Invoke-Uv --directory python/compare run pyrefly check
Write-Output ""

# C++