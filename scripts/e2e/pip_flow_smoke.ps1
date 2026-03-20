#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

function Resolve-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    throw "[e2e] python interpreter not found"
}

function Invoke-CommandChecked {
    param(
        [string[]]$Command,
        [string]$ErrorMessage
    )

    if ($Command.Length -le 1) {
        & $Command[0]
    } else {
        & $Command[0] $Command[1..($Command.Length - 1)]
    }
    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$workDir = New-Item -ItemType Directory -Path (Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString())) -Force
$pythonCmd = Resolve-PythonCommand

Write-Host "[e2e] root: $rootDir"
Write-Host "[e2e] work: $($workDir.FullName)"

try {
    Invoke-CommandChecked -Command (@($pythonCmd) + @("-m", "venv", (Join-Path $workDir.FullName "venv"))) -ErrorMessage "[e2e] failed to create virtual environment"

    . (Join-Path $workDir.FullName "venv\Scripts\Activate.ps1")

    Invoke-CommandChecked -Command @("python", "-m", "pip", "install", "--upgrade", "pip", "build") -ErrorMessage "[e2e] failed to install build dependencies"
    Invoke-CommandChecked -Command @("python", "-m", "build", "--wheel", $rootDir) -ErrorMessage "[e2e] failed to build wheel"

    $wheelPath = Get-ChildItem -Path (Join-Path $rootDir "dist") -Filter "worldflux-*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $wheelPath) {
        throw "[e2e] no wheel found in dist/"
    }

    Invoke-CommandChecked -Command @("python", "-m", "pip", "install", $wheelPath.FullName) -ErrorMessage "[e2e] failed to install wheel"

    $env:WORLDFLUX_INIT_ENSURE_DEPS = "0"
    $env:PYTHONUTF8 = "1"
    Set-Location $workDir.FullName

    @(
        "demo-project",
        "1",
        "1",
        "1",
        "1",
        "2",
        "y"
    ) | worldflux init demo --force

    Set-Location (Join-Path $workDir.FullName "demo")

    worldflux train --steps 2 --device cpu
    if ($LASTEXITCODE -ne 0) {
        throw "[e2e] train failed"
    }

    & worldflux verify --target ./outputs --mode quick --episodes 2 --format json --output verify-quick.json
    $quickStatus = $LASTEXITCODE
    & worldflux verify --target ./outputs --demo --mode proof --format json --output verify-demo.json
    $demoStatus = $LASTEXITCODE

    if ($quickStatus -ne 0 -and $quickStatus -ne 1) {
        throw "[e2e] unexpected quick verify exit code: $quickStatus"
    }
    if ($demoStatus -ne 0 -and $demoStatus -ne 1) {
        throw "[e2e] unexpected demo verify exit code: $demoStatus"
    }

    $quick = Get-Content verify-quick.json -Raw | ConvertFrom-Json
    $demo = Get-Content verify-demo.json -Raw | ConvertFrom-Json

    if (-not $quick.PSObject.Properties.Name.Contains("passed")) {
        throw "[e2e] quick verify payload missing 'passed'"
    }
    if ($quick.env -ne "atari/pong") {
        throw "[e2e] quick verify env mismatch"
    }
    if (-not $quick.stats) {
        throw "[e2e] quick verify payload missing stats"
    }
    if (-not $demo.PSObject.Properties.Name.Contains("passed")) {
        throw "[e2e] demo verify payload missing 'passed'"
    }
    if (-not $demo.stats) {
        throw "[e2e] demo verify payload missing stats"
    }

    Write-Host "[e2e] verification payloads validated"
    Write-Host "[e2e] pip flow smoke passed"
}
finally {
    Set-Location $rootDir
    if (Test-Path $workDir.FullName) {
        Remove-Item -Recurse -Force $workDir.FullName
    }
}
