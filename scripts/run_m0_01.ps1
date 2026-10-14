$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    & .\.venv\Scripts\python.exe -m experiments.m0_01_model_inspection @args
    if ($LASTEXITCODE -ne 0) { throw 'M0-01 failed; inspect the newest results directory.' }
} finally { Pop-Location }
