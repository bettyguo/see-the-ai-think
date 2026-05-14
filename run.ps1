# see-the-ai-think - one-command start for Windows PowerShell.
# Mirrors the Makefile target `run`.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$pyCmd = Get-Command $py -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    Write-Error "python not found. install Python 3.11+ and re-run."
    exit 1
}

if (-not (Test-Path ".venv")) {
    & $py -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
}

& .\.venv\Scripts\python.exe -m pip install -e ".[sae]"
& .\.venv\Scripts\python.exe -m backend.warm

Start-Job -ScriptBlock {
    Start-Sleep -Seconds 1
    Start-Process "http://127.0.0.1:8000"
} | Out-Null

& .\.venv\Scripts\python.exe -m backend --host 127.0.0.1 --port 8000
