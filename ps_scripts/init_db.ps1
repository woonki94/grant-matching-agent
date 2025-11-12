# ───────────────────────────────────────────────
# Initialize PostgreSQL tables via SQLAlchemy Base
# Usage:
#   .\scripts\init_db.ps1
# ───────────────────────────────────────────────

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to project root
Set-Location "$PSScriptRoot\.."

# Load .env variables if present
if (Test-Path ".env") {
    Write-Host "Loading environment variables from .env..."
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*#') { return }  # skip comments
        $parts = $_ -split '=', 2
        if ($parts.Length -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim())
        }
    }
}

Write-Host "Initializing database schema..."
python -m db.init_db
Write-Host "Database initialization complete!"
