# ───────────────────────────────────────────────
# Run the faculty fetch + save pipeline
# Usage:
#   .\scripts\fetch_save_faculty.ps1 [src_url] [limit]
# ───────────────────────────────────────────────

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."

# Load .env if present
if (Test-Path ".env") {
    Write-Host "Loading .env..."
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*#') { return }
        $parts = $_ -split '=', 2
        if ($parts.Length -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim())
        }
    }
}

$SRC_URL = if ($args.Count -ge 1) { $args[0] } else { "https://engineering.oregonstate.edu/people" }
$LIMIT   = if ($args.Count -ge 2) { $args[1] } else { 0 }

Write-Host " Fetching / saving faculty ..."
Write-Host "   Source : $SRC_URL"
Write-Host "   Limit  : $LIMIT"

python -m services.faculty.scrape_faculty $SRC_URL $LIMIT
python -m services.faculty.save_faculty

Write-Host "Faculty fetch + save complete"
