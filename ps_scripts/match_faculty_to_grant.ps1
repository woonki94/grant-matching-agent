# ───────────────────────────────────────────────
# Run the faculty-grant matching pipeline to calculate cosine similarity
# Usage:
#   .\scripts\match_faculty_to_grant.ps1 [batch_size] [k]
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

$BATCH_SIZE = if ($args.Count -ge 1) { $args[0] } else { 10 }
$TOP_K      = if ($args.Count -ge 2) { $args[1] } else { 5 }

Write-Host " Running grant-faculty matching ..."
Write-Host "   Batch size : $BATCH_SIZE"
Write-Host "   Top K      : $TOP_K"
Write-Host ""

python -m services.matching.match_faculty_to_grant $BATCH_SIZE $TOP_K

Write-Host ""
Write-Host "Grant-faculty matching complete."
