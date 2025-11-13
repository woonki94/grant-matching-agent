# ───────────────────────────────────────────────
# Run the faculty keyword generation pipeline
# Usage:
#   .\scripts\generate_faculty_keywords.ps1 [batch_size] [max_keywords]
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

$BATCH_SIZE   = if ($args.Count -ge 1) { $args[0] } else { 50 }
$MAX_KEYWORDS = if ($args.Count -ge 2) { $args[1] } else { 25 }

Write-Host " Generating faculty keywords ..."
Write-Host "   Batch size   : $BATCH_SIZE"
Write-Host "   Max keywords : $MAX_KEYWORDS"
Write-Host ""

python -m services.faculty.generate_faculty_keywords $BATCH_SIZE $MAX_KEYWORDS

Write-Host "Faculty keyword generation complete"
