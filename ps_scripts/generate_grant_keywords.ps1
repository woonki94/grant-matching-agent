# ───────────────────────────────────────────────
# Run the grant keyword mining pipeline
# Usage:
#   .\scripts\generate_grant_keywords.ps1 [batch_size] [max_keywords]
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

Write-Host " Running generate_grant_keywords.py ..."
Write-Host "   Batch size   : $BATCH_SIZE"
Write-Host "   Max keywords : $MAX_KEYWORDS"
Write-Host ""

python -m services.grant.generate_grant_keywords $BATCH_SIZE $MAX_KEYWORDS
