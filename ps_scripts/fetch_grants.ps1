# ───────────────────────────────────────────────
# Run the grant fetch + commit pipeline
# Usage:
#   .\scripts\fetch_commit_grant.ps1 [page_offset] [page_size] [query]
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

# Default arguments
$PAGE_OFFSET = if ($args.Count -ge 1) { $args[0] } else { 1 }
$PAGE_SIZE   = if ($args.Count -ge 2) { $args[1] } else { 5 }
$QUERY       = if ($args.Count -ge 3) { $args[2] } else { "" }

Write-Host " Running fetch_commit_grant.py ..."
Write-Host "   Page offset : $PAGE_OFFSET"
Write-Host "   Page size   : $PAGE_SIZE"
Write-Host "   Query       : $QUERY"
Write-Host ""

python -m services.grant.save_grant $PAGE_OFFSET $PAGE_SIZE $QUERY
