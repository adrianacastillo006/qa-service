# Windows PowerShell: starts the API on port 8000. Loads variables from .env if it exists.
if (Test-Path .env) {
  Get-Content .env | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2; Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
  }
}
if (-not $env:QA_API_TOKEN) { Write-Error "Set QA_API_TOKEN first"; exit 1 }
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
