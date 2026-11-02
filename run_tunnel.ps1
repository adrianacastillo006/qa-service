# Windows PowerShell: opens the public ngrok tunnel to port 8000. Loads variables from .env if it exists.
# With NGROK_DOMAIN set (the free static dev domain of the ngrok account) the public URL never changes.
if (Test-Path .env) {
  Get-Content .env | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2; Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
  }
}
if ($env:NGROK_DOMAIN) { ngrok http 8000 --url="https://$env:NGROK_DOMAIN" }
else { ngrok http 8000 }
