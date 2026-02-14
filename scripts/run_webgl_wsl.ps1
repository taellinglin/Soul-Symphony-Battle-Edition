param(
    [int]$Port = 4747,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'

$projectWslPath = '/mnt/c/Users/User/Programs/Soul Symphony 2'
$buildScript = "$projectWslPath/scripts/freezify_web.py"
$pythonBin = '/root/opt/webpy312/bin/python'
$emsdkEnv = '/root/opt/emsdk/emsdk_env.sh'

if (-not $SkipBuild) {
    Write-Host "[1/3] Building WebGL bundle in WSL venv..."
    wsl -d Arch bash -lc "source $emsdkEnv >/dev/null; $pythonBin $buildScript"
}

Write-Host "[2/3] Starting WSL HTTP server on port $Port..."
wsl -d Arch bash -lc "if ss -ltn | grep -q ':$Port '; then echo 'port already listening'; else cd /mnt/c/Users/User/Programs/Soul\ Symphony\ 2; nohup $pythonBin -m http.server $Port --bind 0.0.0.0 --directory webbuild >/tmp/soul_http_$Port.log 2>&1 & sleep 1; fi"

Write-Host "[3/3] Verifying server..."
wsl -d Arch bash -lc "curl -I -m 5 http://127.0.0.1:$Port/ | sed -n '1,5p'"

Write-Host ""
Write-Host "Done. Open: http://localhost:$Port/"
Write-Host "WSL log: /tmp/soul_http_$Port.log"
Write-Host "Tip: use -SkipBuild to only restart the server."
