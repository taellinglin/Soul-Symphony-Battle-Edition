param(
    [string]$Distro = "Arch",
    [int]$ListenPort = 4747,
    [string]$ListenAddress = "0.0.0.0",
    [string]$WslInterface = "eth0",
    [string]$FirewallRuleName = "WSL-4747-In"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
    throw "Please run this script from an elevated (Administrator) PowerShell session."
}

$ipCmd = "ip -4 -o addr show dev $WslInterface | awk '{print \$4}' | cut -d/ -f1 | head -n1"
$wslIp = (wsl -d $Distro bash -lc $ipCmd).Trim()

if (-not $wslIp -or $wslIp -notmatch '^\d{1,3}(\.\d{1,3}){3}$') {
    throw "Unable to read a valid WSL IPv4 address. Distro=$Distro Interface=$WslInterface Output='$wslIp'"
}

Write-Host "WSL IP: $wslIp"

# 既存ルール削除（存在しない場合も継続）
cmd /c "netsh interface portproxy delete v4tov4 listenaddress=$ListenAddress listenport=$ListenPort" | Out-Null

# 新規ルール追加
cmd /c "netsh interface portproxy add v4tov4 listenaddress=$ListenAddress listenport=$ListenPort connectaddress=$wslIp connectport=$ListenPort"

# FWルール
$existingRule = Get-NetFirewallRule -DisplayName $FirewallRuleName -ErrorAction SilentlyContinue
if ($null -eq $existingRule) {
    New-NetFirewallRule -DisplayName $FirewallRuleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $ListenPort | Out-Null
    Write-Host "Firewall rule created: $FirewallRuleName"
} else {
    Set-NetFirewallRule -DisplayName $FirewallRuleName -Enabled True -Action Allow -Direction Inbound | Out-Null
    Write-Host "Firewall rule ensured: $FirewallRuleName"
}

Write-Host ""
Write-Host "Current portproxy entries:"
cmd /c "netsh interface portproxy show v4tov4"
