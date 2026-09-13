<#
.SYNOPSIS
    Glacier Fleet Benchmark Orchestrator
    Orchestrates distributed LLM/MoE benchmark execution across a heterogeneous Windows 11 fleet via WinRM.

.EXAMPLE
    # Run locally only
    .\orchestrate_fleet.ps1 -ModelPath "path/to/model.gguf"

.EXAMPLE
    # Run across distributed worker nodes
    $cluster = @(
        @{ IP = "localhost";    Role = "Cluster Head"; GPU = "auto";            Engine = "directml" },
        @{ IP = "worker-node-1"; Role = "Worker 1"; GPU = "nvidia-rtx-3060"; Engine = "baremetal" },
        @{ IP = "worker-node-2"; Role = "Worker 2"; GPU = "amd-680m";        Engine = "directml" }
    )
    .\orchestrate_fleet.ps1 -ModelPath "path/to/model.gguf" -Nodes $cluster -DeployBinary
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$ModelPath = "Qwen3-30B-A3B-Q3_K_L.gguf",

    [Parameter(Mandatory = $false)]
    [int]$Tokens = 25,

    [Parameter(Mandatory = $false)]
    [string]$Prompt = "Explain in two sentences what a CPU cache is.",

    [Parameter(Mandatory = $false)]
    [PSCredential]$Credential,

    [Parameter(Mandatory = $false)]
    [array]$Nodes = @(
        @{ IP = "localhost"; Role = "Local Host"; GPU = "auto"; Engine = "directml" }
    ),

    [Parameter(Mandatory = $false)]
    [switch]$DeployBinary,

    [Parameter(Mandatory = $false)]
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "             GLACIER DISTRIBUTED FLEET BENCHMARK                       " -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Model:        $ModelPath"
Write-Host "Tokens:       $Tokens"
Write-Host "Nodes:        $($Nodes.Count) target(s)"
Write-Host ""

# 1. Ensure local TrustedHosts contains configured worker machines
Write-Host "[1/3] Checking WSMan configuration..." -ForegroundColor Yellow
$currentTrusted = (Get-Item WSMan:\localhost\Client\TrustedHosts -ErrorAction SilentlyContinue).Value
if ($currentTrusted) {
    Write-Host "   TrustedHosts configured: $currentTrusted" -ForegroundColor Green
} else {
    Write-Host "   TrustedHosts is empty (remote nodes may require configuration)" -ForegroundColor Yellow
}

# 2. Check reachability
Write-Host "`n[2/3] Checking WinRM reachability across nodes..." -ForegroundColor Yellow
$activeNodes = [System.Collections.Generic.List[hashtable]]::new()

foreach ($node in $Nodes) {
    if ($node.IP -eq "localhost" -or $node.IP -eq "127.0.0.1") {
        $activeNodes.Add($node)
        Write-Host "   [+] $($node.IP) ($($node.Role)): Ready" -ForegroundColor Green
        continue
    }

    $t = Test-NetConnection -ComputerName $node.IP -Port 5985 -WarningAction SilentlyContinue
    if ($t.TcpTestSucceeded) {
        $activeNodes.Add($node)
        Write-Host "   [+] $($node.IP) ($($node.Role)): WinRM Port 5985 Open & Ready" -ForegroundColor Green
    } else {
        Write-Host "   [-] $($node.IP) ($($node.Role)): Port 5985 closed." -ForegroundColor Red
    }
}

if ($CheckOnly) {
    Write-Host "`nAll nodes checked. Fleet is reachable!" -ForegroundColor Green
    return
}

# Prompt for credentials if remote nodes are active and no credential was supplied
if (-not $Credential -and ($activeNodes | Where-Object { $_.IP -ne "localhost" })) {
    Write-Host "`n[!] Enter Windows credentials for the remote worker machines:" -ForegroundColor Yellow
    $Credential = Get-Credential
}

# Local published glacier binary
$localBin = "$PSScriptRoot\..\publish\glacier.exe"
if (-not (Test-Path $localBin)) {
    Write-Host "   Building fresh standalone glacier binary..." -ForegroundColor Cyan
    dotnet publish "$PSScriptRoot\..\src\Glacier.Inference.Cli\Glacier.Inference.Cli.csproj" -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -o "$PSScriptRoot\..\publish\" | Out-Null
}

# 3. Dispatch Benchmark Execution
Write-Host "`n[3/3] Executing benchmarks across active fleet nodes..." -ForegroundColor Yellow
$results = @()

foreach ($node in $activeNodes) {
    Write-Host "`n>> Running benchmark on $($node.IP) ($($node.Role)) with GPU [$($node.GPU)]..." -ForegroundColor Cyan

    if ($node.IP -eq "localhost") {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $out = & $localBin bench $ModelPath -n $Tokens -p $Prompt --device $node.GPU --engine $node.Engine 2>&1 | Out-String
        $sw.Stop()
        $results += [PSCustomObject]@{
            Node     = $node.Role
            IP       = $node.IP
            GPU      = $node.GPU
            Duration = [math]::Round($sw.Elapsed.TotalSeconds, 2)
            Output   = $out
        }
    } else {
        try {
            $sessParams = @{ ComputerName = $node.IP }
            if ($Credential) { $sessParams["Credential"] = $Credential }
            
            $session = New-PSSession @sessParams

            try {
                # Deploy binary if requested or if missing on remote
                $remoteHasBin = Invoke-Command -Session $session -ScriptBlock { Test-Path "C:\Glacier\glacier.exe" }
                if ($DeployBinary -or (-not $remoteHasBin)) {
                    Write-Host "   Deploying standalone glacier.exe to $($node.IP):C:\Glacier\glacier.exe..." -ForegroundColor Cyan
                    Copy-Item -Path $localBin -Destination "C:\Glacier\glacier.exe" -ToSession $session -Force
                    Write-Host "   Deployment complete!" -ForegroundColor Green
                }

                # Query remote GPU devices
                Write-Host "   Enumerating devices on remote host..." -ForegroundColor Cyan
                $devInfo = Invoke-Command -Session $session -ScriptBlock {
                    & "C:\Glacier\glacier.exe" devices 2>&1 | Out-String
                }
                Write-Host $devInfo

                # Run benchmark
                Write-Host "   Launching benchmark on $($node.GPU)..." -ForegroundColor Cyan
                $remOut = Invoke-Command -Session $session -ScriptBlock {
                    param($m, $n, $p, $g, $e)
                    & "C:\Glacier\glacier.exe" bench $m -n $n -p $p --device $g --engine $e 2>&1 | Out-String
                } -ArgumentList $ModelPath, $Tokens, $Prompt, $node.GPU, $node.Engine

                $results += [PSCustomObject]@{
                    Node     = $node.Role
                    IP       = $node.IP
                    GPU      = $node.GPU
                    Duration = "Remote"
                    Output   = $remOut
                }
            } finally {
                Remove-PSSession $session
            }
        } catch {
            Write-Host "   Failed to execute on $($node.IP): $_" -ForegroundColor Red
        }
    }
}

Write-Host "`n=======================================================================" -ForegroundColor Green
Write-Host "                        FLEET RESULTS SUMMARY                          " -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Green
foreach ($r in $results) {
    Write-Host "`nNode: $($r.Node) ($($r.IP)) [Target GPU: $($r.GPU)]" -ForegroundColor Cyan
    Write-Host $r.Output.Trim()
}
