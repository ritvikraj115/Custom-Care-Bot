param(
  [switch]$InitOnly,
  [switch]$NoStart
)

$ErrorActionPreference = 'Stop'
$airflowDir = $PSScriptRoot
Set-Location $airflowDir

function Test-DockerReady {
  try {
    docker info | Out-Null
    return $true
  }
  catch {
    return $false
  }
}

function Invoke-Compose {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Args
  )

  docker compose @Args
  if ($LASTEXITCODE -ne 0) {
    throw "docker compose failed: docker compose $($Args -join ' ')"
  }
}

# Try starting Docker Desktop if not running
if (-not (Test-DockerReady)) {
  $dockerDesktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
  if (Test-Path $dockerDesktop) {
    Start-Process $dockerDesktop | Out-Null
    Write-Host 'Starting Docker Desktop...'
    for ($i = 0; $i -lt 60; $i++) {
      Start-Sleep -Seconds 2
      if (Test-DockerReady) { break }
    }
  }
}

if (-not (Test-DockerReady)) {
  throw 'Docker Engine is not running. Start Docker Desktop and rerun this script.'
}

Write-Host 'Docker is ready.'

# Initialize metadata DB / create default admin
Write-Host 'Running airflow-init...'
Invoke-Compose -Args @('up', 'airflow-init')

if ($InitOnly) {
  Write-Host 'Init completed (InitOnly set).'
  exit 0
}

if (-not $NoStart) {
  Write-Host 'Starting Airflow services...'
  Invoke-Compose -Args @('up', '-d')
}

Write-Host 'Checking DAG registration...'
Invoke-Compose -Args @('exec', 'airflow-scheduler', 'airflow', 'dags', 'list')

 $webPort = '8080'
 $envFile = Join-Path $airflowDir '.env'
 if (Test-Path $envFile) {
   $line = Get-Content $envFile | Where-Object { $_ -match '^AIRFLOW_WEB_PORT=' } | Select-Object -First 1
   if ($line) {
     $parts = $line -split '=', 2
     if ($parts.Count -eq 2 -and [string]::IsNullOrWhiteSpace($parts[1]) -eq $false) {
       $webPort = $parts[1].Trim()
     }
   }
 }

Write-Host ''
Write-Host "Airflow UI: http://localhost:$webPort"
Write-Host 'Default quickstart user: airflow / airflow'
