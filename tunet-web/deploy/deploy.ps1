# Deploy tunet-web to VPS (149.28.38.9 -> https://tunet.cleanplatefx.com).
#
# Default behavior: build web, push web + python repo, restart service.
#
# Usage (from anywhere):
#   pwsh tunet-web/deploy/deploy.ps1               # web + repo
#   pwsh tunet-web/deploy/deploy.ps1 -SkipBuild    # reuse existing .next/standalone
#   pwsh tunet-web/deploy/deploy.ps1 -SkipWeb      # only push python repo
#   pwsh tunet-web/deploy/deploy.ps1 -SkipRepo     # only push web
#   pwsh tunet-web/deploy/deploy.ps1 -PushEnv      # also push deploy/tunet-web.env to /etc/tunet-web/env
#
# Requirements:
#   - PuTTY installed at "C:\Program Files\PuTTY\" (plink + pscp)
#   - VPS / VPS_USER / VPS_PASS set in tunet/.env
#   - Node + npm on PATH for the build
#
# What it does:
#   1. npm run build (unless -SkipBuild or -SkipWeb)
#   2. Stages .next/standalone with .next/static (and public/ if present)
#   3. Tars the python repo (with spark-packer's EXCLUDE_PATTERNS) unless -SkipRepo
#   4. Uploads tarballs to /tmp on the VPS
#   5. Extracts web into /opt/tunet-web, repo into /opt/tunet-web/repo
#   6. chowns everything to tunet:tunet
#   7. Restarts the systemd service
#   8. Smoke-tests https://tunet.cleanplatefx.com (expects 307 -> /sign-in)

[CmdletBinding()]
param(
  [switch]$SkipBuild,
  [switch]$SkipWeb,
  [switch]$SkipRepo,
  [switch]$PushEnv
)

# Note: do NOT set $ErrorActionPreference = 'Stop' globally. In Windows
# PowerShell 5.1 that turns every line of stderr from a native exe (npm, tar,
# plink) into a terminating error, even when the exe returned 0. Instead we
# check $LASTEXITCODE explicitly after each native call.

# ── Paths ────────────────────────────────────────────────────────────────────
$scriptDir = $PSScriptRoot
$webRoot   = Resolve-Path (Join-Path $scriptDir '..')
$repoRoot  = Resolve-Path (Join-Path $webRoot   '..')
$envFile   = Join-Path $repoRoot '.env'

# ── Read VPS credentials from tunet/.env ─────────────────────────────────────
if (-not (Test-Path $envFile)) { throw "Missing $envFile (need VPS / VPS_USER / VPS_PASS)" }
$envMap = @{}
foreach ($line in Get-Content $envFile) {
  if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
    $val = $Matches[2]
    if ($val.StartsWith('"') -and $val.EndsWith('"')) { $val = $val.Substring(1, $val.Length - 2) }
    elseif ($val.StartsWith("'") -and $val.EndsWith("'")) { $val = $val.Substring(1, $val.Length - 2) }
    $envMap[$Matches[1]] = $val
  }
}
$vps     = $envMap['VPS']
$vpsUser = $envMap['VPS_USER']
$vpsPass = $envMap['VPS_PASS']
if (-not $vps -or -not $vpsUser -or -not $vpsPass) { throw "VPS / VPS_USER / VPS_PASS missing in $envFile" }

# Pinned host key from first deploy. If you ever rebuild the VPS, update this.
$hostKey = 'SHA256:p/VnWr2QIWtGTYAZhK1UuPOKgn4PIdbeVYt3Zfoegl8'
$plink   = 'C:\Program Files\PuTTY\plink.exe'
$pscp    = 'C:\Program Files\PuTTY\pscp.exe'
foreach ($exe in @($plink, $pscp)) {
  if (-not (Test-Path $exe)) { throw "Missing $exe -- install PuTTY or update the path in this script" }
}

function Step([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function SshRun([string]$cmd) {
  & $plink -batch -hostkey $hostKey -ssh -pw $vpsPass "$vpsUser@$vps" $cmd
  if ($LASTEXITCODE -ne 0) { throw "SSH failed (exit $LASTEXITCODE)" }
}
function ScpUp([string]$local, [string]$remote) {
  & $pscp -batch -hostkey $hostKey -pw $vpsPass $local "${vpsUser}@${vps}:${remote}"
  if ($LASTEXITCODE -ne 0) { throw "SCP failed (exit $LASTEXITCODE)" }
}

Set-Location $webRoot
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$webTar  = $null
$repoTar = $null

# ── Web: build + stage + tar ─────────────────────────────────────────────────
# Build into .next-prod (not .next) so we never clobber a running `next dev`
# server's .next/ — see next.config.ts distDir + TUNET_DIST_DIR.
$distDir = '.next-prod'
if (-not $SkipWeb) {
  if (-not $SkipBuild) {
    Step "Building tunet-web (next build -> $distDir)"
    $env:NODE_ENV = 'production'
    $env:TUNET_DIST_DIR = $distDir
    npm run build
    $buildExit = $LASTEXITCODE
    Remove-Item Env:\TUNET_DIST_DIR -ErrorAction SilentlyContinue
    if ($buildExit -ne 0) { throw 'next build failed' }
  }

  if (-not (Test-Path "$distDir/standalone/server.js")) {
    throw "Missing $distDir/standalone/server.js -- output: 'standalone' must be set in next.config.ts"
  }

  Step "Staging $distDir/standalone"
  $stagedStatic = "$distDir/standalone/$distDir/static"
  if (Test-Path $stagedStatic) { Remove-Item -Recurse -Force $stagedStatic }
  Copy-Item -Recurse -Force "$distDir/static" $stagedStatic

  $stagedPublic = "$distDir/standalone/public"
  if (Test-Path 'public') {
    if (Test-Path $stagedPublic) { Remove-Item -Recurse -Force $stagedPublic }
    Copy-Item -Recurse -Force 'public' $stagedPublic
  }

  $webTar = Join-Path $env:TEMP "tunet-web-deploy-$stamp.tar.gz"
  Step "Creating web tarball"
  tar -czf $webTar -C "$distDir/standalone" .
  if ($LASTEXITCODE -ne 0) { throw 'tar (web) failed' }
  Write-Host ("    {0:N1} MB" -f ((Get-Item $webTar).Length / 1MB))

  Step "Uploading web tarball"
  ScpUp $webTar '/tmp/tunet-web-deploy.tar.gz'
}

# ── Repo: tar + upload (mirrors spark-packer.ts EXCLUDE_PATTERNS) ────────────
if (-not $SkipRepo) {
  $repoTar = Join-Path $env:TEMP "tunet-repo-$stamp.tar.gz"
  # Keep this list aligned with EXCLUDE_PATTERNS/EXCLUDE_EXTS in
  # tunet-web/src/lib/spark-packer.ts so what we ship matches what the live
  # packer would have produced.
  $excludes = @(
    'node_modules', '.git', '.venv', '.pytest_cache', '__pycache__',
    '.claude', '_archive', '_inference_cache', '_internal',
    'Spark', 'tunet-web', 'docs',
    'output', 'finetuned_outputs', 'data', 'src', 'dst',
    'inputs', 'outputs',
    'paintout-test-A', 'paintout-test-B',
    'finetune-test', 'finetune-test-1024',
    'tunet_session.yaml', 'spark_dashboard_settings.json', '_spark_panel.json',
    'benchmark_chart.png',
    '*.pth', '*.onnx', '*.pyc', '*.log', '*.tar.gz', '*.zip',
    '*.exr', '*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'
  )
  $excludeArgs = $excludes | ForEach-Object { "--exclude=$_" }

  Step 'Packing python repo'
  & tar @excludeArgs -czf $repoTar -C $repoRoot .
  if ($LASTEXITCODE -ne 0) { throw 'tar (repo) failed' }
  Write-Host ("    {0:N1} MB" -f ((Get-Item $repoTar).Length / 1MB))

  Step 'Uploading repo tarball'
  ScpUp $repoTar '/tmp/tunet-repo.tar.gz'
}

# ── Env: optional push ──────────────────────────────────────────────────────
if ($PushEnv) {
  $envSrc = Join-Path $scriptDir 'tunet-web.env'
  if (-not (Test-Path $envSrc)) { throw "Missing $envSrc -- needed for -PushEnv" }
  Step 'Uploading env file'
  ScpUp $envSrc '/tmp/tunet-web.env'
}

# ── Extract + restart (one SSH round-trip) ──────────────────────────────────
Step 'Extracting + restarting tunet-web'
$remoteCmd = "set -e`n"
if (-not $SkipWeb) {
  # Remove the stale .next from pre-distDir deploys (runtime now uses .next-prod).
  $remoteCmd += @'
rm -rf /opt/tunet-web/.next
tar -xzf /tmp/tunet-web-deploy.tar.gz -C /opt/tunet-web
rm -f /tmp/tunet-web-deploy.tar.gz

'@
}
if (-not $SkipRepo) {
  # Wipe the repo dir contents (but keep the dir) so removed files disappear,
  # then extract fresh. Source tree is small (<5 MB), so full replacement is
  # cheaper than diffing.
  $remoteCmd += @'
mkdir -p /opt/tunet-web/repo
find /opt/tunet-web/repo -mindepth 1 -delete
tar -xzf /tmp/tunet-repo.tar.gz -C /opt/tunet-web/repo
rm -f /tmp/tunet-repo.tar.gz
test -f /opt/tunet-web/repo/train.py || { echo 'ERROR: train.py not in repo tarball'; exit 1; }

'@
}
$remoteCmd += @'
chown -R tunet:tunet /opt/tunet-web

'@
if ($PushEnv) {
  $remoteCmd += @'
install -o root -g tunet -m 0640 /tmp/tunet-web.env /etc/tunet-web/env
rm -f /tmp/tunet-web.env

'@
}
$remoteCmd += @'
systemctl restart tunet-web
sleep 2
systemctl is-active tunet-web
'@
SshRun $remoteCmd

if ($webTar)  { Remove-Item $webTar  -ErrorAction SilentlyContinue }
if ($repoTar) { Remove-Item $repoTar -ErrorAction SilentlyContinue }

# ── Smoke test ───────────────────────────────────────────────────────────────
Step 'Smoke-testing https://tunet.cleanplatefx.com'
try {
  $resp = Invoke-WebRequest -Uri 'https://tunet.cleanplatefx.com/' -MaximumRedirection 0 -UseBasicParsing
  $loc  = $resp.Headers['Location']
  Write-Host "    HTTP $($resp.StatusCode)    Location: $loc"
  if ($resp.StatusCode -ne 307 -and $resp.StatusCode -ne 200) {
    Write-Warning "Expected 200 or 307, got $($resp.StatusCode)"
  }
} catch {
  Write-Warning "Smoke test failed: $($_.Exception.Message)"
}

Write-Host ''
Write-Host 'Done. Live at https://tunet.cleanplatefx.com' -ForegroundColor Green
