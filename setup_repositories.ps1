$experimentRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $experimentRoot 'algorithms.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$workspaceRoot = Split-Path -Parent $experimentRoot
$ablationDirectory = Get-ChildItem -LiteralPath $workspaceRoot -Directory |
    Where-Object { $_.Name -like '01*' } |
    Select-Object -First 1
$ablationRoot = if ($ablationDirectory) { $ablationDirectory.FullName } else { $null }

function Write-CommitRecord {
    param(
        [string]$Target,
        [string]$RecordPath,
        [string]$Source
    )
    $commit = & git -C $Target rev-parse HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $commit) {
        "source=$Source`ncommit=$commit`nrecorded_at=$([DateTime]::Now.ToString('s'))" | Set-Content -LiteralPath $RecordPath -Encoding UTF8
    } else {
        "source=$Source`ncommit=unavailable`nrecorded_at=$([DateTime]::Now.ToString('s'))" | Set-Content -LiteralPath $RecordPath -Encoding UTF8
    }
}

foreach ($method in $manifest.methods) {
    $repoDirectory = if ($method.repo_dir) { [string]$method.repo_dir } else { 'repo' }
    $sourceMode = if ($method.source_mode) { [string]$method.source_mode } else { 'external' }
    $target = Join-Path (Join-Path $experimentRoot $method.directory) $repoDirectory
    $recordPath = Join-Path (Join-Path $experimentRoot $method.directory) 'source_commit.txt'

    if ($sourceMode -eq 'bundled') {
        if (-not (Test-Path -LiteralPath $target)) {
            Write-Error "Bundled source is missing for $($method.name): $target"
            exit 1
        }
        Write-Host "[bundled] $($method.id) $($method.name): $target"
        if (-not (Test-Path -LiteralPath $recordPath)) {
            "source=bundled in comparison repository`ncommit=tracked with comparison repository`nrecorded_at=$([DateTime]::Now.ToString('s'))" | Set-Content -LiteralPath $recordPath -Encoding UTF8
        }
        continue
    }

    if ($ablationRoot -and $method.url -eq 'local implementation') {
        $localSource = Join-Path $ablationRoot 'our-gaussian-splatting'
    } elseif ($ablationRoot -and $method.id -eq '02') {
        $localSource = Join-Path $ablationRoot 'external_methods\2dgs\repo'
    } elseif ($ablationRoot -and $method.id -eq '03') {
        $localSource = Join-Path $ablationRoot 'external_methods\gaussian_surfels\repo'
    } else {
        $localSource = $null
    }

    if (Test-Path -LiteralPath $target) {
        Write-Host "[exists] $($method.id) $($method.name): $target"
        if (Test-Path -LiteralPath (Join-Path $target '.git')) {
            Write-CommitRecord -Target $target -RecordPath $recordPath -Source $method.url
            if ($localSource) {
                "overlay=$localSource" | Add-Content -LiteralPath $recordPath -Encoding UTF8
            }
        } elseif (-not (Test-Path -LiteralPath $recordPath)) {
            # Preserve provenance for local implementations that are copied into
            # the experiment directory instead of being cloned from GitHub.
            Write-CommitRecord -Target $target -RecordPath $recordPath -Source $method.url
        }
        continue
    }

    if ($localSource -and (Test-Path -LiteralPath $localSource)) {
        Write-Host "[copy] $($method.id) $($method.name) from existing local source"
        Copy-Item -LiteralPath $localSource -Destination $target -Recurse -Force
        Write-CommitRecord -Target $target -RecordPath $recordPath -Source "copied from $localSource"
        continue
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Write-Host "[clone] $($method.id) $($method.name)"
    & git clone --depth 1 --recurse-submodules $method.url $target
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Clone failed for $($method.name): $($method.url)"
        exit $LASTEXITCODE
    }
    Write-CommitRecord -Target $target -RecordPath $recordPath -Source $method.url
}

Write-Host 'Repository setup finished. No training or rendering was started.'
