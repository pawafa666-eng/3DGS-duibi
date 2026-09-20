# Compatibility entrypoint: use the same implementation as the Python/BAT entrypoints.
$scriptPath = Join-Path $PSScriptRoot 'setup_repositories.py'
& python -u $scriptPath @args
exit $LASTEXITCODE
