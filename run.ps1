# Choose an existing Python with NumPy; never installs packages or changes settings.
$ErrorActionPreference = 'Stop'
$candidatePaths = @((Join-Path $PSScriptRoot '.venv\Scripts\python.exe'))
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand -and $pythonCommand.Source -notlike '*\WindowsApps\*') {
    $candidatePaths += $pythonCommand.Source
}
$candidatePaths += Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$environmentCheckPath = Join-Path $PSScriptRoot 'scripts\check_environment.py'
$selectedPython = $null
foreach ($candidatePath in $candidatePaths) {
    if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
        # Windows PowerShell 5 strips embedded quotes in native -c arguments.
        # A file also lets an unsuitable candidate exit quietly and fall back.
        try {
            & $candidatePath $environmentCheckPath 2>$null
        } catch {
            continue
        }
        if ($LASTEXITCODE -eq 0) {
            $selectedPython = $candidatePath
            break
        }
    }
}
if (-not $selectedPython) {
    throw 'Python 3.10+ with NumPy 2.x was not found. Follow README.md to create .venv.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($args.Count -eq 0) {
        & $selectedPython -m recsys --help
    } else {
        & $selectedPython -m recsys @args
    }
    $commandExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $commandExitCode
