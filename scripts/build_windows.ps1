$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Icon = Join-Path $Root "static\icons\icon.ico"
$DistApp = Join-Path $Root "dist\BatchBench"
$BuildDir = Join-Path $Root "build"
$Exe = Join-Path $DistApp "BatchBench.exe"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$BuildLogDir = Join-Path $Root "_work\build_logs"
$BuildLog = Join-Path $BuildLogDir ("build_windows_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
if ($env:BATCHBENCH_BUILD_PYTHON) {
    $Python = $env:BATCHBENCH_BUILD_PYTHON
}
elseif (Test-Path $VenvPython) {
    $Python = $VenvPython
}
else {
    $Python = "python"
}

function Write-BuildLogLine {
    param([AllowEmptyString()][string]$Message)
    Write-Host $Message
    Add-Content -LiteralPath $BuildLog -Encoding UTF8 -Value $Message
}

function Invoke-Logged {
    param(
        [string]$CommandPath,
        [string[]]$Arguments,
        [string]$FailureMessage
    )
    Write-BuildLogLine ("> {0} {1}" -f $CommandPath, ($Arguments -join " "))
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $CommandPath @Arguments 2>&1 | ForEach-Object {
            Write-BuildLogLine ([string]$_)
        }
        $Code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    Write-BuildLogLine ("Exit code: {0}" -f $Code)
    if ($Code -ne 0) {
        throw $FailureMessage
    }
}

if (!(Test-Path $Icon)) {
    throw "Missing required icon: static\icons\icon.ico"
}

Push-Location $Root
try {
    New-Item -ItemType Directory -Force -Path $BuildLogDir | Out-Null
    Set-Content -LiteralPath $BuildLog -Encoding UTF8 -Value @(
        "BatchBench Windows build log",
        ("Started: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
        "Root: $Root",
        "Python: $Python",
        ""
    )
    Write-BuildLogLine "Build log: $BuildLog"
    Invoke-Logged -CommandPath $Python -Arguments @("-c", "import importlib.util, sys; mods=['flask','werkzeug','dotenv','PIL','numpy','torch','transformers','huggingface_hub','safetensors','tokenizers','timm','hf_xet','httpx','httpcore','fsspec','requests','urllib3','charset_normalizer','pypresence','pystray']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('Missing dependencies: ' + ', '.join(missing)) if missing else None; sys.exit(1 if missing else 0)") -FailureMessage "Python cannot import required dependencies."

    Invoke-Logged -CommandPath $Python -Arguments @("-c", "from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification; import torch; import huggingface_hub; import safetensors; import tokenizers; import timm; print('Offline tagger dependencies OK')") -FailureMessage "Python cannot import required dependencies."

    Invoke-Logged -CommandPath $Python -Arguments @("-c", "import PyInstaller") -FailureMessage "PyInstaller is missing. Install requirements-dev.txt first."

    Invoke-Logged -CommandPath $Python -Arguments @("scripts\export_discord_asset.py") -FailureMessage "Failed to generate Discord Presence asset."

    foreach ($Target in @($DistApp, $BuildDir)) {
        if (Test-Path $Target) {
            $Resolved = (Resolve-Path $Target).Path
            if (!$Resolved.StartsWith($Root.Path, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to remove path outside project: $Resolved"
            }
            Remove-Item -LiteralPath $Resolved -Recurse -Force
        }
    }

    $AddData = @(
        "templates;templates",
        "static;static"
    )
    foreach ($Optional in @("presets")) {
        if (Test-Path (Join-Path $Root $Optional)) {
            $AddData += "$Optional;$Optional"
        }
    }
    foreach ($Optional in @(
        "README.md",
        "tag_editor_glossary.json",
        "preset_keep_warm_balanced.json",
        "preset_neutral_daylight.json",
        "preset_greyscale.json",
        "custom.json"
    )) {
        if (Test-Path (Join-Path $Root $Optional)) {
            $AddData += "$Optional;."
        }
    }

    $Args = @(
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name", "BatchBench",
        "--icon", "static\icons\icon.ico",
        "--noupx",
        "--runtime-hook", "scripts\pyinstaller_runtime_hook.py",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "timm",
        "--hidden-import", "pystray._win32"
    )
    $CollectAll = @(
        "transformers",
        "huggingface_hub",
        "safetensors",
        "tokenizers",
        "httpx",
        "httpcore",
        "fsspec",
        "requests",
        "urllib3",
        "charset_normalizer",
        "idna",
        "certifi",
        "anyio",
        "h11",
        "filelock",
        "packaging",
        "yaml",
        "regex",
        "tqdm",
        "typer",
        "shellingham",
        "hf_xet"
    )
    foreach ($Package in $CollectAll) {
        $Args += @("--collect-all", $Package)
    }
    $HiddenImports = @(
        "transformers.models.swinv2.configuration_swinv2",
        "transformers.models.vit.configuration_vit",
        "transformers.models.vit.image_processing_vit",
        "huggingface_hub._snapshot_download"
    )
    foreach ($Import in $HiddenImports) {
        $Args += @("--hidden-import", $Import)
    }
    foreach ($Item in $AddData) {
        $Args += @("--add-data", $Item)
    }
    $Args += "run_batchbench.pyw"

    Invoke-Logged -CommandPath $Python -Arguments (@("-m", "PyInstaller") + $Args) -FailureMessage "PyInstaller build failed."
    if (!(Test-Path $Exe)) { throw "Generated executable was not created: dist\BatchBench\BatchBench.exe" }

    Write-Host "Build complete."
    Write-Host "Run: dist\BatchBench\BatchBench.exe"
    Write-Host "Configure Discord Rich Presence in .env before launching."
}
finally {
    Pop-Location
}
