# Get the project root directory
$env:PROJECT_ROOT = $PSScriptRoot

# Create necessary directories
@(
    "$env:PROJECT_ROOT/data",
    "$env:PROJECT_ROOT/data/raw",
    "$env:PROJECT_ROOT/data/processed",
    "$env:PROJECT_ROOT/models",
    "$env:PROJECT_ROOT/logs",
    "$env:PROJECT_ROOT/results"
) | ForEach-Object {
    if (!(Test-Path $_)) {
        New-Item -ItemType Directory -Path $_
    }
}

# Set environment variables
$env:DATA_DIR = "$env:PROJECT_ROOT/data"
$env:RAW_DATA_DIR = "$env:PROJECT_ROOT/data/raw"
$env:PROCESSED_DATA_DIR = "$env:PROJECT_ROOT/data/processed"
$env:MODEL_DIR = "$env:PROJECT_ROOT/models"
$env:LOG_DIR = "$env:PROJECT_ROOT/logs"
$env:RESULTS_DIR = "$env:PROJECT_ROOT/results"


# Run the script
python -m src.main