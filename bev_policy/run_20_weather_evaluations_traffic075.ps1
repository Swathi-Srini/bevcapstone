$ErrorActionPreference = 'Stop'

$python = '.\venv\Scripts\python.exe'
$checkpoint = '.\bev_policy\checkpoints\ppo_bev_adaptive_safety_v1.zip'
$outputRoot = '.\bev_policy\outputs\ppo_weather_20_traffic075_h1000'

$conditions = @(
    @{ Name = 'clear_0.00'; Weather = 'clear'; Level = '0.0' },
    @{ Name = 'fog_0.30';   Weather = 'fog';   Level = '0.3' },
    @{ Name = 'fog_1.00';   Weather = 'fog';   Level = '1.0' },
    @{ Name = 'rain_0.30';  Weather = 'rain';  Level = '0.3' },
    @{ Name = 'rain_1.00';  Weather = 'rain';  Level = '1.0' }
)

foreach ($condition in $conditions) {
    $outputDir = Join-Path $outputRoot $condition.Name
    & $python -m bev_policy.evaluate `
        --checkpoint $checkpoint `
        --episodes 20 `
        --max-steps 1000 `
        --seed 30000 `
        --num-scenarios 30050 `
        --traffic-density 0.075 `
        --weather $condition.Weather `
        --weather-level $condition.Level `
        --inference-size 320 `
        --confidence-threshold 0.25 `
        --stereo-scale 0.5 `
        --output-dir $outputDir
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $($condition.Name) with exit code $LASTEXITCODE"
    }
}

& $python -m bev_policy.validate_weather_energy_outputs `
    --input-root $outputRoot `
    --expected-episodes 20
if ($LASTEXITCODE -ne 0) {
    throw "Energy trace validation failed with exit code $LASTEXITCODE. Inspect $outputRoot\energy_validation_report.md"
}

Write-Host "All five 20-episode weather evaluations and energy validations finished. Results: $outputRoot"
