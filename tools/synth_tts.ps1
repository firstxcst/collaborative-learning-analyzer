param(
    [Parameter(Mandatory = $true)][string]$JobsFile
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$jobs = Get-Content -Raw -Encoding UTF8 $JobsFile | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

foreach ($job in $jobs) {
    $synth.SelectVoice($job.voice)
    $synth.Rate = [int]$job.rate      # -10..10
    $synth.Volume = 100
    $synth.SetOutputToWaveFile($job.out)
    $synth.Speak($job.text)
    $synth.SetOutputToNull()
}
$synth.Dispose()
Write-Output "SYNTH_DONE"
