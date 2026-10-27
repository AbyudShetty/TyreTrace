param([string]$Root = 'C:\ABYUD\HACKATHONS\TRACKSHIFT')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.SelectVoice('Microsoft David Desktop')
$speaker.Rate = 3
$segments = Get-Content -LiteralPath (Join-Path $Root 'out/demo/narration.json') -Raw | ConvertFrom-Json
try {
    for ($i=0; $i -lt $segments.Count; $i++) {
        $destination = Join-Path $Root ('out/demo/voice-{0:D2}.wav' -f $i)
        $speaker.SetOutputToWaveFile($destination)
        $speaker.Speak($segments[$i].text)
        $speaker.SetOutputToNull()
    }
} finally {
    $speaker.Dispose()
}
Write-Output 'Local synthetic narration written.'
