param([string]$OutputDirectory = (Join-Path $PSScriptRoot '../fixtures/asr'))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$fixtureRoot = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $fixtureRoot) { throw "Refusing to overwrite fixture directory: $fixtureRoot" }
New-Item -ItemType Directory -Path $fixtureRoot | Out-Null
$format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)
$cases = @(
    @{ name = 'zh_system_tts'; voice = 'Microsoft Huihui Desktop'; text = '这是语音识别功能测试。今天我们在国产算力平台上验证中文语音识别。' },
    @{ name = 'en_system_tts'; voice = 'Microsoft Zira Desktop'; text = 'This is a speech recognition test. We are checking a model on a single accelerator.' }
)
$records = foreach ($case in $cases) {
    $synth = [System.Speech.Synthesis.SpeechSynthesizer]::new()
    try {
        $synth.SelectVoice($case.voice)
        $synth.Rate = 0
        $path = Join-Path $fixtureRoot ($case.name + '.wav')
        $synth.SetOutputToWaveFile($path, $format)
        $synth.Speak($case.text)
        $synth.SetOutputToNull()
        @{
            file = $case.name + '.wav'; expected_text = $case.text; voice = $case.voice
            sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
            sample_rate_hz = 16000; channels = 1; sample_bits = 16
        }
    } finally { $synth.Dispose() }
}
$manifest = @{
    scope = 'Locally synthesized transport and transcription diagnostics; not an official dataset or accuracy score.'
    source = 'Windows System.Speech installed desktop voices'; cases = @($records)
}
$json = $manifest | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path $fixtureRoot 'manifest.json'), $json + "`n", [Text.UTF8Encoding]::new($false))
$records | Select-Object file,sha256
