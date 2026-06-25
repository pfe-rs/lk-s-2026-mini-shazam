# Ova skripta skida FFmpeg 8.1.2 i stavlja ga u src/tools/.
# Posle toga normalizer.py ce ga automatski naci i koristiti.
#
# Verzija je fiksna (autobuild-2026-06-24-13-41, FFmpeg n8.1.2)
# tako da svi na projektu koriste potpuno isti FFmpeg.
# Ovo je vazno za determinizam — two-pass loudnorm mora da bude
# identican na svakoj masini.
#
# Ako nesto ne radi, kontaktiraj Lazara (kripticni).

$ReleaseTag = "autobuild-2026-06-24-13-41"
$BaseUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/$ReleaseTag"

# SHA256 arhive — ako se ne poklapa, arhiva je izmenjena ili je
# skidanje bilo neispravno. U tom slucaju javi Lazu.
$ExpectedChecksum = "7a722cf2c8d45531bb89ddc902d8144b076547aa0ab6c3a9250436a0ed96f3f5"

$ToolsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FfmpegDst = Join-Path $ToolsDir "ffmpeg.exe"

if (Test-Path $FfmpegDst) {
    Write-Host "FFmpeg vec postoji na $FfmpegDst"
    & $FfmpegDst -version 2>&1 | Select-Object -First 1
    exit 0
}

$Archive = "ffmpeg-n8.1.2-win64-gpl-8.1.zip"

Write-Host "Skidam FFmpeg 8.1.2 (win64) sa BtbN/FFmpeg-Builds ..."

$ZipPath = Join-Path $env:TEMP $Archive
Invoke-WebRequest -Uri "${BaseUrl}/${Archive}" -OutFile $ZipPath

# Proveri da li je arhiva stigla cela i neizmenjena
Write-Host "Proveravam SHA256 checksum ..."
$Computed = (Get-FileHash -Algorithm SHA256 -Path $ZipPath).Hash.ToLower()
if ($Computed -ne $ExpectedChecksum) {
    Write-Host ""
    Write-Host "GREŠKA: SHA256 se ne poklapa!"
    Write-Host "  Ocekivan:  $ExpectedChecksum"
    Write-Host "  Dobijen:   $Computed"
    Write-Host ""
    Write-Host "Arhiva je mozda izmenjena ili je download bio neispravan."
    Write-Host "Kontaktiraj Lazara (kripticni) da dobijes ispravan ffmpeg."
    Remove-Item $ZipPath
    exit 1
}
Write-Host "  OK ($Computed)"

# Raspakuj i nadji ffmpeg.exe
Write-Host "Raspakujem arhivu ..."
$ExtractDir = Join-Path $env:TEMP "ffmpeg-extract"
Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force
$Exe = Get-ChildItem -Recurse -Filter "ffmpeg.exe" -Path $ExtractDir | Select-Object -First 1
if (-not $Exe) {
    Write-Error "ffmpeg.exe nije nadjen u arhivi."
    exit 1
}
Copy-Item $Exe.FullName -Destination $FfmpegDst

# Ocisti temp fajlove
Remove-Item $ZipPath
Remove-Item $ExtractDir -Recurse -Force

Write-Host ""
Write-Host "Gotovo:"
& $FfmpegDst -version 2>&1 | Select-Object -First 1
