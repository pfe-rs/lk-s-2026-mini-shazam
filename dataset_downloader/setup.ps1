$md5Expected = "d3ebfd86e283345ee2366a5492495935"

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

if (-not (Test-Path "fma")) {
    if (-not (Test-Path "fma_metadata.zip")) {
        Write-Host "Preuzimanje fma_metadata.zip sa FMA servera..."
        try {
            Invoke-WebRequest -Uri https://os.unil.cloud.switch.ch/fma/fma_metadata.zip -OutFile fma_metadata.zip
        } catch {
            Write-Host "Neuspesno preuzimanje: $_"
        }
    }

    if (Test-Path "fma_metadata.zip") {
        $md5Actual = (Get-FileHash -Algorithm MD5 -Path fma_metadata.zip).Hash.ToLower()
        if ($md5Actual -eq $md5Expected) {
            Expand-Archive -Path fma_metadata.zip -DestinationPath .
            Remove-Item fma_metadata.zip
        } else {
            Write-Host "MD5 ne odgovara. Ocekivano: $md5Expected, dobijeno: $md5Actual"
            Remove-Item fma_metadata.zip
            Write-Host "Kontaktiraj Lazara (kripticni) da dobijes fma_metadata.zip (MD5: $md5Expected)"
            exit 1
        }
    } else {
        Write-Host "Kontaktiraj Lazara (kripticni) da dobijes fma_metadata.zip (MD5: $md5Expected)"
        exit 1
    }
}

Write-Host "Spremno. Aktiviraj venv sa: .\venv\Scripts\Activate.ps1"
