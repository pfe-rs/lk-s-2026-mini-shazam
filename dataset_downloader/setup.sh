#!/usr/bin/env bash
set -euo pipefail

MD5_FMA_METADATA="d3ebfd86e283345ee2366a5492495935"

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

if [ ! -d "fma" ]; then
    if [ ! -f "fma_metadata.zip" ]; then
        echo "Preuzimanje fma_metadata.zip sa FMA servera..."
        curl -LO https://os.unil.cloud.switch.ch/fma/fma_metadata.zip || true
    fi

    if [ -f "fma_metadata.zip" ]; then
        echo "${MD5_FMA_METADATA}  fma_metadata.zip" | md5sum -c - && {
            unzip fma_metadata.zip
            rm fma_metadata.zip
        } || {
            echo "MD5 ne odgovara. FMA je verovatno azurirao fajl."
            rm -f fma_metadata.zip
            echo "Kontaktiraj Lazara (kripticni) da dobijes fma_metadata.zip (MD5: ${MD5_FMA_METADATA})"
            exit 1
        }
    else
        echo "Kontaktiraj Lazara (kripticni) da dobijes fma_metadata.zip (MD5: ${MD5_FMA_METADATA})"
        exit 1
    fi
fi

echo "Spremno. Aktiviraj venv sa: source venv/bin/activate"
