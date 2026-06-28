#!/usr/bin/env bash
set -euo pipefail

# Ova skripta skida FFmpeg 8.1.2 i stavlja ga u src/tools/.
# Posle toga normalizer.py ce ga automatski naci i koristiti.
#
# Verzija je fiksna (autobuild-2026-06-24-13-41, FFmpeg n8.1.2)
# tako da svi na projektu koriste potpuno isti FFmpeg.
# Ovo je vazno za determinizam — two-pass loudnorm mora da bude
# identican na svakoj masini.
#
# Ako nesto ne radi, kontaktiraj Lazara (kripticni).

RELEASE_TAG="autobuild-2026-06-24-13-41"
BASE_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/${RELEASE_TAG}"

# SHA256 arhive — ako se ne poklapa, arhiva je izmenjena ili je
# skidanje bilo neispravno. U tom slucaju javi Lazu.
EXPECTED_CHECKSUM="db78ca2f02f39666fac26a6f31c3816447a0a668d05279d61aed5023ce55783b"

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
FFMPEG_DST="$TOOLS_DIR/ffmpeg"

if [ -f "$FFMPEG_DST" ] && [ -x "$FFMPEG_DST" ]; then
    echo "FFmpeg vec postoji na $FFMPEG_DST"
    "$FFMPEG_DST" -version 2>&1 | head -1
    exit 0
fi

# Podrzane arhitekture: x86_64 (Intel/AMD), aarch64 (Apple Silicon/RPi) nema
# n8.1.2 build, samo master.
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)  BF="linux64"   ARCHIVE="ffmpeg-n8.1.2-linux64-gpl-8.1.tar.xz"  ;;
    aarch64)
        echo "Nema n8.1.2 za arm64. Koristi RELEASE_TAG=latest za master build."
        exit 1
        ;;
    *)
        echo "Nepoznata arhitektura: $ARCH"
        exit 1
        ;;
esac

echo "Skidam FFmpeg 8.1.2 ($BF) sa BtbN/FFmpeg-Builds ..."
curl -fL "${BASE_URL}/${ARCHIVE}" -o /tmp/ffmpeg.tar.xz

# Proveri da li je arhiva stigla cela i neizmenjena
echo "Proveravam SHA256 checksum ..."
COMPUTED=$(sha256sum /tmp/ffmpeg.tar.xz | cut -d' ' -f1)
if [ "$COMPUTED" != "$EXPECTED_CHECKSUM" ]; then
    echo ""
    echo "GREŠKA: SHA256 se ne poklapa!"
    echo "  Ocekivan:  $EXPECTED_CHECKSUM"
    echo "  Dobijen:   $COMPUTED"
    echo ""
    echo "Arhiva je mozda izmenjena ili je download bio neispravan."
    echo "Kontaktiraj Lazara (kripticni) da dobijes ispravan ffmpeg."
    rm -f /tmp/ffmpeg.tar.xz
    exit 1
fi
echo "  OK ($COMPUTED)"

# Raspakuj i nadji ffmpeg binarni fajl
echo "Raspakujem arhivu ..."
mkdir -p /tmp/ffmpeg-extract
tar xf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg-extract
mkdir -p "$TOOLS_DIR"
extracted=$(find /tmp/ffmpeg-extract -type f -name "ffmpeg" | head -1)
if [ -z "$extracted" ]; then
    echo "GREŠKA: ffmpeg binarni fajl nije nadjen u arhivi."
    exit 1
fi
cp "$extracted" "$FFMPEG_DST"
chmod +x "$FFMPEG_DST"

# Ocisti temp fajlove
rm -rf /tmp/ffmpeg.tar.xz /tmp/ffmpeg-extract

echo ""
echo "Gotovo:"
"$FFMPEG_DST" -version 2>&1 | head -1
