#!/usr/bin/env sh
# JPL DE406: spans -3000 .. +3000, ~287 MiB. Not committed (see .gitignore).
set -eu
URL="https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de406.bsp"
OUT="${1:-de406.bsp}"
[ -f "$OUT" ] && { echo "$OUT already present"; exit 0; }
echo "Fetching $URL -> $OUT"
curl -fSL --progress-bar -o "$OUT" "$URL"
