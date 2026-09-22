#!/bin/sh
# Copy the static build into a directory that can be served as-is,
# e.g. ./deploy.sh ~/public_html/hespectra
set -eu

if [ $# -ne 1 ]; then
    echo "usage: $0 <target-directory>" >&2
    exit 2
fi

target=$1
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")

mkdir -p "$target"

for f in index.html style.css app.js bridge.py; do
    cp "$here/$f" "$target/$f"
done

# The physics module lives in the repository root and is copied in unchanged,
# so the deployed directory is self-contained.
cp "$repo/helium_spectra_calc.py" "$target/helium_spectra_calc.py"

chmod a+rx "$target"
chmod a+r "$target"/*

echo "Deployed to $target:"
ls -1 "$target"
