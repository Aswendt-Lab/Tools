#!/bin/bash

# Requirements check
if ! command -v pigz &> /dev/null; then
  echo "❌ pigz not found. Install it with: brew install pigz"
  exit 1
fi

if ! command -v pv &> /dev/null; then
  echo "❌ pv not found. Install it with: brew install pv"
  exit 1
fi

WORKDIR="${1:-.}"
cd "$WORKDIR" || exit 1

for folder in */ ; do
  foldername="${folder%/}"
  zipname="${foldername}.zip"

  echo "📦 Preparing: $foldername -> $zipname"

  # Use macOS-compatible du to get size in bytes
  total_size=$(du -sk "$foldername" | awk '{print $1}')
  total_size_bytes=$((total_size * 1024))  # Convert KB to bytes

  # Create a temporary uncompressed zip archive
  tmpfile=$(mktemp "${zipname}.tmp.XXXXXX")

  echo "⚙️  Archiving (no compression)..."
  zip -r -0 -X - "$foldername" | pv -s "$total_size_bytes" > "$tmpfile"

  echo "⚙️  Compressing with pigz..."
  pigz -3 -f "$tmpfile"  # creates "$tmpfile.gz"

  # Rename the compressed file to .zip
  mv "${tmpfile}.gz" "$zipname"

  echo "✅ Done: $zipname"
  echo
done

echo "🎉 All folders zipped with progress bar in $WORKDIR"
