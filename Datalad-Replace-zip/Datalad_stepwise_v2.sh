#!/usr/bin/env bash

set +e  # never exit on error

BASE_DIR="input/MRI/proc_data"
BATCH_SIZE=5
count=0

echo "======================================"
echo "PHASE 1: datalad get + unlock (NO unzip)"
echo "Batch size: $BATCH_SIZE"
echo "======================================"

for zip in "$BASE_DIR"/*.zip; do
    zipname=$(basename "$zip")

    echo
    echo "--------------------------------------"
    echo "ZIP: $zipname"
    echo "--------------------------------------"

    # Skip if file exists AND is unlocked
    if [ -f "$zip" ] && datalad status "$zip" 2>/dev/null | grep -q "unlocked"; then
        echo "Already present & unlocked, skipping get/unlock"
    else
        datalad get "$zip"
        if [ $? -ne 0 ]; then
            echo "ERROR: datalad get failed for $zipname"
            continue
        fi

        datalad unlock "$zip"
        if [ $? -ne 0 ]; then
            echo "ERROR: datalad unlock failed for $zipname"
            continue
        fi
    fi

    count=$((count + 1))

    if (( count % BATCH_SIZE == 0 )); then
        echo
        echo "======================================"
        echo "Processed $count ZIP files."
        echo "👉 Inspect, unzip, verify manually."
        echo "Press ENTER to continue."
        echo "======================================"
        read
    fi
done

echo
echo "======================================"
echo "PHASE 1 complete."
echo "Manually unzip and verify folders now."
echo "Press ENTER to start PHASE 2 (save/push/drop)."
echo "======================================"
read

echo
echo "======================================"
echo "PHASE 2: datalad save + push + drop"
echo "======================================"

for zip in "$BASE_DIR"/*.zip; do
    zipname=$(basename "$zip")
    folder="${zipname%.zip}"
    folder_path="$BASE_DIR/$folder"

    echo
    echo "--------------------------------------"
    echo "ZIP: $zipname"
    echo "Folder: $folder"
    echo "--------------------------------------"

    if [ ! -d "$folder_path" ]; then
        echo "WARNING: Folder missing, skipping save/push/drop"
        continue
    fi

    datalad save -m "replace zip file $folder" "$folder_path"
    if [ $? -ne 0 ]; then
        echo "ERROR: datalad save failed for $folder"
        continue
    fi

    datalad push --to origin
    if [ $? -ne 0 ]; then
        echo "ERROR: datalad push failed for $folder"
    fi

    datalad drop "$folder_path" "$zip"
    if [ $? -ne 0 ]; then
        echo "ERROR: datalad drop failed for $folder"
    fi

    echo "Done: $folder"
done

echo
echo "======================================"
echo "ALL DONE"
echo "======================================"
