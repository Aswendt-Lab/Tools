#!/usr/bin/env bash

set -u
IFS=$'\n\t'

for zip in input/MRI/proc_data/*.zip; do
    folder="${zip%.zip}"

    echo "======================================"
    echo "Processing: $zip"
    echo "Target folder: $folder"
    echo "======================================"

    failed=0

    # 1. datalad get
    if ! datalad get "$zip"; then
        echo "ERROR: datalad get failed for $zip"
        failed=1
    fi

    # 2. datalad unlock
    if ! datalad unlock "$zip"; then
        echo "ERROR: datalad unlock failed for $zip"
        failed=1
    fi

    # 3. unzip (only if folder does not exist)
    if [[ -d "$folder" ]]; then
        echo "Folder already exists, skipping unzip: $folder"
    else
        if ! unzip -o "$zip"; then
            echo "WARNING: unzip failed for $zip"
            failed=1
        fi
    fi

    # Skip remaining steps if something failed
    if [[ $failed -ne 0 ]]; then
        echo "Skipping save/push/drop for $zip due to errors"
        echo
        continue
    fi

    # 4. save extracted content + zip state
    if ! datalad save -m "replace zip file $folder" "$folder" "$zip"; then
        echo "ERROR: datalad save failed for $zip"
        echo
        continue
    fi

    # 5. push to origin
    if ! datalad push --to origin; then
        echo "ERROR: datalad push failed for $zip"
        echo
        continue
    fi

    # 6. drop local content (ONLY after successful push)
    if ! datalad drop "$folder" "$zip"; then
        echo "WARNING: datalad drop failed for $zip or $folder"
        echo
        continue
    fi

    echo "Done (saved, pushed, dropped): $zip"
    echo
done
