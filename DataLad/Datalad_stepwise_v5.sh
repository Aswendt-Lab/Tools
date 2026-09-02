#!/usr/bin/env bash

set +e  # continue on errors; process remaining files/datasets

# Usage:
#   ./Datalad_stepwise_v5.sh [PARENT_DIR]
#
# PARENT_DIR should contain one or more DataLad datasets as immediate subdirectories.
# The script lets you:
#   - choose one dataset or all datasets
#   - choose operation mode per dataset:
#       A) datalad get + datalad unlock on ALL files in the dataset
#       B) datalad save + datalad push + datalad drop --force on changed content
#
# Important behavior for mode A:
#   - each file is processed sequentially
#   - datalad get fully completes before datalad unlock starts
#   - next file starts only after current file is done
#   - next dataset starts only after current dataset is done
#
# Compatibility:
#   - avoids 'mapfile' so it works with the older Bash version on macOS
#   - includes both regular files and symlinks, because annexed content is often symlinked

PARENT_DIR="${1:-.}"
BATCH_SIZE=100

if [ ! -d "$PARENT_DIR" ]; then
    echo "ERROR: Parent directory not found: $PARENT_DIR"
    exit 1
fi

# ------------------------------------------------------------
# Discover datasets (immediate subdirectories containing .git)
# ------------------------------------------------------------
datasets=()
while IFS= read -r ds; do
    datasets+=("$ds")
done < <(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d -exec test -d "{}/.git" ';' -print | sort)

if [ ${#datasets[@]} -eq 0 ]; then
    echo "ERROR: No DataLad datasets found in: $PARENT_DIR"
    exit 1
fi

select_datasets() {
    echo "======================================"
    echo "Found DataLad datasets in: $PARENT_DIR"
    echo "======================================"
    for i in "${!datasets[@]}"; do
        printf "%2d) %s\n" "$((i + 1))" "$(basename "${datasets[$i]}")"
    done
    echo " a) ALL datasets"
    echo "======================================"
    read -rp "Select dataset number or 'a' for all: " selection

    selected_datasets=()
    if [[ "$selection" == "a" || "$selection" == "A" || "$selection" == "all" || "$selection" == "ALL" ]]; then
        selected_datasets=("${datasets[@]}")
    else
        if ! [[ "$selection" =~ ^[0-9]+$ ]]; then
            echo "ERROR: Invalid selection: $selection"
            exit 1
        fi
        idx=$((selection - 1))
        if [ "$idx" -lt 0 ] || [ "$idx" -ge ${#datasets[@]} ]; then
            echo "ERROR: Selection out of range: $selection"
            exit 1
        fi
        selected_datasets=("${datasets[$idx]}")
    fi
}

ask_mode_for_dataset() {
    local dataset_root="$1"

    echo
    echo "======================================"
    echo "Dataset: $(basename "$dataset_root")"
    echo "Choose mode:"
    echo "  A) datalad get + datalad unlock on ALL files in this dataset"
    echo "  B) datalad save + datalad push + datalad drop --force"
    echo "  S) skip this dataset"
    echo "======================================"

    local mode
    while true; do
        read -rp "Select mode [A/B/S]: " mode
        case "$mode" in
            A|a) DATASET_MODE="A"; return 0 ;;
            B|b) DATASET_MODE="B"; return 0 ;;
            S|s) DATASET_MODE="S"; return 0 ;;
            *) echo "Invalid choice. Please enter A, B, or S." ;;
        esac
    done
}

collect_all_files() {
    local dataset_root="$1"
    find "$dataset_root" \
        \( -type f -o -type l \) \
        ! -path "$dataset_root/.git/*" \
        ! -path "$dataset_root/.datalad/*" \
        | sort
}

is_unlocked() {
    local dataset_root="$1"
    local relpath="$2"
    datalad -C "$dataset_root" status -- "$relpath" 2>/dev/null | grep -q "unlocked"
}

load_files_into_array() {
    local dataset_root="$1"
    local __resultvar="$2"
    local arr=()
    local item

    while IFS= read -r item; do
        [ -n "$item" ] && arr+=("$item")
    done < <(collect_all_files "$dataset_root")

    eval "$__resultvar=(\"\${arr[@]}\")"
}

load_changed_paths_into_array() {
    local dataset_root="$1"
    local __resultvar="$2"
    local arr=()
    local item

    while IFS= read -r item; do
        [ -n "$item" ] && arr+=("$item")
    done < <(
        datalad -C "$dataset_root" status --untracked=all 2>/dev/null \
            | awk '$1 != "" && $1 != "clean" {print $NF}' \
            | sed 's#/$##' \
            | sort -u
    )

    eval "$__resultvar=(\"\${arr[@]}\")"
}

process_mode_a() {
    local dataset_root="$1"
    local count=0
    local files=()

    echo
    echo "======================================"
    echo "MODE A: datalad get + unlock"
    echo "Dataset: $(basename "$dataset_root")"
    echo "Scope: all files in dataset"
    echo "Batch size: $BATCH_SIZE"
    echo "======================================"

    load_files_into_array "$dataset_root" files

    if [ ${#files[@]} -eq 0 ]; then
        echo "WARNING: No files found in dataset $(basename "$dataset_root")"
        return 0
    fi

    echo "Found ${#files[@]} files/symlinks to process"

    local file relpath
    for file in "${files[@]}"; do
        relpath="${file#$dataset_root/}"

        echo
        echo "--------------------------------------"
        echo "DATASET: $(basename "$dataset_root")"
        echo "FILE:    $relpath"
        echo "--------------------------------------"

        if is_unlocked "$dataset_root" "$relpath"; then
            echo "Already unlocked, skipping"
            count=$((count + 1))
        else
            echo "Running: datalad get -- $relpath"
            datalad -C "$dataset_root" get -- "$relpath"
            if [ $? -ne 0 ]; then
                echo "ERROR: datalad get failed for $relpath"
                continue
            fi

            echo "Running: datalad unlock -- $relpath"
            datalad -C "$dataset_root" unlock -- "$relpath"
            if [ $? -ne 0 ]; then
                echo "ERROR: datalad unlock failed for $relpath"
                continue
            fi

            count=$((count + 1))
        fi

        if (( count % BATCH_SIZE == 0 )); then
            echo
            echo "======================================"
            echo "Processed $count files in dataset $(basename "$dataset_root")."
            echo "Press ENTER to continue."
            echo "======================================"
            read -r
        fi
    done

    echo
    echo "======================================"
    echo "MODE A complete for dataset: $(basename "$dataset_root")"
    echo "======================================"
}

process_mode_b() {
    local dataset_root="$1"
    local changed_paths=()
    local path

    echo
    echo "======================================"
    echo "MODE B: datalad save + push + drop --force"
    echo "Dataset: $(basename "$dataset_root")"
    echo "Scope: changed content in dataset"
    echo "======================================"

    load_changed_paths_into_array "$dataset_root" changed_paths

    if [ ${#changed_paths[@]} -eq 0 ]; then
        echo "No changed or untracked content found in dataset $(basename "$dataset_root")"
        return 0
    fi

    echo "Changed paths to process: ${#changed_paths[@]}"

    for path in "${changed_paths[@]}"; do
        echo
        echo "--------------------------------------"
        echo "DATASET: $(basename "$dataset_root")"
        echo "PATH:    $path"
        echo "--------------------------------------"

        datalad -C "$dataset_root" save -m "save $path" -- "$path"
        if [ $? -ne 0 ]; then
            echo "ERROR: datalad save failed for $path"
            continue
        fi

        datalad -C "$dataset_root" push --to origin
        if [ $? -ne 0 ]; then
            echo "ERROR: datalad push failed for dataset $(basename "$dataset_root")"
            continue
        fi

        datalad -C "$dataset_root" drop --force -- "$path"
        if [ $? -ne 0 ]; then
            echo "ERROR: datalad drop failed for $path"
            continue
        fi

        echo "Done: $path"
    done

    echo
    echo "======================================"
    echo "MODE B complete for dataset: $(basename "$dataset_root")"
    echo "======================================"
}

select_datasets

for ds in "${selected_datasets[@]}"; do
    echo
    echo "############################################################"
    echo "START DATASET: $(basename "$ds")"
    echo "PATH: $ds"
    echo "############################################################"

    ask_mode_for_dataset "$ds"

    case "$DATASET_MODE" in
        A) process_mode_a "$ds" ;;
        B) process_mode_b "$ds" ;;
        S) echo "Skipping dataset $(basename "$ds")" ;;
    esac

    echo
    echo "############################################################"
    echo "FINISHED DATASET: $(basename "$ds")"
    echo "############################################################"
done

echo
echo "======================================"
echo "ALL SELECTED DATASETS DONE"
echo "======================================"
