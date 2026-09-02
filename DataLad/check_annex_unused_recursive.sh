#!/usr/bin/env bash

set -u
set -o pipefail

ROOT="${1:-.}"
LOGFILE="${2:-annex_unused_report.tsv}"
TMP_SEEN="$(mktemp)"

cleanup() {
    rm -f "$TMP_SEEN"
}
trap cleanup EXIT

ROOT="$(cd "$ROOT" && pwd)"

printf "dataset\tstatus\tmessage\n" > "$LOGFILE"

count_total=0
count_annex=0
count_no_annex=0
count_error=0

echo "Scanning for datasets under: $ROOT"
echo "Writing report to: $LOGFILE"
echo

# Find candidate dataset roots from .git or .datalad markers
find "$ROOT" \( -name .git -o -name .datalad \) -print | while read -r marker; do
    ds="$(dirname "$marker")"

    # Deduplicate
    if grep -Fxq "$ds" "$TMP_SEEN"; then
        continue
    fi
    echo "$ds" >> "$TMP_SEEN"

    count_total=$((count_total + 1))
    echo "[$count_total] Checking: $ds"

    # Confirm it is a git repo
    if ! git -C "$ds" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf "%s\terror\tnot a git repository\n" "$ds" >> "$LOGFILE"
        count_error=$((count_error + 1))
        continue
    fi

    # Check whether annex is initialized
    if ! git -C "$ds" annex info >/dev/null 2>&1; then
        printf "%s\tno-annex\tgit-annex not initialized\n" "$ds" >> "$LOGFILE"
        count_no_annex=$((count_no_annex + 1))
        continue
    fi

    count_annex=$((count_annex + 1))

    # Run git annex unused and capture output
    output="$(git -C "$ds" annex unused 2>&1)"
    exitcode=$?

    if [[ $exitcode -ne 0 ]]; then
        printf "%s\terror\t%s\n" "$ds" "$(echo "$output" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')" >> "$LOGFILE"
        count_error=$((count_error + 1))
        continue
    fi

    # Decide whether unused objects were found
    # If numbered unused entries appear, record them
    unused_lines="$(echo "$output" | grep -E '^[[:space:]]*[0-9]+[[:space:]]')"

    if [[ -n "$unused_lines" ]]; then
        summary="$(echo "$unused_lines" | tr '\n' '; ' | sed 's/[[:space:]]\+/ /g')"
        printf "%s\tunused-found\t%s\n" "$ds" "$summary" >> "$LOGFILE"
        echo "    -> unused objects found"
    else
        printf "%s\tok\tno unused annex objects\n" "$ds" >> "$LOGFILE"
        echo "    -> none"
    fi
done

echo
echo "Done."
echo "Report: $LOGFILE"
echo
echo "Tip: sort the report with:"
echo "  column -t -s \$'\\t' \"$LOGFILE\""