#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${1:-.}"
ORG_NAME="Aswendt_Lab"
GIN_API="https://gin.g-node.org/api/v1"
TOKEN_FILE="/Users/maswendt/SynologyDrive/Transfer/GIN.txt"

if ! command -v curl >/dev/null 2>&1; then
  echo "❌ 'curl' is required."
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "❌ 'jq' is required."
  exit 1
fi

if ! command -v datalad >/dev/null 2>&1; then
  echo "❌ 'datalad' is not installed or not on PATH."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "❌ 'git' is not installed or not on PATH."
  exit 1
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "❌ Local directory does not exist: $LOCAL_DIR"
  exit 1
fi

if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "❌ Token file not found: $TOKEN_FILE"
  exit 1
fi

GIN_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"

if [[ -z "$GIN_TOKEN" ]]; then
  echo "❌ Token file is empty: $TOKEN_FILE"
  exit 1
fi

echo "🌐 Fetching remote repositories for '$ORG_NAME'..."
REMOTE_REPOS="$(
  curl -fsSL \
    -H "Authorization: token ${GIN_TOKEN}" \
    "${GIN_API}/orgs/${ORG_NAME}/repos?limit=1000" \
    | jq -r '.[].name' \
    | sort
)"

if [[ -z "$REMOTE_REPOS" ]]; then
  echo "❌ No remote repositories found, or API request failed."
  exit 1
fi

echo "💽 Scanning local repositories in '$LOCAL_DIR'..."
LOCAL_REPOS="$(
  find "$LOCAL_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort
)"

echo
echo "📂 Local repositories not found remotely:"
comm -23 <(printf '%s\n' "$LOCAL_REPOS") <(printf '%s\n' "$REMOTE_REPOS") || true

echo
echo "🌐 Remote repositories not found locally:"
MISSING_REPOS="$(comm -13 <(printf '%s\n' "$LOCAL_REPOS") <(printf '%s\n' "$REMOTE_REPOS"))"
printf '%s\n' "$MISSING_REPOS"

if [[ -n "$MISSING_REPOS" ]]; then
  missing_array=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && missing_array+=("$line")
  done <<< "$MISSING_REPOS"

  echo
  echo "Available remote datasets to clone:"
  i=1
  for repo in "${missing_array[@]}"; do
    printf "  [%d] %s\n" "$i" "$repo"
    i=$((i + 1))
  done

  echo
  echo "Enter the numbers to clone, separated by spaces."
  echo "Examples:"
  echo "  1 3 5     -> clone datasets 1, 3, and 5"
  echo "  all       -> clone all missing datasets"
  echo "  none      -> skip cloning"
  read -r -p "❓ Your selection: " selection

  selected_indices=()

  if [[ "$selection" == "all" ]]; then
    i=1
    while [[ $i -le ${#missing_array[@]} ]]; do
      selected_indices+=("$i")
      i=$((i + 1))
    done
  elif [[ "$selection" == "none" || -z "$selection" ]]; then
    :
  else
    for idx in $selection; do
      selected_indices+=("$idx")
    done
  fi

  if [[ ${#selected_indices[@]} -eq 0 ]]; then
    echo "❌ Skipping cloning."
  else
    echo
    for idx in "${selected_indices[@]}"; do
      if ! [[ "$idx" =~ ^[0-9]+$ ]]; then
        echo "⚠️  Skipping invalid selection: $idx"
        continue
      fi

      if (( idx < 1 || idx > ${#missing_array[@]} )); then
        echo "⚠️  Selection out of range: $idx"
        continue
      fi

      repo="${missing_array[$((idx - 1))]}"
      SSH_URL="git@gin.g-node.org:/${ORG_NAME}/${repo}.git"

      if [[ -d "$LOCAL_DIR/$repo" ]]; then
        echo "⏭️  Already exists locally, skipping: $repo"
        continue
      fi

      echo "🚀 Cloning $repo..."
      datalad clone "$SSH_URL" "$LOCAL_DIR/$repo"
    done
  fi
else
  echo "✅ All remote repositories already exist locally."
fi

echo
read -r -p "❓ Do you want to run 'git annex dropunused all' on all local annex repositories in '$LOCAL_DIR'? [y/N]: " DROP_UNUSED

if [[ "$DROP_UNUSED" =~ ^[Yy]$ ]]; then
  echo
  echo "🧹 Checking local repositories for unused annex objects..."

  find "$LOCAL_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r repo_path; do
    repo_name="$(basename "$repo_path")"
    echo
    echo "🔍 Processing: $repo_name"

    if ! git -C "$repo_path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "   ⏭️  Skipping: not a Git repository"
      continue
    fi

    if ! git -C "$repo_path" annex info >/dev/null 2>&1; then
      echo "   ⏭️  Skipping: git-annex not initialized"
      continue
    fi

    UNUSED_OUTPUT="$(git -C "$repo_path" annex unused 2>&1 || true)"

    if echo "$UNUSED_OUTPUT" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]'; then
      echo "   ⚠️  Unused annex objects found"
      echo "   🗑️  Running: git annex dropunused all"
      git -C "$repo_path" annex dropunused all
      echo "   ✅ Finished cleanup"
    else
      echo "   ✅ No unused annex objects found"
    fi
  done
else
  echo "❌ Skipping annex cleanup."
fi