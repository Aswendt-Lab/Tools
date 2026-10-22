#!/bin/bash
# One-way GIN mirror. macOS Bash 3.2 compatible. Never pushes or saves.
set -uo pipefail
ROOT="${MIRROR_ROOT:-/Volumes/Publications_GIN}"
APPLY=0; DISCARD=0; UNLOCK=1; BATCH=200; ONLY=""
usage() { cat <<'EOF'
Usage: Datalad_GIN_Mirror_v2.sh [--apply] [--discard-local-changes] [--no-unlock]
       [--root PATH] [--batch-size N] [--dataset NAME]
Default: preview only. --apply enables one-way fetch/download/unlock.
--discard-local-changes allows reset of tracked files ONLY when remote advances.
Never run against working copies. Does not delete untracked files.
EOF
}
while [ "$#" -gt 0 ]; do
 case "$1" in
 --apply) APPLY=1;; --discard-local-changes) DISCARD=1;; --no-unlock) UNLOCK=0;;
 --root) shift; ROOT="${1:?missing root}";; --batch-size) shift; BATCH="${1:?missing size}";;
 --dataset) shift; ONLY="${1:?missing dataset}";; -h|--help) usage; exit 0;;
 *) echo "Unknown option: $1" >&2; usage; exit 2;; esac
 shift
done
case "$BATCH" in *[!0-9]*|'') echo 'Invalid batch size' >&2; exit 2;; esac
[ "$BATCH" -ge 1 ] || exit 2
for cmd in git datalad python3; do command -v "$cmd" >/dev/null || { echo "Missing $cmd" >&2; exit 2; }; done
[ -d "$ROOT" ] || { echo "Missing root: $ROOT" >&2; exit 2; }
ROOT="$(cd "$ROOT" && pwd -P)"
if [ "$APPLY" -eq 1 ] && [ ! -f "$ROOT/.datalad-one-way-mirror" ]; then
 echo "Create $ROOT/.datalad-one-way-mirror only after verifying mirror-only use" >&2; exit 2
fi
LOGDIR="$ROOT/.mirror_logs"
if [ "$APPLY" -eq 1 ]; then mkdir -p "$LOGDIR" || exit 2; fi
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s %s\n' "$(stamp)" "$*"; }
# Exclude accidental concurrent mirror runs.
if [ "$APPLY" -eq 1 ]; then
 if ! mkdir "$LOGDIR/.running" 2>/dev/null; then
   echo "Mirror already running, or stale lock: $LOGDIR/.running" >&2; exit 3
 fi
 trap 'rmdir "$LOGDIR/.running" 2>/dev/null || true' EXIT
fi
failed=0
for ds in "$ROOT"/*; do
 [ -d "$ds" ] || continue
 [ -e "$ds/.git" ] || continue
 name="$(basename "$ds")"
 [ -z "$ONLY" ] || [ "$name" = "$ONLY" ] || continue
 log "=== $name ==="
 if [ "$APPLY" -eq 0 ]; then
   (cd "$ds" && git status -sb && echo "Would fetch, update, get and optionally unlock in batches of $BATCH")
   continue
 fi
 # Tee writes to automatically created root-level .mirror_logs; pipefail captures failure.
 (
  cd "$ds" || exit 1
  [ "$(git rev-parse --show-toplevel)" = "$PWD" ] || exit 4
  branch="$(git symbolic-ref -q --short HEAD)" || { echo 'Detached HEAD: skip'; exit 4; }
  git remote get-url origin >/dev/null || exit 4
  git config remote.origin.pushurl DISABLED
  log 'FETCH origin'
  # Abort this dataset on failed fetch, rather than trusting stale remote refs.
  git -c fetch.prune=true fetch origin || exit 5
  git show-ref --verify --quiet "refs/remotes/origin/$branch" || exit 5
  local_sha="$(git rev-parse HEAD)"; remote_sha="$(git rev-parse "origin/$branch")"
  if [ "$local_sha" != "$remote_sha" ]; then
    git merge-base --is-ancestor HEAD "origin/$branch" || { echo 'Divergent history: manual review'; exit 6; }
    if [ -n "$(git status --porcelain -uno)" ]; then
      if [ "$DISCARD" -eq 0 ]; then
        echo 'Local tracked changes block update; --discard-local-changes required'; exit 6
      fi
      log 'RESET tracked files to fetched remote (mirror only)'
      git reset --hard "origin/$branch" || exit 6
    else
      git merge --ff-only "origin/$branch" || exit 6
    fi
  fi
  # Python batches the annex file list without per-file shell process overhead.
  # Only missing content is downloaded; only locked, present annex files are unlocked.
  python3 - "$BATCH" "$UNLOCK" <<'PY'
import os, sys, subprocess, time
batch_size=int(sys.argv[1]); do_unlock=bool(int(sys.argv[2]))
def annex_files(*args):
    p=subprocess.run(['git','annex','find',*args,'--print0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError('git annex find failed: '+p.stderr.decode(errors='replace'))
    return [os.fsdecode(x) for x in p.stdout.split(b'\0') if x]
def batches(seq):
    for i in range(0,len(seq),batch_size): yield seq[i:i+batch_size]
def run_batch(cmd, items):
    p=subprocess.run(cmd+['--']+items)
    return p.returncode==0
missing=annex_files('--not','--in','here')
print(f'MISSING at start: {len(missing)} annex files; batch size: {batch_size}',flush=True)
errors=0
for idx, group in enumerate(batches(missing),1):
    print(f'GET batch {idx}/{(len(missing)+batch_size-1)//batch_size} ({len(group)} files)',flush=True)
    if not run_batch(['datalad','get'],group):
        errors+=1; print('WARN get batch failed; remaining files retried next run',flush=True)
# On unlocked files, git annex find --in here includes files that are available.
if do_unlock:
    present=annex_files('--in','here')
    # Only real symlinks with existing target; do not touch already unlocked files.
    locked=[f for f in present if os.path.islink(f) and os.path.exists(f)]
    print(f'LOCKED local annex files to unlock: {len(locked)}',flush=True)
    for idx,group in enumerate(batches(locked),1):
        print(f'UNLOCK batch {idx}/{(len(locked)+batch_size-1)//batch_size} ({len(group)} files)',flush=True)
        if not run_batch(['git','annex','unlock'],group):
            errors+=1; print('WARN unlock batch failed',flush=True)
remaining=annex_files('--not','--in','here')
print(f'RESULT missing={len(remaining)} batch_errors={errors}',flush=True)
if remaining or errors: sys.exit(7)
PY
 ) 2>&1 | tee -a "$LOGDIR/$name.log"
 rc=${PIPESTATUS[0]}
 if [ "$rc" -ne 0 ]; then log "INCOMPLETE $name (exit $rc)"; failed=1; else log "COMPLETE $name"; fi
done
exit "$failed"
