#!/bin/bash
# One-way GIN mirror, optimized for macOS Bash 3.2 and large datasets.
# Never saves or pushes. GIN is authoritative. No routine full git-annex find scan.
# v5: if DataLad retrieval fails, retry still-missing files via git annex --from origin.
set -uo pipefail
ROOT="${MIRROR_ROOT:-/Volumes/Publications_GIN}"
APPLY=0; DISCARD=0; UNLOCK=1; BATCH=200; ONLY=""; RETRIES=3; VERIFY=0
SCHEDULED=0
GIN_TOKEN_FILE="${GIN_TOKEN_FILE:-}"
usage() { cat <<'EOF'
Usage: Datalad_GIN_Mirror_v5.sh [--apply] [--dataset NAME] [--batch-size N]
  [--retries N] [--discard-local-changes] [--no-unlock] [--verify] [--root PATH]
  [--gin-token-file PATH] [--scheduled]
Default is read-only preview. --apply requires .datalad-one-way-mirror in root.
--apply --verify scans all annexed paths for missing content before and after
  retrieval. This is slower and includes unlocked files.
--gin-token-file PATH reads a GIN API token from a plain-text file and uses
  HTTPS for origins hosted at gin.g-node.org. Save only the token in the file.
--scheduled requires --apply and runs at most once every two calendar days.
--discard-local-changes permits git reset --hard when remote has advanced;
  ONLY for a dedicated disposable mirror. No untracked files are deleted.
No save/push is performed. Only immediate child datasets are processed.
EOF
}
while [ "$#" -gt 0 ]; do
 case "$1" in
 --apply) APPLY=1;; --discard-local-changes) DISCARD=1;; --no-unlock) UNLOCK=0;; --verify) VERIFY=1;;
 --scheduled) SCHEDULED=1;;
 --root|--batch-size|--dataset|--retries|--gin-token-file)
   opt="$1"; shift; [ "$#" -gt 0 ] || { echo "Missing value for $opt" >&2; exit 2; }
   case "$opt" in --root) ROOT="$1";; --batch-size) BATCH="$1";; --dataset) ONLY="$1";; --retries) RETRIES="$1";; --gin-token-file) GIN_TOKEN_FILE="$1";; esac;;
 -h|--help) usage; exit 0;; *) echo "Unknown option: $1" >&2; exit 2;; esac
 shift
done
for n in "$BATCH" "$RETRIES"; do case "$n" in ''|*[!0-9]*) echo 'Batch/retries must be positive integers' >&2; exit 2;; esac; done
[ "$BATCH" -ge 1 ] && [ "$RETRIES" -ge 1 ] || exit 2
for c in git git-annex datalad python3; do command -v "$c" >/dev/null || { echo "Missing $c" >&2; exit 2; }; done
[ "$SCHEDULED" -eq 0 ] || [ "$APPLY" -eq 1 ] || { echo '--scheduled requires --apply' >&2; exit 2; }
[ -d "$ROOT" ] || { echo "Missing root: $ROOT" >&2; exit 2; }
ROOT="$(cd "$ROOT" && pwd -P)"; LOGDIR="$ROOT/.mirror_logs"
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
if [ "$APPLY" -eq 1 ]; then
 [ -f "$ROOT/.datalad-one-way-mirror" ] || { echo "Missing mirror safety marker" >&2; exit 2; }
 if [ -n "$GIN_TOKEN_FILE" ]; then
  [ -r "$GIN_TOKEN_FILE" ] && [ -s "$GIN_TOKEN_FILE" ] || { echo "Missing or empty GIN token file: $GIN_TOKEN_FILE" >&2; exit 2; }
  GIN_TOKEN_FILE="$(cd "$(dirname "$GIN_TOKEN_FILE")" && pwd -P)/$(basename "$GIN_TOKEN_FILE")"
  export GIN_TOKEN_FILE
 fi
 mkdir -p "$LOGDIR/missing_files" || exit 2
 if ! mkdir "$LOGDIR/.running" 2>/dev/null; then echo "Mirror already running or stale lock: $LOGDIR/.running" >&2; exit 3; fi
 trap 'rmdir "$LOGDIR/.running" 2>/dev/null || true' EXIT
fi
if [ "$SCHEDULED" -eq 1 ]; then
 python3 - "$LOGDIR/.last_scheduled_run" <<'PY'
from datetime import date
from pathlib import Path
import sys

stamp = Path(sys.argv[1])
today = date.today()
if stamp.exists() and (today - date.fromisoformat(stamp.read_text().strip())).days < 2:
    sys.exit(10)
temporary = stamp.with_suffix('.tmp')
temporary.write_text(today.isoformat() + '\n', encoding='ascii')
temporary.replace(stamp)
PY
 case "$?" in 0) log 'Starting scheduled mirror run';; 10) exit 0;; *) exit 2;; esac
 if [ -z "${SSH_AUTH_SOCK:-}" ] && command -v launchctl >/dev/null; then
  agent_socket="$(launchctl print "gui/$(id -u)/com.openssh.ssh-agent" 2>/dev/null |
   sed -n 's/^[[:space:]]*SSH_AUTH_SOCK => //p' | head -n 1)"
  if [ -S "$agent_socket" ]; then export SSH_AUTH_SOCK="$agent_socket"; fi
 fi
fi

git_config_override() {
 local index="${GIT_CONFIG_COUNT:-0}"
 export "GIT_CONFIG_KEY_$index=$1" "GIT_CONFIG_VALUE_$index=$2"
 export GIT_CONFIG_COUNT="$((index+1))"
}

use_gin_https() {
 local origin="$1" path owner repo
 case "$origin" in
  git@gin.g-node.org:*) path="${origin#*:}"; path="${path#/}";;
  ssh://git@gin.g-node.org/*|ssh://git@gin.g-node.org:22/*|https://gin.g-node.org/*)
   path="${origin#*://}"; path="${path#*/}";;
  *) echo 'ERROR GIN API token mode requires an origin at gin.g-node.org'; return 1;;
 esac
 if [[ "$path" != */* ]]; then
  echo 'ERROR GIN origin must identify a repository as OWNER/REPOSITORY'
  return 1
 fi
 owner="${path%%/*}"
 repo="${path#*/}"
 repo="${repo%.git}"
 case "$owner" in ''|.|..|*[!a-zA-Z0-9_.-]*) echo 'ERROR invalid GIN owner name'; return 1;; esac
 case "$repo" in ''|.|..|*[!a-zA-Z0-9_.-]*) echo 'ERROR invalid GIN repository name'; return 1;; esac
 git_config_override remote.origin.url "https://gin.g-node.org/$owner/$repo"
 git_config_override remote.origin.annexurl "https://gin.g-node.org/$owner/$repo"
 git_config_override credential.helper ''
 git_config_override credential.https://gin.g-node.org.helper '!f() {
  [ "$1" = get ] || return 0
  protocol= host=
  while IFS="=" read -r key value && [ -n "$key" ]; do
   case "$key" in protocol) protocol="$value";; host) host="$value";; esac
  done
  [ "$protocol" = https ] && [ "$host" = gin.g-node.org ] || return 0
  token=$(tr -d "\r\n" < "$GIN_TOKEN_FILE") && [ -n "$token" ] || return 1
  printf "username=%s\npassword=\n" "$token"
 }; f'
 export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND=/usr/bin/false
}
summary() {
 python3 - "$LOGDIR/mirror_summary.csv" "$1" "$2" "$3" "$4" "$5" <<'PY'
import csv,sys,os,tempfile,datetime
out,name,status,missing,metadata,detail=sys.argv[1:]
cols=['dataset','status','missing_scientific','missing_metadata','detail','checked_at']
rows={}
if os.path.isfile(out):
 with open(out,encoding='utf-8',newline='') as f:
  for row in csv.DictReader(f):
   if row.get('dataset'):rows[row['dataset']]={c:row.get(c,'') for c in cols}
rows[name]=dict(zip(cols,[name,status,missing,metadata,detail,datetime.datetime.now().isoformat(timespec='seconds')]))
fd,tmp=tempfile.mkstemp(dir=os.path.dirname(out),prefix='.summary_',suffix='.csv')
try:
 with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for key in sorted(rows):w.writerow(rows[key])
 os.replace(tmp,out)
finally:
 if os.path.exists(tmp):os.unlink(tmp)
PY
}
failed=0;matched=0
for ds in "$ROOT"/*; do
 [ -d "$ds" ] && [ -e "$ds/.git" ] || continue
 name="$(basename "$ds")"
 [ -z "$ONLY" ] || [ "$name" = "$ONLY" ] || continue
 matched=1; log "=== $name ==="
 if [ "$APPLY" -eq 0 ]; then
  if [ "$VERIFY" -eq 1 ]; then
   echo "PREVIEW: --apply --verify would fetch, scan all annexed paths, retrieve missing scientific content, and verify availability."
  else
   echo "PREVIEW: would fetch, scan Git index once, get missing annex symlinks, unlock in batches."
  fi
  continue
 fi
 resultfile="$LOGDIR/.result_${name}_$$"
 (
  cd "$ds" || exit 4
  [ "$(git rev-parse --show-toplevel 2>/dev/null)" = "$PWD" ] || exit 4
  branch="$(git symbolic-ref -q --short HEAD)" || { echo 'ERROR detached HEAD'; exit 4; }
  origin="$(git remote get-url origin)" || exit 4
  git config remote.origin.pushurl DISABLED || exit 4
  export GIT_SSH_COMMAND='ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=20 -o ServerAliveCountMax=3'
  if [ -n "$GIN_TOKEN_FILE" ]; then
   export GIN_CONFIG_COUNT_BEFORE="${GIT_CONFIG_COUNT:-0}" GIN_SSH_FALLBACK_COMMAND="$GIT_SSH_COMMAND"
   use_gin_https "$origin" || exit 4
  fi
  fetched=0; i=1
  while [ "$i" -le "$RETRIES" ]; do
   log "FETCH origin ($i/$RETRIES)"
   if git -c fetch.prune=true fetch origin; then fetched=1; break; fi
   [ "$i" -eq "$RETRIES" ] || sleep $((i*10)); i=$((i+1))
  done
  [ "$fetched" -eq 1 ] || { echo 'ERROR fetch failed'; exit 5; }
  git show-ref --verify --quiet "refs/remotes/origin/$branch" || exit 5
  local_sha="$(git rev-parse HEAD)";remote_sha="$(git rev-parse "origin/$branch")"
  if [ "$local_sha" != "$remote_sha" ]; then
   git merge-base --is-ancestor HEAD "origin/$branch" || { echo 'ERROR diverged; manual review required'; exit 6; }
   if [ -n "$(git status --porcelain -uno)" ]; then
    [ "$DISCARD" -eq 1 ] || { echo 'ERROR tracked local changes block update; use --discard-local-changes only for mirror'; exit 6; }
    log 'RESET tracked files to remote (remote advanced)'
    git reset --hard "origin/$branch" || exit 6
   else
    git merge --ff-only "origin/$branch" || exit 6
   fi
  fi
  log 'SCAN Git index (no git-annex find; fast symlink-based check)'
  python3 - "$BATCH" "$UNLOCK" "$RETRIES" "$VERIFY" "$LOGDIR/missing_files" "$name" "$resultfile" <<'PY'
import os,sys,subprocess,time,json
batch_size=int(sys.argv[1]);unlock=int(sys.argv[2]);retries=int(sys.argv[3]);verify=int(sys.argv[4]);outdir,name,resultfile=sys.argv[5:]
def metadata(path):
 base=os.path.basename(path)
 return base=='.DS_Store' or base in ('Icon','Icon\r') or base.startswith('._') or base=='.AppleDouble'
def batches(seq):
 for k in range(0,len(seq),batch_size):yield seq[k:k+batch_size]
def run_batch(cmd,files,progress=None,env=None):
 for n in range(retries):
  with subprocess.Popen(cmd+['--']+files,env=env) as process:
   while True:
    try:
     code=process.wait(timeout=15 if progress else None)
     break
    except subprocess.TimeoutExpired:
     progress()
  if code==0:return True
  if n+1<retries:
   delay=(n+1)*10
   if progress:
    print(f'GET retry {n+2}/{retries} in {delay}s (exit {code})',flush=True)
   time.sleep(delay)
 return False
def write_report(kind,paths):
 with open(os.path.join(outdir,name+'_'+kind+'.txt'),'w',encoding='utf-8') as f:
  for p in paths:f.write(p.replace('\r','\\r')+'\n')
def find_missing(paths=(),timeout=None):
 result=subprocess.run(
  ['git','annex','find','--not','--in=here','--print0','--',*paths],
  stdout=subprocess.PIPE,check=True,timeout=timeout)
 return [os.fsdecode(path) for path in result.stdout.split(b'\0') if path]

def remaining(paths,timeout=None):
 if verify:return find_missing(paths,timeout=timeout)
 return [path for path in paths if not os.path.exists(path)]

def ssh_fallback_environment():
 count=os.environ.get('GIN_CONFIG_COUNT_BEFORE')
 if count is None:return None
 return dict(os.environ,GIT_CONFIG_COUNT=count,
             GIT_SSH_COMMAND=os.environ['GIN_SSH_FALLBACK_COMMAND'])

def show_progress(label,stage,files,started,missing_count=None):
 elapsed=int(time.monotonic()-started)
 minutes,seconds=divmod(elapsed,60)
 if missing_count is None:
  try:
   missing_count=len(remaining(files,timeout=5))
  except (OSError,subprocess.SubprocessError):
   print(f'{label} ({stage}): checking availability timed out or failed, elapsed {minutes}m {seconds:02d}s',flush=True)
   return
 available=len(files)-missing_count
 filled=available*20//len(files)
 bar='#'*filled+'-'*(20-filled)
 print(f'{label} ({stage}): [{bar}] {available}/{len(files)} files available, elapsed {minutes}m {seconds:02d}s',flush=True)

# Parse the Git index in one subprocess. The mode 120000 represents symlinks.
# Only annex symlinks are eligible; ordinary symlinks are left untouched.
p=subprocess.run(['git','ls-files','--stage','-z'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
annex=[]; regular=0
for rec in p.stdout.split(b'\0'):
 if not rec:continue
 try:meta,path=rec.split(b'\t',1)
 except ValueError:continue
 if meta.split(b' ',1)[0]!=b'120000':continue
 file=os.fsdecode(path)
 if not os.path.islink(file):continue
 try:target=os.readlink(file)
 except OSError:continue
 if '/annex/objects/' in target or target.startswith('.git/annex/objects/'):
  annex.append(file)
 else:regular+=1
print(f'SCAN complete: {len(annex)} locked annex symlinks; {regular} other symlinks',flush=True)
if verify:
 print('VERIFY: scanning all annexed paths before retrieval',flush=True)
 missing=find_missing()
else:
 missing=remaining(annex)
missing_meta=[f for f in missing if metadata(f)]
missing_science=[f for f in missing if not metadata(f)]
coverage='all annexed paths' if verify else 'locked annex symlinks'
print(f'MISSING ({coverage}): scientific={len(missing_science)} metadata={len(missing_meta)}',flush=True)
errors=0
# Try DataLad first. If it reports failure, retry ONLY the files that are
# still absent using the explicitly named annex remote `origin`.
# A command failure is not counted if the content was ultimately retrieved.
unavailable=[]
ssh_env=ssh_fallback_environment()
for i,group in enumerate(batches(missing_science),1):
 total=(len(missing_science)+batch_size-1)//batch_size
 label=f'GET batch {i}/{total}'
 started=time.monotonic()
 print(f'{label}: {len(group)} files (DataLad)',flush=True)
 datalad_ok=run_batch(['datalad','get'],group,
                      lambda:show_progress(label,'DataLad',group,started))
 still_missing=remaining(group)
 if still_missing:
  print(f'FALLBACK origin batch {i}/{total}: {len(still_missing)} files still missing',flush=True)
  # Retry only still-missing paths, not the whole original batch.
  # Use `git annex get --from origin` because GIN may serve the key even
  # when DataLad's default remote selection fails.
  run_batch(['git','annex','get','--from','origin'],still_missing,
            lambda:show_progress(label,'origin',group,started))
  still_missing=remaining(still_missing)
 if still_missing and ssh_env is not None:
  print(f'FALLBACK configured origin batch {i}/{total}: {len(still_missing)} files still missing',flush=True)
  run_batch(['git','annex','get','--from','origin'],still_missing,
            lambda:show_progress(label,'configured origin',group,started),env=ssh_env)
  still_missing=remaining(still_missing)
 show_progress(label,'finished',group,started,len(still_missing))
 if still_missing:
  unavailable.extend(still_missing)
  print(f'WARN GET batch {i}/{total}: {len(still_missing)} files unavailable after fallback',flush=True)
 elif not datalad_ok:
  print(f'GET batch {i}/{total}: recovered via origin fallback',flush=True)
# Recheck actual availability, rather than trusting process return codes.
missing_science=unavailable
# Symlinks with available targets can be unlocked in one annex command per batch.
if unlock:
 locked_present=[f for f in annex if os.path.islink(f) and os.path.exists(f)]
 print(f'UNLOCK available symlinks: {len(locked_present)}',flush=True)
 for i,group in enumerate(batches(locked_present),1):
  print(f'UNLOCK batch {i}/{(len(locked_present)+batch_size-1)//batch_size}',flush=True)
  if not run_batch(['git','annex','unlock'],group):errors+=1
# A fast index/symlink scan cannot detect missing content for already-unlocked
# pointer files (mode 100644), nor validate checksums. Report this explicitly.
if verify:
 print('VERIFY: scanning all annexed paths after retrieval',flush=True)
 all_missing=find_missing()
 missing_science=[x for x in all_missing if not metadata(x)]
 missing_meta=[x for x in all_missing if metadata(x)]
write_report('scientific',missing_science);write_report('metadata',missing_meta)
status='INCOMPLETE' if missing_science or errors else ('COMPLETE_WITH_WARNINGS' if missing_meta else 'COMPLETE')
if not verify:status='FAST_'+status
print(f'RESULT {status}: missing_scientific={len(missing_science)}, missing_metadata={len(missing_meta)}, failed_batches={errors}',flush=True)
if not verify:print('NOTE fast status covers locked annex symlinks only; use --apply --verify to check and retrieve missing scientific content for all annexed paths.',flush=True)
with open(resultfile,'w',encoding='utf-8') as f:json.dump(dict(status=status,science=len(missing_science),meta=len(missing_meta),errors=errors),f)
if missing_science or errors:sys.exit(7)
PY
 ) 2>&1 | tee -a "$LOGDIR/$name.log"
 rc=${PIPESTATUS[0]}
 if [ -f "$resultfile" ]; then
  values="$(python3 - "$resultfile" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]));print(v['status'],v['science'],v['meta'],v['errors'])
PY
)"
  read -r status sci meta errs <<< "$values"
 else
  status='INCOMPLETE';sci='unknown';meta='unknown';errs='unknown'
 fi
 rm -f "$resultfile"
 summary "$name" "$status" "$sci" "$meta" "exit=$rc;failed_batches=$errs"
 if [ "$rc" -ne 0 ]; then failed=1;log "INCOMPLETE $name (exit $rc)"; else log "$status $name";fi
done
[ "$matched" -eq 1 ] || { echo 'No matching dataset found' >&2; exit 2; }
exit "$failed"
