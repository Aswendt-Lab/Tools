# DataLad helper scripts

Utilities for uploading, preparing, checking, and comparing DataLad/git-annex datasets. These scripts can transfer large amounts of data and some run `drop`; verify remotes and backups before use.

## Upload scripts

`DataLadUpload.py` is the current sequential uploader. It finds folders at an exact depth below a dataset root, processes only folders of at least 5 GiB, and performs `datalad save`, `datalad push --to origin`, then `datalad drop --what filecontent --recursive`. Older upload versions remain available in Git history.

```bash
python DataLadUpload.py -i /path/to/dataset -d 2
```

The depth is the number of wildcard directory levels below the initial path. Processing aborts on the first failed save, push, or drop.

## One-way GIN mirror

### Download and verify all datasets with v5

From `Publications_GIN`, run the updated [`Datalad_GIN_Mirror_v5.sh`](Datalad_GIN_Mirror_v5.sh) copy. The mirror root must contain the `.datalad-one-way-mirror` marker. Run this after any existing mirror job finishes.

```bash
bash ./Datalad_GIN_Mirror_v5.sh --root "$PWD" --apply --verify --no-unlock
```

Omitting `--dataset` processes every immediate, non-hidden child Git repository. Add `--dataset NAME` with the exact dataset directory name to process a single dataset. The script fetches from `origin` and updates the checked-out branch when it can fast-forward. `--no-unlock` skips the unlock step.

With `--apply --verify`, v5 scans all annexed paths, including unlocked files, for missing local content. It attempts to retrieve missing scientific files with DataLad, tries `origin` for files still missing, and scans again to report what remains. This checks local availability. `--verify` alone previews the operation, and `FAST_COMPLETE` from a run without verification covers only locked annex symlinks.

During a download batch, v5 prints a progress bar about every 15 seconds and a final count when the batch finishes. The bar counts files whose content is available locally. A large file can be transferring while the count stays steady.

The results are recorded under `.mirror_logs/` in the mirror root:

- `mirror_summary.csv` lists the latest status and missing counts per dataset.
- `<dataset>.log` contains the per-dataset run output, including the final `RESULT` line.
- `missing_files/<dataset>_scientific.txt` and `missing_files/<dataset>_metadata.txt` list the paths still missing after the scan.

`COMPLETE` means the final scan found all annex content locally. `COMPLETE_WITH_WARNINGS` means only files classified as macOS metadata artifacts are missing. `INCOMPLETE` means scientific content is still missing or an operation failed.

### Set up a GIN API token

1. Sign in with a GIN account that can read the repositories you want to mirror. Open **Your Settings → Applications** and choose **Generate New Token**. [GIN application settings](https://gin.g-node.org/user/settings/applications)
2. Save only the generated token in a plain-text file outside the Git checkout. Use a file readable only by your account, for example with `chmod 600 /path/to/private/token-file`.
3. Pass the file path with `--gin-token-file /path/to/private/token-file` or set `GIN_TOKEN_FILE` to that path. For a scheduled run, set the path in the installed LaunchAgent.

v5 reads the owner or organization and repository from each dataset's `origin` at `gin.g-node.org` and uses the API token for HTTPS fetches. When HTTPS cannot retrieve annex content, v5 retries through the configured origin. An SSH origin requires a key available to the SSH agent for unattended downloads.

### Scheduled v5 mirror on macOS

The [LaunchAgent template](org.gin-mirror.plist) checks daily at 02:30 local time and runs `--apply --verify --scheduled` on `/Volumes/Publications_GIN` when at least two calendar days have passed since its last scheduled start. The mounted mirror root must contain the marker file. Each run downloads missing scientific content and unlocks locally available annex symlinks. The calendar check runs in the user's macOS login session.

On the target Mac, copy the template from the `DataLad` folder into your LaunchAgents directory.

```bash
mkdir -p "$HOME/Library/LaunchAgents"
install -m 0644 org.gin-mirror.plist "$HOME/Library/LaunchAgents/"
```

Edit the installed copy to replace the placeholders for the script, token file, DataLad bin directory, and both log paths. Use absolute paths and adjust the mirror root if needed. Then validate and load the configured copy.

```bash
agent="$HOME/Library/LaunchAgents/org.gin-mirror.plist"
plutil -lint "$agent" && launchctl bootstrap "gui/$(id -u)" "$agent"
```

Check `~/Library/Logs/gin-mirror.log` for scheduled job output and `.mirror_logs/mirror_summary.csv` in the mirror root for per-dataset results. Unlocking can leave tracked changes. If GIN advances and tracked changes block the update, the scheduled run reports that dataset as incomplete and continues with the others.

### v2 reference

[`Datalad_GIN_Mirror_v2.sh`](Datalad_GIN_Mirror_v2.sh) updates existing mirror datasets from `origin`, downloads missing annex content, and unlocks locally available annex files in batches. It supports macOS Bash 3.2 and never runs `save`, `push`, or `drop`. Use it only for dedicated mirror copies.

The script scans immediate, non-hidden child directories of the mirror root that contain a `.git` entry. It does not clone missing repositories or recursively process subdatasets. Each dataset must have a checked-out branch with a matching branch on `origin`, initialized git-annex, and working credentials for fetching and downloading content.

Run these commands from the `DataLad` folder. The script requires `git`, `git-annex`, `datalad`, and Python 3 on `PATH`; invoking it with `bash` does not require executable permissions.

### Preview and apply

The default root is `/Volumes/Publications_GIN`. Preview lists each selected dataset's Git status and the intended operations; it does not fetch, download, unlock, create logs, or check whether the remote has advanced.

```bash
bash Datalad_GIN_Mirror_v2.sh
bash Datalad_GIN_Mirror_v2.sh --root /path/to/mirror --dataset dataset_name
```

After verifying that the root contains dedicated mirror copies, create the required marker file and apply:

```bash
touch /path/to/mirror/.datalad-one-way-mirror
bash Datalad_GIN_Mirror_v2.sh --root /path/to/mirror --apply
```

### Options

| Option | Behavior |
|---|---|
| `--apply` | Enable fetch, branch update, content download, and optional unlock. Requires `.datalad-one-way-mirror` in the root. |
| `--root PATH` | Select the mirror root. Overrides the `MIRROR_ROOT` environment variable and the default `/Volumes/Publications_GIN`. |
| `--dataset NAME` | Process only the immediate child directory with this exact name. |
| `--batch-size N` | Process up to `N` files per download/unlock command; positive integer, default `200`. |
| `--no-unlock` | Download missing content without unlocking files. |
| `--discard-local-changes` | Permit `git reset --hard origin/<branch>` when the remote has advanced and tracked files have local changes. |
| `-h`, `--help` | Show usage. |

For example, download one dataset in smaller batches while keeping annex files locked:

```bash
bash Datalad_GIN_Mirror_v2.sh --root /path/to/mirror --dataset dataset_name --batch-size 100 --no-unlock --apply
```

### Update behavior and local changes

In apply mode, the script sets each selected dataset's `remote.origin.pushurl` to `DISABLED` before fetching. This persists in the local Git configuration and blocks ordinary pushes to `origin` through that push URL; it does not prevent pushes via other remotes or explicit URLs.

A successful fetch and an `origin` branch matching the current local branch are required. Detached HEADs and local history that cannot fast-forward to the remote are skipped as incomplete. When the remote advances, clean tracked files are updated with `git merge --ff-only`. Tracked changes block the update unless `--discard-local-changes` is supplied; that option allows a hard reset and can destroy local edits. Git hard reset can also remove untracked files or directories that obstruct writing tracked files. There is no general untracked-file cleanup.

When local and remote commits already match, tracked changes are not reset, even with `--discard-local-changes`; download and unlock still proceed. Unlocking can itself leave tracked changes, so a later remote update may require explicitly discarding those changes on a dedicated mirror.

Downloads target annex files whose content is missing locally. By default, the script then unlocks only locally available annex files that are still symlinks with existing targets. Failed download batches do not stop later batches; rerunning the script retries content that remains missing. Available files are not downloaded again.

### Logs and completion

Apply mode appends per-dataset output to `<root>/.mirror_logs/<dataset>.log` and prints `COMPLETE` or `INCOMPLETE` for each dataset. The `RESULT missing=... batch_errors=...` line reports remaining missing content and failed download/unlock batches. A root-level `.mirror_logs/.running` directory prevents concurrent apply runs; if an interrupted run leaves this lock behind, confirm no mirror process is active before removing the empty directory with `rmdir`.

The overall exit status is `0` when all selected datasets complete, `1` when any dataset is incomplete, `2` for argument or setup errors, and `3` when the apply lock cannot be acquired. A selector that matches no dataset performs no work and still exits `0`; check that the expected dataset appears in the output. Preview status does not establish that a mirror is current or complete.

## Stepwise workflows

`Datalad_stepwise.sh` discovers immediate child datasets and interactively runs either:

- mode A: sequential `get` then `unlock` for each file; or
- mode B: `save`, `push`, then forced `drop`.

It prompts for a batch size in mode A.

```bash
./Datalad_stepwise.sh /path/containing/datasets
```

Older stepwise versions remain available in Git history.

## Inspection and comparison

`check_annex_unused_recursive.sh` discovers nested Git/DataLad datasets, runs `git annex unused`, and writes a TSV report:

```bash
./check_annex_unused_recursive.sh /path/to/root annex_unused_report.tsv
```

It reports unused objects but does not remove them.

`compare_local_remote_GIN_repos.sh` compares immediate local directories with repositories in the configured GIN organization and can interactively clone missing repositories:

```bash
./compare_local_remote_GIN_repos.sh /path/to/local/repos
```

It requires `curl`, `jq`, DataLad, Git, and a GIN API token. Set `GIN_ORG` to your GIN organization name and `GIN_TOKEN_FILE` to the path of your private token file.

## Requirements

Install DataLad and git-annex and configure the required remotes before use. Bash scripts expect a Unix-like environment; some interactive scripts are specifically written for the older Bash shipped with macOS.
