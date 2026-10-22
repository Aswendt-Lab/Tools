# DataLad helper scripts

Utilities for uploading, preparing, checking, and comparing DataLad/git-annex datasets. These scripts can transfer large amounts of data and some run `drop`; verify remotes and backups before use.

## Upload scripts

`DataLadUpload.py` is the current sequential uploader. It finds folders at an exact depth below a dataset root, processes only folders of at least 5 GiB, and performs `datalad save`, `datalad push --to origin`, then `datalad drop --what filecontent --recursive`. Older upload versions remain available in Git history.

```bash
python DataLadUpload.py -i /path/to/dataset -d 2
```

The depth is the number of wildcard directory levels below the initial path. Processing aborts on the first failed save, push, or drop.

## One-way GIN mirror

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

It requires `curl`, `jq`, DataLad, Git, and a GIN token. The organization and token-file path are currently hard-coded near the top and must be adapted for another account or computer.

## Requirements

Install DataLad and git-annex and configure the required remotes before use. Bash scripts expect a Unix-like environment; some interactive scripts are specifically written for the older Bash shipped with macOS.
