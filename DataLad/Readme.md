# DataLad helper scripts

Utilities for uploading, preparing, checking, and comparing DataLad/git-annex datasets. These scripts can transfer large amounts of data and some run `drop`; verify remotes and backups before use.

## Upload scripts

`DataLadUpload.py` is the current sequential uploader. It finds folders at an exact depth below a dataset root, processes only folders of at least 5 GiB, and performs `datalad save`, `datalad push --to origin`, then `datalad drop --what filecontent --recursive`. Older upload versions remain available in Git history.

```bash
python DataLadUpload.py -i /path/to/dataset -d 2
```

The depth is the number of wildcard directory levels below the initial path. Processing aborts on the first failed save, push, or drop.

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
