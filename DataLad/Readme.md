# DataLad helper scripts

Utilities for uploading, preparing, checking, and comparing DataLad/git-annex datasets. These scripts can transfer large amounts of data and some run `drop`; verify remotes and backups before use.

## Upload scripts

`DataLadUpload_v1.py` and `v2.py` are historical upload workflows. `DataLadUpload_v3.py` is the current sequential uploader. It finds folders at an exact depth below a dataset root, processes only folders of at least 5 GiB, and performs `datalad save`, `datalad push --to origin`, then `datalad drop --what filecontent --recursive`.

```bash
python DataLadUpload_v3.py -i /path/to/dataset -d 2
```

The depth is the number of wildcard directory levels below the initial path. Processing aborts on the first failed save, push, or drop.

## Stepwise workflows

`Datalad_stepwise_v5.sh` discovers immediate child datasets and interactively runs either:

- mode A: sequential `get` then `unlock` for each file; or
- mode B: `save`, `push`, then forced `drop`.

Its fixed batch size is currently 100 files.

```bash
./Datalad_stepwise_v5.sh /path/containing/datasets
```

`Datalad_stepwise_v6.sh` implements the same modes but prompts for the batch size. `Datalad_stepwise_v2.sh` is an older project-specific script with a hard-coded relative base directory and should be reviewed before use.

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
