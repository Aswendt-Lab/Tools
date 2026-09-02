# DataLad ZIP preparation workflow

`Datalad_replace_zip.sh` is a two-phase workflow for replacing annexed ZIP files with manually extracted folders. In phase 1 it runs `datalad get` and `datalad unlock` for each ZIP and pauses after each batch of five. It then waits for the user to extract and verify the archives manually. In phase 2, for each ZIP whose corresponding folder now exists, it runs `datalad save` on that folder, pushes to `origin`, and drops the folder and ZIP content locally.

The script does not unzip archives itself. Errors are reported and processing continues, so its final completion message does not guarantee that every operation succeeded.

## Usage

The current script uses this path relative to the working directory:

```bash
BASE_DIR="input/MRI/proc_data"
```

Run it from the dataset root that contains that path:

```bash
chmod +x Datalad_replace_zip.sh
./Datalad_replace_zip.sh
```

Requirements are Bash, DataLad, git-annex, an initialized dataset, and a correctly configured `origin`. Review `BASE_DIR` and `BATCH_SIZE` before use. Phase 2 changes the dataset, pushes it, and drops local annex content; inspect the extracted folders and remote configuration carefully before confirming that phase.
