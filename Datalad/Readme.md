# DataLad Helper Scripts

This repository contains helper scripts for working with large files and folders in existing DataLad datasets.

## Scripts

### `Datalad_stepwise`
Bash script to **replace zip files, unzip them, and upload the extracted content again** in a controlled, stepwise workflow.

Main purpose:
- get file content locally
- unlock files
- manually unzip and inspect
- save changes
- push to remote
- drop content again safely

### `DataLadUpload`
Script for **uploading data** to a DataLad dataset.

---

# Sequential DataLad Stepwise Workflow

`Datalad_stepwise` is designed for folders that contain **many DataLad datasets** and allows processing either:

- **one selected dataset**
- or **all datasets in the parent folder**

It is especially useful when zip files inside a dataset need to be replaced by their extracted folder contents in a **controlled sequential workflow**.

---

## ✨ Key Features

- ✅ Detects DataLad datasets inside a parent folder
- ✅ Lets the user choose **one dataset or all datasets**
- ✅ Works in **two separate modes per dataset**
- ✅ Processes files **strictly sequentially**
- ✅ Ensures **`datalad get` finishes before `datalad unlock` starts**
- ✅ Pauses after a configurable batch size in Mode A
- ✅ Uses safe order for upload workflow: **save → push → drop**
- ✅ Uses `datalad drop --force` where needed

---

## 🧠 Workflow Logic

For each selected dataset, the user can choose between:

### Mode A — `datalad get` + `datalad unlock`
This mode prepares files for manual work.

Workflow:
1. process files one by one
2. run `datalad get` on a file
3. wait until `get` is complete
4. run `datalad unlock` on the same file
5. continue with the next file only after both steps finish

This is useful before manually:
- unzipping archives
- inspecting content
- replacing zip files with extracted folders

### Mode B — `datalad save` + `push` + `drop`
This mode uploads the modified content back to the remote.

Workflow:
1. save dataset changes
2. push to remote (`origin`)
3. drop local content only after push

This follows the recommended git-annex/DataLad logic:

**save → push → drop**

---

## 📦 Requirements

- Bash
- DataLad
- git-annex
- A DataLad dataset with a configured remote such as `origin`

Verify setup with:

```bash
datalad --version
git annex version
bash --version
```

> Note: the script was adapted to work on older macOS Bash versions as well, where features like `mapfile` are unavailable.

---

## 🚀 Usage

### Run on a folder containing multiple datasets

```bash
chmod +x Datalad_stepwise_v5.sh
./Datalad_stepwise_v5.sh /path/to/folder/with/datasets
```

The script will:
1. detect datasets in the given parent folder
2. ask whether to process one dataset or all datasets
3. ask for each dataset whether to run:
   - `A` = get + unlock
   - `B` = save + push + drop
   - `S` = skip

---

## ⚙️ Batch Size

In Mode A, the script pauses after every defined number of processed files.

Current default:

```bash
BATCH_SIZE=5
```

That means:
- 5 files are processed sequentially
- then the script pauses
- the user can inspect/unzip/verify before continuing

---

## 🔐 Why this workflow is useful

Large datasets and annexed content can be error-prone if handled too quickly or in the wrong order.

This workflow helps avoid:
- unlocking files before content is present
- pushing incomplete changes
- dropping content before it exists on the remote
- running too many heavy operations at once

---

## ⚠️ Notes & Caveats

- The script operates on files inside already existing DataLad datasets.
- Many DataLad-tracked files may appear as symlinks; the script accounts for that.
- Mode A is intentionally sequential and can take a long time.
- Manual unzipping or inspection is expected between preparation and upload.
- `drop --force` is powerful and should only be used when you are sure the pushed content is safely available remotely.
- The remote must be reachable and writable.

---

## Example Use Case

A common use case is:

1. dataset contains `.zip` files tracked by DataLad
2. run **Mode A** to get and unlock the files
3. manually unzip and replace the zip files with extracted folders
4. run **Mode B** to save, push, and drop content again

---

## Related Script

### `DataLadUpload`
Use this script when the main goal is simply:

- uploading data
- saving changes
- pushing content to the configured remote

without the stepwise get/unlock preparation workflow.
