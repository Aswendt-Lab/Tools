Datalad_stepwise => bash script to replace zip files, unzip, and upload again
DataLadUpload => uploading data

# Sequential DataLad Folder Uploader (≥ 5 GB)

This repository contains a helper script for **safely uploading large folders to a DataLad dataset** and pushing them to a remote (e.g. `origin`) using **git-annex best practices**.

The script is designed for datasets that contain **many large subfolders** (e.g. MRI data) and avoids common issues such as remote unpacker errors or unsafe `git-annex drop` operations.

---

## ✨ Key Features

- ✅ Operates **only on directories**
- ✅ Processes folders **≥ 5 GB**
- ✅ Folders are inside an existing DataLad dataset (no subdatasets created)
- ✅ Runs `datalad save` **per folder**
- ✅ Executes **a single `datalad push --to origin`**
- ✅ Drops file content **only after push** (safe workflow)
- ✅ Informative commit messages including folder paths

---

## 🧠 Workflow Logic

The script follows the recommended git-annex pattern: save → push → drop

This ensures that file content is safely stored on the remote before any local data is dropped, preventing `unsafe` or `numcopies` errors.

---

## 📦 Requirements

- Python ≥ 3.7
- DataLad
- git-annex
- A DataLad dataset with a configured remote named `origin`

Verify setup with:

```bash
datalad --version
git annex version
```

## 🚀 Usage

### 1️⃣ Clone or place the script inside your dataset

```bash
cd /path/to/your/dataset
python DataLadUpload_v2.py \
    --initial_path /path/to/your/dataset \
    --depth 3
```
| Argument         | Description                                         |
| ---------------- | --------------------------------------------------- |
| `--initial_path` | Path to the root of the DataLad dataset             |
| `--depth`        | Folder depth to search (e.g. `3` → `dataset/*/*/*`) |

⚠️ Notes & Caveats

- Folder size is calculated recursively, which may take time for very large directory trees.
- Nested folders may be saved separately if they also meet the size criterion.
- The script assumes: numcopies = 1, the remote is reachable and writable
- Dropping happens only after a successful push.




