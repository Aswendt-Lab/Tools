# BulkzipFolder

`bulk_zip_folders.sh` creates one compressed archive for every immediate subfolder of a working directory. It builds an uncompressed ZIP stream, displays progress with `pv`, compresses the stream with `pigz -3`, and saves the result as `<folder>.zip`.

## Requirements

macOS or a Unix-like shell with `bash`, `zip`, `pigz`, and `pv`. On macOS:

```bash
brew install pigz pv
```

## Usage

```bash
chmod +x bulk_zip_folders.sh
./bulk_zip_folders.sh /path/to/parent-folder
```

If no path is supplied, the current directory is used. Existing ZIP files with matching names are replaced. The source folders are not deleted.
