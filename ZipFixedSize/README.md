# ZipFixedSize

`zip_files.py` recursively groups input files into ZIP archives whose **uncompressed source-size total** is capped approximately by a requested number of GiB. Existing `.zip` files are excluded.

```bash
pip install tqdm
python zip_files.py -i /path/to/input -o /path/to/output -c 5 -k
```

`-c/--chunk_size` is an integer GiB limit (default 5). `-k/--keep_structure` preserves relative paths; without it, files are flattened and duplicate basenames can collide. A single file larger than the limit is still placed in one archive. Output is named `archive_part_N.zip`.
