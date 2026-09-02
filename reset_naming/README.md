# Reset stroke-mask naming

`reset_naming_masks.py` finds top-level files matching `*Stroke_mask*` and renames them by replacing the first three underscores, in order, with `s`, `c`, and `m`.

```bash
python reset_naming_masks.py -i /path/to/mask-folder
```

Example transformation logic: the first `_` becomes `s`, the next becomes `c`, and the next becomes `m`. This is destructive, project-specific, and has no dry-run or collision check; test it on a copy first.
