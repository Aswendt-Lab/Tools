# Data management plan

Edit [AswendtLab_Data_Management_Plan_v1.md](AswendtLab_Data_Management_Plan_v1.md) and keep its figures in `assets/`. The final [PDF](../output/pdf/AswendtLab_Data_Management_Plan_v1.pdf) is versioned alongside the source. Figure sources and attribution are recorded in the plan captions.

Work in the Git checkout (for example, `~/Documents/GitHub/Tools`), commit changes on a branch, and push to `Aswendt-Lab/Tools`. Keep the older Nextcloud copy as an archive rather than a second working copy.

## Rebuild the PDF

Use a Python environment with `reportlab` installed, then run from the repository root:

```bash
python DataManagement/build_plan_pdf.py
```

The builder resolves paths from its own location and uses GitHub URLs for repository-file hyperlinks in the PDF. Markdown links remain relative for navigation on GitHub and in a local checkout. Render and inspect the resulting PDF after changing content or figures.
