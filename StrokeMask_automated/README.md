# Automated stroke-mask processing

Three command-line scripts form a longitudinal mouse MRI mask workflow:

1. `DistributeStrokeMasks_1.py` transforms each `*Stroke_mask.nii.gz` into incidence space and resamples it into other sessions for the same subject.
2. `GeneratePerilesionalMask_2.py` renames masks according to its session mapping, creates dilated perilesional ring masks, registers them across sessions, and clips outputs to whole-brain masks.
3. `GenerateContralesionalStrokeMask_3.py` registers anatomy to a template, estimates the template midline, flips lesion masks across it, transforms them back to native space, and clips them to the brain mask.

## Requirements

Python dependencies include NumPy, SciPy, and nibabel. NiftyReg commands such as `reg_aladin`, `reg_resample`, and `reg_transform` must be on `PATH`; the contralesional script also recognizes `NIFTYREGDIR`.

The expected layout is broadly:

```text
<root>/<subject>/<session>/anat/
```

Filenames must match the project-specific `Stroke_mask`, `IncidenceData`, transformation-matrix, `BiasBet`, and brain-mask patterns used in the scripts.

## Usage

```bash
python DistributeStrokeMasks_1.py -i /path/to/root
python GeneratePerilesionalMask_2.py -i /path/to/root
python GenerateContralesionalStrokeMask_3.py -i /path/to/root -tpl /path/to/template.nii.gz
```

Run the steps in order. The scripts use the first glob match for several required inputs, create and overwrite derived NIfTI/transformation files, and log or continue past some missing inputs. Test on a copy and inspect every generated mask before quantitative analysis.
