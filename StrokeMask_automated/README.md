# Stroke Mask Processing Scripts

This folder contains three Python scripts used for processing stroke-related MRI masks across longitudinal mouse brain imaging sessions.

---

## Workflow Overview

```text
1. DistributeStrokeMasks_1.py
        ↓
2. GeneratePerilesionalMask_2.py
        ↓
3. GenerateContralesionalStrokeMask_3.py
```

---

## Script Summary

| Step | Script | What it does |
|---:|---|---|
| **1** | `DistributeStrokeMasks_1.py` | Finds existing stroke masks and registers them from their original session to the other timepoints of the same subject. |
| **2** | `GeneratePerilesionalMask_2.py` | Creates perilesional stroke masks by expanding the stroke area and removing the original lesion region. It also registers these masks across timepoints and corrects them with the whole-brain mask. |
| **3** | `GenerateContralesionalStrokeMask_3.py` | Creates contralesional control masks by registering the brain to a template, estimating the midline, flipping the stroke mask to the opposite hemisphere, and transforming it back to native space. |

---

## Short Description of Each Script

### `DistributeStrokeMasks_1.py`

This script distributes stroke masks across different timepoints of the same subject.

It first transforms each stroke mask into incidence space and then registers it to the anatomical space of the other sessions. This allows the same lesion mask to be compared across longitudinal MRI timepoints.

---

### `GeneratePerilesionalMask_2.py`

This script generates perilesional stroke masks.

The perilesional mask represents the tissue surrounding the lesion. The script creates this region by dilating the original stroke mask and subtracting the lesion itself. For chronic timepoints, it adjusts the mask so that the remaining local stroke region is excluded. It also applies whole-brain correction so the final mask remains inside the brain.

---

### `GenerateContralesionalStrokeMask_3.py`

This script generates contralesional stroke masks.

It registers the subject’s brain image to an in-house template, calculates the midline, flips the stroke mask to the opposite hemisphere, and then transforms the flipped mask back to the native image space. The generated contralesional mask can be used as a control region corresponding to the lesion location on the opposite side of the brain.

---

## Overall Purpose

Together, these scripts support longitudinal stroke MRI analysis by preparing three important mask types:

```text
Stroke mask
Perilesional mask
Contralesional mask
```

These masks can then be used for region-based analysis of tissue changes after stroke.
