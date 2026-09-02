# Aswendt Lab Tools

Helper scripts for AIDA data management, MRI processing, behavioral video processing, DataLad workflows, and meeting utilities.

## Tool index

| Folder | Purpose |
|---|---|
| [`AIDAmri_helper_tools`](AIDAmri_helper_tools/) | Validate Bruker-style acquisitions for missing method files and malformed subject metadata. |
| [`Assign_Speaker_LabMeeting`](Assign_Speaker_LabMeeting/) | Randomly assign lab-meeting presenters while avoiding consecutive assignments. |
| [`BulkzipFolder`](BulkzipFolder/) | Archive every immediate subfolder using `zip`, `pigz`, and `pv`. |
| [`CitationsReport`](CitationsReport/) | Generate OpenAlex-based citation reports from an ORCID/name query. |
| [`CopyFiles`](CopyFiles/) | GUI for copying matching files and folders while retaining relative paths. |
| [`CopyFolders`](CopyFolders/) | GUI variants for copying matching folders or files. |
| [`CorrectBIDS_AIDAmri`](CorrectBIDS_AIDAmri/) | Apply project-specific cleanup to mouse MRI BIDS datasets. |
| [`CreateGroupMapping`](CreateGroupMapping/) | Build a text list of matching subfolder names. |
| [`CropVideo4DLC`](CropVideo4DLC/) | Crop, rotate, and change the frame rate of videos for DeepLabCut. |
| [`CropVideo_mulitpleFiles_option`](CropVideo_mulitpleFiles_option/) | Historical and current multi-video crop tools plus a behavior-video copier. |
| [`CropVideo_singleFiles`](CropVideo_singleFiles/) | Historical single- and folder-based crop/FPS utilities. |
| [`CropVideos`](CropVideos/) | OCR video labels and optionally rename matching videos. |
| [`DataLad`](DataLad/) | Upload, inspect, compare, and process DataLad/git-annex datasets. |
| [`Datalad-Replace-zip`](Datalad-Replace-zip/) | Prepare annexed ZIP files for manual replacement by extracted folders. |
| [`IdentifyVideoFileName`](IdentifyVideoFileName/) | OCR experiment labels and create standardized behavioral-video names. |
| [`MeetingTranscriber`](MeetingTranscriber/) | Local transcription, speaker diarization, and meeting summaries on Apple Silicon. |
| [`StrokeMask_automated`](StrokeMask_automated/) | Register and generate longitudinal stroke-related MRI masks. |
| [`ZipFixedSize`](ZipFixedSize/) | Split a directory into size-limited ZIP archives. |
| [`check_files_aidamri`](check_files_aidamri/) | Create a CSV inventory of BIDS-like NIfTI filenames. |
| [`copy_stroke_masks`](copy_stroke_masks/) | Move stroke masks into matching subject/session folders. |
| [`reset_naming`](reset_naming/) | Apply a project-specific rename transformation to stroke-mask files. |

Each folder has its own README with requirements, usage, outputs, and safety notes. Several tools modify, move, or delete data; test them on a copy first.
