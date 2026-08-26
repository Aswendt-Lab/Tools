# MeetingTranscriber

**MeetingTranscriber** is a local meeting-processing pipeline for
**Apple Silicon macOS**. It turns meeting recordings into searchable
transcripts with approximate speaker labels and can generate structured
meeting summaries without sending the meeting audio or transcript to a
cloud transcription or LLM service.

The project is designed especially for recorded Zoom meetings, but it
also works with ordinary video and audio files.

## How it works

The processing pipeline is:

``` text
Video / Zoom recording / audio file
              |
              v
     Audio selection/extraction
       ffmpeg -> 16 kHz WAV
              |
              v
       Speech transcription
         MLX Whisper
              |
              v
       Speaker diarization
 WebRTC VAD + Resemblyzer + clustering
              |
              v
      Transcript cleanup
 repetition/hallucination filtering
              |
              +--------------------+
              |                    |
              v                    v
      Markdown transcript      Local Ollama
              |               summarization
              v                    |
           JSON                    v
                         Markdown meeting summary
```

MLX Whisper is optimized for Apple Silicon. Speaker diarization is
performed locally, and Ollama provides local LLM-based summarization.

After the required models have been downloaded, meeting processing can
be performed entirely locally. The batch script also provides an
`--offline` option to prevent accidental model downloads.

## Scripts

Two scripts are provided.

### `MeetingTranscriber_v2.py`

Processes **one video or audio file**.

Use this when you want to manually process an individual recording and
explicitly choose the input and output location.

### `MeetingTranscriber_v2_batch.py`

Processes **multiple recordings automatically**.

Give it a parent directory, such as your Zoom recordings directory. It
recursively searches recording folders, detects videos and audio files,
prefers an existing Zoom audio recording when available, and skips
meetings for which a transcript already exists.

This is the recommended script for routinely processing a collection of
meetings.

------------------------------------------------------------------------

# Requirements

-   macOS on Apple Silicon (`arm64`)
-   Python environment, for example Conda or venv
-   `ffmpeg`
-   `ollama` for local meeting summaries

Install system tools:

``` bash
brew install ffmpeg ollama
```

Install Python dependencies:

``` bash
pip install mlx-whisper resemblyzer webrtcvad scikit-learn soundfile numpy requests
```

For summary generation, start Ollama:

``` bash
ollama serve
```

If this reports that port `11434` is already in use, Ollama is usually
already running.

Install at least one local summary model, for example:

``` bash
ollama pull llama3.1:8b
```

------------------------------------------------------------------------

# Language

Use Whisper language codes:

``` text
German   de
English  en
French   fr
Spanish  es
```

For English use `en`, **not** `eng`.

If `--language` is omitted, Whisper can detect the language
automatically.

------------------------------------------------------------------------

# Single-file mode

Use `MeetingTranscriber_v2.py` for one recording.

German meeting:

``` bash
python MeetingTranscriber_v2.py /path/to/meeting.mp4 \
  --out-dir /path/to/output_folder \
  --language de \
  --min-speakers 2 \
  --max-speakers 4
```

English meeting:

``` bash
python MeetingTranscriber_v2.py /path/to/meeting.mp4 \
  --out-dir /path/to/output_folder \
  --language en \
  --min-speakers 2 \
  --max-speakers 4
```

An audio file can also be processed directly:

``` bash
python MeetingTranscriber_v2.py /path/to/audio_only.m4a \
  --out-dir /path/to/output_folder \
  --language de \
  --min-speakers 2 \
  --max-speakers 4
```

For a quick transcription test without running Ollama:

``` bash
python MeetingTranscriber_v2.py /path/to/meeting.mp4 \
  --out-dir /path/to/output_folder \
  --language de \
  --min-speakers 2 \
  --max-speakers 4 \
  --skip-summary
```

------------------------------------------------------------------------

# Batch mode

Use `MeetingTranscriber_v2_batch.py` for directories containing multiple
recordings.

A typical Zoom directory might look like:

``` text
Zoom/
├── 2026-05-07 Meeting A/
│   ├── 2026-05-07 Meeting A.mp4
│   └── audio_only.m4a
│
├── 2026-05-13 Meeting B/
│   ├── 2026-05-13 Meeting B.mp4
│   └── audio_only.m4a
│
└── 2026-05-20 Meeting C/
    └── 2026-05-20 Meeting C.mp4
```

The batch script:

1.  Recursively searches the input directory.
2.  Identifies supported video and audio recordings.
3.  Checks whether a separate Zoom audio recording exists.
4.  Prefers the separate audio file when `--audio-source auto` is used.
5.  Checks whether the meeting has already been transcribed.
6.  Processes only missing transcripts unless `--force` is specified.
7.  Continues with the next recording if one recording fails.

Basic batch processing:

``` bash
python MeetingTranscriber_v2_batch.py /Users/ma_work/Documents/Zoom \
  --language de \
  --model auto \
  --summary-model auto \
  --min-speakers 2 \
  --max-speakers 4
```

## Dry run

Before processing a large directory, check what the script finds:

``` bash
python MeetingTranscriber_v2_batch.py /Users/ma_work/Documents/Zoom \
  --language de \
  --dry-run
```

The script reports which recordings would be processed and which would
be skipped.

## Transcript-only batch run

Useful for checking transcription quality before spending time
generating summaries:

``` bash
python MeetingTranscriber_v2_batch.py /Users/ma_work/Documents/Zoom \
  --language de \
  --model auto \
  --min-speakers 2 \
  --max-speakers 4 \
  --skip-summary
```

## Force re-processing

Existing transcripts are skipped by default.

To deliberately process them again:

``` bash
python MeetingTranscriber_v2_batch.py /Users/ma_work/Documents/Zoom \
  --language de \
  --force
```

------------------------------------------------------------------------

# Transcription model selection

The batch script provides model presets so that the transcription model
can be changed without modifying the code.

``` text
Preset      Model
----------  ------------------------------------------
auto        mlx-community/whisper-large-v3-turbo
balanced    mlx-community/whisper-large-v3-turbo
accurate    mlx-community/whisper-large-v3-mlx
fast        mlx-community/whisper-medium-mlx
```

For normal use:

``` bash
--model auto
```

`auto` currently selects **Whisper Large V3 Turbo**, which provides a
good balance between multilingual transcription quality and speed on
Apple Silicon.

For difficult recordings where transcription quality is more important
than processing time:

``` bash
--model accurate
```

For faster processing:

``` bash
--model fast
```

A compatible MLX Whisper model can also be specified directly:

``` bash
--model mlx-community/whisper-large-v3-mlx
```

The older explicit option remains available for compatibility:

``` bash
--whisper-model mlx-community/whisper-large-v3-mlx
```

Explicit model arguments take precedence over presets.

## Check available models

The batch script can inspect model availability:

``` bash
python MeetingTranscriber_v2_batch.py --check-models
```

This shows the configured transcription presets, indicates which Whisper
models are already available locally when detectable, and reports Ollama
models installed on the machine.

The script does **not** automatically switch to an arbitrary newly
released model. Model presets are intentionally validated choices so
that transcription behavior remains reproducible.

------------------------------------------------------------------------

# Summary model selection

Meeting summaries are generated locally through Ollama.

The batch script supports:

``` text
--summary-model auto
--summary-model accurate
--summary-model balanced
--summary-model fast
```

For normal use:

``` bash
--summary-model auto
```

`auto` examines the models already installed in the local Ollama
instance and selects a suitable available model. It does not
automatically download a new summary model.

You can always select an Ollama model explicitly:

``` bash
--summary-model llama3.1:8b
```

The older option remains supported:

``` bash
--ollama-model llama3.1:8b
```

To see locally installed Ollama models:

``` bash
ollama list
```

------------------------------------------------------------------------

# Offline mode

The first use of an MLX Whisper model may require downloading its model
files.

After the models are cached, batch processing can be explicitly
restricted to local models:

``` bash
python MeetingTranscriber_v2_batch.py /Users/ma_work/Documents/Zoom \
  --language de \
  --model auto \
  --summary-model auto \
  --offline
```

With `--offline`, the transcription model must already be available
locally. The script should stop rather than silently downloading a
missing model.

This is the recommended mode when recordings must remain in a strictly
offline processing environment.

------------------------------------------------------------------------

# Zoom audio handling

Zoom often stores a separate audio recording such as:

``` text
audio_only.m4a
```

This is usually preferable to extracting the audio track from the MP4.

Batch mode therefore defaults to:

``` bash
--audio-source auto
```

Available modes are:

``` text
auto    Prefer separate Zoom audio when available; otherwise use video audio.
video   Always use the video's audio track.
audio   Require a separate audio recording next to the video.
```

Example:

``` bash
python MeetingTranscriber_v2_batch.py /Users/ma_work/Documents/Zoom \
  --language de \
  --audio-source auto
```

------------------------------------------------------------------------

# Speaker diarization

The script uses:

-   WebRTC VAD for speech detection
-   Resemblyzer for speaker embeddings
-   agglomerative clustering for speaker grouping

Speaker names are not identified automatically. They are represented as:

``` text
SPEAKER_00
SPEAKER_01
SPEAKER_02
```

If you know the approximate number of participants, provide it:

``` bash
--min-speakers 2 --max-speakers 4
```

If the exact number is known, use the same value for both:

``` bash
--min-speakers 3 --max-speakers 3
```

Diarization is approximate. Overlapping speech, very short responses,
poor microphones, and similar voices can reduce accuracy.

------------------------------------------------------------------------

# Output

For each recording the scripts create files such as:

``` text
<recording_name>.16k.wav
<recording_name>.transcript.md
<recording_name>.summary.md
<recording_name>.meeting.json
```

The Markdown transcript contains timestamps, speaker labels, and cleaned
transcription text.

The JSON file contains structured transcript and diarization information
for further processing.

The generated 16 kHz WAV is intentionally retained because it is useful
for checking whether transcription problems originate from the source
audio or from the speech-recognition model.

In batch mode, the default structure is:

``` text
Recording Folder/
└── meeting_transcript/
    ├── <recording_name>.16k.wav
    ├── <recording_name>.transcript.md
    ├── <recording_name>.summary.md
    └── <recording_name>.meeting.json
```

A central output directory can instead be specified with:

``` bash
--out-dir /path/to/MeetingTranscripts
```

------------------------------------------------------------------------

# Recommended workflows

## Normal batch processing

``` bash
python MeetingTranscriber_v2_batch.py /Users/ma_work/Documents/Zoom \
  --language de \
  --model auto \
  --summary-model auto \
  --min-speakers 2 \
  --max-speakers 4
```

## Highest transcription accuracy

``` bash
python MeetingTranscriber_v2_batch.py /Users/ma_work/Documents/Zoom \
  --language de \
  --model accurate \
  --summary-model auto \
  --min-speakers 2 \
  --max-speakers 4
```

## Faster processing

``` bash
python MeetingTranscriber_v2_batch.py /Users/ma_work/Documents/Zoom \
  --language de \
  --model fast \
  --summary-model fast \
  --min-speakers 2 \
  --max-speakers 4
```

## Strictly local/offline processing

``` bash
python MeetingTranscriber_v2_batch.py /Users/ma_work/Documents/Zoom \
  --language de \
  --model auto \
  --summary-model auto \
  --min-speakers 2 \
  --max-speakers 4 \
  --offline
```

------------------------------------------------------------------------

# Troubleshooting

## `zsh: command not found: --language`

A multi-line shell command must end each continued line with `\`.

Correct:

``` bash
python MeetingTranscriber_v2.py meeting.mp4 \
  --out-dir ./meeting_transcript \
  --language de
```

Incorrect:

``` bash
python MeetingTranscriber_v2.py meeting.mp4
--out-dir ./meeting_transcript
--language de
```

Alternatively, put the entire command on one line.

## Transcript contains repeated phrases

First listen to the generated WAV:

``` bash
open /path/to/meeting_transcript/*.16k.wav
```

If the WAV itself is correct, the repetition is likely a Whisper
decoding artifact. The current scripts include repetition cleanup and
anti-hallucination settings, but difficult recordings can still
occasionally cause errors.

Try:

``` bash
--model accurate
```

and make sure the correct language is specified.

## Speaker labels are wrong

Speaker diarization is independent of Whisper transcription.

If the participant count is known, specify the exact number:

``` bash
--min-speakers 3 --max-speakers 3
```

## Ollama port already in use

If:

``` bash
ollama serve
```

reports that port `11434` is already in use, Ollama is probably already
running.

Check:

``` bash
ollama list
```

## Processing is slow

Transcription model size has the largest effect.

Try:

``` bash
--model fast
```

You can also use:

``` bash
--summary-model fast
```

For initial testing, skip summary generation entirely:

``` bash
--skip-summary
```

------------------------------------------------------------------------

# Privacy and local processing

The transcription, diarization, cleanup, and Ollama summarization stages
are designed to run locally.

The first use of a model may contact its model repository to download
model weights. After the required models are cached, use:

``` bash
--offline
```

to enforce local transcription-model use in batch mode.

For sensitive recordings, verify that all required models are already
installed before disconnecting the machine from the network.
