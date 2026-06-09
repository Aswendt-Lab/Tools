# MeetingTranscriber

Local meeting transcription and summarization for **Apple Silicon macOS**.

The script extracts audio from a video file, transcribes it locally with MLX Whisper, assigns rough speaker labels, removes common Whisper repetition artifacts, and optionally creates a meeting summary using a local Ollama model.

## Requirements

- macOS on Apple Silicon (`arm64`)
- Python environment, for example Conda or venv
- `ffmpeg`
- `ollama` for local summaries

Install system tools:

```bash
brew install ffmpeg ollama
```

Install Python dependencies:

```bash
pip install mlx-whisper resemblyzer webrtcvad scikit-learn soundfile numpy requests
```

For summary generation, start Ollama and pull a model:

```bash
ollama serve
ollama pull llama3.1:8b
```

If `ollama serve` says the port is already in use, Ollama is probably already running.

## Usage

Basic German meeting:

```bash
python MeetingTranscriber_v2.py /path/to/meeting.mp4 \
  --out-dir /path/to/output_folder \
  --language de \
  --min-speakers 2 \
  --max-speakers 4
```

English meeting:

```bash
python MeetingTranscriber_v2.py /path/to/meeting.mp4 \
  --out-dir /path/to/output_folder \
  --language en \
  --min-speakers 2 \
  --max-speakers 4
```

Use `en` for English and `de` for German. Do not use `eng`.

## Faster test run

To check the transcript first without running the summary:

```bash
python MeetingTranscriber_v2.py /path/to/meeting.mp4 \
  --out-dir /path/to/output_folder \
  --language de \
  --min-speakers 2 \
  --max-speakers 4 \
  --skip-summary
```

## Output files

The script writes files into the output folder:

```text
<video_name>.16k.wav
<video_name>.transcript.md
<video_name>.summary.md
<video_name>.meeting.json
```

The WAV file is useful for checking whether the extracted audio is correct.

## Model options

Default transcription model:

```bash
--whisper-model mlx-community/whisper-large-v3-mlx
```

Faster alternatives:

```bash
--whisper-model mlx-community/whisper-medium-mlx
--whisper-model mlx-community/whisper-small-mlx
```

For faster summaries, use a smaller Ollama model:

```bash
ollama pull llama3.2:3b
```

Then run with:

```bash
--ollama-model llama3.2:3b
```

## Notes

- Processing is local after the models have been downloaded.
- The first run may download Whisper model files.
- Speaker labels are generic, for example `SPEAKER_00`, `SPEAKER_01`.
- Diarization is approximate and may be less accurate with overlapping speakers.
- If the transcript contains repeated phrases, first check the extracted `.16k.wav` file to confirm that the audio is clear.
- If the transcript looks good, rerun without `--skip-summary` to create the summary.
