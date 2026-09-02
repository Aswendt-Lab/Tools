# MeetingTranscriber

Local meeting transcription, approximate speaker diarization, and optional Ollama-based summarization for Apple Silicon macOS. Audio and transcript processing can remain local after models have been downloaded.

## Scripts

- `MeetingTranscriber_v1.py`: first single-recording pipeline retained for reproducibility.
- `MeetingTranscriber_v2.py`: current single-recording pipeline with improved chunking and repetition filtering.
- `MeetingTranscriber_v2_batch.py`: recommended batch workflow; recursively discovers recordings, prefers separate Zoom audio where available, skips existing transcripts, supports model presets and offline operation, and continues after individual failures.

## Requirements

```bash
brew install ffmpeg ollama
pip install mlx-whisper resemblyzer webrtcvad scikit-learn soundfile numpy requests
```

The scripts enforce Apple Silicon because transcription uses MLX. For summaries, start Ollama and install a local model:

```bash
ollama serve
ollama pull llama3.1:8b
```

## Single recording

```bash
python MeetingTranscriber_v2.py /path/to/meeting.mp4 \
  --out-dir /path/to/output \
  --language de \
  --min-speakers 2 \
  --max-speakers 4
```

Use `--skip-summary` to produce transcripts without Ollama. Supported inputs include common video and audio formats; audio is normalized to a 16 kHz WAV with ffmpeg.

## Batch processing

Preview discovery first:

```bash
python MeetingTranscriber_v2_batch.py /path/to/recordings --language de --dry-run
```

Then process:

```bash
python MeetingTranscriber_v2_batch.py /path/to/recordings \
  --language de \
  --model auto \
  --summary-model auto \
  --min-speakers 2 \
  --max-speakers 4
```

Use `--offline` after the Whisper model is cached to prevent model downloads, `--skip-summary` for transcripts only, and `--force` to reprocess completed meetings. Run the selected script with `--help` for all current tuning and output options.

Outputs include Markdown/plain transcripts, JSON metadata, and optional Markdown summaries. Speaker labels are clustering estimates, not verified identities; review important decisions and action items against the recording.
