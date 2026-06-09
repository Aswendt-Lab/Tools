""""
Created on 15.05.2026

@authors: Markus Aswendt, ChatGPT v5.4
Department of Neurology
University Hospital Frankfurt
Theodor-Stern-Kai 7
D-60590 Frankfurt am Main

"""

#!/usr/bin/env python3

import argparse
import contextlib
import json
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import mlx_whisper
import numpy as np
import requests
import soundfile as sf
import webrtcvad

from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


FILLER_ONLY_WORDS = {
    "yes", "yeah", "mhm", "hm", "hmm", "uh", "um", "okay", "ok",
    "ja", "jaja", "genau", "äh", "ähm", "hmhm", "nein", "also",
    "thank", "you", "thanks", "danke", "bitte",
}


def require_apple_silicon() -> None:
    system = subprocess.check_output(["uname", "-s"], text=True).strip()
    machine = subprocess.check_output(["uname", "-m"], text=True).strip()

    if system != "Darwin" or machine != "arm64":
        raise RuntimeError(
            "This script is intentionally limited to Apple Silicon macOS only. "
            f"Detected system={system}, machine={machine}."
        )


def run_command(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_audio(video_path: Path, wav_path: Path) -> None:
    """
    Converts input video/audio to 16 kHz mono 16-bit PCM WAV.
    This format is required by WebRTC VAD and works well for Whisper.
    """
    run_command([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-f", "wav",
        str(wav_path),
    ])


def normalize_word(word: str) -> str:
    return re.sub(r"[^A-Za-zÄÖÜäöüß0-9]+", "", word).lower()


def collapse_consecutive_words(text: str, max_repeat: int = 2) -> str:
    """
    Example:
      "yes yes yes yes"
    becomes:
      "yes yes"

    Keeps up to max_repeat repetitions because natural speech can repeat once.
    """
    tokens = text.split()

    if not tokens:
        return text

    output = []
    last_norm = None
    repeat_count = 0

    for token in tokens:
        norm = normalize_word(token)

        if norm and norm == last_norm:
            repeat_count += 1
        else:
            repeat_count = 1
            last_norm = norm

        if repeat_count <= max_repeat:
            output.append(token)

    return " ".join(output)


def collapse_repeated_phrases(text: str, max_phrase_words: int = 10) -> str:
    """
    Removes repeated phrase loops.

    Example:
      "Then early registration. Then early registration. Then early registration."
    becomes:
      "Then early registration."
    """
    words = text.split()

    if len(words) < 6:
        return text

    i = 0
    output = []

    while i < len(words):
        removed_loop = False

        for phrase_len in range(max_phrase_words, 1, -1):
            if i + phrase_len * 2 > len(words):
                continue

            phrase = words[i:i + phrase_len]
            next_phrase = words[i + phrase_len:i + phrase_len * 2]

            phrase_norm = [normalize_word(w) for w in phrase]
            next_norm = [normalize_word(w) for w in next_phrase]

            if phrase_norm == next_norm:
                output.extend(phrase)

                i += phrase_len * 2

                while i + phrase_len <= len(words):
                    candidate = words[i:i + phrase_len]
                    candidate_norm = [normalize_word(w) for w in candidate]

                    if candidate_norm == phrase_norm:
                        i += phrase_len
                    else:
                        break

                removed_loop = True
                break

        if not removed_loop:
            output.append(words[i])
            i += 1

    return " ".join(output)


def remove_repeated_sentences(text: str, max_repeat: int = 1) -> str:
    """
    Removes repeated adjacent sentences.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())

    if len(parts) <= 1:
        return text

    output = []
    last_norm = None
    repeat_count = 0

    for part in parts:
        norm = re.sub(r"\s+", " ", part.strip().lower())

        if norm == last_norm:
            repeat_count += 1
        else:
            repeat_count = 1
            last_norm = norm

        if repeat_count <= max_repeat:
            output.append(part)

    return " ".join(output)


def remove_low_diversity_tail(text: str, min_tail_words: int = 12, diversity_threshold: float = 0.25) -> str:
    """
    Removes obvious hallucinated tails such as:
      "... then 3 3 3 3 3 3 3 3 ..."
      "... morph morph morph morph ..."
    """
    words = text.split()

    if len(words) < min_tail_words:
        return text

    for start in range(len(words) - min_tail_words):
        tail = words[start:]
        norm_tail = [normalize_word(w) for w in tail if normalize_word(w)]

        if len(norm_tail) < min_tail_words:
            continue

        unique_ratio = len(set(norm_tail)) / len(norm_tail)

        if unique_ratio < diversity_threshold:
            return " ".join(words[:start]).strip()

    return text


def clean_repeated_text(text: str) -> str:
    """
    Main cleanup function for Whisper repetition artifacts.
    """
    text = re.sub(r"\s+", " ", text).strip()

    previous = None

    for _ in range(4):
        if text == previous:
            break

        previous = text
        text = collapse_consecutive_words(text, max_repeat=2)
        text = collapse_repeated_phrases(text, max_phrase_words=10)
        text = remove_repeated_sentences(text, max_repeat=1)
        text = remove_low_diversity_tail(text)
        text = re.sub(r"\s+", " ", text).strip()

    return text


def should_keep_transcript_segment(text: str) -> bool:
    """
    Drops low-information hallucination-like segments.
    Keeps normal speech.
    """
    clean = re.sub(r"[^A-Za-zÄÖÜäöüß0-9\s]", " ", text).strip()
    words = [normalize_word(w) for w in clean.split()]
    words = [w for w in words if w]

    if not words:
        return False

    normalized_text = " ".join(words)

    # Common Whisper hallucinations during silence / low-volume audio.
    silence_hallucinations = {
        "thank you",
        "thank you thank you",
        "thanks",
        "thanks for watching",
        "thank you for watching",
        "danke",
        "danke schön",
        "vielen dank",
    }

    if normalized_text in silence_hallucinations:
        return False

    if len(words) <= 5 and all(w in FILLER_ONLY_WORDS for w in words):
        return False

    unique_ratio = len(set(words)) / len(words)

    if len(words) >= 8 and unique_ratio < 0.25:
        return False

    return True


def segment_text_key(text: str) -> str:
    """
    Normalized key used to detect repeated transcript loops across segments.
    """
    words = [normalize_word(w) for w in text.split()]
    words = [w for w in words if w]
    return " ".join(words)


def remove_repeated_segment_loops(
    segments: List[TranscriptSegment],
    max_consecutive_repeats: int = 2,
) -> List[TranscriptSegment]:
    """
    Removes transcript loops across segment boundaries.

    This catches cases where Whisper outputs the same phrase for many consecutive
    timestamps, e.g. "Ja, ich hab noch eine E-Mail" for minutes.
    """
    cleaned: List[TranscriptSegment] = []
    last_key = None
    repeat_count = 0

    for seg in segments:
        key = segment_text_key(seg.text)

        if key and key == last_key:
            repeat_count += 1
        else:
            repeat_count = 1
            last_key = key

        if repeat_count <= max_consecutive_repeats:
            cleaned.append(seg)

    return cleaned


def merge_regions_for_transcription(
    regions: List[SpeakerSegment],
    max_chunk_seconds: float = 30.0,
    max_gap_seconds: float = 1.0,
    pad_seconds: float = 0.25,
    total_duration: Optional[float] = None,
) -> List[SpeakerSegment]:
    """
    Merges VAD speech regions into chunks for Whisper.

    This prevents Whisper from transcribing long silence/noise stretches, which
    is the main cause of repeated hallucinations such as "Thank you" or
    "Das ist ganz doof".
    """
    if not regions:
        return []

    regions = sorted(regions, key=lambda r: r.start)
    merged: List[SpeakerSegment] = []

    cur_start = regions[0].start
    cur_end = regions[0].end

    for region in regions[1:]:
        gap = region.start - cur_end
        new_len = region.end - cur_start

        if gap <= max_gap_seconds and new_len <= max_chunk_seconds:
            cur_end = max(cur_end, region.end)
        else:
            start = max(0.0, cur_start - pad_seconds)
            end = cur_end + pad_seconds
            if total_duration is not None:
                end = min(total_duration, end)
            merged.append(SpeakerSegment(start=start, end=end, speaker="UNKNOWN"))
            cur_start = region.start
            cur_end = region.end

    start = max(0.0, cur_start - pad_seconds)
    end = cur_end + pad_seconds
    if total_duration is not None:
        end = min(total_duration, end)
    merged.append(SpeakerSegment(start=start, end=end, speaker="UNKNOWN"))

    return merged


def transcribe_audio_chunk(
    chunk_wav_path: Path,
    whisper_model: str,
    language: Optional[str],
) -> List[TranscriptSegment]:
    """
    Transcribes one speech chunk with anti-loop decoding settings.
    """
    kwargs: Dict[str, Any] = {
        "path_or_hf_repo": whisper_model,
        "verbose": False,

        # Deterministic decoding.
        "temperature": 0.0,

        # Crucial: do not let a hallucinated phrase carry into the next segment.
        "condition_on_previous_text": False,

        # Stricter hallucination filtering than the original version.
        "compression_ratio_threshold": 2.0,
        "logprob_threshold": -0.7,
        "no_speech_threshold": 0.85,

        # Important: transcribe, do not translate.
        "task": "transcribe",
    }

    if language:
        kwargs["language"] = language

    result = mlx_whisper.transcribe(str(chunk_wav_path), **kwargs)

    segments: List[TranscriptSegment] = []

    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()

        if not text:
            continue

        text = clean_repeated_text(text)

        if should_keep_transcript_segment(text):
            segments.append(
                TranscriptSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=text,
                )
            )

    return segments


def transcribe_audio(
    wav_path: Path,
    whisper_model: str,
    language: Optional[str],
) -> List[TranscriptSegment]:
    """
    Local transcription using MLX Whisper on Apple Silicon.

    This version first cuts the audio into VAD speech chunks and only sends those
    chunks to Whisper. This greatly reduces repeated hallucinations in silence.
    """
    print(f"Transcribing with MLX Whisper model: {whisper_model}")
    print("Detecting speech regions before transcription to reduce hallucinations")

    duration = read_wav_duration(wav_path)

    speech_regions = collect_speech_regions(
        wav_path=wav_path,
        aggressiveness=3,
        frame_ms=30,
        padding_ms=300,
        min_region_seconds=0.6,
    )

    transcript_chunks = merge_regions_for_transcription(
        regions=speech_regions,
        max_chunk_seconds=30.0,
        max_gap_seconds=1.0,
        pad_seconds=0.25,
        total_duration=duration,
    )

    if not transcript_chunks:
        print("No speech regions detected. Falling back to whole-file transcription.")
        transcript_chunks = [SpeakerSegment(start=0.0, end=duration, speaker="UNKNOWN")]

    wav, sample_rate = sf.read(str(wav_path))
    if wav.ndim > 1:
        wav = wav[:, 0]

    all_segments: List[TranscriptSegment] = []

    with tempfile.TemporaryDirectory(prefix="meeting_transcribe_chunks_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for idx, chunk in enumerate(transcript_chunks, start=1):
            print(f"Transcribing speech chunk {idx}/{len(transcript_chunks)} "
                  f"[{format_timestamp(chunk.start)} - {format_timestamp(chunk.end)}]")

            start_sample = max(0, int(chunk.start * sample_rate))
            end_sample = min(len(wav), int(chunk.end * sample_rate))
            audio_chunk = wav[start_sample:end_sample]

            if len(audio_chunk) < int(0.5 * sample_rate):
                continue

            chunk_path = tmpdir_path / f"chunk_{idx:05d}.wav"
            sf.write(str(chunk_path), audio_chunk, sample_rate, subtype="PCM_16")

            chunk_segments = transcribe_audio_chunk(
                chunk_wav_path=chunk_path,
                whisper_model=whisper_model,
                language=language,
            )

            for seg in chunk_segments:
                seg.start += chunk.start
                seg.end += chunk.start
                all_segments.append(seg)

    all_segments = sorted(all_segments, key=lambda s: s.start)
    all_segments = remove_repeated_segment_loops(all_segments, max_consecutive_repeats=2)

    return all_segments


def read_wav_duration(wav_path: Path) -> float:
    with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def collect_speech_regions(
    wav_path: Path,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    padding_ms: int = 300,
    min_region_seconds: float = 0.8,
) -> List[SpeakerSegment]:
    """
    Detects speech regions using WebRTC VAD.

    aggressiveness:
      0 = least aggressive
      3 = most aggressive

    frame_ms must be 10, 20, or 30.
    """
    if frame_ms not in (10, 20, 30):
        raise ValueError("frame_ms must be 10, 20, or 30.")

    vad = webrtcvad.Vad(aggressiveness)

    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        audio = wf.readframes(wf.getnframes())

    if sample_rate != 16000 or channels != 1 or sample_width != 2:
        raise RuntimeError(
            "Expected 16 kHz mono 16-bit PCM WAV. "
            "The ffmpeg extraction step should create this format."
        )

    frame_bytes = int(sample_rate * frame_ms / 1000) * sample_width
    padding_frames = max(1, padding_ms // frame_ms)

    frames = []

    for i in range(0, len(audio) - frame_bytes + 1, frame_bytes):
        start = (i / sample_width) / sample_rate
        end = start + frame_ms / 1000
        frame = audio[i:i + frame_bytes]
        is_speech = vad.is_speech(frame, sample_rate)
        frames.append((start, end, is_speech))

    regions: List[SpeakerSegment] = []

    active = False
    region_start = 0.0
    silence_count = 0

    for start, end, is_speech in frames:
        if is_speech and not active:
            active = True
            region_start = start
            silence_count = 0

        elif is_speech and active:
            silence_count = 0

        elif not is_speech and active:
            silence_count += 1

            if silence_count >= padding_frames:
                region_end = end - padding_frames * frame_ms / 1000

                if region_end - region_start >= min_region_seconds:
                    regions.append(
                        SpeakerSegment(
                            start=region_start,
                            end=region_end,
                            speaker="UNKNOWN",
                        )
                    )

                active = False
                silence_count = 0

    if active:
        duration = read_wav_duration(wav_path)

        if duration - region_start >= min_region_seconds:
            regions.append(
                SpeakerSegment(
                    start=region_start,
                    end=duration,
                    speaker="UNKNOWN",
                )
            )

    return regions


def split_long_regions(
    regions: List[SpeakerSegment],
    max_len: float = 6.0,
    min_len: float = 1.0,
) -> List[SpeakerSegment]:
    """
    Speaker embeddings work better on moderately short chunks.
    """
    chunks: List[SpeakerSegment] = []

    for region in regions:
        cur = region.start

        while cur < region.end:
            nxt = min(cur + max_len, region.end)

            if nxt - cur >= min_len:
                chunks.append(
                    SpeakerSegment(
                        start=cur,
                        end=nxt,
                        speaker="UNKNOWN",
                    )
                )

            cur = nxt

    return chunks


def estimate_num_speakers(
    embeddings: np.ndarray,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> int:
    """
    Simple fallback speaker-count heuristic.

    Best result:
      --min-speakers 3 --max-speakers 3
    """
    if min_speakers is not None and max_speakers is not None:
        if min_speakers == max_speakers:
            return min_speakers

    if max_speakers is not None:
        return max_speakers

    if min_speakers is not None:
        return min_speakers

    return min(4, max(2, len(embeddings) // 8))


def merge_speaker_segments(
    segments: List[SpeakerSegment],
    max_gap_seconds: float = 0.7,
) -> List[SpeakerSegment]:
    """
    Merges adjacent speaker chunks with the same speaker label.
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: s.start)
    merged: List[SpeakerSegment] = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]

        same_speaker = seg.speaker == prev.speaker
        small_gap = seg.start - prev.end <= max_gap_seconds

        if same_speaker and small_gap:
            prev.end = seg.end
        else:
            merged.append(seg)

    return merged


def diarize_audio(
    wav_path: Path,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    vad_aggressiveness: int = 2,
) -> List[SpeakerSegment]:
    """
    Hugging Face-free local diarization using:
      - WebRTC VAD
      - Resemblyzer speaker embeddings
      - Agglomerative clustering
    """
    print("Diarizing locally with Resemblyzer + WebRTC VAD + clustering")

    speech_regions = collect_speech_regions(
        wav_path=wav_path,
        aggressiveness=vad_aggressiveness,
    )

    chunks = split_long_regions(speech_regions)

    if not chunks:
        print("No speech chunks detected.")
        return []

    wav, sample_rate = sf.read(str(wav_path))

    if wav.ndim > 1:
        wav = wav[:, 0]

    encoder = VoiceEncoder()

    embeddings = []
    valid_chunks: List[SpeakerSegment] = []

    for chunk in chunks:
        start_sample = int(chunk.start * sample_rate)
        end_sample = int(chunk.end * sample_rate)

        audio_chunk = wav[start_sample:end_sample]

        if len(audio_chunk) < sample_rate:
            continue

        processed = preprocess_wav(audio_chunk, source_sr=sample_rate)
        embedding = encoder.embed_utterance(processed)

        embeddings.append(embedding)
        valid_chunks.append(chunk)

    if not embeddings:
        print("No valid speaker embeddings created.")
        return []

    embeddings_array = np.vstack(embeddings)

    n_speakers = estimate_num_speakers(
        embeddings=embeddings_array,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    n_speakers = max(1, min(n_speakers, len(valid_chunks)))

    clustering = AgglomerativeClustering(
        n_clusters=n_speakers,
        metric="cosine",
        linkage="average",
    )

    labels = clustering.fit_predict(embeddings_array)

    diarized: List[SpeakerSegment] = []

    for chunk, label in zip(valid_chunks, labels):
        diarized.append(
            SpeakerSegment(
                start=chunk.start,
                end=chunk.end,
                speaker=f"SPEAKER_{int(label):02d}",
            )
        )

    return merge_speaker_segments(diarized)


def overlap_seconds(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    transcript_segments: List[TranscriptSegment],
    speaker_segments: List[SpeakerSegment],
) -> List[TranscriptSegment]:
    """
    Assigns each Whisper transcript segment to the speaker segment
    with the greatest timestamp overlap.
    """
    if not speaker_segments:
        return transcript_segments

    for t in transcript_segments:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for s in speaker_segments:
            overlap = overlap_seconds(
                t.start,
                t.end,
                s.start,
                s.end,
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = s.speaker

        t.speaker = best_speaker

    return transcript_segments


def merge_adjacent_transcript_segments(
    segments: List[TranscriptSegment],
    max_gap_seconds: float = 1.5,
) -> List[TranscriptSegment]:
    """
    Combines adjacent transcript segments from the same speaker.
    """
    if not segments:
        return []

    merged: List[TranscriptSegment] = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]

        same_speaker = seg.speaker == prev.speaker
        small_gap = seg.start - prev.end <= max_gap_seconds

        if same_speaker and small_gap:
            prev.end = seg.end
            prev.text = f"{prev.text} {seg.text}".strip()
        else:
            merged.append(seg)

    return merged


def clean_transcript_segments_after_merge(
    segments: List[TranscriptSegment],
) -> List[TranscriptSegment]:
    """
    Cleans repetition artifacts after speaker assignment and segment merging.
    """
    cleaned: List[TranscriptSegment] = []

    for seg in segments:
        seg.text = clean_repeated_text(seg.text)

        if should_keep_transcript_segment(seg.text):
            cleaned.append(seg)

    cleaned = remove_repeated_segment_loops(cleaned, max_consecutive_repeats=2)
    return cleaned


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def transcript_to_markdown(segments: List[TranscriptSegment]) -> str:
    lines = ["# Meeting Transcript", ""]

    for seg in segments:
        lines.append(
            f"**[{format_timestamp(seg.start)} - {format_timestamp(seg.end)}] "
            f"{seg.speaker}:** {seg.text}"
        )
        lines.append("")

    return "\n".join(lines)


def transcript_to_plain_text(segments: List[TranscriptSegment]) -> str:
    lines = []

    for seg in segments:
        lines.append(
            f"[{format_timestamp(seg.start)} - {format_timestamp(seg.end)}] "
            f"{seg.speaker}: {seg.text}"
        )

    return "\n".join(lines)


def chunk_text(
    text: str,
    max_chars: int = 18000,
) -> List[str]:
    """
    Simple text chunking for long meetings.
    """
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines()
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1

        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def call_ollama(
    prompt: str,
    ollama_model: str,
    ollama_url: str,
    temperature: float = 0.2,
    num_ctx: int = 32768,
) -> str:
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        },
        timeout=900,
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def summarize_chunk_with_ollama(
    transcript_chunk: str,
    ollama_model: str,
    ollama_url: str,
    chunk_number: int,
    total_chunks: int,
) -> str:
    prompt = f"""
You are a precise meeting assistant.

Summarize transcript chunk {chunk_number} of {total_chunks}.

Return:

# Chunk Summary
A concise summary of this part.

# Key Points
Bullet list of important points.

# Decisions
Explicit decisions, if any.

# Action Items
Action items with owner and due date if mentioned.

# Open Questions
Open questions, risks, blockers, or dependencies.

Transcript chunk:

{transcript_chunk}
""".strip()

    return call_ollama(
        prompt=prompt,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
    )


def combine_summaries_with_ollama(
    chunk_summaries: List[str],
    ollama_model: str,
    ollama_url: str,
) -> str:
    combined = "\n\n---\n\n".join(
        f"PART {i + 1}\n\n{summary}"
        for i, summary in enumerate(chunk_summaries)
    )

    prompt = f"""
You are a precise meeting assistant.

Create the final meeting summary from the partial summaries below.

Use this exact structure:

# Executive Summary
Summarize the whole meeting in 5-8 sentences.

# Key Topics
List the main topics discussed.

# Decisions
List explicit decisions. If none, say "No explicit decisions captured."

# Action Items
For each action item, include:
- Owner, if mentioned
- Task
- Due date, if mentioned

# Risks / Blockers
List risks, blockers, open questions, or dependencies.

# Speaker Contributions
Summarize what each speaker contributed. Use speaker labels such as SPEAKER_00.

# Suggested Follow-Up
List useful follow-up steps.

Partial summaries:

{combined}
""".strip()

    return call_ollama(
        prompt=prompt,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
    )


def summarize_with_ollama(
    transcript_text: str,
    ollama_model: str,
    ollama_url: str,
    max_chars_per_chunk: int,
) -> str:
    chunks = chunk_text(transcript_text, max_chars=max_chars_per_chunk)

    if len(chunks) == 1:
        prompt = f"""
You are a precise meeting assistant.

Summarize the following meeting transcript.

Use this exact structure:

# Executive Summary
Summarize the meeting in 5-8 sentences.

# Key Topics
List the main topics discussed.

# Decisions
List explicit decisions. If none, say "No explicit decisions captured."

# Action Items
For each action item, include:
- Owner, if mentioned
- Task
- Due date, if mentioned

# Risks / Blockers
List risks, blockers, open questions, or dependencies.

# Speaker Contributions
Summarize what each speaker contributed. Use speaker labels such as SPEAKER_00.

# Suggested Follow-Up
List useful follow-up steps.

Transcript:

{transcript_text}
""".strip()

        return call_ollama(
            prompt=prompt,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
        )

    print(f"Transcript is long. Summarizing in {len(chunks)} chunks.")

    chunk_summaries = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {i}/{len(chunks)}")

        chunk_summary = summarize_chunk_with_ollama(
            transcript_chunk=chunk,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
            chunk_number=i,
            total_chunks=len(chunks),
        )

        chunk_summaries.append(chunk_summary)

    return combine_summaries_with_ollama(
        chunk_summaries=chunk_summaries,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
    )


def save_json(
    output_path: Path,
    transcript_segments: List[TranscriptSegment],
    speaker_segments: List[SpeakerSegment],
    summary: str,
) -> None:
    payload = {
        "summary": summary,
        "transcript": [
            {
                "start": seg.start,
                "end": seg.end,
                "start_timestamp": format_timestamp(seg.start),
                "end_timestamp": format_timestamp(seg.end),
                "speaker": seg.speaker,
                "text": seg.text,
            }
            for seg in transcript_segments
        ],
        "diarization": [
            {
                "start": seg.start,
                "end": seg.end,
                "start_timestamp": format_timestamp(seg.start),
                "end_timestamp": format_timestamp(seg.end),
                "speaker": seg.speaker,
            }
            for seg in speaker_segments
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )



VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg"}

SKIP_DIRECTORY_NAMES = {
    "meeting_transcript",
    "meeting_output",
    "meeting_output_clean",
    "meeting_output_fixed",
    "meeting_output_vad_clean",
    "__pycache__",
}


@dataclass
class ProcessingJob:
    """
    Represents one recording to process.

    recording_path:
      The original video/audio file used as the logical recording name.

    source_audio_or_video:
      The actual file converted to 16 kHz WAV. For Zoom recordings this may be
      audio_only.m4a instead of the MP4.
    """
    recording_path: Path
    source_audio_or_video: Path
    output_dir: Path
    base_name: str


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def is_generated_or_temporary_file(path: Path) -> bool:
    """
    Avoid re-processing files created by this script.
    """
    name = path.name.lower()
    stem = path.stem.lower()

    if any(part in SKIP_DIRECTORY_NAMES for part in path.parts):
        return True

    if name.endswith(".16k.wav") or name.endswith("_16k.wav"):
        return True

    if stem.endswith(".transcript") or stem.endswith("_transcript"):
        return True

    if stem.endswith(".summary") or stem.endswith("_summary"):
        return True

    if stem.endswith(".meeting") or stem.endswith("_meeting"):
        return True

    return False


def find_zoom_audio_file(video_path: Path) -> Optional[Path]:
    """
    Zoom recordings often contain a separate audio file in the same folder.
    Prefer that over extracting audio from the MP4.

    Typical Zoom names:
      audio_only.m4a
      audio_only.mp4
    """
    folder = video_path.parent

    preferred_names = [
        "audio_only.m4a",
        "audio_only.mp4",
        "audio.m4a",
        "audio.wav",
        "Audio Record.m4a",
    ]

    for name in preferred_names:
        candidate = folder / name
        if candidate.exists() and candidate.is_file():
            return candidate

    candidates = [
        p for p in folder.iterdir()
        if is_audio_file(p)
        and not is_generated_or_temporary_file(p)
        and p.resolve() != video_path.resolve()
    ]

    if not candidates:
        return None

    # Prefer audio-ish names first.
    for candidate in sorted(candidates):
        lower = candidate.name.lower()
        if "audio" in lower or "zoom" in lower or "record" in lower:
            return candidate

    # If there is only one separate audio file, use it.
    if len(candidates) == 1:
        return candidates[0]

    return None


def transcript_output_paths(out_dir: Path, base_name: str) -> List[Path]:
    """
    Supports both historical dot naming and newer underscore naming.
    The current script writes dot naming by default.
    """
    return [
        out_dir / f"{base_name}.transcript.md",
        out_dir / f"{base_name}_transcript.md",
    ]


def transcript_already_exists(out_dir: Path, base_name: str) -> bool:
    return any(path.exists() for path in transcript_output_paths(out_dir, base_name))


def default_output_dir_for_recording(recording_path: Path) -> Path:
    return recording_path.parent / "meeting_transcript"


def safe_folder_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "recording"


def output_dir_for_job(
    recording_path: Path,
    input_root: Path,
    batch_out_root: Optional[Path],
    output_folder_name: str,
) -> Path:
    """
    For batch mode:
      - default: put output into each recording folder / output_folder_name
      - if --out-dir is given: mirror the relative folder structure under that root
    """
    if batch_out_root is None:
        return recording_path.parent / output_folder_name

    try:
        relative_parent = recording_path.parent.relative_to(input_root)
    except ValueError:
        relative_parent = Path(".")

    if str(relative_parent) == ".":
        return batch_out_root / safe_folder_name(recording_path.stem)

    return batch_out_root / relative_parent / safe_folder_name(recording_path.stem)


def choose_source_for_recording(
    recording_path: Path,
    audio_source_mode: str,
) -> Path:
    """
    audio_source_mode:
      auto  = prefer separate Zoom audio for video, otherwise use video/audio input
      video = always use the video/audio file provided
      audio = require/prefer a separate audio file for videos
    """
    if is_audio_file(recording_path):
        return recording_path

    if not is_video_file(recording_path):
        return recording_path

    if audio_source_mode == "video":
        return recording_path

    zoom_audio = find_zoom_audio_file(recording_path)

    if zoom_audio:
        return zoom_audio

    if audio_source_mode == "audio":
        raise FileNotFoundError(
            f"No separate audio file found next to video: {recording_path}"
        )

    return recording_path


def discover_recordings(
    input_path: Path,
    recursive: bool,
    batch_out_root: Optional[Path],
    output_folder_name: str,
    audio_source_mode: str,
) -> List[ProcessingJob]:
    """
    Finds recordings from a file or directory.

    Directory behavior:
      - scans subfolders
      - if a folder contains videos, processes the videos and prefers Zoom audio
      - if a folder contains no videos but contains audio files, processes audio files
      - skips generated output files and output folders
    """
    input_path = input_path.expanduser().resolve()

    if input_path.is_file():
        if not (is_video_file(input_path) or is_audio_file(input_path)):
            raise ValueError(f"Unsupported input file type: {input_path}")

        source = choose_source_for_recording(input_path, audio_source_mode)
        out_dir = batch_out_root if batch_out_root else default_output_dir_for_recording(input_path)

        return [
            ProcessingJob(
                recording_path=input_path,
                source_audio_or_video=source,
                output_dir=out_dir,
                base_name=input_path.stem,
            )
        ]

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    if recursive:
        folders = [input_path] + [
            p for p in input_path.rglob("*")
            if p.is_dir() and not any(part in SKIP_DIRECTORY_NAMES for part in p.parts)
        ]
    else:
        folders = [input_path] + [
            p for p in input_path.iterdir()
            if p.is_dir() and p.name not in SKIP_DIRECTORY_NAMES
        ]

    jobs: List[ProcessingJob] = []

    for folder in sorted(folders):
        if any(part in SKIP_DIRECTORY_NAMES for part in folder.parts):
            continue

        files = [
            p for p in folder.iterdir()
            if p.is_file() and not is_generated_or_temporary_file(p)
        ]

        videos = sorted([p for p in files if is_video_file(p)])
        audios = sorted([p for p in files if is_audio_file(p)])

        if videos:
            for video in videos:
                source = choose_source_for_recording(video, audio_source_mode)
                out_dir = output_dir_for_job(
                    recording_path=video,
                    input_root=input_path,
                    batch_out_root=batch_out_root,
                    output_folder_name=output_folder_name,
                )
                jobs.append(
                    ProcessingJob(
                        recording_path=video,
                        source_audio_or_video=source,
                        output_dir=out_dir,
                        base_name=video.stem,
                    )
                )
        else:
            for audio in audios:
                out_dir = output_dir_for_job(
                    recording_path=audio,
                    input_root=input_path,
                    batch_out_root=batch_out_root,
                    output_folder_name=output_folder_name,
                )
                jobs.append(
                    ProcessingJob(
                        recording_path=audio,
                        source_audio_or_video=audio,
                        output_dir=out_dir,
                        base_name=audio.stem,
                    )
                )

    # De-duplicate by recording path.
    unique: Dict[str, ProcessingJob] = {}
    for job in jobs:
        unique[str(job.recording_path.resolve())] = job

    return list(unique.values())


def process_recording_job(
    job: ProcessingJob,
    args: argparse.Namespace,
) -> None:
    """
    Runs the full transcription/summarization pipeline for one recording.
    """
    out_dir = job.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = job.base_name

    wav_path = out_dir / f"{base_name}.16k.wav"
    transcript_md_path = out_dir / f"{base_name}.transcript.md"
    summary_md_path = out_dir / f"{base_name}.summary.md"
    json_path = out_dir / f"{base_name}.meeting.json"

    if transcript_already_exists(out_dir, base_name) and not args.force:
        print()
        print(f"Skipping existing transcript: {job.recording_path}")
        print(f"Transcript already exists in: {out_dir}")
        return

    print()
    print("=" * 80)
    print(f"Recording: {job.recording_path}")
    print(f"Audio source: {job.source_audio_or_video}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)
    print()

    extract_audio(job.source_audio_or_video, wav_path)

    transcript_segments = transcribe_audio(
        wav_path=wav_path,
        whisper_model=args.whisper_model,
        language=args.language,
    )

    if args.skip_diarization:
        print("Skipping diarization.")
        speaker_segments = []
        for seg in transcript_segments:
            seg.speaker = "SPEAKER_00"
    else:
        speaker_segments = diarize_audio(
            wav_path=wav_path,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            vad_aggressiveness=args.vad_aggressiveness,
        )

        transcript_segments = assign_speakers(
            transcript_segments=transcript_segments,
            speaker_segments=speaker_segments,
        )

    transcript_segments = merge_adjacent_transcript_segments(transcript_segments)
    transcript_segments = clean_transcript_segments_after_merge(transcript_segments)

    transcript_md = transcript_to_markdown(transcript_segments)
    transcript_md_path.write_text(transcript_md, encoding="utf-8")

    transcript_text = transcript_to_plain_text(transcript_segments)

    if args.skip_summary:
        summary = "Summary skipped."
        summary_md_path.write_text(
            "# Meeting Summary\n\nSummary skipped.\n",
            encoding="utf-8",
        )
    else:
        summary = summarize_with_ollama(
            transcript_text=transcript_text,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            max_chars_per_chunk=args.max_chars_per_summary_chunk,
        )

        summary_md = f"# Meeting Summary\n\n{summary}\n"
        summary_md_path.write_text(summary_md, encoding="utf-8")

    save_json(
        output_path=json_path,
        transcript_segments=transcript_segments,
        speaker_segments=speaker_segments,
        summary=summary,
    )

    print()
    print("Done.")
    print(f"Audio:      {wav_path}")
    print(f"Transcript: {transcript_md_path}")
    print(f"Summary:    {summary_md_path}")
    print(f"JSON:       {json_path}")



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Local Apple Silicon batch meeting transcription. "
            "Accepts one video/audio file or a parent folder with Zoom/session subfolders."
        )
    )

    parser.add_argument(
        "input_path",
        help="Path to one video/audio file or to a folder containing recording subfolders.",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. For one file: writes directly there. "
            "For a folder: mirrors recording folders under this output root. "
            "If omitted in folder mode, each recording gets a 'meeting_transcript' subfolder."
        ),
    )

    parser.add_argument(
        "--output-folder-name",
        default="meeting_transcript",
        help=(
            "Folder name created next to each recording when --out-dir is not used "
            "in folder mode. Default: meeting_transcript."
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan input folders. Enabled by default.",
    )

    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the input folder and its direct subfolders.",
    )

    parser.add_argument(
        "--audio-source",
        choices=["auto", "video", "audio"],
        default="auto",
        help=(
            "auto: for videos, prefer separate Zoom audio file if present; "
            "video: always extract from video; "
            "audio: require separate audio next to videos."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process recordings even if a transcript already exists.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list what would be processed. Do not transcribe.",
    )

    parser.add_argument(
        "--whisper-model",
        default="mlx-community/whisper-large-v3-mlx",
        help=(
            "MLX Whisper model or local model path. Examples: "
            "mlx-community/whisper-small-mlx, "
            "mlx-community/whisper-medium-mlx, "
            "mlx-community/whisper-large-v3-mlx"
        ),
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code, e.g. en, de, fr. Use 'en' for English, not 'eng'.",
    )

    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Optional minimum number of speakers.",
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional maximum number of speakers.",
    )

    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="WebRTC VAD aggressiveness for diarization: 0 least aggressive, 3 most aggressive.",
    )

    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="Skip speaker diarization and label all transcript segments as SPEAKER_00.",
    )

    parser.add_argument(
        "--ollama-model",
        default="llama3.1:8b",
        help="Local Ollama model for summarization.",
    )

    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL.",
    )

    parser.add_argument(
        "--max-chars-per-summary-chunk",
        type=int,
        default=18000,
        help="Maximum transcript characters per summarization chunk.",
    )

    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Only create transcript and JSON. Do not summarize with Ollama.",
    )

    args = parser.parse_args()

    require_apple_silicon()

    input_path = Path(args.input_path).expanduser().resolve()
    batch_out_root = Path(args.out_dir).expanduser().resolve() if args.out_dir else None

    jobs = discover_recordings(
        input_path=input_path,
        recursive=args.recursive,
        batch_out_root=batch_out_root,
        output_folder_name=args.output_folder_name,
        audio_source_mode=args.audio_source,
    )

    if not jobs:
        print(f"No video/audio recordings found in: {input_path}")
        return

    print(f"Found {len(jobs)} recording(s).")

    pending_jobs: List[ProcessingJob] = []
    skipped_count = 0

    for job in jobs:
        if transcript_already_exists(job.output_dir, job.base_name) and not args.force:
            skipped_count += 1
        else:
            pending_jobs.append(job)

    print(f"Already processed: {skipped_count}")
    print(f"To process:         {len(pending_jobs)}")

    if args.dry_run:
        print()
        print("Dry run. No files will be processed.")
        for job in jobs:
            status = "SKIP exists" if transcript_already_exists(job.output_dir, job.base_name) and not args.force else "PROCESS"
            print(f"[{status}]")
            print(f"  recording: {job.recording_path}")
            print(f"  source:    {job.source_audio_or_video}")
            print(f"  output:    {job.output_dir}")
        return

    failures = 0

    for index, job in enumerate(jobs, start=1):
        print()
        print(f"Job {index}/{len(jobs)}")

        try:
            process_recording_job(job, args)
        except Exception as exc:
            failures += 1
            print()
            print(f"ERROR while processing: {job.recording_path}")
            print(f"{type(exc).__name__}: {exc}")
            print("Continuing with next recording.")

    print()
    print("=" * 80)
    print("Batch finished.")
    print(f"Total recordings found: {len(jobs)}")
    print(f"Failures:               {failures}")
    print("=" * 80)


if __name__ == "__main__":
    main()
