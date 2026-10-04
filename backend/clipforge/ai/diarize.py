"""Speaker detection.

Two real implementations:

* **whisperx** (optional) - pyannote based diarization, used when installed and
  configured.
* **energy/spectral clustering** (default, always available) - utterance level
  acoustic features (level, zero-crossing rate, spectral centroid/rolloff and a
  filter-bank cepstral summary) clustered with k-means; ``k`` is chosen by
  silhouette score so a solo video stays a single speaker instead of being
  chopped into fake speakers.

The result is a speaker label on every word, which the pipeline then uses for
reframing and for "who is talking" aware caption/zoom decisions.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..logging_setup import get_logger
from .transcribe import Utterance, Word

log = get_logger("clipforge.ai")

SAMPLE_RATE = 16000
FRAME_MS = 25
HOP_MS = 10


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #


def read_wav_mono(path: str | Path, *, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file into a float32 mono array in [-1, 1]."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        if max_seconds:
            frames = min(frames, int(max_seconds * rate))
        raw = handle.readframes(frames)

    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        data = np.zeros(0, dtype=np.float32)

    if channels > 1 and data.size:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and data.size:
        data = resample_linear(data, rate, SAMPLE_RATE)
        rate = SAMPLE_RATE
    return data.astype(np.float32, copy=False), rate


def resample_linear(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or data.size == 0:
        return data
    duration = data.size / float(source_rate)
    target_length = max(1, int(duration * target_rate))
    source_positions = np.linspace(0.0, data.size - 1, num=target_length)
    return np.interp(source_positions, np.arange(data.size), data).astype(np.float32)


# --------------------------------------------------------------------------- #
# Acoustic features
# --------------------------------------------------------------------------- #


@dataclass
class UtteranceFeatures:
    start: float
    end: float
    vector: np.ndarray
    energy: float


def _hann(window: int) -> np.ndarray:
    return np.hanning(window).astype(np.float32)


def frame_signal(samples: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if samples.size < frame_len:
        padded = np.pad(samples, (0, frame_len - samples.size))
        return padded.reshape(1, -1)
    count = 1 + (samples.size - frame_len) // hop
    indices = np.arange(frame_len)[None, :] + hop * np.arange(count)[:, None]
    return samples[indices]


def mel_filterbank(n_filters: int, n_fft: int, sample_rate: int, low_hz: float = 60.0, high_hz: float = 7600.0) -> np.ndarray:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), n_filters + 2)
    hz_points = np.array([mel_to_hz(point) for point in mel_points])
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for index in range(1, n_filters + 1):
        left, center, right = bins[index - 1], bins[index], bins[index + 1]
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        for bin_index in range(left, min(center, bank.shape[1])):
            bank[index - 1, bin_index] = (bin_index - left) / max(center - left, 1)
        for bin_index in range(center, min(right, bank.shape[1])):
            bank[index - 1, bin_index] = (right - bin_index) / max(right - center, 1)
    return bank


def estimate_pitch(samples: np.ndarray, sample_rate: int = SAMPLE_RATE, fmin: float = 70.0, fmax: float = 400.0) -> float:
    """Autocorrelation pitch estimate (median F0 proxy for speaker identity)."""
    if samples.size < sample_rate // 40:
        return 0.0
    windowed = samples - float(np.mean(samples))
    energy = float(np.dot(windowed, windowed))
    if energy <= 1e-8:
        return 0.0
    min_lag = int(sample_rate / fmax)
    max_lag = min(int(sample_rate / fmin), windowed.size - 1)
    if max_lag <= min_lag:
        return 0.0
    correlation = np.correlate(windowed, windowed, mode="full")[windowed.size - 1:]
    segment = correlation[min_lag:max_lag]
    if segment.size == 0:
        return 0.0
    peak = int(np.argmax(segment)) + min_lag
    if correlation[peak] < 0.3 * correlation[0]:
        return 0.0
    return sample_rate / peak


def utterance_features(samples: np.ndarray, sample_rate: int, start: float, end: float) -> UtteranceFeatures | None:
    """Compute a compact acoustic fingerprint for one utterance."""
    first = max(0, int(start * sample_rate))
    last = min(samples.size, int(end * sample_rate))
    if last - first < int(0.12 * sample_rate):
        return None
    segment = samples[first:last]

    frame_len = max(16, int(sample_rate * FRAME_MS / 1000))
    hop = max(8, int(sample_rate * HOP_MS / 1000))
    frames = frame_signal(segment, frame_len, hop)
    window = _hann(frame_len)
    frames = frames * window[None, :]

    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    log_rms = np.log(rms + 1e-8)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

    spectrum = np.abs(np.fft.rfft(frames, n=frame_len))
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sample_rate)
    power = spectrum**2 + 1e-10
    centroid = float(np.mean((power * freqs[None, :]).sum(axis=1) / power.sum(axis=1)))
    cumulative = np.cumsum(power, axis=1)
    total = cumulative[:, -1:]
    rolloff_bins = np.argmax(cumulative >= 0.85 * total, axis=1)
    rolloff = float(np.mean(freqs[np.clip(rolloff_bins, 0, len(freqs) - 1)]))

    bank = mel_filterbank(13, frame_len, sample_rate)
    mel_energy = np.log(spectrum @ bank.T + 1e-8)
    cepstral = np.mean(np.log(np.abs(np.fft.rfft(mel_energy, axis=1)) + 1e-8)[:, :8], axis=0)

    pitch = estimate_pitch(segment, sample_rate)

    vector = np.concatenate(
        [
            np.array(
                [
                    float(np.mean(log_rms)),
                    float(np.std(log_rms)),
                    float(np.mean(zcr)),
                    math.log1p(centroid / 1000.0),
                    math.log1p(rolloff / 1000.0),
                    math.log1p(pitch / 100.0),
                ],
                dtype=np.float64,
            ),
            cepstral.astype(np.float64),
        ]
    )
    return UtteranceFeatures(start=start, end=end, vector=vector, energy=float(np.mean(rms)))


# --------------------------------------------------------------------------- #
# Clustering (k-means++ with silhouette model selection)
# --------------------------------------------------------------------------- #


def _kmeans(matrix: np.ndarray, k: int, *, iterations: int = 60, restarts: int = 8, seed: int = 1234) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_inertia = float("inf")
    n_samples = matrix.shape[0]
    if n_samples == 0:
        return np.zeros(0, dtype=int), 0.0
    k = max(1, min(k, n_samples))

    for _ in range(restarts):
        # k-means++ initialisation.
        centres = [matrix[rng.integers(n_samples)]]
        for _ in range(1, k):
            distances = np.min(
                np.stack([np.sum((matrix - centre) ** 2, axis=1) for centre in centres], axis=1),
                axis=1,
            )
            total = float(distances.sum())
            probabilities = distances / total if total > 0 else np.full(n_samples, 1.0 / n_samples)
            centres.append(matrix[rng.choice(n_samples, p=probabilities)])
        centres_array = np.stack(centres)

        labels = np.zeros(n_samples, dtype=int)
        for _ in range(iterations):
            distances = np.stack([np.sum((matrix - centre) ** 2, axis=1) for centre in centres_array], axis=1)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for index in range(k):
                members = matrix[labels == index]
                if members.size:
                    centres_array[index] = members.mean(axis=0)

        inertia = float(
            np.sum(np.min(np.stack([np.sum((matrix - centre) ** 2, axis=1) for centre in centres_array], axis=1), axis=1))
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return (best_labels if best_labels is not None else np.zeros(n_samples, dtype=int)), best_inertia


def _silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    n_samples = matrix.shape[0]
    unique = np.unique(labels)
    if n_samples < 3 or unique.size < 2:
        return -1.0
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    scores = np.zeros(n_samples)
    for index in range(n_samples):
        own = labels == labels[index]
        own_count = max(int(own.sum()) - 1, 1)
        a = distances[index][own].sum() / own_count if own_count else 0.0
        b = float("inf")
        for other in unique:
            if other == labels[index]:
                continue
            mask = labels == other
            if mask.any():
                b = min(b, float(distances[index][mask].mean()))
        scores[index] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(scores.mean())


def cluster_speakers(
    features: list[UtteranceFeatures],
    *,
    max_speakers: int = 6,
    min_silhouette: float = 0.18,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster utterances into speakers, refusing to invent speakers when unsure."""
    if not features:
        return np.zeros(0, dtype=int), {"speakers": 0, "silhouette": 0.0, "mode": "energy"}
    matrix = np.stack([feature.vector for feature in features])
    if matrix.shape[0] < 4:
        return np.zeros(matrix.shape[0], dtype=int), {"speakers": 1, "silhouette": 0.0, "mode": "energy", "reason": "too_few_utterances"}

    # Standardise so no single feature dominates the distance metric.
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    normalised = (matrix - mean) / std

    best: tuple[float, int, np.ndarray] | None = None
    for k in range(2, min(max_speakers, normalised.shape[0] - 1) + 1):
        labels, _ = _kmeans(normalised, k)
        counts = np.bincount(labels, minlength=k)
        if counts.min() < max(2, int(0.05 * len(features))):  # refuse tiny phantom clusters
            continue
        score = _silhouette(normalised, labels)
        if best is None or score > best[0]:
            best = (score, k, labels)

    if best is None or best[0] < min_silhouette:
        return np.zeros(normalised.shape[0], dtype=int), {
            "speakers": 1,
            "silhouette": round(best[0], 3) if best else 0.0,
            "mode": "energy",
            "reason": "single_speaker" if best is None else "low_separation",
        }

    score, k, labels = best
    log.info("diarization: %d speakers (silhouette %.2f)", k, score)
    return labels, {"speakers": int(k), "silhouette": round(score, 3), "mode": "energy"}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def diarize(
    audio_path: str | Path,
    utterances: Sequence[Utterance],
    *,
    max_speakers: int = 6,
    mode: str = "energy",
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[Utterance], dict[str, Any]]:
    """Assign speaker labels to every word. Returns ``(utterances, report)``."""
    if not utterances:
        return list(utterances), {"speakers": 0, "mode": mode}

    if mode in {"auto", "whisperx"}:
        result = _try_whisperx_diarization(audio_path, utterances, max_speakers)
        if result is not None:
            return result
        if mode == "whisperx":
            log.warning("WhisperX diarization unavailable; using acoustic clustering instead")

    if mode == "off":
        for utterance in utterances:
            utterance.speaker = "SPEAKER_01"
            for word in utterance.words:
                word.speaker = "SPEAKER_01"
        return list(utterances), {"speakers": 1, "mode": "off"}

    if progress:
        progress(0.1, "reading audio for speaker analysis")
    try:
        samples, rate = read_wav_mono(audio_path)
    except (OSError, wave.Error) as exc:
        log.warning("cannot read audio for diarization: %s", exc)
        for utterance in utterances:
            utterance.speaker = "SPEAKER_01"
        return list(utterances), {"speakers": 1, "mode": "unavailable"}

    if samples.size == 0:
        return list(utterances), {"speakers": 1, "mode": "empty"}

    # Group words into acoustically coherent utterances for clustering.
    blocks = _group_blocks(utterances)
    features: list[UtteranceFeatures] = []
    block_refs: list[tuple[int, int]] = []
    for start, end in blocks:
        feature = utterance_features(samples, rate, start, end)
        block_refs.append((start, end))
        if feature is not None:
            features.append(feature)

    if not features:
        return list(utterances), {"speakers": 1, "mode": "energy"}

    if progress:
        progress(0.5, f"clustering {len(features)} speech segments")
    labels, report = cluster_speakers(features, max_speakers=max_speakers)

    # Map cluster ids to stable SPEAKER_XX labels ordered by total speaking time.
    durations: dict[int, float] = {}
    for label, (start, end) in zip(labels, block_refs):
        durations[int(label)] = durations.get(int(label), 0.0) + (end - start)
    ordered = [cluster for cluster, _ in sorted(durations.items(), key=lambda item: -item[1])]
    mapping = {cluster: f"SPEAKER_{index + 1:02d}" for index, cluster in enumerate(ordered)}

    by_start = {start: int(label) for label, (start, _) in zip(labels, block_refs)}
    for utterance in utterances:
        cluster = _nearest_block_label(utterance.start, by_start)
        label = mapping.get(cluster, "SPEAKER_01")
        utterance.speaker = label
        for word in utterance.words:
            word.speaker = label

    report = {**report, "speakers": max(len(mapping), 1), "utterances": len(features)}
    if progress:
        progress(1.0, f"detected {report['speakers']} speaker(s)")
    return list(utterances), report


def _nearest_block_label(start: float, by_start: dict[float, int]) -> int:
    if not by_start:
        return 0
    nearest = min(by_start, key=lambda block_start: abs(block_start - start))
    return by_start[nearest]


def _group_blocks(utterances: Sequence[Utterance], *, gap: float = 0.45, max_block: float = 12.0) -> list[tuple[float, float]]:
    """Merge neighbouring utterances into blocks of continuous speech."""
    blocks: list[tuple[float, float]] = []
    for utterance in utterances:
        if not utterance.words and utterance.end <= utterance.start:
            continue
        start, end = utterance.start, utterance.end
        if blocks and start - blocks[-1][1] <= gap and (blocks[-1][1] - blocks[-1][0] + (end - start)) <= max_block:
            blocks[-1] = (blocks[-1][0], end)
        else:
            blocks.append((start, end))
    return blocks


def _try_whisperx_diarization(
    audio_path: str | Path,
    utterances: Sequence[Utterance],
    max_speakers: int,
) -> tuple[list[Utterance], dict[str, Any]] | None:
    try:
        import whisperx  # type: ignore
    except ImportError:
        return None
    try:
        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = whisperx.DiarizationPipeline(use_auth_token=None, device=device)
        diarize_segments = pipeline(str(audio_path), max_speakers=max_speakers if max_speakers > 1 else None)
    except Exception as exc:  # noqa: BLE001
        log.info("WhisperX diarization unavailable (%s)", exc)
        return None

    spans: list[tuple[float, float, str]] = []
    for _, row in diarize_segments.iterrows():
        spans.append((float(row["start"]), float(row["end"]), str(row["speaker"])))

    speakers = sorted({span[2] for span in spans})
    mapping = {speaker: f"SPEAKER_{index + 1:02d}" for index, speaker in enumerate(speakers)}
    for utterance in utterances:
        best_speaker, best_overlap = "SPEAKER_01", 0.0
        for start, end, speaker in spans:
            overlap = min(utterance.end, end) - max(utterance.start, start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, mapping[speaker]
        utterance.speaker = best_speaker
        for word in utterance.words:
            word.speaker = best_speaker
    return list(utterances), {"speakers": max(len(mapping), 1), "mode": "whisperx"}


def speaker_turn_map(utterances: Sequence[Utterance]) -> list[dict[str, Any]]:
    """Collapse consecutive same-speaker utterances into turns (for reframing)."""
    turns: list[dict[str, Any]] = []
    for utterance in utterances:
        if turns and turns[-1]["speaker"] == utterance.speaker and utterance.start - turns[-1]["end"] <= 0.8:
            turns[-1]["end"] = utterance.end
            turns[-1]["utterances"] += 1
        else:
            turns.append(
                {
                    "speaker": utterance.speaker,
                    "start": utterance.start,
                    "end": utterance.end,
                    "utterances": 1,
                }
            )
    return turns


__all__ = [
    "UtteranceFeatures",
    "cluster_speakers",
    "diarize",
    "estimate_pitch",
    "read_wav_mono",
    "speaker_turn_map",
    "utterance_features",
]
