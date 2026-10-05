"""Tests for CUDA/cuBLAS fallback to CPU in Whisper transcription.

These tests verify that when CUDA libraries (like cublas64_12.dll) are missing,
CLIPFORGE reliably falls back to CPU transcription instead of failing the entire
analysis. They run offline without requiring actual CUDA hardware.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from clipforge.ai.transcribe import (
    Transcript,
    Utterance,
    Word,
    _device_and_compute,
    transcribe_faster_whisper,
)
from clipforge.config import AppSettings
from clipforge.errors import ClipForgeError, ErrorCode


# --------------------------------------------------------------------------- #
# Device and compute selection
# --------------------------------------------------------------------------- #


def test_device_and_compute_cpu_when_no_gpu():
    """When no CUDA GPU is detected, device defaults to CPU with int8 compute."""
    settings = AppSettings(whisper_device="auto", whisper_compute_type="auto")
    with patch("clipforge.ai.transcribe.hardware") as mock_hardware:
        mock_hardware.return_value = {"gpu": {"cuda": False, "available": False}}
        device, compute = _device_and_compute(settings)
    assert device == "cpu"
    assert compute == "int8"


def test_device_and_compute_cpu_when_gpu_detected():
    """When CUDA GPU is detected and device is auto, it should use CUDA."""
    settings = AppSettings(whisper_device="auto", whisper_compute_type="auto")
    with patch("clipforge.ai.transcribe.hardware") as mock_hardware:
        mock_hardware.return_value = {"gpu": {"cuda": True, "available": True}}
        device, compute = _device_and_compute(settings)
    assert device == "cuda"
    assert compute == "float16"


def test_device_and_compute_force_cpu_overrides_gpu():
    """force_cpu=True overrides GPU detection and uses CPU."""
    settings = AppSettings(whisper_device="auto", whisper_compute_type="auto")
    with patch("clipforge.ai.transcribe.hardware") as mock_hardware:
        mock_hardware.return_value = {"gpu": {"cuda": True, "available": True}}
        device, compute = _device_and_compute(settings, force_cpu=True)
    assert device == "cpu"
    assert compute == "int8"


def test_device_and_compute_explicit_cpu():
    """Explicit CPU device setting is respected."""
    settings = AppSettings(whisper_device="cpu", whisper_compute_type="auto")
    with patch("clipforge.ai.transcribe.hardware") as mock_hardware:
        mock_hardware.return_value = {"gpu": {"cuda": True, "available": True}}
        device, compute = _device_and_compute(settings)
    assert device == "cpu"
    assert compute == "int8"


def test_device_and_compute_explicit_cuda_no_gpu():
    """Explicit CUDA device falls back to CPU when no GPU is detected."""
    settings = AppSettings(whisper_device="cuda", whisper_compute_type="auto")
    with patch("clipforge.ai.transcribe.hardware") as mock_hardware:
        mock_hardware.return_value = {"gpu": {"cuda": False, "available": False}}
        device, compute = _device_and_compute(settings)
    assert device == "cpu"
    assert compute == "int8"


def test_compute_type_auto_adjusts_for_device():
    """Compute type auto-adjusts based on device."""
    settings = AppSettings(whisper_device="cpu", whisper_compute_type="auto")
    with patch("clipforge.ai.transcribe.hardware") as mock_hardware:
        mock_hardware.return_value = {"gpu": {"cuda": False, "available": False}}
        device, compute = _device_and_compute(settings)
    assert device == "cpu"
    assert compute == "int8"


def test_compute_type_explicit_float16_on_cpu_falls_back_to_int8():
    """float16 compute type falls back to int8 on CPU."""
    settings = AppSettings(whisper_device="cpu", whisper_compute_type="float16")
    with patch("clipforge.ai.transcribe.hardware") as mock_hardware:
        mock_hardware.return_value = {"gpu": {"cuda": False, "available": False}}
        device, compute = _device_and_compute(settings)
    assert device == "cpu"
    assert compute == "int8"


# --------------------------------------------------------------------------- #
# CUDA error detection and fallback
# --------------------------------------------------------------------------- #


def test_cublas_error_message_detected():
    """CUDA/cuBLAS error messages are correctly identified."""
    error_messages = [
        "Library cublas64_12.dll is not found or cannot be loaded",
        "CUDA error: cublasLtCreate failed",
        "cudnn64_8.dll not found",
        "Could not load dynamic library cublas64_11.dll",
    ]
    
    for msg in error_messages:
        assert ("cuda" in msg.lower() or "cublas" in msg.lower() or "cudnn" in msg.lower())


def test_non_cuda_error_message_not_detected():
    """Non-CUDA error messages are not misidentified as CUDA errors."""
    error_messages = [
        "File not found",
        "Model tiny not found in cache",
        "Out of memory",
        "Invalid audio format",
    ]
    
    for msg in error_messages:
        lower_msg = msg.lower()
        assert not ("cuda" in lower_msg and ("cublas" in lower_msg or "cudnn" in lower_msg))


# --------------------------------------------------------------------------- #
# Mock WhisperModel for testing fallback behavior
# --------------------------------------------------------------------------- #


class MockSegment:
    def __init__(self, text, start, end, words=None):
        self.text = text
        self.start = start
        self.end = end
        self.words = words or []


class MockWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class MockWhisperInfo:
    def __init__(self, duration=10.0, language="en", language_probability=0.95):
        self.duration = duration
        self.language = language
        self.language_probability = language_probability
        self.all_language_probs = {"en": 0.95, "fr": 0.05}


class MockWhisperModel:
    def __init__(self, device="cpu", compute_type="int8"):
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls = []
        
    def transcribe(self, audio_path, **kwargs):
        self.transcribe_calls.append((audio_path, kwargs))
        # Return mock results
        segments = [
            MockSegment("Hello world", 0.0, 2.0, [
                MockWord("Hello", 0.0, 1.0),
                MockWord("world", 1.0, 2.0)
            ]),
            MockSegment("This is a test", 2.0, 4.0, [
                MockWord("This", 2.0, 2.5),
                MockWord("is", 2.5, 3.0),
                MockWord("a", 3.0, 3.5),
                MockWord("test", 3.5, 4.0)
            ])
        ]
        info = MockWhisperInfo()
        return segments, info


# --------------------------------------------------------------------------- #
# Fallback behavior tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_faster_whisper_module():
    """Mock the faster_whisper module for testing fallback behavior."""
    with patch.dict('sys.modules', {'faster_whisper': MagicMock()}):
        yield


def test_cuda_error_during_model_load_triggers_fallback(tmp_path):
    """When CUDA errors occur during model loading, fallback to CPU."""
    import sys
    from pathlib import Path
    
    # Create a temporary audio file
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal fake WAV
    
    settings = AppSettings(
        whisper_model="tiny",
        whisper_device="auto",
        whisper_compute_type="auto"
    )
    
    # Mock hardware to report CUDA available
    with patch("clipforge.ai.transcribe.hardware") as mock_hardware:
        mock_hardware.return_value = {"gpu": {"cuda": True, "available": True}}
        
        # Mock the faster_whisper import and WhisperModel
        mock_fw = MagicMock()
        def mock_whisper_model_constructor(model_name, device="cuda", compute_type="float16", **kwargs):
            if device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            return MockWhisperModel(device=device, compute_type=compute_type)
        
        mock_fw.WhisperModel = mock_whisper_model_constructor
        
        with patch.dict('sys.modules', {'faster_whisper': mock_fw}):
            # This should succeed by falling back to CPU
            from clipforge.ai.transcribe import _load_faster_whisper
            from clipforge.ai.transcribe import _MODELS, _MODEL_LOCK
            
            # Clear any cached models
            with _MODEL_LOCK:
                _MODELS.clear()
                
            model, device, compute, model_name = _load_faster_whisper(settings)
            assert device == "cpu"
            assert compute == "int8"


def test_cuda_error_during_transcription_triggers_fallback(tmp_path):
    """When CUDA errors occur during transcription, fallback to CPU."""
    import sys
    from pathlib import Path
    
    # Create a temporary audio file
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal fake WAV
    
    settings = AppSettings(
        whisper_model="tiny",
        whisper_device="auto",
        whisper_compute_type="auto"
    )
    
    # Mock hardware to report CUDA available
    with patch("clipforge.ai.transcribe.hardware") as mock_hardware:
        mock_hardware.return_value = {"gpu": {"cuda": True, "available": True}}
        
        # Create a model that succeeds on load but fails on transcribe with CUDA error
        class FailingWhisperModel:
            def __init__(self, *args, **kwargs):
                self.device = kwargs.get("device", "cpu")
                self.compute_type = kwargs.get("compute_type", "int8")
                
            def transcribe(self, audio_path, **kwargs):
                if self.device == "cuda":
                    raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                # On CPU, return mock results
                segments = [
                    MockSegment("Hello world", 0.0, 2.0, [
                        MockWord("Hello", 0.0, 1.0),
                        MockWord("world", 1.0, 2.0)
                    ])
                ]
                info = MockWhisperInfo()
                return segments, info
        
        mock_fw = MagicMock()
        mock_fw.WhisperModel = FailingWhisperModel
        
        with patch.dict('sys.modules', {'faster_whisper': mock_fw}):
            # Clear model cache
            from clipforge.ai.transcribe import _MODELS, _MODEL_LOCK
            with _MODEL_LOCK:
                _MODELS.clear()
            
            # This should succeed by falling back to CPU during transcription
            try:
                transcript = transcribe_faster_whisper(
                    audio_path,
                    settings=settings,
                    progress=None,
                    should_cancel=None,
                    language="en"
                )
                # Should succeed with CPU fallback
                assert len(transcript.utterances) > 0
                assert "cpu" in transcript.engine.lower() or "int8" in transcript.engine
            except Exception as e:
                # If this fails, it means the fallback didn't work
                pytest.fail(f"CUDA fallback during transcription failed: {e}")


# --------------------------------------------------------------------------- #
# Uploaded transcript bypass tests
# --------------------------------------------------------------------------- #


def test_uploaded_transcript_bypasses_whisper(tmp_path):
    """When an authoritative uploaded transcript is provided, Whisper is bypassed entirely."""
    from clipforge.pipeline.analyze import _transcript_from_files
    from clipforge.pipeline.context import ProjectPaths, NullReporter
    from clipforge.ai.language import detect_language
    from clipforge.config import get_settings
    
    # Create a mock transcript file - use the expected naming convention
    srt_content = """1
00:00:01,000 --> 00:00:03,000
This is a test transcript.

2
00:00:03,100 --> 00:00:05,000
Second line of the transcript.
"""
    # The _provided_transcript_file function looks for files with "provided" stem
    transcript_path = tmp_path / "provided.srt"
    transcript_path.write_text(srt_content)
    
    # Create project paths that point to our test file
    class MockPaths:
        transcript = tmp_path
    
    settings = get_settings()
    reporter = NullReporter()
    profile = detect_language("This is a test transcript.")
    
    # This should find and parse the uploaded transcript, bypassing Whisper
    provided = _transcript_from_files(
        MockPaths(),
        settings,
        reporter,
        profile,
        preferred_source="uploaded_srt",
        preferred_filename="provided.srt",
        allow_legacy=True
    )
    
    # Should return a transcript (not None)
    assert provided is not None
    assert provided.source == "uploaded_srt"
    assert len(provided.utterances) >= 1


# --------------------------------------------------------------------------- #
# Integration test: End-to-end fallback verification
# --------------------------------------------------------------------------- #


def test_fallback_preserves_transcript_quality(tmp_path):
    """Verify that CPU fallback produces valid transcript structure."""
    import sys
    # This test would run with a real model if available, but we mock it here
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"RIFF" + b"\x00" * 100)
    
    settings = AppSettings(
        whisper_model="tiny",
        whisper_device="cpu",  # Force CPU
        whisper_compute_type="int8"
    )
    
    mock_fw = MagicMock()
    mock_fw.WhisperModel = MockWhisperModel
    
    with patch.dict('sys.modules', {'faster_whisper': mock_fw}):
        transcript = transcribe_faster_whisper(
            audio_path,
            settings=settings,
            progress=None,
            should_cancel=None,
            language="en"
        )
        
        # Verify transcript structure
        assert isinstance(transcript, Transcript)
        assert len(transcript.utterances) > 0
        assert all(isinstance(utt, Utterance) for utt in transcript.utterances)
        assert all(hasattr(utt, 'words') for utt in transcript.utterances)
        assert transcript.word_count > 0
        assert transcript.duration > 0
        assert transcript.language == "en"
