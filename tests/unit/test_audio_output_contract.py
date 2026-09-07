"""Contract tests for validating and exporting separated audio."""

import importlib
import os
from pathlib import Path
import shutil
import stat
from unittest.mock import Mock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from pydub import AudioSegment

from audio_separator.separator.audio_io import normalize_output_subtype, resolve_output_subtype, validate_audio_source
from audio_separator.separator.architectures.demucs_separator import DemucsSeparator
from audio_separator.separator.common_separator import CommonSeparator
from audio_separator.separator.uvr_lib_v5 import spec_utils
from audio_separator.separator.exceptions import AudioExportError, BatchSeparationError, InvalidAudioDataError


requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for pydub encoding")


@pytest.mark.parametrize("subtype", ["AUTO", "PCM_16", "PCM_24", "PCM_32", "FLOAT"])
def test_wav_accepts_supported_output_subtypes(subtype):
    assert normalize_output_subtype(subtype.lower(), "WAV") == subtype


@pytest.mark.parametrize(
    ("subtype", "output_format", "message"),
    [
        ("PCM_24", "MP3", "WAV or FLAC"),
        ("PCM_32", "FLAC", "does not support"),
        ("FLOAT", "FLAC", "does not support"),
        ("PCM_20", "WAV", "must be one of"),
    ],
)
def test_invalid_output_subtype_combinations_fail_early(subtype, output_format, message):
    with pytest.raises(ValueError, match=message):
        normalize_output_subtype(subtype, output_format)


@pytest.mark.parametrize(("output_format", "expected"), [("wav", "PCM_24"), ("flac", "PCM_24")])
def test_auto_vorbis_falls_back_to_container_compatible_pcm(output_format, expected):
    assert resolve_output_subtype("AUTO", "VORBIS", 24, output_format) == expected


@pytest.fixture
def common_separator(tmp_path):
    config = {
        "logger": Mock(),
        "log_level": 20,
        "torch_device": Mock(),
        "torch_device_cpu": Mock(),
        "torch_device_mps": Mock(),
        "onnx_execution_provider": Mock(),
        "model_name": "test_model",
        "model_path": "/path/to/model",
        "model_data": {"training": {"instruments": ["vocals", "other"]}},
        "output_dir": str(tmp_path),
        "output_format": "wav",
        "output_bitrate": None,
        "normalization_threshold": 0.9,
        "amplification_threshold": 0.1,
        "enable_denoise": False,
        "output_single_stem": None,
        "invert_using_spec": False,
        "sample_rate": 44100,
        "use_soundfile": True,
    }
    return CommonSeparator(config)


def test_normalize_preserves_silence_when_amplification_is_enabled():
    wave = np.zeros((16, 2), dtype=np.float32)

    normalized = spec_utils.normalize(wave, max_peak=0.9, min_peak=0.1)

    np.testing.assert_array_equal(normalized, wave)
    assert np.isfinite(normalized).all()


@pytest.mark.parametrize(
    ("wave", "message"),
    [
        (np.array([], dtype=np.float32), "empty"),
        (np.array([0.0, np.nan], dtype=np.float32), "finite"),
        (np.array([-np.inf, np.inf], dtype=np.float32), "finite"),
    ],
)
def test_normalize_rejects_invalid_audio(wave, message):
    with pytest.raises(InvalidAudioDataError, match=message):
        spec_utils.normalize(wave)


def test_soundfile_writer_exports_silent_stereo_audio(common_separator, tmp_path):
    frames = 441

    common_separator.write_audio_soundfile("silent.wav", np.zeros((frames, 2), dtype=np.float32))

    output_path = tmp_path / "silent.wav"
    info = sf.info(output_path)
    assert output_path.stat().st_size > 0
    assert info.frames == frames
    assert info.channels == 2
    assert info.samplerate == common_separator.sample_rate


def test_prepare_mix_accepts_nonempty_silent_audio(common_separator, tmp_path):
    frames = 441
    input_path = tmp_path / "silent-input.wav"
    sf.write(input_path, np.zeros((frames, 2), dtype=np.float32), common_separator.sample_rate)

    mix = common_separator.prepare_mix(str(input_path))

    assert mix.shape == (2, frames)
    assert np.isfinite(mix).all()
    assert np.count_nonzero(mix) == 0


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (np.empty((0, 2), dtype=np.float32), "no audio frames"),
        (np.array([[0.0, np.nan]], dtype=np.float32), "non-finite samples"),
        (np.array([[np.inf, 0.0]], dtype=np.float32), "non-finite samples"),
    ],
)
def test_prepare_mix_rejects_invalid_numpy_audio(common_separator, audio, message):
    with pytest.raises(InvalidAudioDataError, match=message):
        common_separator.prepare_mix(audio)


def test_prepare_mix_accepts_nonempty_silent_numpy_audio(common_separator):
    frames = 32

    mix = common_separator.prepare_mix(np.zeros((frames, 2), dtype=np.float32))

    assert mix.shape == (2, frames)
    assert np.isfinite(mix).all()
    assert np.count_nonzero(mix) == 0


def test_demucs_receives_finite_normalized_silent_audio():
    separator = Mock(spec=DemucsSeparator)
    separator.logger = Mock()
    separator.demucs_model_instance = Mock()
    separator.demucs_model_instance.sources = ["drums", "bass", "other", "vocals"]
    separator.demucs_model_instance.models = []
    separator.shifts = 0
    separator.segments_enabled = True
    separator.overlap = 0.25
    separator.torch_device = torch.device("cpu")

    def apply_silent_model(*, mix, **_kwargs):
        assert torch.isfinite(mix).all()
        return torch.zeros((1, 4, 2, mix.shape[-1]), dtype=torch.float32)

    with patch("audio_separator.separator.architectures.demucs_separator.apply_model", side_effect=apply_silent_model):
        sources = DemucsSeparator.demix_demucs(separator, np.zeros((2, 32), dtype=np.float32))

    assert np.isfinite(sources).all()


@requires_ffmpeg
def test_pydub_writer_exports_silent_stereo_audio(common_separator, tmp_path):
    frames = 441

    common_separator.write_audio_pydub("silent-pydub.wav", np.zeros((frames, 2), dtype=np.float32))

    output_path = tmp_path / "silent-pydub.wav"
    info = sf.info(output_path)
    assert output_path.stat().st_size > 0
    assert info.frames == frames
    assert info.channels == 2
    assert info.samplerate == common_separator.sample_rate


@requires_ffmpeg
def test_pydub_writer_exports_redecodable_near_silent_m4a(common_separator, tmp_path):
    frames = 4410
    common_separator.amplification_threshold = 0.0
    near_silent = np.full((frames, 2), 5e-7, dtype=np.float32)

    common_separator.write_audio_pydub("near-silent.m4a", near_silent)

    output_path = tmp_path / "near-silent.m4a"
    decoded = AudioSegment.from_file(output_path)
    assert output_path.stat().st_size > 0
    assert decoded.channels == 2
    assert abs(len(decoded) - 100) <= 30


@requires_ffmpeg
def test_pydub_writer_preserves_mono_audio(common_separator, tmp_path):
    frames = 441

    common_separator.write_audio_pydub("mono.wav", np.zeros(frames, dtype=np.float32))

    info = sf.info(tmp_path / "mono.wav")
    assert info.frames == frames
    assert info.channels == 1


def test_audio_source_validation_rejects_invalid_channel_shape():
    with pytest.raises(InvalidAudioDataError, match="shape"):
        validate_audio_source(np.zeros((32, 3), dtype=np.float32))


def test_soundfile_failure_raises_audio_export_error_with_context(common_separator, tmp_path):
    backend_error = OSError("disk full")

    with patch("audio_separator.separator.common_separator.sf.write", side_effect=backend_error):
        with pytest.raises(AudioExportError) as raised:
            common_separator.write_audio_soundfile("broken.wav", np.zeros((32, 2), dtype=np.float32))

    assert raised.value.path == str(tmp_path / "broken.wav")
    assert raised.value.backend == "soundfile"
    assert raised.value.__cause__ is backend_error


def test_soundfile_failure_preserves_existing_target_and_removes_partial_file(common_separator, tmp_path):
    output_path = tmp_path / "existing.wav"
    output_path.write_bytes(b"original")

    def write_partial_then_fail(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    with patch("audio_separator.separator.common_separator.sf.write", side_effect=write_partial_then_fail):
        with pytest.raises(AudioExportError):
            common_separator.write_audio_soundfile(output_path.name, np.zeros((32, 2), dtype=np.float32))

    assert output_path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [output_path]


def test_pydub_failure_raises_audio_export_error_with_context(common_separator, tmp_path):
    backend_error = OSError("ffmpeg failed")

    with patch.object(AudioSegment, "export", side_effect=backend_error):
        with pytest.raises(AudioExportError) as raised:
            common_separator.write_audio_pydub("broken-pydub.wav", np.zeros((32, 2), dtype=np.float32))

    assert raised.value.path == str(tmp_path / "broken-pydub.wav")
    assert raised.value.backend == "pydub"
    assert raised.value.__cause__ is backend_error


def test_pydub_export_handle_is_closed_before_publishing(common_separator):
    export_handle = Mock()

    def export_audio(path, **_kwargs):
        Path(path).write_bytes(b"encoded audio")
        return export_handle

    with patch.object(AudioSegment, "export", side_effect=export_audio):
        common_separator.write_audio_pydub("closed-handle.wav", np.zeros((32, 2), dtype=np.float32))

    export_handle.close.assert_called_once_with()


def test_pydub_constructor_failure_raises_audio_export_error(common_separator, tmp_path):
    backend_error = ValueError("invalid raw audio")

    with patch("audio_separator.separator.common_separator.AudioSegment", side_effect=backend_error):
        with pytest.raises(AudioExportError) as raised:
            common_separator.write_audio_pydub("constructor.wav", np.zeros((32, 2), dtype=np.float32))

    assert raised.value.path == str(tmp_path / "constructor.wav")
    assert raised.value.backend == "pydub"
    assert raised.value.__cause__ is backend_error


def test_soundfile_directory_failure_raises_audio_export_error(common_separator, tmp_path):
    filesystem_error = OSError("permission denied")

    with patch("audio_separator.separator.common_separator.os.makedirs", side_effect=filesystem_error):
        with pytest.raises(AudioExportError) as raised:
            common_separator.write_audio_soundfile("directory.wav", np.zeros((32, 2), dtype=np.float32))

    assert raised.value.path == str(tmp_path / "directory.wav")
    assert raised.value.backend == "soundfile"
    assert raised.value.__cause__ is filesystem_error


def test_temporary_file_failure_raises_audio_export_error(common_separator, tmp_path):
    filesystem_error = OSError("too many open files")

    with patch("audio_separator.separator.audio_io.tempfile.mkstemp", side_effect=filesystem_error):
        with pytest.raises(AudioExportError) as raised:
            common_separator.write_audio_soundfile("temporary.wav", np.zeros((32, 2), dtype=np.float32))

    assert raised.value.path == str(tmp_path / "temporary.wav")
    assert raised.value.backend == "soundfile"
    assert raised.value.__cause__ is filesystem_error


def test_atomic_replace_failure_preserves_target_and_removes_temporary_file(common_separator, tmp_path):
    output_path = tmp_path / "replace.wav"
    output_path.write_bytes(b"original")
    filesystem_error = OSError("replace failed")

    with patch("audio_separator.separator.audio_io.os.replace", side_effect=filesystem_error):
        with pytest.raises(AudioExportError) as raised:
            common_separator.write_audio_soundfile(output_path.name, np.zeros((32, 2), dtype=np.float32))

    assert raised.value.path == str(output_path)
    assert raised.value.backend == "soundfile"
    assert raised.value.__cause__ is filesystem_error
    assert output_path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [output_path]


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not available on Windows")
def test_atomic_publish_preserves_existing_target_mode(common_separator, tmp_path):
    output_path = tmp_path / "existing-mode.wav"
    output_path.write_bytes(b"original")
    output_path.chmod(0o640)

    common_separator.write_audio_soundfile(output_path.name, np.zeros((32, 2), dtype=np.float32))

    assert stat.S_IMODE(output_path.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not available on Windows")
def test_atomic_publish_applies_umask_to_new_target(common_separator, tmp_path):
    previous_umask = os.umask(0o027)
    try:
        common_separator.write_audio_soundfile("new-mode.wav", np.zeros((32, 2), dtype=np.float32))
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((tmp_path / "new-mode.wav").stat().st_mode) == 0o640


def test_zero_byte_backend_output_is_rejected_and_removed(common_separator, tmp_path):
    with patch("audio_separator.separator.common_separator.sf.write", return_value=None):
        with pytest.raises(AudioExportError, match="empty output"):
            common_separator.write_audio_soundfile("zero-byte.wav", np.zeros((32, 2), dtype=np.float32))

    assert list(tmp_path.iterdir()) == []


def test_audio_output_errors_are_exported_from_separator_package_only():
    package = importlib.import_module("audio_separator.separator")

    assert package.InvalidAudioDataError is InvalidAudioDataError
    assert package.AudioExportError is AudioExportError
    assert package.BatchSeparationError is BatchSeparationError
    assert issubclass(package.InvalidAudioDataError, ValueError)
    assert issubclass(package.AudioExportError, RuntimeError)
    assert issubclass(package.BatchSeparationError, RuntimeError)
