"""
Unit tests for bit depth preservation in audio writing functions.

Tests that the write_audio functions preserve the input bit depth.
"""

import os
import pytest
import tempfile
import shutil
import soundfile as sf
import numpy as np
from unittest.mock import Mock

from audio_separator.separator.common_separator import CommonSeparator


# Check if FFmpeg is available (required for pydub tests)
def is_ffmpeg_available():
    """Check if FFmpeg is available in the system."""
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


FFMPEG_AVAILABLE = is_ffmpeg_available()
requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE,
    reason="FFmpeg not available (required for pydub tests)"
)


def create_test_audio_file(output_path, sample_rate=44100, duration=0.5, bit_depth=16):
    """Create a test audio file with a specific bit depth."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    stereo_audio = np.column_stack([audio, audio])
    
    if bit_depth == 16:
        subtype = 'PCM_16'
    elif bit_depth == 24:
        subtype = 'PCM_24'
    elif bit_depth == 32:
        subtype = 'PCM_32'
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    sf.write(output_path, stereo_audio, sample_rate, subtype=subtype)
    return output_path


def get_audio_bit_depth(file_path):
    """Get the bit depth of an audio file."""
    info = sf.info(file_path)
    subtype = info.subtype
    
    if 'PCM_16' in subtype or subtype == 'PCM_S8':
        return 16
    elif 'PCM_24' in subtype:
        return 24
    elif 'PCM_32' in subtype or 'FLOAT' in subtype or 'DOUBLE' in subtype:
        return 32
    else:
        return None


@pytest.fixture(name="temp_dir")
def fixture_temp_dir():
    """Fixture providing a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(name="mock_separator_config")
def fixture_mock_separator_config(temp_dir):
    """Fixture providing a mock separator configuration."""
    return {
        "logger": Mock(),
        "log_level": 20,
        "torch_device": Mock(),
        "torch_device_cpu": Mock(),
        "torch_device_mps": Mock(),
        "onnx_execution_provider": Mock(),
        "model_name": "test_model",
        "model_path": "/path/to/model",
        "model_data": {"training": {"instruments": ["vocals", "other"]}},
        "output_dir": temp_dir,
        "output_format": "wav",
        "output_bitrate": None,
        "normalization_threshold": 0.9,
        "amplification_threshold": 0.0,
        "enable_denoise": False,
        "output_single_stem": None,
        "invert_using_spec": False,
        "sample_rate": 44100,
        "use_soundfile": False,
    }


@requires_ffmpeg
def test_write_16bit_with_pydub(temp_dir, mock_separator_config):
    """Test that 16-bit audio is written correctly with pydub."""
    print("\n>>> TEST: Write 16-bit audio with pydub")
    
    # Create a 16-bit test input file
    input_file = os.path.join(temp_dir, "input_16bit.wav")
    create_test_audio_file(input_file, bit_depth=16)
    
    # Create CommonSeparator and prepare mix
    separator = CommonSeparator(mock_separator_config)
    separator.audio_file_path = input_file
    mix = separator.prepare_mix(input_file)
    
    print(f"Input bit depth detected: {separator.input_bit_depth}")
    
    # Create output audio data (simulated separation output)
    # The mix is in format [channels, samples], we need [samples, channels] for writing
    output_audio = mix.T
    
    # Write audio using pydub
    output_file = "test_output_16bit.wav"
    separator.write_audio_pydub(output_file, output_audio)
    
    # Check the output file bit depth
    full_output_path = os.path.join(temp_dir, output_file)
    assert os.path.exists(full_output_path), f"Output file not created: {full_output_path}"
    
    output_bit_depth = get_audio_bit_depth(full_output_path)
    print(f"Output bit depth: {output_bit_depth}")
    
    assert output_bit_depth == 16, f"Expected 16-bit output, got {output_bit_depth}"
    print("✅ Test passed: 16-bit audio written correctly with pydub")


@requires_ffmpeg
def test_write_24bit_with_pydub(temp_dir, mock_separator_config):
    """Test that 24-bit audio is written correctly with pydub."""
    print("\n>>> TEST: Write 24-bit audio with pydub")
    
    # Create a 24-bit test input file
    input_file = os.path.join(temp_dir, "input_24bit.wav")
    create_test_audio_file(input_file, bit_depth=24)
    
    # Create CommonSeparator and prepare mix
    separator = CommonSeparator(mock_separator_config)
    separator.audio_file_path = input_file
    mix = separator.prepare_mix(input_file)
    
    print(f"Input bit depth detected: {separator.input_bit_depth}")
    
    # Create output audio data
    output_audio = mix.T
    
    # Write audio using pydub
    output_file = "test_output_24bit.wav"
    separator.write_audio_pydub(output_file, output_audio)
    
    # Check the output file bit depth
    full_output_path = os.path.join(temp_dir, output_file)
    assert os.path.exists(full_output_path), f"Output file not created: {full_output_path}"
    
    output_bit_depth = get_audio_bit_depth(full_output_path)
    print(f"Output bit depth: {output_bit_depth}")
    
    assert output_bit_depth == 24, f"Expected 24-bit output, got {output_bit_depth}"
    print("✅ Test passed: 24-bit audio written correctly with pydub")


@requires_ffmpeg
def test_write_32bit_with_pydub(temp_dir, mock_separator_config):
    """Test that 32-bit audio is written correctly with pydub."""
    print("\n>>> TEST: Write 32-bit audio with pydub")
    
    # Create a 32-bit test input file
    input_file = os.path.join(temp_dir, "input_32bit.wav")
    create_test_audio_file(input_file, bit_depth=32)
    
    # Create CommonSeparator and prepare mix
    separator = CommonSeparator(mock_separator_config)
    separator.audio_file_path = input_file
    mix = separator.prepare_mix(input_file)
    
    print(f"Input bit depth detected: {separator.input_bit_depth}")
    
    # Create output audio data
    output_audio = mix.T
    
    # Write audio using pydub
    output_file = "test_output_32bit.wav"
    separator.write_audio_pydub(output_file, output_audio)
    
    # Check the output file bit depth
    full_output_path = os.path.join(temp_dir, output_file)
    assert os.path.exists(full_output_path), f"Output file not created: {full_output_path}"
    
    output_bit_depth = get_audio_bit_depth(full_output_path)
    print(f"Output bit depth: {output_bit_depth}")
    
    assert output_bit_depth == 32, f"Expected 32-bit output, got {output_bit_depth}"
    print("✅ Test passed: 32-bit audio written correctly with pydub")


def test_write_24bit_with_soundfile(temp_dir, mock_separator_config):
    """Test that 24-bit audio is written correctly with soundfile."""
    print("\n>>> TEST: Write 24-bit audio with soundfile")
    
    # Update config to use soundfile
    mock_separator_config["use_soundfile"] = True
    
    # Create a 24-bit test input file
    input_file = os.path.join(temp_dir, "input_24bit.wav")
    create_test_audio_file(input_file, bit_depth=24)
    
    # Create CommonSeparator and prepare mix
    separator = CommonSeparator(mock_separator_config)
    separator.audio_file_path = input_file
    mix = separator.prepare_mix(input_file)
    
    print(f"Input bit depth detected: {separator.input_bit_depth}")
    
    # Create output audio data
    output_audio = mix.T
    
    # Write audio using soundfile
    output_file = "test_output_24bit_sf.wav"
    separator.write_audio_soundfile(output_file, output_audio)
    
    # Check the output file bit depth
    full_output_path = os.path.join(temp_dir, output_file)
    assert os.path.exists(full_output_path), f"Output file not created: {full_output_path}"
    
    output_bit_depth = get_audio_bit_depth(full_output_path)
    print(f"Output bit depth: {output_bit_depth}")
    
    assert output_bit_depth == 24, f"Expected 24-bit output, got {output_bit_depth}"
    print("✅ Test passed: 24-bit audio written correctly with soundfile")


def test_write_16bit_with_soundfile(temp_dir, mock_separator_config):
    """Test that 16-bit audio is written correctly with soundfile."""
    print("\n>>> TEST: Write 16-bit audio with soundfile")
    
    # Update config to use soundfile
    mock_separator_config["use_soundfile"] = True
    
    # Create a 16-bit test input file
    input_file = os.path.join(temp_dir, "input_16bit.wav")
    create_test_audio_file(input_file, bit_depth=16)
    
    # Create CommonSeparator and prepare mix
    separator = CommonSeparator(mock_separator_config)
    separator.audio_file_path = input_file
    mix = separator.prepare_mix(input_file)
    
    print(f"Input bit depth detected: {separator.input_bit_depth}")
    
    # Create output audio data
    output_audio = mix.T
    
    # Write audio using soundfile
    output_file = "test_output_16bit_sf.wav"
    separator.write_audio_soundfile(output_file, output_audio)
    
    # Check the output file bit depth
    full_output_path = os.path.join(temp_dir, output_file)
    assert os.path.exists(full_output_path), f"Output file not created: {full_output_path}"
    
    output_bit_depth = get_audio_bit_depth(full_output_path)
    print(f"Output bit depth: {output_bit_depth}")
    
    assert output_bit_depth == 16, f"Expected 16-bit output, got {output_bit_depth}"
    print("✅ Test passed: 16-bit audio written correctly with soundfile")


def test_explicit_24bit_output_preserves_model_precision(temp_dir, mock_separator_config):
    """An explicit high-depth output must bypass pydub's int16 conversion."""
    mock_separator_config["output_subtype"] = "PCM_24"
    separator = CommonSeparator(mock_separator_config)
    separator.input_bit_depth = 16
    separator.input_subtype = "PCM_16"

    samples = np.array([[0.123456, -0.123456], [0.234567, -0.234567]], dtype=np.float32)
    separator.write_audio_pydub("explicit-24bit.wav", samples)

    output_path = os.path.join(temp_dir, "explicit-24bit.wav")
    assert sf.info(output_path).subtype == "PCM_24"
    decoded, _ = sf.read(output_path, dtype="float32")
    np.testing.assert_allclose(decoded, samples, atol=1 / 2**23)


@requires_ffmpeg
def test_explicit_16bit_output_overrides_24bit_input_metadata(temp_dir, mock_separator_config):
    """Honor an explicit PCM_16 export even when the input metadata reports PCM_24."""
    mock_separator_config["output_subtype"] = "PCM_16"
    separator = CommonSeparator(mock_separator_config)
    separator.input_bit_depth = 24
    separator.input_subtype = "PCM_24"

    samples = np.array([[0.25, -0.25], [0.5, -0.5]], dtype=np.float32)
    separator.write_audio_pydub("explicit-16bit.wav", samples)

    output_path = os.path.join(temp_dir, "explicit-16bit.wav")
    assert sf.info(output_path).subtype == "PCM_16"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
