import json
import pytest
import logging
from pathlib import Path
from audio_separator.utils.cli import main
import subprocess
import importlib.metadata
from unittest import mock
from unittest.mock import patch, MagicMock, mock_open

from audio_separator.separator.exceptions import AudioExportError


# Mock metadata.distribution for tests to avoid PackageNotFoundError in environment without installed package
@pytest.fixture(autouse=True)
def mock_distribution():
    original_distribution = importlib.metadata.distribution

    def side_effect(package_name):
        if package_name == "audio-separator":
            mock_dist = MagicMock()
            mock_dist.version = "0.42.1"
            return mock_dist
        return original_distribution(package_name)

    with patch("importlib.metadata.distribution", side_effect=side_effect):
        yield


# Common fixture for expected arguments
@pytest.fixture
def common_expected_args():
    return {
        "log_formatter": mock.ANY,
        "log_level": logging.INFO,
        "model_file_dir": "/tmp/audio-separator-models/",
        "output_dir": None,
        "output_format": "FLAC",
        "output_bitrate": None,
        "normalization_threshold": 0.9,
        "amplification_threshold": 0.0,
        "output_single_stem": None,
        "invert_using_spec": False,
        "sample_rate": 44100,
        "use_soundfile": False,
        "use_autocast": False,
        "use_native_fp16": False,
        "use_torch_compile": False,
        "use_directml": False,
        "chunk_duration": None,
        "ensemble_algorithm": None,
        "ensemble_weights": None,
        "ensemble_preset": None,
        "mdx_params": {"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False},
        "vr_params": {"batch_size": 1, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
        "demucs_params": {"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True},
        "mdxc_params": {"segment_size": 256, "batch_size": None, "overlap": None, "override_model_segment_size": False, "pitch_shift": 0},
    }


# Test the CLI with version argument using subprocess
def test_cli_version_subprocess():
    # Skip subprocess CLI tests - require proper CLI installation
    pytest.skip("CLI subprocess tests require proper installation")


# Test the CLI with no arguments
def test_cli_no_args(capsys):
    """Print usage and fail before importing inference backends when no input is supplied."""
    test_args = ["cli.py"]

    with patch("sys.argv", test_args), patch.dict("sys.modules", {"audio_separator.separator": None}):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Separate audio file into different stems." in captured.out


def test_cli_demucs_segment_size_help(capsys):
    """Keep CLI help and README examples aligned on Demucs segment seconds and the Default sentinel."""
    with patch("sys.argv", ["cli.py", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 0
    help_output = capsys.readouterr().out
    assert "Use 'Default' to use the model's configured segment length" in help_output
    assert "Example: --demucs_segment_size=40" in help_output

    readme = Path(__file__).parents[2].joinpath("README.md").read_text(encoding="utf-8")
    assert "Use 'Default' to use the model's configured segment length" in readme
    assert "Example: --demucs_segment_size=40" in readme


# Test with multiple filename arguments
def test_cli_multiple_filenames():
    """Forward every input filename in one separation call and log the resulting outputs."""
    test_args = ["cli.py", "test1.mp3", "test2.mp3"]

    # Mock the open function to prevent actual file operations
    mock_file = mock_open()

    # Create a mock logger
    mock_logger = MagicMock()

    # Patch multiple functions to prevent actual file operations and separations
    with patch("sys.argv", test_args), patch("builtins.open", mock_file), patch("audio_separator.separator.Separator.separate") as mock_separate, patch(
        "audio_separator.separator.Separator.load_model"
    ), patch("logging.getLogger", return_value=mock_logger):

        # Mock the separate method to return some dummy output
        mock_separate.return_value = ["output_file1.mp3", "output_file2.mp3"]

        # Call the main function
        main()

        mock_separate.assert_called_once()
        args, kwargs = mock_separate.call_args
        assert args[0] == ["test1.mp3", "test2.mp3"]

        # Check if the logger captured information about both files
        log_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("test1.mp3" in msg and "test2.mp3" in msg for msg in log_messages)
        assert any("Separation complete" in msg for msg in log_messages)


def test_cli_exits_one_when_audio_export_fails():
    test_args = ["cli.py", "bad.wav"]
    export_error = AudioExportError("writer failed", path="bad_(Vocals).flac", backend="pydub")

    with patch("sys.argv", test_args), patch("audio_separator.separator.Separator") as separator_class:
        separator_class.return_value.separate.side_effect = export_error

        with pytest.raises(SystemExit) as raised:
            main()

    assert raised.value.code == 1


# Test the CLI with a specific audio file
def test_cli_with_audio_file(capsys, common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--model_filename=UVR-MDX-NET-Inst_HQ_4.onnx"]
    with patch("audio_separator.separator.Separator.separate") as mock_separate:
        mock_separate.return_value = ["output_file.mp3"]
        with patch("sys.argv", test_args):
            # Call the main function in cli.py
            main()

    # Update expected args for this specific test
    common_expected_args["model_file_dir"] = "/tmp/audio-separator-models/"

    # Check if the separate method was called with the correct arguments
    mock_separate.assert_called_once()

    # Assertions
    assert mock_separate.called


# Test the CLI with invalid log level
def test_cli_invalid_log_level():
    test_args = ["cli.py", "test_audio.mp3", "--log_level=invalid"]
    with patch("sys.argv", test_args):
        # Assert an attribute error is raised due to the invalid LogLevel
        with pytest.raises(AttributeError):
            # Call the main function in cli.py
            main()


# Test using model name argument
def test_cli_model_filename_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--model_filename=Custom_Model.onnx"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)
            mock_separator_instance.load_model.assert_called_once_with(model_filename="Custom_Model.onnx")


# Test using output directory argument
def test_cli_output_dir_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--output_dir=/custom/output/dir"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["output_dir"] = "/custom/output/dir"

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using output format argument
def test_cli_output_format_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--output_format=MP3"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["output_format"] = "MP3"

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using normalization_threshold argument
def test_cli_normalization_threshold_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--normalization=0.75"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["normalization_threshold"] = 0.75

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using amplification_threshold argument
def test_cli_amplification_threshold_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--amplification=0.75"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["amplification_threshold"] = 0.75

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using single stem argument
def test_cli_single_stem_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--single_stem=instrumental"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["output_single_stem"] = "instrumental"

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using invert spectrogram argument
def test_cli_invert_spectrogram_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--invert_spect"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["invert_using_spec"] = True

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using use_autocast argument
def test_cli_use_autocast_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--use_autocast"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["use_autocast"] = True

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


def test_cli_use_native_fp16_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--use_native_fp16"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            expected_args = common_expected_args.copy()
            expected_args["use_native_fp16"] = True

            mock_separator.assert_called_once_with(**expected_args)


def test_cli_rejects_multiple_precision_modes():
    test_args = ["cli.py", "test_audio.mp3", "--use_autocast", "--use_native_fp16"]

    with patch("sys.argv", test_args), pytest.raises(SystemExit, match="2"):
        main()


@pytest.mark.parametrize(
    ("precision_flag", "expected_key"),
    [
        ("--use_autocast", "use_autocast"),
        ("--use_native_fp16", "use_native_fp16"),
    ],
)
def test_cli_precision_mode_can_be_combined_with_torch_compile(
    common_expected_args,
    precision_flag,
    expected_key,
):
    test_args = ["cli.py", "test_audio.mp3", precision_flag, "--use_torch_compile"]

    with patch("sys.argv", test_args), patch("audio_separator.separator.Separator") as mock_separator:
        mock_separator.return_value.separate.return_value = ["output_file.mp3"]
        main()

    expected_args = common_expected_args.copy()
    expected_args[expected_key] = True
    expected_args["use_torch_compile"] = True
    mock_separator.assert_called_once_with(**expected_args)


def test_cli_rejects_multiple_precision_modes_even_with_torch_compile():
    test_args = [
        "cli.py",
        "test_audio.mp3",
        "--use_autocast",
        "--use_native_fp16",
        "--use_torch_compile",
    ]

    with patch("sys.argv", test_args), pytest.raises(SystemExit, match="2"):
        main()


def test_cli_use_torch_compile_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--use_torch_compile"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            expected_args = common_expected_args.copy()
            expected_args["use_torch_compile"] = True

            mock_separator.assert_called_once_with(**expected_args)


# Test using use_directml argument
def test_cli_use_directml_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--use_directml"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["use_directml"] = True

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using custom_output_names arguments
def test_cli_custom_output_names_argument(common_expected_args):
    custom_names = {
        "Vocals": "vocals_output",
        "Instrumental": "instrumental_output",
    }
    test_args = ["cli.py", "test_audio.mp3", f"--custom_output_names={json.dumps(custom_names)}"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)
            mock_separator_instance.separate.assert_called_once_with(["test_audio.mp3"], custom_output_names=custom_names)


# Test using custom_output_names arguments
def test_cli_demucs_output_names_argument(common_expected_args):
    demucs_output_names = {
        "Vocals": "vocals_output",
        "Drums": "drums_output",
        "Bass": "bass_output",
        "Other": "other_output",
        "Guitar": "guitar_output",
        "Piano": "piano_output"
    }
    test_args = ["cli.py", "test_audio.mp3", f"--custom_output_names={json.dumps(demucs_output_names)}", "--model_filename=htdemucs_6s.yaml"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)
            mock_separator_instance.separate.assert_called_once_with(["test_audio.mp3"], custom_output_names=demucs_output_names)


# Test using --extra_models for ensemble mode
def test_cli_extra_models_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "-m", "model1.onnx", "--extra_models", "model2.onnx", "model3.onnx"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)
            mock_separator_instance.load_model.assert_called_once_with(model_filename=["model1.onnx", "model2.onnx", "model3.onnx"])


# Test that -m with single model still passes a string (backward compat)
def test_cli_single_model_passes_string(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "-m", "my_model.onnx"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            mock_separator_instance.load_model.assert_called_once_with(model_filename="my_model.onnx")


# Test old CLI syntax: -m model audio.wav (model before audio file)
def test_cli_old_syntax_model_before_audio(common_expected_args):
    test_args = ["cli.py", "-m", "my_model.onnx", "test_audio.mp3"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            mock_separator_instance.load_model.assert_called_once_with(model_filename="my_model.onnx")
            mock_separator_instance.separate.assert_called_once_with(["test_audio.mp3"], custom_output_names=None)


# Test --ensemble_preset passes preset to Separator and calls load_model() with default
def test_cli_ensemble_preset(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--ensemble_preset", "vocal_balanced"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            expected_args = common_expected_args.copy()
            expected_args["ensemble_preset"] = "vocal_balanced"
            mock_separator.assert_called_once_with(**expected_args)
            # With preset and no explicit models, load_model() called with default
            mock_separator_instance.load_model.assert_called_once_with()


# Test --list_presets exits cleanly
def test_cli_list_presets(capsys):
    test_args = ["cli.py", "--list_presets"]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "vocal_balanced" in captured.out
    assert "karaoke" in captured.out
