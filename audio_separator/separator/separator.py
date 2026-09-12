""" This file contains the Separator class, to facilitate the separation of stems from audio. """

from importlib import metadata, resources
import os
import sys
import platform
import subprocess
import time
import logging
import warnings
import importlib
import io
import re
import librosa
import numpy as np
from typing import Optional

import hashlib
import json
import yaml
import requests
import torch
import torch.amp.autocast_mode as autocast_mode
import onnxruntime as ort
from tqdm import tqdm
from audio_separator.separator.audio_io import atomic_output_path, validate_audio_source
from audio_separator.separator.ensembler import Ensembler
from audio_separator.separator.exceptions import AudioExportError, BatchSeparationError, InvalidAudioDataError
from audio_separator.separator.execution_policy import AUTOCAST, FP32, NATIVE_FP16

# Mapping of common stem name variations to canonical names for ensemble grouping.
STEM_NAME_MAP = {
    "vocals": "Vocals",
    "instrumental": "Instrumental",
    "inst": "Instrumental",
    "karaoke": "Instrumental",
    "other": "Other",
    "no_vocals": "Instrumental",
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "piano": "Piano",
    "synthesizer": "Synthesizer",
    "strings": "Strings",
    "woodwinds": "Woodwinds",
    "brass": "Brass",
    "wind inst": "Wind Inst",
    "lead vocals": "Lead Vocals",
    "backing vocals": "Backing Vocals",
    "primary stem": "Primary Stem",
    "secondary stem": "Secondary Stem",
}

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3")


def _validate_cuda_device_index(index):
    """Reject explicit CUDA indices unless they are non-negative integers; allow None for automatic selection."""
    if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
        raise ValueError("cuda_device_index must be a non-negative integer or None")


def _iter_directory_audio_files(directory):
    """Yield supported audio paths recursively, matching filename extensions case-insensitively."""
    for root, _dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
                yield os.path.join(root, filename)


class Separator:
    """
    The Separator class is designed to facilitate the separation of audio sources from a given audio file.
    It supports various separation architectures and models, including MDX, VR, and Demucs. The class provides
    functionalities to configure separation parameters, load models, and perform audio source separation.
    It also handles logging, normalization, and output formatting of the separated audio stems.

    The actual separation task is handled by one of the architecture-specific classes in the `architectures` module;
    this class is responsible for initialising logging, configuring hardware acceleration, loading the model,
    initiating the separation process and passing outputs back to the caller.

    Common Attributes:
        log_level (int): The logging level.
        log_formatter (logging.Formatter): The logging formatter.
        model_file_dir (str): The directory where model files are stored.
        output_dir (str): The directory where output files will be saved.
        output_format (str): The format of the output audio file.
        output_bitrate (str): The bitrate of the output audio file.
        amplification_threshold (float): The threshold for audio amplification.
        normalization_threshold (float): The threshold for audio normalization.
        output_single_stem (str): Option to output a single stem.
        invert_using_spec (bool): Flag to invert using spectrogram.
        sample_rate (int): The sample rate of the audio.
        cuda_device_index (int or None): Optional visible CUDA index used by PyTorch and ONNX Runtime.
        use_soundfile (bool): Use soundfile for audio writing, can solve OOM issues.
        use_autocast (bool): Use PyTorch autocast when the loaded model and device support it.
        use_torch_compile (bool): Compile verified repeated model blocks when supported.
        use_native_fp16 (bool): Convert a verified model to native float16 inference when supported.

    MDX Architecture Specific Attributes:
        hop_length (int): The hop length for STFT.
        segment_size (int): The segment size for processing.
        overlap (float): The overlap between segments.
        batch_size (int): The batch size for processing.
        enable_denoise (bool): Flag to enable or disable denoising.

    VR Architecture Specific Attributes & Defaults:
        batch_size: 16
        window_size: 512
        aggression: 5
        enable_tta: False
        enable_post_process: False
        post_process_threshold: 0.2
        high_end_process: False

    Demucs Architecture Specific Attributes & Defaults:
        segment_size: "Default"
        shifts: 2
        overlap: 0.25
        segments_enabled: True

    MDXC Architecture Specific Attributes & Defaults:
        segment_size: 256
        override_model_segment_size: False
        batch_size: None (uses inference.batch_size from the model YAML, falling back to 1)
        overlap: None (uses inference.num_overlap from the model YAML, falling back to 8)
        pitch_shift: 0
    """

    def __init__(
        self,
        log_level=logging.INFO,
        log_formatter=None,
        model_file_dir="/tmp/audio-separator-models/",
        output_dir=None,
        output_format="WAV",
        output_bitrate=None,
        normalization_threshold=0.9,
        amplification_threshold=0.0,
        output_single_stem=None,
        invert_using_spec=False,
        sample_rate=44100,
        use_soundfile=False,
        use_autocast=False,
        use_directml=False,
        chunk_duration=None,
        mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False},
        vr_params={"batch_size": 1, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
        demucs_params={"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True},
        mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": None, "overlap": None, "pitch_shift": 0},
        ensemble_algorithm=None,
        ensemble_weights=None,
        ensemble_preset=None,
        info_only=False,
        use_torch_compile=False,
        use_native_fp16=False,
        cuda_device_index=None,
    ):
        """Initialize the separator."""
        _validate_cuda_device_index(cuda_device_index)
        self.cuda_device_index = cuda_device_index
        if use_autocast and use_native_fp16:
            raise ValueError("use_autocast and use_native_fp16 are mutually exclusive precision modes.")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.log_level = log_level
        self.log_formatter = log_formatter

        self.log_handler = logging.StreamHandler()

        if self.log_formatter is None:
            self.log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

        self.log_handler.setFormatter(self.log_formatter)

        if not self.logger.hasHandlers():
            self.logger.addHandler(self.log_handler)

        # Filter out noisy warnings from PyTorch for users who don't care about them
        if log_level > logging.DEBUG:
            warnings.filterwarnings("ignore")

        # Skip initialization logs if info_only is True
        if not info_only:
            package_version = self.get_package_distribution("audio-separator").version
            self.logger.info(f"Separator version {package_version} instantiating with output_dir: {output_dir}, output_format: {output_format}")

        if output_dir is None:
            output_dir = os.getcwd()
            if not info_only:
                self.logger.info("Output directory not specified. Using current working directory.")

        self.output_dir = output_dir

        # Check for environment variable to override model_file_dir
        env_model_dir = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR")
        if env_model_dir:
            self.model_file_dir = env_model_dir
            self.logger.info(f"Using model directory from AUDIO_SEPARATOR_MODEL_DIR env var: {self.model_file_dir}")
            if not os.path.exists(self.model_file_dir):
                raise FileNotFoundError(f"The specified model directory does not exist: {self.model_file_dir}")
        else:
            self.logger.info(f"Using model directory from model_file_dir parameter: {model_file_dir}")
            self.model_file_dir = model_file_dir

        # Create the model directory if it does not exist
        os.makedirs(self.model_file_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.output_format = output_format
        self.output_bitrate = output_bitrate

        if self.output_format is None:
            self.output_format = "WAV"

        self.normalization_threshold = normalization_threshold
        if normalization_threshold <= 0 or normalization_threshold > 1:
            raise ValueError("The normalization_threshold must be greater than 0 and less than or equal to 1.")

        self.amplification_threshold = amplification_threshold
        if amplification_threshold < 0 or amplification_threshold > 1:
            raise ValueError("The amplification_threshold must be greater than or equal to 0 and less than or equal to 1.")

        self.output_single_stem = output_single_stem
        if output_single_stem is not None:
            self.logger.debug(f"Single stem output requested, so only one output file ({output_single_stem}) will be written")

        self.invert_using_spec = invert_using_spec
        if self.invert_using_spec:
            self.logger.debug(f"Secondary step will be inverted using spectogram rather than waveform. This may improve quality but is slightly slower.")

        try:
            self.sample_rate = int(sample_rate)
            if self.sample_rate <= 0:
                raise ValueError(f"The sample rate setting is {self.sample_rate} but it must be a non-zero whole number.")
            if self.sample_rate > 12800000:
                raise ValueError(f"The sample rate setting is {self.sample_rate}. Enter something less ambitious.")
        except ValueError:
            raise ValueError("The sample rate must be a non-zero whole number. Please provide a valid integer.")

        self.use_soundfile = use_soundfile
        self.use_autocast = use_autocast
        self.use_native_fp16 = use_native_fp16
        self.use_torch_compile = use_torch_compile
        self.use_directml = use_directml

        self.chunk_duration = chunk_duration
        if chunk_duration is not None:
            if chunk_duration <= 0:
                raise ValueError("chunk_duration must be greater than 0")

        self.ensemble_algorithm = ensemble_algorithm
        self.ensemble_weights = ensemble_weights
        self.ensemble_preset = ensemble_preset
        self._ensemble_preset_models = None

        # If an ensemble preset is specified, load it and apply defaults
        if ensemble_preset is not None:
            preset_data = self._load_ensemble_preset(ensemble_preset)
            self._ensemble_preset_models = preset_data["models"]
            # Preset values are defaults — explicit user args (non-None) take priority
            if ensemble_algorithm is None:
                self.ensemble_algorithm = preset_data["algorithm"]
            if ensemble_weights is None and preset_data.get("weights") is not None:
                self.ensemble_weights = preset_data["weights"]

        # Apply default algorithm if still not set (no preset, no explicit arg)
        if self.ensemble_algorithm is None:
            self.ensemble_algorithm = "avg_wave"

        # These are parameters which users may want to configure so we expose them to the top-level Separator class,
        # even though they are specific to a single model architecture
        self.arch_specific_params = {"MDX": mdx_params, "VR": vr_params, "Demucs": demucs_params, "MDXC": mdxc_params}

        self.torch_device = None
        self.torch_device_cpu = None
        self.torch_device_mps = None

        self.onnx_execution_provider = None
        self.model_instance = None
        self.model_filename = None
        self.model_filenames = []
        self._loaded_model_filename = None
        self._loaded_model_friendly_name = None
        self._loaded_model_is_uvr_vip = False

        self.model_is_uvr_vip = False
        self.model_friendly_name = None

        if not info_only:
            self.setup_accelerated_inferencing_device()

    @property
    def effective_precision(self):
        """Return the precision mode selected for the currently loaded model."""
        if len(getattr(self, "model_filenames", ())) > 1:
            return FP32
        return getattr(self.model_instance, "effective_precision", FP32)

    @property
    def effective_torch_compile(self):
        """Return whether regional compilation is active for the loaded model."""
        if len(getattr(self, "model_filenames", ())) > 1:
            return False
        return bool(getattr(self.model_instance, "effective_torch_compile", False))

    VALID_ENSEMBLE_ALGORITHMS = [
        "avg_wave", "median_wave", "min_wave", "max_wave",
        "avg_fft", "median_fft", "min_fft", "max_fft",
        "uvr_max_spec", "uvr_min_spec", "ensemble_wav",
    ]

    def _load_ensemble_preset(self, preset_name):
        """
        Load and validate an ensemble preset from ensemble_presets.json.

        Returns a dict with keys: name, description, models, algorithm, weights, contributor.
        Raises ValueError if the preset is not found or fails validation.
        """
        try:
            with resources.open_text("audio_separator", "ensemble_presets.json") as f:
                presets_data = json.load(f)
        except FileNotFoundError:
            raise ValueError("Ensemble presets file not found. The package may be corrupted or improperly installed.")

        presets = presets_data.get("presets", {})
        if preset_name not in presets:
            available = ", ".join(sorted(presets.keys()))
            raise ValueError(f"Unknown ensemble preset: '{preset_name}'. Available presets: {available}")

        preset = presets[preset_name]

        # Validate models
        models = preset.get("models", [])
        if not isinstance(models, list) or len(models) < 2:
            raise ValueError(f"Ensemble preset '{preset_name}' must specify at least 2 models, got {len(models) if isinstance(models, list) else 0}")

        # Validate algorithm
        algorithm = preset.get("algorithm", "avg_wave")
        if algorithm not in self.VALID_ENSEMBLE_ALGORITHMS:
            raise ValueError(f"Ensemble preset '{preset_name}' has unknown algorithm: '{algorithm}'")

        # Validate weights
        weights = preset.get("weights")
        if weights is not None:
            if not isinstance(weights, list) or len(weights) != len(models):
                raise ValueError(f"Ensemble preset '{preset_name}' weights length ({len(weights) if isinstance(weights, list) else 'N/A'}) must match models count ({len(models)})")

        self.logger.info(f"Loaded ensemble preset '{preset_name}': {preset.get('name', preset_name)} — {preset.get('description', '')}")
        return preset

    def list_ensemble_presets(self):
        """
        List all available ensemble presets.

        Returns a dict mapping preset IDs to their full preset data.
        """
        try:
            with resources.open_text("audio_separator", "ensemble_presets.json") as f:
                presets_data = json.load(f)
        except FileNotFoundError:
            return {}
        return presets_data.get("presets", {})

    def setup_accelerated_inferencing_device(self):
        """
        This method sets up the PyTorch and/or ONNX Runtime inferencing device, using GPU hardware acceleration if available.
        """
        system_info = self.get_system_info()
        self.check_ffmpeg_installed()
        self.log_onnxruntime_packages()
        self.preload_onnxruntime_dependencies()
        self.setup_torch_device(system_info)

    def preload_onnxruntime_dependencies(self):
        """
        Preload ONNX Runtime shared library dependencies when supported by the installed package.

        This helps pip-installed CUDA/cuDNN runtime wheels become visible to ONNX Runtime before
        the first CUDAExecutionProvider session is created.
        """
        if not hasattr(ort, "preload_dlls"):
            self.logger.debug("Installed ONNX Runtime does not provide preload_dlls(); skipping dependency preload.")
            return

        try:
            ort.preload_dlls()
            self.logger.debug("Preloaded ONNX Runtime shared library dependencies.")
        except Exception as exc:
            self.logger.warning(f"Unable to preload ONNX Runtime shared library dependencies: {exc}")

    def get_system_info(self):
        """
        This method logs the system information, including the operating system, CPU archutecture and Python version
        """
        os_name = platform.system()
        os_version = platform.version()
        self.logger.info(f"Operating System: {os_name} {os_version}")

        system_info = platform.uname()
        self.logger.info(f"System: {system_info.system} Node: {system_info.node} Release: {system_info.release} Machine: {system_info.machine} Proc: {system_info.processor}")

        python_version = platform.python_version()
        self.logger.info(f"Python Version: {python_version}")

        pytorch_version = torch.__version__
        self.logger.info(f"PyTorch Version: {pytorch_version}")
        return system_info

    def check_ffmpeg_installed(self):
        """
        This method checks if ffmpeg is installed and logs its version.
        """
        try:
            ffmpeg_version_output = subprocess.check_output(["ffmpeg", "-version"], text=True)
            first_line = ffmpeg_version_output.splitlines()[0]
            self.logger.info(f"FFmpeg installed: {first_line}")
        except FileNotFoundError:
            self.logger.error("FFmpeg is not installed. Please install FFmpeg to use this package.")
            # Raise an exception if this is being run by a user, as ffmpeg is required for pydub to write audio
            # but if we're just running unit tests in CI, no reason to throw
            if "PYTEST_CURRENT_TEST" not in os.environ:
                raise

    def log_onnxruntime_packages(self):
        """
        This method logs the ONNX Runtime package versions, including the GPU and Silicon packages if available.
        """
        onnxruntime_gpu_package = self.get_package_distribution("onnxruntime-gpu")
        onnxruntime_silicon_package = self.get_package_distribution("onnxruntime-silicon")
        onnxruntime_cpu_package = self.get_package_distribution("onnxruntime")
        onnxruntime_dml_package = self.get_package_distribution("onnxruntime-directml")

        if onnxruntime_gpu_package is not None:
            self.logger.info(f"ONNX Runtime GPU package installed with version: {onnxruntime_gpu_package.version}")
        if onnxruntime_silicon_package is not None:
            self.logger.info(f"ONNX Runtime Silicon package installed with version: {onnxruntime_silicon_package.version}")
        if onnxruntime_cpu_package is not None:
            self.logger.info(f"ONNX Runtime CPU package installed with version: {onnxruntime_cpu_package.version}")
        if onnxruntime_dml_package is not None:
            self.logger.info(f"ONNX Runtime DirectML package installed with version: {onnxruntime_dml_package.version}")

    def setup_torch_device(self, system_info):
        """
        This method sets up the PyTorch and/or ONNX Runtime inferencing device, using GPU hardware acceleration if available.
        """
        if self.cuda_device_index is not None and not torch.cuda.is_available():
            raise ValueError("cuda_device_index was specified, but CUDA is not available")
        hardware_acceleration_enabled = False
        ort_providers = ort.get_available_providers()
        has_torch_dml_installed = self.get_package_distribution("torch_directml")

        self.torch_device_cpu = torch.device("cpu")

        if torch.cuda.is_available():
            self.configure_cuda(ort_providers)
            hardware_acceleration_enabled = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and system_info.processor == "arm":
            self.configure_mps(ort_providers)
            hardware_acceleration_enabled = True
        elif self.use_directml and has_torch_dml_installed:
            import torch_directml
            if torch_directml.is_available():
                self.configure_dml(ort_providers)
                hardware_acceleration_enabled = True

        if not hardware_acceleration_enabled:
            self.logger.info("No hardware acceleration could be configured, running in CPU mode")
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]

            # Discoverability hint: DirectML is an explicit opt-in (experimental). If the
            # DirectML packages are installed but the feature wasn't enabled, tell the user how.
            if not self.use_directml and (has_torch_dml_installed or self.get_package_distribution("onnxruntime-directml") is not None):
                self.logger.info(
                    "DirectML packages detected but DirectML is not enabled. "
                    "Pass use_directml=True (or --use_directml on the CLI) to enable experimental DirectML acceleration."
                )

    def configure_cuda(self, ort_providers, device_index=None):
        """
        This method configures the CUDA device for PyTorch and ONNX Runtime, if available.
        """
        index = self.cuda_device_index if device_index is None else device_index
        _validate_cuda_device_index(index)
        if index is not None and index >= torch.cuda.device_count():
            raise ValueError(f"cuda_device_index {index} is outside the available CUDA device range")
        self.torch_device = torch.device("cuda" if index is None else f"cuda:{index}")
        self.logger.info(f"CUDA is available in Torch, setting Torch device to {self.torch_device}")
        if "CUDAExecutionProvider" in ort_providers:
            self.logger.info("ONNXruntime has CUDAExecutionProvider available, enabling acceleration")
            self.onnx_execution_provider = ["CUDAExecutionProvider"] if index is None else [("CUDAExecutionProvider", {"device_id": index})]
        else:
            self.logger.warning("CUDAExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled")

    def configure_mps(self, ort_providers):
        """
        This method configures the Apple Silicon MPS/CoreML device for PyTorch and ONNX Runtime, if available.
        """
        self.logger.info("Apple Silicon MPS/CoreML is available in Torch and processor is ARM, setting Torch device to MPS")
        self.torch_device_mps = torch.device("mps")

        self.torch_device = self.torch_device_mps

        if "CoreMLExecutionProvider" in ort_providers:
            self.logger.info("ONNXruntime has CoreMLExecutionProvider available, enabling acceleration")
            self.onnx_execution_provider = ["CoreMLExecutionProvider"]
        else:
            self.logger.warning("CoreMLExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled")

    def configure_dml(self, ort_providers):
        """
        This method configures the DirectML device for PyTorch and ONNX Runtime, if available.
        """
        import torch_directml
        self.logger.info("DirectML is available in Torch, setting Torch device to DirectML")
        self.torch_device_dml = torch_directml.device() 
        self.torch_device = self.torch_device_dml

        if "DmlExecutionProvider" in ort_providers:
            self.logger.info("ONNXruntime has DmlExecutionProvider available, enabling acceleration")
            self.onnx_execution_provider = ["DmlExecutionProvider"]
        else:
            self.logger.warning("DmlExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled")

    def get_package_distribution(self, package_name):
        """
        This method returns the package distribution for a given package name if installed, or None otherwise.
        """
        try:
            return metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            self.logger.debug(f"Python package: {package_name} not installed")
            return None

    def get_model_hash(self, model_path):
        """
        This method returns the MD5 hash of a given model file.
        """
        self.logger.debug(f"Calculating hash of model file {model_path}")
        # Use the specific byte count from the original logic
        BYTES_TO_HASH = 10000 * 1024  # 10,240,000 bytes

        try:
            file_size = os.path.getsize(model_path)

            with open(model_path, "rb") as f:
                if file_size < BYTES_TO_HASH:
                    # Hash the entire file if smaller than the target byte count
                    self.logger.debug(f"File size {file_size} < {BYTES_TO_HASH}, hashing entire file.")
                    hash_value = hashlib.md5(f.read()).hexdigest()
                else:
                    # Seek to the specific position before the end (from the beginning) and hash
                    seek_pos = file_size - BYTES_TO_HASH
                    self.logger.debug(f"File size {file_size} >= {BYTES_TO_HASH}, seeking to {seek_pos} and hashing remaining bytes.")
                    f.seek(seek_pos, io.SEEK_SET)
                    hash_value = hashlib.md5(f.read()).hexdigest()

            # Log the calculated hash
            self.logger.info(f"Hash of model file {model_path} is {hash_value}")
            return hash_value

        except FileNotFoundError:
            self.logger.error(f"Model file not found at {model_path}")
            raise # Re-raise the specific error
        except Exception as e:
            # Catch other potential errors (e.g., permissions, other IOErrors)
            self.logger.error(f"Error calculating hash for {model_path}: {e}")
            raise # Re-raise other errors

    def download_file_if_not_exists(self, url, output_path):
        """
        This method downloads a file from a given URL to a given output path, if the file does not already exist.
        """

        if os.path.isfile(output_path):
            self.logger.debug(f"File already exists at {output_path}, skipping download")
            return

        self.logger.debug(f"Downloading file from {url} to {output_path} with timeout 300s")
        response = requests.get(url, stream=True, timeout=300)

        if response.status_code == 200:
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        else:
            raise RuntimeError(f"Failed to download file from {url}, response code: {response.status_code}")

    def list_supported_model_files(self):
        """
        This method lists the supported model files for audio-separator, by fetching the same file UVR uses to list these.
        Also includes model performance scores where available.

        Example response object:

        {
            "MDX": {
                "MDX-Net Model VIP: UVR-MDX-NET-Inst_full_292": {
                "filename": "UVR-MDX-NET-Inst_full_292.onnx",
                "scores": {
                    "vocals": {
                    "SDR": 10.6497,
                    "SIR": 20.3786,
                    "SAR": 10.692,
                    "ISR": 14.848
                    },
                    "instrumental": {
                    "SDR": 15.2149,
                    "SIR": 25.6075,
                    "SAR": 17.1363,
                    "ISR": 17.7893
                    }
                },
                "download_files": [
                    "UVR-MDX-NET-Inst_full_292.onnx"
                ]
                }
            },
            "Demucs": {
                "Demucs v4: htdemucs_ft": {
                "filename": "htdemucs_ft.yaml",
                "scores": {
                    "vocals": {
                    "SDR": 11.2685,
                    "SIR": 21.257,
                    "SAR": 11.0359,
                    "ISR": 19.3753
                    },
                    "drums": {
                    "SDR": 13.235,
                    "SIR": 23.3053,
                    "SAR": 13.0313,
                    "ISR": 17.2889
                    },
                    "bass": {
                    "SDR": 9.72743,
                    "SIR": 19.5435,
                    "SAR": 9.20801,
                    "ISR": 13.5037
                    }
                },
                "download_files": [
                    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
                    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
                    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th",
                    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th",
                    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/htdemucs_ft.yaml"
                ]
                }
            },
            "MDXC": {
                "MDX23C Model: MDX23C-InstVoc HQ": {
                "filename": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
                "scores": {
                    "vocals": {
                    "SDR": 11.9504,
                    "SIR": 23.1166,
                    "SAR": 12.093,
                    "ISR": 15.4782
                    },
                    "instrumental": {
                    "SDR": 16.3035,
                    "SIR": 26.6161,
                    "SAR": 18.5167,
                    "ISR": 18.3939
                    }
                },
                "download_files": [
                    "MDX23C-8KFFT-InstVoc_HQ.ckpt",
                    "model_2_stem_full_band_8k.yaml"
                ]
                }
            }
        }
        """
        download_checks_path = os.path.join(self.model_file_dir, "download_checks.json")

        self.download_file_if_not_exists("https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json", download_checks_path)

        model_downloads_list = json.load(open(download_checks_path, encoding="utf-8"))
        self.logger.debug(f"UVR model download list loaded")

        # Load the model scores with error handling
        model_scores = {}
        try:
            with resources.open_text("audio_separator", "models-scores.json") as f:
                model_scores = json.load(f)
            self.logger.debug(f"Model scores loaded")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to load model scores: {str(e)}")
            self.logger.warning("Continuing without model scores")

        # Only show Demucs v4 models as we've only implemented support for v4
        filtered_demucs_v4 = {key: value for key, value in model_downloads_list["demucs_download_list"].items() if key.startswith("Demucs v4")}

        # Modified Demucs handling to use YAML files as identifiers and include download files
        demucs_models = {}
        for name, files in filtered_demucs_v4.items():
            # Find the YAML file in the model files
            yaml_file = next((filename for filename in files.keys() if filename.endswith(".yaml")), None)
            if yaml_file:
                model_score_data = model_scores.get(yaml_file, {})
                demucs_models[name] = {
                    "filename": yaml_file,
                    "scores": model_score_data.get("median_scores", {}),
                    "stems": model_score_data.get("stems", []),
                    "target_stem": model_score_data.get("target_stem"),
                    "download_files": list(files.values()),  # List of all download URLs/filenames
                }

        # Load the JSON file using importlib.resources
        with resources.open_text("audio_separator", "models.json") as f:
            audio_separator_models_list = json.load(f)
        self.logger.debug(f"Audio-Separator model list loaded")

        # Return object with list of model names
        model_files_grouped_by_type = {
            "VR": {
                name: {
                    "filename": filename,
                    "scores": model_scores.get(filename, {}).get("median_scores", {}),
                    "stems": model_scores.get(filename, {}).get("stems", []),
                    "target_stem": model_scores.get(filename, {}).get("target_stem"),
                    "download_files": [filename],
                }  # Just the filename for VR models
                for name, filename in {**model_downloads_list["vr_download_list"], **audio_separator_models_list["vr_download_list"]}.items()
            },
            "MDX": {
                name: {
                    "filename": filename,
                    "scores": model_scores.get(filename, {}).get("median_scores", {}),
                    "stems": model_scores.get(filename, {}).get("stems", []),
                    "target_stem": model_scores.get(filename, {}).get("target_stem"),
                    "download_files": [filename],
                }  # Just the filename for MDX models
                for name, filename in {**model_downloads_list["mdx_download_list"], **model_downloads_list["mdx_download_vip_list"], **audio_separator_models_list["mdx_download_list"]}.items()
            },
            "Demucs": demucs_models,
            "MDXC": {
                name: {
                    "filename": next(iter(files.keys())),
                    "scores": model_scores.get(next(iter(files.keys())), {}).get("median_scores", {}),
                    "stems": model_scores.get(next(iter(files.keys())), {}).get("stems", []),
                    "target_stem": model_scores.get(next(iter(files.keys())), {}).get("target_stem"),
                    "download_files": list(files.keys()) + list(files.values()),  # List of both model filenames and config filenames
                }
                for name, files in {
                    **model_downloads_list["mdx23c_download_list"],
                    **model_downloads_list["mdx23c_download_vip_list"],
                    **model_downloads_list["roformer_download_list"],
                    **audio_separator_models_list["mdx23c_download_list"],
                    **audio_separator_models_list["roformer_download_list"],
                }.items()
            },
        }

        return model_files_grouped_by_type

    def print_uvr_vip_message(self):
        """
        This method prints a message to the user if they have downloaded a VIP model, reminding them to support Anjok07 on Patreon.
        """
        if self.model_is_uvr_vip:
            self.logger.warning(f"The model: '{self.model_friendly_name}' is a VIP model, intended by Anjok07 for access by paying subscribers only.")
            self.logger.warning("If you are not already subscribed, please consider supporting the developer of UVR, Anjok07 by subscribing here: https://patreon.com/uvr")

    def download_model_files(self, model_filename):
        """
        This method downloads the model files for a given model filename, if they are not already present.
        Returns tuple of (model_filename, model_type, model_friendly_name, model_path, yaml_config_filename)
        """
        model_path = os.path.join(self.model_file_dir, f"{model_filename}")

        supported_model_files_grouped = self.list_supported_model_files()
        public_model_repo_url_prefix = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
        vip_model_repo_url_prefix = "https://github.com/Anjok0109/ai_magic/releases/download/v5"
        audio_separator_models_repo_url_prefix = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"

        yaml_config_filename = None

        self.logger.debug(f"Searching for model_filename {model_filename} in supported_model_files_grouped")

        # Iterate through model types (MDX, Demucs, MDXC)
        for model_type, models in supported_model_files_grouped.items():
            # Iterate through each model in this type
            for model_friendly_name, model_info in models.items():
                self.model_is_uvr_vip = "VIP" in model_friendly_name
                model_repo_url_prefix = vip_model_repo_url_prefix if self.model_is_uvr_vip else public_model_repo_url_prefix

                # Check if this model matches our target filename
                if model_info["filename"] == model_filename or model_filename in model_info["download_files"]:
                    self.logger.debug(f"Found matching model: {model_friendly_name}")
                    self.model_friendly_name = model_friendly_name
                    self.print_uvr_vip_message()

                    # Download each required file for this model
                    for file_to_download in model_info["download_files"]:
                        # For URLs, extract just the filename portion
                        if file_to_download.startswith("http"):
                            filename = file_to_download.split("/")[-1]
                            download_path = os.path.join(self.model_file_dir, filename)
                            self.download_file_if_not_exists(file_to_download, download_path)
                            continue

                        download_path = os.path.join(self.model_file_dir, file_to_download)

                        # For MDXC models, handle YAML config files specially
                        if model_type == "MDXC" and file_to_download.endswith(".yaml"):
                            yaml_config_filename = file_to_download
                            try:
                                yaml_url = f"{model_repo_url_prefix}/mdx_model_data/mdx_c_configs/{file_to_download}"
                                self.download_file_if_not_exists(yaml_url, download_path)
                            except RuntimeError:
                                self.logger.debug("YAML config not found in UVR repo, trying audio-separator models repo...")
                                yaml_url = f"{audio_separator_models_repo_url_prefix}/{file_to_download}"
                                self.download_file_if_not_exists(yaml_url, download_path)
                            continue

                        # For regular model files, try UVR repo first, then audio-separator repo
                        try:
                            download_url = f"{model_repo_url_prefix}/{file_to_download}"
                            self.download_file_if_not_exists(download_url, download_path)
                        except RuntimeError:
                            self.logger.debug("Model not found in UVR repo, trying audio-separator models repo...")
                            download_url = f"{audio_separator_models_repo_url_prefix}/{file_to_download}"
                            self.download_file_if_not_exists(download_url, download_path)

                    return model_filename, model_type, model_friendly_name, model_path, yaml_config_filename

        raise ValueError(f"Model file {model_filename} not found in supported model files")

    def load_model_data_from_yaml(self, yaml_config_filename):
        """
        This method loads model-specific parameters from the YAML file for that model.
        The parameters in the YAML are critical to inferencing, as they need to match whatever was used during training.
        """
        # Verify if the YAML filename includes a full path or just the filename
        if not os.path.exists(yaml_config_filename):
            model_data_yaml_filepath = os.path.join(self.model_file_dir, yaml_config_filename)
        else:
            model_data_yaml_filepath = yaml_config_filename

        self.logger.debug(f"Loading model data from YAML at path {model_data_yaml_filepath}")

        model_data = yaml.load(open(model_data_yaml_filepath, encoding="utf-8"), Loader=yaml.FullLoader)
        self.logger.debug(f"Model data loaded from YAML file: {model_data}")

        from .roformer.configuration_normalizer import ConfigurationNormalizer

        configured_roformer_type = ConfigurationNormalizer().detect_model_type(model_data)
        yaml_filename = os.path.basename(model_data_yaml_filepath).lower()
        if configured_roformer_type is not None or "roformer" in yaml_filename:
            model_data["is_roformer"] = True

        return model_data

    def load_model_data_using_hash(self, model_path):
        """
        This method loads model-specific parameters from UVR model data files.
        These parameters are critical to inferencing using a given model, as they need to match whatever was used during training.
        The correct parameters are identified by calculating the hash of the model file and looking up the hash in the UVR data files.
        """
        # Model data and configuration sources from UVR
        model_data_url_prefix = "https://raw.githubusercontent.com/TRvlvr/application_data/main"

        vr_model_data_url = f"{model_data_url_prefix}/vr_model_data/model_data_new.json"
        mdx_model_data_url = f"{model_data_url_prefix}/mdx_model_data/model_data_new.json"

        # Calculate hash for the downloaded model
        self.logger.debug("Calculating MD5 hash for model file to identify model parameters from UVR data...")
        model_hash = self.get_model_hash(model_path)
        self.logger.debug(f"Model {model_path} has hash {model_hash}")

        # Setting up the path for model data and checking its existence
        vr_model_data_path = os.path.join(self.model_file_dir, "vr_model_data.json")
        self.logger.debug(f"VR model data path set to {vr_model_data_path}")
        self.download_file_if_not_exists(vr_model_data_url, vr_model_data_path)

        mdx_model_data_path = os.path.join(self.model_file_dir, "mdx_model_data.json")
        self.logger.debug(f"MDX model data path set to {mdx_model_data_path}")
        self.download_file_if_not_exists(mdx_model_data_url, mdx_model_data_path)

        # Loading model data from UVR
        self.logger.debug("Loading MDX and VR model parameters from UVR model data files...")
        vr_model_data_object = json.load(open(vr_model_data_path, encoding="utf-8"))
        mdx_model_data_object = json.load(open(mdx_model_data_path, encoding="utf-8"))

        # Load additional model data from audio-separator
        self.logger.debug("Loading additional model parameters from audio-separator model data file...")
        with resources.open_text("audio_separator", "model-data.json") as f:
            audio_separator_model_data = json.load(f)

        # Merge the model data objects, with audio-separator data taking precedence
        vr_model_data_object = {**vr_model_data_object, **audio_separator_model_data.get("vr_model_data", {})}
        mdx_model_data_object = {**mdx_model_data_object, **audio_separator_model_data.get("mdx_model_data", {})}

        if model_hash in mdx_model_data_object:
            model_data = mdx_model_data_object[model_hash]
        elif model_hash in vr_model_data_object:
            model_data = vr_model_data_object[model_hash]
        else:
            raise ValueError(f"Unsupported Model File: parameters for MD5 hash {model_hash} could not be found in UVR model data file for MDX or VR arch.")

        self.logger.debug(f"Model data loaded using hash {model_hash}: {model_data}")

        return model_data

    def load_model(self, model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt", force_reload=False):
        """
        This method instantiates the architecture-specific separation class,
        loading the separation model into memory, downloading it first if necessary.

        Consecutive calls with the same single model reuse the loaded instance. Set
        ``force_reload`` to ``True`` after changing settings captured at load time.

        Args:
            model_filename (str or list): The model filename, or filenames for an ensemble.
            force_reload (bool): Reload a matching single model instead of reusing it.
        """
        # If an ensemble preset was loaded and no explicit model list was provided, use preset models
        if self._ensemble_preset_models is not None and model_filename == "model_bs_roformer_ep_317_sdr_12.9755.ckpt":
            model_filename = self._ensemble_preset_models

        if isinstance(model_filename, list):
            if len(model_filename) > 1:
                self.model_filename = list(model_filename)
                self.model_filenames = list(model_filename)
                self.logger.info(f"Multiple models specified for ensembling: {self.model_filenames}")
                return
            model_filename = model_filename[0]

        if not force_reload and self.model_instance is not None and self._loaded_model_filename == model_filename:
            self.model_filename = model_filename
            self.model_filenames = [model_filename]
            self.model_friendly_name = self._loaded_model_friendly_name
            self.model_is_uvr_vip = self._loaded_model_is_uvr_vip
            self.logger.info(f"Model {model_filename} is already loaded; reusing the existing instance.")
            return

        self.logger.info(f"Loading model {model_filename}...")

        load_model_start_time = time.perf_counter()
        selected_model_filename = model_filename
        previous_model_friendly_name = self.model_friendly_name
        previous_model_is_uvr_vip = self.model_is_uvr_vip

        try:
            # Setting up the model path
            model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)
            model_is_uvr_vip = self.model_is_uvr_vip
            model_name = model_filename.split(".")[0]
            self.logger.debug(f"Model downloaded, friendly name: {model_friendly_name}, model_path: {model_path}")

            if model_path.lower().endswith(".yaml"):
                yaml_config_filename = model_path

            if yaml_config_filename is not None:
                model_data = self.load_model_data_from_yaml(yaml_config_filename)
            else:
                model_data = self.load_model_data_using_hash(model_path)

            common_params = {
                "logger": self.logger,
                "log_level": self.log_level,
                "torch_device": self.torch_device,
                "torch_device_cpu": self.torch_device_cpu,
                "torch_device_mps": self.torch_device_mps,
                "onnx_execution_provider": self.onnx_execution_provider,
                "model_name": model_name,
                "model_path": model_path,
                "model_data": model_data,
                "output_format": self.output_format,
                "output_bitrate": self.output_bitrate,
                "output_dir": self.output_dir,
                "normalization_threshold": self.normalization_threshold,
                "amplification_threshold": self.amplification_threshold,
                "output_single_stem": self.output_single_stem,
                "invert_using_spec": self.invert_using_spec,
                "sample_rate": self.sample_rate,
                "use_soundfile": self.use_soundfile,
                "use_autocast": self.use_autocast,
                "use_native_fp16": self.use_native_fp16,
                "use_torch_compile": self.use_torch_compile,
            }

            # Instantiate the appropriate separator class depending on the model type
            separator_classes = {"MDX": "mdx_separator.MDXSeparator", "VR": "vr_separator.VRSeparator", "Demucs": "demucs_separator.DemucsSeparator", "MDXC": "mdxc_separator.MDXCSeparator"}

            if model_type not in self.arch_specific_params or model_type not in separator_classes:
                # Enhanced error message for Roformer models
                if "roformer" in model_filename.lower() or (model_data and model_data.get("is_roformer", False)):
                    error_msg = (f"Roformer model type not properly configured: {model_type}. "
                               f"This may indicate a configuration validation failure. "
                               f"Please check the model file and YAML configuration.")
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    raise ValueError(f"Model type not supported (yet): {model_type}")

            if model_type == "Demucs" and sys.version_info < (3, 10):
                raise Exception("Demucs models require Python version 3.10 or newer.")

            self.logger.debug(f"Importing module for model type {model_type}: {separator_classes[model_type]}")

            module_name, class_name = separator_classes[model_type].split(".")
            module = importlib.import_module(f"audio_separator.separator.architectures.{module_name}")
            separator_class = getattr(module, class_name)

            self.logger.debug(f"Instantiating separator class for model type {model_type}: {separator_class}")

            try:
                model_instance = separator_class(common_config=common_params, arch_config=self.arch_specific_params[model_type])
            except Exception as e:
                # Enhanced error handling for Roformer models
                if "roformer" in model_filename.lower() or (model_data and model_data.get("is_roformer", False)):
                    error_msg = (f"Failed to instantiate Roformer model: {e}. "
                               f"This may be due to missing parameters or configuration validation failures.")
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                else:
                    raise

            resolve_policy = getattr(model_instance, "resolve_execution_policy", None)
            if callable(resolve_policy) and not getattr(model_instance, "_execution_policy_resolved", False):
                resolve_policy(model_type.lower())

            # Log Roformer implementation version if applicable
            if hasattr(model_instance, 'is_roformer_model') and model_instance.is_roformer_model:
                roformer_stats = model_instance.get_roformer_loading_stats()
                if roformer_stats:
                    self.logger.info(f"Roformer loading stats: {roformer_stats}")
        except BaseException:
            # Model discovery updates these fields; restore them with the prior
            # selection so callers never observe a partially loaded model.
            self.model_friendly_name = previous_model_friendly_name
            self.model_is_uvr_vip = previous_model_is_uvr_vip
            raise

        self.model_instance = model_instance
        self.model_filename = selected_model_filename
        self.model_filenames = [selected_model_filename]
        self._loaded_model_filename = model_filename
        self.model_friendly_name = model_friendly_name
        self.model_is_uvr_vip = model_is_uvr_vip
        self._loaded_model_friendly_name = model_friendly_name
        self._loaded_model_is_uvr_vip = model_is_uvr_vip

        # Log the completion of the model load process
        self.logger.debug("Loading model completed.")
        self.logger.info(f'Load model duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - load_model_start_time)))}')

    def separate(self, audio_file_path, custom_output_names=None):
        """
        Separates the audio file(s) into different stems (e.g., vocals, instruments) using the loaded model.

        This method takes the path to an audio file or a directory containing audio files, processes them through
        the loaded separation model, and returns the paths to the output files containing the separated audio stems.
        It handles the entire flow from loading the audio, running the separation, clearing up resources, and logging the process.

        Parameters:
        - audio_file_path (str or list): The path to the audio file or directory, or a list of paths.
        - custom_output_names (dict, optional): Custom names for the output files. Defaults to None.

        Returns:
        - output_files (list of str): Fully written output paths produced by inputs that completed
          their full stem set. Valid silent and near-silent stems are included.

        Raises:
        - InvalidAudioDataError: Decoded or generated audio data was empty, non-finite, or structurally invalid.
        - AudioExportError: An audio backend or filesystem failed to publish an output file.
        - BatchSeparationError: A list or directory input had one or more failures. All discoverable
          inputs are attempted first; inspect ``successful_files`` and the ordered ``failures`` list.

        A single file fails immediately with its original exception. Import the public exception
        types from ``audio_separator.separator``.
        """
        # Check if the model and device are properly initialized
        if not (self.torch_device and (self.model_instance or (isinstance(self.model_filename, list) and len(self.model_filename) > 0))):
            raise ValueError("Initialization failed or model not loaded. Please load a model before attempting to separate.")

        if isinstance(self.model_filename, list) and len(self.model_filename) > 1:
            if isinstance(audio_file_path, str) and not os.path.isdir(audio_file_path):
                return self._separate_ensemble(audio_file_path, custom_output_names)

            ensemble_inputs = []
            failures = []
            input_paths = [audio_file_path] if isinstance(audio_file_path, str) else audio_file_path
            for path in input_paths:
                if os.path.isdir(path):
                    ensemble_inputs.extend(_iter_directory_audio_files(path))
                else:
                    ensemble_inputs.append(path)

            output_files = []
            for path in ensemble_inputs:
                try:
                    output_files.extend(self._separate_ensemble(path, custom_output_names))
                except Exception as e:
                    self.logger.error(f"Failed to process ensemble file {path}: {e}", exc_info=True)
                    failures.append((path, e))

            if failures:
                raise BatchSeparationError(output_files, failures)
            return output_files

        if isinstance(audio_file_path, str) and not os.path.isdir(audio_file_path):
            self.logger.info(f"Processing file: {audio_file_path}")
            return self._separate_file(audio_file_path, custom_output_names)

        # If audio_file_path is a string, convert it to a list for uniform processing
        if isinstance(audio_file_path, str):
            audio_file_path = [audio_file_path]

        # Initialize a list to store paths of all output files
        output_files = []
        failures = []

        # Process each path in the list
        for path in audio_file_path:
            if os.path.isdir(path):
                # If the path is a directory, recursively search for all audio files
                for full_path in _iter_directory_audio_files(path):
                    self.logger.info(f"Processing file: {full_path}")
                    try:
                        # Perform separation for each file
                        files_output = self._separate_file(full_path, custom_output_names)
                        output_files.extend(files_output)
                    except Exception as e:
                        self.logger.error(f"Failed to process file {full_path}: {e}", exc_info=True)
                        failures.append((full_path, e))
            else:
                # If the path is a file, process it directly
                self.logger.info(f"Processing file: {path}")
                try:
                    files_output = self._separate_file(path, custom_output_names)
                    output_files.extend(files_output)
                except Exception as e:
                    self.logger.error(f"Failed to process file {path}: {e}", exc_info=True)
                    failures.append((path, e))

        if failures:
            raise BatchSeparationError(output_files, failures)
        return output_files

    def _separate_file(self, audio_file_path, custom_output_names=None):
        """
        Internal method to handle separation for a single audio file.
        This method performs the actual separation process for a single audio file. It logs the start and end of the process,
        handles autocast if enabled, and ensures GPU cache is cleared after processing.
        Parameters:
        - audio_file_path (str): The path to the audio file.
        - custom_output_names (dict, optional): Custom names for the output files. Defaults to None.
        Returns:
        - output_files (list of str): A list containing the paths to the separated audio stem files.
        """
        # Check if chunking is enabled and file is large enough
        if self.chunk_duration is not None:
            import librosa
            duration = librosa.get_duration(path=audio_file_path)

            from audio_separator.separator.audio_chunking import AudioChunker
            chunker = AudioChunker(self.chunk_duration, self.logger)

            if chunker.should_chunk(duration):
                self.logger.info(f"File duration {duration:.1f}s exceeds chunk size {self.chunk_duration}s, using chunked processing")
                return self._process_with_chunking(audio_file_path, custom_output_names)

        # Log the start of the separation process
        self.logger.info(f"Starting separation process for audio_file_path: {audio_file_path}")
        separate_start_time = time.perf_counter()

        # Log normalization and amplification thresholds
        self.logger.debug(f"Normalization threshold set to {self.normalization_threshold}, waveform will be lowered to this max amplitude to avoid clipping.")
        self.logger.debug(f"Amplification threshold set to {self.amplification_threshold}, waveform will be scaled up to this max amplitude if below it.")

        # Run separation using the policy resolved for the loaded model's actual
        # inference device. This matters when an architecture falls back to CPU.
        output_files = None
        effective_precision = self.effective_precision
        inference_device = getattr(self.model_instance, "torch_device", self.torch_device)
        inference_device_type = getattr(inference_device, "type", str(inference_device))
        try:
            if effective_precision == NATIVE_FP16:
                self.logger.debug("Using native float16 inference.")
                output_files = self.model_instance.separate(audio_file_path, custom_output_names)
            elif effective_precision == AUTOCAST and inference_device_type != "privateuseone":
                self.logger.debug("Using autocast inference on %s.", inference_device_type)
                with autocast_mode.autocast(inference_device_type):
                    output_files = self.model_instance.separate(audio_file_path, custom_output_names)
            else:
                self.logger.debug("Using float32 inference.")
                output_files = self.model_instance.separate(audio_file_path, custom_output_names)
        finally:
            # Clear per-file state even when inference or audio export fails.
            active_error = sys.exc_info()[1]
            cleanup_errors = []
            for cleanup_name in ("clear_gpu_cache", "clear_file_specific_paths"):
                try:
                    getattr(self.model_instance, cleanup_name)()
                except Exception as cleanup_error:
                    cleanup_errors.append((cleanup_name, cleanup_error))
                    self.logger.error(f"Cleanup step {cleanup_name} failed: {cleanup_error}", exc_info=True)

            if cleanup_errors:
                if active_error is not None:
                    if hasattr(active_error, "add_note"):
                        for cleanup_name, cleanup_error in cleanup_errors:
                            active_error.add_note(f"Cleanup step {cleanup_name} failed: {cleanup_error}")
                else:
                    cleanup_name, cleanup_error = cleanup_errors[0]
                    if hasattr(cleanup_error, "add_note"):
                        cleanup_error.add_note(f"Cleanup step failed: {cleanup_name}")
                    raise cleanup_error

        # Remind the user one more time if they used a VIP model, so the message doesn't get lost in the logs
        self.print_uvr_vip_message()

        # Log the completion of the separation process
        self.logger.debug("Separation process completed.")
        self.logger.info(f'Separation duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - separate_start_time)))}')

        return output_files

    def _process_with_chunking(self, audio_file_path, custom_output_names=None):
        """
        Process large file by splitting into chunks.

        This method splits a large audio file into smaller chunks, processes each chunk
        separately, and merges the results back together. This helps prevent out-of-memory
        errors when processing very long audio files.

        Parameters:
        - audio_file_path (str): The path to the audio file.
        - custom_output_names (dict, optional): Custom names for the output files. Defaults to None.

        Returns:
        - output_files (list of str): A list containing the paths to the separated audio stem files.
        """
        import tempfile
        import shutil
        from audio_separator.separator.audio_chunking import AudioChunker

        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp(prefix="audio-separator-chunks-")
        self.logger.debug(f"Created temporary directory for chunks: {temp_dir}")

        try:
            # Split audio into chunks
            chunker = AudioChunker(self.chunk_duration, self.logger)
            chunk_paths = chunker.split_audio(audio_file_path, temp_dir)

            # Process each chunk
            processed_chunks_by_stem = {}
            stem_names_by_chunk = []

            for i, chunk_path in enumerate(chunk_paths):
                self.logger.info(f"Processing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")

                original_chunk_duration = self.chunk_duration
                original_output_dir = self.output_dir
                self.chunk_duration = None
                self.output_dir = temp_dir

                if self.model_instance:
                    original_model_output_dir = self.model_instance.output_dir
                    self.model_instance.output_dir = temp_dir

                try:
                    output_files = self._separate_file(chunk_path)
                    current_stem_names = set()

                    # Dynamically group chunks by stem name
                    for stem_index, stem_path in enumerate(output_files):
                        # Extract stem name from filename: "chunk_0000_(Vocals).wav" → "Vocals"
                        filename = os.path.basename(stem_path)
                        match = re.search(r'_\(([^)]+)\)', filename)
                        if match:
                            stem_name = match.group(1)
                        else:
                            # Fallback: use index-based name if pattern not found
                            stem_name = f"stem_{stem_index}"
                            self.logger.warning(f"Could not extract stem name from {filename}, using {stem_name}")

                        if stem_name in current_stem_names:
                            raise InvalidAudioDataError(f"Chunk {i} produced duplicate stem output: {stem_name}")
                        current_stem_names.add(stem_name)
                        if stem_name not in processed_chunks_by_stem:
                            processed_chunks_by_stem[stem_name] = []

                        # Ensure absolute path
                        abs_path = stem_path if os.path.isabs(stem_path) else os.path.join(temp_dir, stem_path)
                        processed_chunks_by_stem[stem_name].append(abs_path)

                    if not output_files:
                        self.logger.warning(f"Chunk {i+1} produced no output files")
                    stem_names_by_chunk.append(current_stem_names)

                finally:
                    self.chunk_duration = original_chunk_duration
                    self.output_dir = original_output_dir
                    if self.model_instance:
                        self.model_instance.output_dir = original_model_output_dir

                # Clear GPU cache between chunks
                if self.model_instance:
                    self.model_instance.clear_gpu_cache()

            all_stem_names = set().union(*stem_names_by_chunk) if stem_names_by_chunk else set()
            if not all_stem_names:
                raise InvalidAudioDataError("Chunked separation produced no stems")

            missing_stems = {
                chunk_index: sorted(all_stem_names - chunk_stems)
                for chunk_index, chunk_stems in enumerate(stem_names_by_chunk)
                if chunk_stems != all_stem_names
            }
            if missing_stems:
                details = "; ".join(f"chunk {chunk_index}: {', '.join(stems)}" for chunk_index, stems in missing_stems.items())
                raise InvalidAudioDataError(f"Chunked separation is missing stem output(s): {details}")

            # Merge chunks for each stem dynamically
            base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
            output_files = []

            for stem_name in sorted(processed_chunks_by_stem.keys()):
                chunk_paths_for_stem = processed_chunks_by_stem[stem_name]

                if not chunk_paths_for_stem:
                    self.logger.warning(f"No chunks found for stem: {stem_name}")
                    continue

                # Determine output filename
                if custom_output_names and stem_name in custom_output_names:
                    output_filename = custom_output_names[stem_name]
                else:
                    output_filename = f"{base_name}_({stem_name})"

                output_path = os.path.join(self.output_dir, f"{output_filename}.{self.output_format.lower()}")

                self.logger.info(f"Merging {len(chunk_paths_for_stem)} chunks for stem: {stem_name}")
                chunker.merge_chunks(chunk_paths_for_stem, output_path)
                output_files.append(output_path)

            self.logger.info(f"Chunked processing completed. Output files: {output_files}")
            return output_files

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                self.logger.debug(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)

    def download_model_and_data(self, model_filename):
        """
        Downloads the model file without loading it into memory.
        """
        self.logger.info(f"Downloading model {model_filename}...")

        model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)

        if model_path.lower().endswith(".yaml"):
            yaml_config_filename = model_path

        if yaml_config_filename is not None:
            model_data = self.load_model_data_from_yaml(yaml_config_filename)
        else:
            model_data = self.load_model_data_using_hash(model_path)

        model_data_dict_size = len(model_data)

        self.logger.info(f"Model downloaded, type: {model_type}, friendly name: {model_friendly_name}, model_path: {model_path}, model_data: {model_data_dict_size} items")

    def get_simplified_model_list(self, filter_sort_by: Optional[str] = None):
        """
        Returns a simplified, user-friendly list of models with their key metrics.
        Optionally sorts the list based on the specified criteria.

        :param sort_by: Criteria to sort by. Can be "name", "filename", or any stem name
        """
        model_files = self.list_supported_model_files()
        simplified_list = {}

        for model_type, models in model_files.items():
            for name, data in models.items():
                filename = data["filename"]
                scores = data.get("scores") or {}
                stems = data.get("stems") or []
                target_stem = data.get("target_stem")

                # Format stems with their SDR scores where available
                stems_with_scores = []
                stem_sdr_dict = {}

                # Process each stem from the model's stem list
                for stem in stems:
                    stem_scores = scores.get(stem, {})
                    # Add asterisk if this is the target stem
                    stem_display = f"{stem}*" if stem == target_stem else stem

                    if isinstance(stem_scores, dict) and "SDR" in stem_scores:
                        sdr = round(stem_scores["SDR"], 1)
                        stems_with_scores.append(f"{stem_display} ({sdr})")
                        stem_sdr_dict[stem.lower()] = sdr
                    else:
                        # Include stem without SDR score
                        stems_with_scores.append(stem_display)
                        stem_sdr_dict[stem.lower()] = None

                # If no stems listed, mark as Unknown
                if not stems_with_scores:
                    stems_with_scores = ["Unknown"]
                    stem_sdr_dict["unknown"] = None

                simplified_list[filename] = {"Name": name, "Type": model_type, "Stems": stems_with_scores, "SDR": stem_sdr_dict}

        # Sort and filter the list if a sort_by parameter is provided
        if filter_sort_by:
            if filter_sort_by == "name":
                return dict(sorted(simplified_list.items(), key=lambda x: x[1]["Name"]))
            elif filter_sort_by == "filename":
                return dict(sorted(simplified_list.items()))
            else:
                # Convert sort_by to lowercase for case-insensitive comparison
                sort_by_lower = filter_sort_by.lower()
                # Filter out models that don't have the specified stem
                filtered_list = {k: v for k, v in simplified_list.items() if sort_by_lower in v["SDR"]}

                # Sort by SDR score if available, putting None values last
                def sort_key(item):
                    sdr = item[1]["SDR"][sort_by_lower]
                    return (0 if sdr is None else 1, sdr if sdr is not None else float("-inf"))

                return dict(sorted(filtered_list.items(), key=sort_key, reverse=True))

        return simplified_list

    def _separate_ensemble(self, audio_file_path, custom_output_names=None):
        """
        Internal method to handle ensembling of multiple models.
        """
        import tempfile
        import shutil

        if isinstance(audio_file_path, str):
            audio_file_path = [audio_file_path]

        output_files = []

        original_model_filename = self.model_filename
        original_model_filenames = self.model_filenames

        for path in audio_file_path:
            self.logger.info(f"Ensemble processing for file: {path}")

            # Create temporary directory for intermediate stems
            temp_dir = tempfile.mkdtemp(prefix="audio-separator-ensemble-")
            self.logger.debug(f"Created temporary directory for ensemble: {temp_dir}")

            try:
                # Store paths of intermediate stems grouped by stem name
                # { "Vocals": ["temp_dir/model1_Vocals.wav", "temp_dir/model2_Vocals.wav"], ... }
                stems_by_type = {}
                original_output_dir = self.output_dir

                for model_filename in original_model_filenames:
                    self.logger.info(f"Processing with model: {model_filename}")

                    # Load the model
                    self.load_model(model_filename)

                    # Set temporary output directory for this model
                    self.output_dir = temp_dir
                    if self.model_instance:
                        self.model_instance.output_dir = temp_dir

                    try:
                        # Perform separation WITHOUT custom_output_names for intermediate files.
                        # Intermediate stems must use the default "base_(StemType)_model.ext" naming
                        # so the regex below can extract stem types for classification.
                        # custom_output_names is applied later to the final ensembled output.
                        model_stems = self._separate_file(path, None)

                        # Extract and normalize stem names from this model's outputs
                        model_stem_names = []
                        for stem_path in model_stems:
                            filename = os.path.basename(stem_path)
                            match = re.search(r'_\(([^)]+)\)', filename)
                            stem_name = match.group(1) if match else "Unknown"
                            model_stem_names.append(stem_name)

                        # Normalize stem names with context about how many stems this model produced
                        num_model_stems = len(model_stem_names)
                        has_vocal_stem = any(
                            "vocal" in s.lower() or s.lower() in ("vocals",)
                            for s in model_stem_names
                        )

                        for stem_path, raw_stem_name in zip(model_stems, model_stem_names):
                            lower_name = raw_stem_name.lower()

                            if "vocal" in lower_name and "lead" not in lower_name and "backing" not in lower_name:
                                stem_name = "Vocals"
                            elif lower_name == "other" and num_model_stems == 2 and has_vocal_stem:
                                # For 2-stem models where one stem is vocals, "other" is the instrumental
                                stem_name = "Instrumental"
                                self.logger.debug(f"Mapped 'other' → 'Instrumental' for 2-stem model (model produced: {model_stem_names})")
                            elif lower_name in STEM_NAME_MAP:
                                stem_name = STEM_NAME_MAP[lower_name]
                            else:
                                stem_name = raw_stem_name.title()

                            if stem_name not in stems_by_type:
                                stems_by_type[stem_name] = []

                            abs_path = stem_path if os.path.isabs(stem_path) else os.path.join(temp_dir, stem_path)
                            stems_by_type[stem_name].append(abs_path)
                    finally:
                        self.output_dir = original_output_dir

                # Perform ensembling for each stem type
                ensembler = Ensembler(self.logger, self.ensemble_algorithm, self.ensemble_weights)
                base_name = os.path.splitext(os.path.basename(path))[0]

                for stem_name, stem_paths in stems_by_type.items():
                    self.logger.info(f"Ensembling {len(stem_paths)} stems for type: {stem_name}")

                    waveforms = []
                    original_channels = None
                    for sp in stem_paths:
                        wav, _ = librosa.load(sp, mono=False, sr=self.sample_rate)
                        if wav.ndim == 1:
                            if original_channels is None:
                                original_channels = 1
                            wav = np.asfortranarray([wav, wav])
                        elif original_channels is None:
                            original_channels = wav.shape[0]
                        waveforms.append(wav)

                    ensembled_wav = ensembler.ensemble(waveforms)

                    # Restore original channel count (avoid fake stereo from mono input)
                    if original_channels == 1 and ensembled_wav.shape[0] > 1:
                        ensembled_wav = ensembled_wav[:1, :]

                    # Determine output filename
                    if custom_output_names and stem_name in custom_output_names:
                        output_filename = custom_output_names[stem_name]
                    elif self.ensemble_preset:
                        output_filename = f"{base_name}_({stem_name})_preset_{self.ensemble_preset}"
                    else:
                        # Build descriptive name from model filenames
                        model_slugs = []
                        for mf in original_model_filenames:
                            # Remove extension, then truncate to keep filenames reasonable
                            name = os.path.splitext(mf)[0]
                            # Remove common verbose prefixes
                            for prefix in ["mel_band_roformer_", "melband_roformer_", "bs_roformer_", "model_bs_roformer_", "UVR-MDX-NET-", "UVR_MDXNET_"]:
                                if name.startswith(prefix):
                                    name = name[len(prefix):]
                                    break
                            model_slugs.append(name[:12])
                        slugs_str = "_".join(model_slugs)
                        output_filename = f"{base_name}_({stem_name})_custom_ensemble_{slugs_str}"

                    output_path = f"{output_filename}.{self.output_format.lower()}"

                    # Use a dummy model instance to write the audio if necessary,
                    # or just use the last model instance we had.
                    # Actually, we can use the write_audio method from the last model_instance
                    if self.model_instance:
                        # Ensure the model instance has the correct audio_file_path and output_dir
                        self.model_instance.audio_file_path = path
                        self.model_instance.output_dir = self.output_dir
                        self.model_instance.write_audio(output_path, ensembled_wav.T)
                        final_output_path = os.path.join(self.output_dir, output_path)
                        output_files.append(final_output_path)
                    else:
                        # Fallback writer if no model instance is available
                        self.logger.warning(f"No model instance available to write ensembled audio. Using fallback writer for {output_path}")
                        final_output_path = os.path.join(self.output_dir, output_path)

                        import soundfile as sf
                        from audio_separator.separator.uvr_lib_v5 import spec_utils

                        ensemble_audio = validate_audio_source(ensembled_wav.T)
                        ensemble_audio = spec_utils.normalize(
                            wave=ensemble_audio,
                            max_peak=self.normalization_threshold,
                            min_peak=self.amplification_threshold,
                        )
                        try:
                            with atomic_output_path(final_output_path, "soundfile") as temp_output_path:
                                self.logger.debug(f"Attempting to write ensembled audio to {final_output_path}...")
                                sf.write(temp_output_path, ensemble_audio, self.sample_rate)
                        except AudioExportError as requested_format_error:
                            self.logger.error(
                                f"Error writing {self.output_format} format: {requested_format_error}. Falling back to WAV."
                            )
                            final_output_path = final_output_path.rsplit(".", 1)[0] + ".wav"
                            with atomic_output_path(final_output_path, "soundfile") as temp_output_path:
                                sf.write(temp_output_path, ensemble_audio, self.sample_rate)

                        output_files.append(final_output_path)

            finally:
                # Restore original model filenames state
                self.model_filename = original_model_filename
                self.model_filenames = original_model_filenames

                # Clear model instance reference
                self.model_instance = None

                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    self.logger.debug(f"Cleaning up temporary directory: {temp_dir}")
                    shutil.rmtree(temp_dir, ignore_errors=True)

        return output_files
