""" This file contains the CommonSeparator class, common to all architecture-specific Separator classes. """

from logging import Logger
import os
import re
import gc
import numpy as np
import librosa
import torch
from pydub import AudioSegment
import soundfile as sf
from audio_separator.separator.audio_io import (
    atomic_output_path,
    normalize_output_subtype,
    resolve_output_subtype,
    validate_audio_source,
)
from audio_separator.separator.exceptions import AudioExportError, InvalidAudioDataError
from audio_separator.separator.uvr_lib_v5 import spec_utils
from audio_separator.separator.execution_policy import FP32, resolve_execution_policy


class CommonSeparator:
    """
    This class contains the common methods and attributes common to all architecture-specific Separator classes.
    """

    ALL_STEMS = "All Stems"
    VOCAL_STEM = "Vocals"
    INST_STEM = "Instrumental"
    OTHER_STEM = "Other"
    BASS_STEM = "Bass"
    DRUM_STEM = "Drums"
    GUITAR_STEM = "Guitar"
    PIANO_STEM = "Piano"
    SYNTH_STEM = "Synthesizer"
    STRINGS_STEM = "Strings"
    WOODWINDS_STEM = "Woodwinds"
    BRASS_STEM = "Brass"
    WIND_INST_STEM = "Wind Inst"
    NO_OTHER_STEM = "No Other"
    NO_BASS_STEM = "No Bass"
    NO_DRUM_STEM = "No Drums"
    NO_GUITAR_STEM = "No Guitar"
    NO_PIANO_STEM = "No Piano"
    NO_SYNTH_STEM = "No Synthesizer"
    NO_STRINGS_STEM = "No Strings"
    NO_WOODWINDS_STEM = "No Woodwinds"
    NO_WIND_INST_STEM = "No Wind Inst"
    NO_BRASS_STEM = "No Brass"
    PRIMARY_STEM = "Primary Stem"
    SECONDARY_STEM = "Secondary Stem"
    LEAD_VOCAL_STEM = "lead_only"
    BV_VOCAL_STEM = "backing_only"
    LEAD_VOCAL_STEM_I = "with_lead_vocals"
    BV_VOCAL_STEM_I = "with_backing_vocals"
    LEAD_VOCAL_STEM_LABEL = "Lead Vocals"
    BV_VOCAL_STEM_LABEL = "Backing Vocals"
    NO_STEM = "No "

    STEM_PAIR_MAPPER = {VOCAL_STEM: INST_STEM, INST_STEM: VOCAL_STEM, LEAD_VOCAL_STEM: BV_VOCAL_STEM, BV_VOCAL_STEM: LEAD_VOCAL_STEM, PRIMARY_STEM: SECONDARY_STEM}

    NON_ACCOM_STEMS = (VOCAL_STEM, OTHER_STEM, BASS_STEM, DRUM_STEM, GUITAR_STEM, PIANO_STEM, SYNTH_STEM, STRINGS_STEM, WOODWINDS_STEM, BRASS_STEM, WIND_INST_STEM)

    def __init__(self, config):

        self.logger: Logger = config.get("logger")
        self.log_level: int = config.get("log_level")

        # Inferencing device / acceleration config
        self.torch_device = config.get("torch_device")
        self.requested_torch_device = self.torch_device
        self.torch_device_cpu = config.get("torch_device_cpu")
        self.torch_device_mps = config.get("torch_device_mps")
        self.onnx_execution_provider = config.get("onnx_execution_provider")

        # Model data
        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.model_data = config.get("model_data")

        # Output directory and format
        self.output_dir = config.get("output_dir")
        self.output_format = config.get("output_format")
        self.output_bitrate = config.get("output_bitrate")
        self.output_subtype = normalize_output_subtype(config.get("output_subtype", "AUTO"), self.output_format)

        # Functional options which are applicable to all architectures and the user may tweak to affect the output
        self.normalization_threshold = config.get("normalization_threshold")
        self.amplification_threshold = config.get("amplification_threshold")
        self.enable_denoise = config.get("enable_denoise")
        self.output_single_stem = config.get("output_single_stem")
        self.invert_using_spec = config.get("invert_using_spec")
        self.sample_rate = config.get("sample_rate")
        self.use_soundfile = config.get("use_soundfile")
        self.use_autocast = config.get("use_autocast", False)
        self.use_native_fp16 = config.get("use_native_fp16", False)
        self.use_torch_compile = config.get("use_torch_compile", False)
        self.uses_pytorch_inference = True
        self.is_native_fp16 = False
        self.effective_precision = FP32
        self.effective_torch_compile = False
        self._should_torch_compile = False
        self._execution_policy_resolved = False
        
        # Roformer-specific loading support
        self.roformer_loader = None
        self.is_roformer_model = self._detect_roformer_model()
        if self.is_roformer_model:
            self._initialize_roformer_loader()

        # Model specific properties

        # Check if model_data has a "training" key with "instruments" list
        self.primary_stem_name = None
        self.secondary_stem_name = None
        
        # Audio bit depth tracking for preserving input quality
        self.input_bit_depth = None
        self.input_subtype = None

        if "training" in self.model_data and "instruments" in self.model_data["training"]:
            instruments = self.model_data["training"]["instruments"]
            if instruments:
                target_instrument = self.model_data["training"].get("target_instrument")

                # When target_instrument is set and doesn't match instruments[0],
                # the model's prediction would be labeled with the wrong stem name.
                # Swap so primary_stem_name always matches the model's actual target output.
                if target_instrument and len(instruments) >= 2 and instruments[0] != target_instrument and instruments[1] == target_instrument:
                    self.logger.debug(f"Swapping stem names: target_instrument '{target_instrument}' doesn't match instruments[0] '{instruments[0]}'")
                    self.primary_stem_name = instruments[1]
                    self.secondary_stem_name = instruments[0]
                else:
                    self.primary_stem_name = instruments[0]
                    self.secondary_stem_name = instruments[1] if len(instruments) > 1 else self.secondary_stem(self.primary_stem_name)

        if self.primary_stem_name is None:
            self.primary_stem_name = self.model_data.get("primary_stem", "Vocals")
            self.secondary_stem_name = self.secondary_stem(self.primary_stem_name)

        self.is_karaoke = self.model_data.get("is_karaoke", False)
        self.is_bv_model = self.model_data.get("is_bv_model", False)
        self.bv_model_rebalance = self.model_data.get("is_bv_model_rebalanced", 0)

        self.logger.debug(f"Common params: model_name={self.model_name}, model_path={self.model_path}")
        self.logger.debug(f"Common params: output_dir={self.output_dir}, output_format={self.output_format}")
        self.logger.debug(f"Common params: normalization_threshold={self.normalization_threshold}, amplification_threshold={self.amplification_threshold}")
        self.logger.debug(f"Common params: enable_denoise={self.enable_denoise}, output_single_stem={self.output_single_stem}")
        self.logger.debug(f"Common params: invert_using_spec={self.invert_using_spec}, sample_rate={self.sample_rate}")

        self.logger.debug(f"Common params: primary_stem_name={self.primary_stem_name}, secondary_stem_name={self.secondary_stem_name}")
        self.logger.debug(f"Common params: is_karaoke={self.is_karaoke}, is_bv_model={self.is_bv_model}, bv_model_rebalance={self.bv_model_rebalance}")

        # File-specific variables which need to be cleared between processing different audio inputs
        self.audio_file_path = None
        self.audio_file_base = None

        self.primary_source = None
        self.secondary_source = None

        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None

        self.cached_sources_map = {}

    def resolve_execution_policy(self, model_family):
        """Resolve the requested precision and compilation settings for this model."""
        policy = resolve_execution_policy(
            device=self.torch_device,
            requested_device=self.requested_torch_device,
            model_family=model_family,
            use_autocast=self.use_autocast,
            use_native_fp16=self.use_native_fp16,
            use_torch_compile=self.use_torch_compile,
            uses_pytorch_inference=getattr(self, "uses_pytorch_inference", True),
            logger=self.logger,
        )
        self.effective_precision = policy.precision
        self.effective_torch_compile = False
        self._should_torch_compile = policy.use_torch_compile
        self._execution_policy_resolved = True
        return policy

    def secondary_stem(self, primary_stem: str):
        """Determines secondary stem name based on the primary stem name."""
        primary_stem = primary_stem if primary_stem else self.NO_STEM

        if primary_stem in self.STEM_PAIR_MAPPER:
            secondary_stem = self.STEM_PAIR_MAPPER[primary_stem]
        else:
            secondary_stem = primary_stem.replace(self.NO_STEM, "") if self.NO_STEM in primary_stem else f"{self.NO_STEM}{primary_stem}"

        return secondary_stem

    def separate(self, audio_file_path):
        """
        Placeholder method for separating audio sources. Should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def final_process(self, stem_path, source, stem_name):
        """
        Finalizes the processing of a stem by writing the audio to a file and returning the processed source.
        """
        self.logger.debug(f"Finalizing {stem_name} stem processing and writing audio...")
        self.write_audio(stem_path, source)

        return {stem_name: source}

    def cached_sources_clear(self):
        """
        Clears the cache dictionaries for VR, MDX, and Demucs models.

        This function is essential for ensuring that the cache does not hold outdated or irrelevant data
        between different processing sessions or when a new batch of audio files is processed.
        It helps in managing memory efficiently and prevents potential errors due to stale data.
        """
        self.cached_sources_map = {}

    def cached_source_callback(self, model_architecture, model_name=None):
        """
        Retrieves the model and sources from the cache based on the processing method and model name.

        Args:
            model_architecture: The architecture type (VR, MDX, or Demucs) being used for processing.
            model_name: The specific model name within the architecture type, if applicable.

        Returns:
            A tuple containing the model and its sources if found in the cache; otherwise, None.

        This function is crucial for optimizing performance by avoiding redundant processing.
        If the requested model and its sources are already in the cache, they can be reused directly,
        saving time and computational resources.
        """
        model, sources = None, None

        mapper = self.cached_sources_map[model_architecture]

        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value

        return model, sources

    def cached_model_source_holder(self, model_architecture, sources, model_name=None):
        """
        Update the dictionary for the given model_architecture with the new model name and its sources.
        Use the model_architecture as a key to access the corresponding cache source mapper dictionary.
        """
        self.cached_sources_map[model_architecture] = {**self.cached_sources_map.get(model_architecture, {}), **{model_name: sources}}

    def prepare_mix(self, mix):
        """
        Prepares the mix for processing. This includes loading the audio from a file if necessary,
        ensuring the mix is in the correct format, and converting mono to stereo if needed.
        """
        # Store the original path or the mix itself for later checks
        audio_path = mix

        # Check if the input is a file path (string) and needs to be loaded
        if not isinstance(mix, np.ndarray):
            self.logger.debug(f"Loading audio from file: {mix}")
            
            # Get audio file info to capture bit depth before loading
            try:
                audio_info = sf.info(mix)
                self.input_subtype = audio_info.subtype
                self.logger.info(f"Input audio subtype: {self.input_subtype}")
                
                # Map subtype to bit depth
                if 'PCM_16' in self.input_subtype or self.input_subtype == 'PCM_S8':
                    self.input_bit_depth = 16
                elif 'PCM_24' in self.input_subtype:
                    self.input_bit_depth = 24
                elif 'PCM_32' in self.input_subtype or 'FLOAT' in self.input_subtype or 'DOUBLE' in self.input_subtype:
                    self.input_bit_depth = 32
                else:
                    # Default to 16-bit for unknown formats
                    self.input_bit_depth = 16
                    self.logger.warning(f"Unknown audio subtype {self.input_subtype}, defaulting to 16-bit output")
                
                self.logger.info(f"Detected input bit depth: {self.input_bit_depth}-bit")
            except Exception as e:
                self.logger.warning(f"Could not read audio file info, defaulting to 16-bit output: {e}")
                self.input_bit_depth = 16
                self.input_subtype = 'PCM_16'
            
            mix, sr = librosa.load(mix, mono=False, sr=self.sample_rate)
            self.logger.debug(f"Audio loaded. Sample rate: {sr}, Audio shape: {mix.shape}")
        else:
            # Transpose the mix if it's already an ndarray (expected shape: [channels, samples])
            self.logger.debug("Transposing the provided mix array.")
            # Default to 16-bit if numpy array provided directly
            if self.input_bit_depth is None:
                self.input_bit_depth = 16
                self.input_subtype = 'PCM_16'
            mix = mix.T
            self.logger.debug(f"Transposed mix shape: {mix.shape}")

        source_description = "Audio array" if isinstance(audio_path, np.ndarray) else f"Audio file {audio_path}"
        if mix.size == 0:
            error_msg = f"{source_description} contains no audio frames"
            self.logger.error(error_msg)
            raise InvalidAudioDataError(error_msg)
        elif not np.isfinite(mix).all():
            error_msg = f"{source_description} contains non-finite samples"
            self.logger.error(error_msg)
            raise InvalidAudioDataError(error_msg)
        else:
            self.logger.debug("Audio input is valid and contains audio frames.")

        # Ensure the mix is in stereo format
        if mix.ndim == 1:
            self.logger.debug("Mix is mono. Converting to stereo.")
            mix = np.asfortranarray([mix, mix])
            self.logger.debug("Converted to stereo mix.")

        # Final log indicating successful preparation of the mix
        self.logger.debug("Mix preparation completed.")
        return mix

    def write_audio(self, stem_path: str, stem_source):
        """
        Writes the separated audio source to a file using pydub or soundfile
        Pydub supports a much wider range of audio formats and produces better encoded lossy files for some formats.
        Soundfile is used for very large files (longer than 1 hour), as pydub has memory issues with large files:
        https://github.com/jiaaro/pydub/issues/135
        """
        # Get the duration of the input audio file
        duration_seconds = librosa.get_duration(path=self.audio_file_path)
        duration_hours = duration_seconds / 3600
        self.logger.info(f"Audio duration is {duration_hours:.2f} hours ({duration_seconds:.2f} seconds).")

        if self.use_soundfile:
            self.logger.warning(f"Using soundfile for writing.")
            self.write_audio_soundfile(stem_path, stem_source)
        else:
            self.logger.info(f"Using pydub for writing.")
            self.write_audio_pydub(stem_path, stem_source)

    def write_audio_pydub(self, stem_path: str, stem_source):
        """
        Writes the separated audio source to a file using pydub (ffmpeg)
        """
        self.logger.debug(f"Entering write_audio_pydub with stem_path: {stem_path}")

        stem_source = validate_audio_source(stem_source)
        stem_source = spec_utils.normalize(wave=stem_source, max_peak=self.normalization_threshold, min_peak=self.amplification_threshold)

        # If output_dir is specified, create it and join it with stem_path
        if self.output_dir:
            stem_path = os.path.join(self.output_dir, stem_path)
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e:
                raise AudioExportError(f"Failed to prepare output directory for {stem_path}: {e}", path=stem_path, backend="pydub") from e

        self.logger.debug(f"Audio data shape before processing: {stem_source.shape}")
        self.logger.debug(f"Data type before conversion: {stem_source.dtype}")

        file_format = stem_path.lower().split(".")[-1]
        output_subtype = resolve_output_subtype(self.output_subtype, self.input_subtype, self.input_bit_depth, file_format)

        # Pydub starts from int16 samples, so higher-depth lossless output must
        # write the model's floating-point samples directly through soundfile.
        if file_format in ("wav", "flac") and output_subtype != "PCM_16":
            with atomic_output_path(stem_path, "soundfile") as temp_path:
                sf.write(temp_path, stem_source, self.sample_rate, subtype=output_subtype)
            self.logger.debug(f"Exported audio file successfully to {stem_path} with subtype {output_subtype}")
            return

        output_bit_depth = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}.get(output_subtype, 16)
        self.logger.info(f"Writing output with {output_bit_depth}-bit depth")

        # For pydub, we always convert to int16 for the AudioSegment creation
        # Then let ffmpeg handle the conversion to the target bit depth during export
        if stem_source.dtype != np.int16:
            stem_source = (stem_source * 32767).astype(np.int16)
            self.logger.debug("Converted stem_source to int16 for pydub processing.")

        channels = 1 if stem_source.ndim == 1 else stem_source.shape[1]
        stem_source_interleaved = np.ascontiguousarray(stem_source).reshape(-1)

        self.logger.debug(f"Interleaved audio data shape: {stem_source_interleaved.shape}")

        # Create a pydub AudioSegment (always from 16-bit data)
        try:
            audio_segment = AudioSegment(stem_source_interleaved.tobytes(), frame_rate=self.sample_rate, sample_width=2, channels=channels)
            self.logger.debug("Created AudioSegment successfully.")
        except Exception as e:
            raise AudioExportError(f"Failed to create audio for {stem_path} with pydub: {e}", path=stem_path, backend="pydub") from e

        # For m4a files, specify mp4 as the container format as the extension doesn't match the format name
        if file_format == "m4a":
            file_format = "mp4"
        elif file_format == "mka":
            file_format = "matroska"

        # Set the bitrate to 320k for mp3 files if output_bitrate is not specified
        bitrate = "320k" if file_format == "mp3" and self.output_bitrate is None else self.output_bitrate

        target_path = stem_path

        # Export using the determined format
        with atomic_output_path(target_path, "pydub") as temp_path:
            # Pass codec parameters to ffmpeg to enforce bit depth for lossless formats
            export_params = {"format": file_format}
            
            if bitrate:
                export_params["bitrate"] = bitrate
            
            # For lossless formats (WAV/FLAC), specify the codec parameters to enforce bit depth
            if file_format in ["wav", "flac"]:
                if output_bit_depth == 16:
                    export_params["parameters"] = ["-sample_fmt", "s16"]
                elif output_bit_depth == 24:
                    export_params["parameters"] = ["-sample_fmt", "s32"]
                    # For 24-bit, we also need to specify the bit depth explicitly
                    if file_format == "wav":
                        export_params["codec"] = "pcm_s24le"
                    elif file_format == "flac":
                        # FLAC supports 24-bit natively, no special handling needed
                        pass
                elif output_bit_depth == 32:
                    export_params["parameters"] = ["-sample_fmt", "s32"]
                    if file_format == "wav":
                        export_params["codec"] = "pcm_s32le"
            
            export_handle = audio_segment.export(temp_path, **export_params)
            close_export = getattr(export_handle, "close", None)
            if callable(close_export):
                close_export()
        self.logger.debug(f"Exported audio file successfully to {target_path} with {output_bit_depth}-bit depth")

    def write_audio_soundfile(self, stem_path: str, stem_source):
        """
        Writes the separated audio source to a file using soundfile library.
        """
        self.logger.debug(f"Entering write_audio_soundfile with stem_path: {stem_path}")

        stem_source = validate_audio_source(stem_source)
        stem_source = spec_utils.normalize(wave=stem_source, max_peak=self.normalization_threshold, min_peak=self.amplification_threshold)

        # If output_dir is specified, create it and join it with stem_path
        if self.output_dir:
            stem_path = os.path.join(self.output_dir, stem_path)
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e:
                raise AudioExportError(f"Failed to prepare output directory for {stem_path}: {e}", path=stem_path, backend="soundfile") from e

        file_format = stem_path.lower().split(".")[-1]
        output_subtype = resolve_output_subtype(self.output_subtype, self.input_subtype, self.input_bit_depth, file_format)
        self.logger.info(f"Using output subtype: {output_subtype}")

        # Correctly interleave stereo channels if needed
        if stem_source.ndim == 2 and stem_source.shape[1] == 2:
            # If the audio is already interleaved, ensure it's in the correct order
            # Check if the array is Fortran contiguous (column-major)
            if stem_source.flags["F_CONTIGUOUS"]:
                # Convert to C contiguous (row-major)
                stem_source = np.ascontiguousarray(stem_source)
            # No need to manually interleave for soundfile - it handles multi-channel properly
            # Just ensure we don't have the wrong shape

        self.logger.debug(f"Audio data shape for soundfile: {stem_source.shape}")

        """
        Write audio using soundfile (for formats other than M4A).
        """
        target_path = stem_path
        with atomic_output_path(target_path, "soundfile") as temp_path:
            # Specify the subtype to match input bit depth
            sf.write(temp_path, stem_source, self.sample_rate, subtype=output_subtype)
        self.logger.debug(f"Exported audio file successfully to {target_path} with subtype {output_subtype}")

    def clear_gpu_cache(self):
        """
        This method clears the GPU cache to free up memory.
        """
        self.logger.debug("Running garbage collection...")
        gc.collect()
        if self.torch_device == torch.device("mps"):
            self.logger.debug("Clearing MPS cache...")
            torch.mps.empty_cache()
        if self.torch_device == torch.device("cuda"):
            self.logger.debug("Clearing CUDA cache...")
            torch.cuda.empty_cache()

    def clear_file_specific_paths(self):
        """
        Clears the file-specific variables which need to be cleared between processing different audio inputs.
        """
        self.logger.info("Clearing input audio file paths, sources and stems...")

        self.audio_file_path = None
        self.audio_file_base = None

        self.primary_source = None
        self.secondary_source = None

        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None

    def sanitize_filename(self, filename):
        """
        Cleans the filename by replacing invalid characters with underscores.
        """
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_. ')
        return sanitized

    def get_stem_output_path(self, stem_name, custom_output_names):
        """
        Gets the output path for a stem based on the stem name and custom output names.
        """
        # Convert custom_output_names keys to lowercase for case-insensitive comparison
        if custom_output_names:
            custom_output_names_lower = {k.lower(): v for k, v in custom_output_names.items()}
            stem_name_lower = stem_name.lower()
            if stem_name_lower in custom_output_names_lower:
                sanitized_custom_name = self.sanitize_filename(custom_output_names_lower[stem_name_lower])
                return os.path.join(f"{sanitized_custom_name}.{self.output_format.lower()}")

        sanitized_audio_base = self.sanitize_filename(self.audio_file_base)
        sanitized_stem_name = self.sanitize_filename(stem_name)
        sanitized_model_name = self.sanitize_filename(self.model_name) 

        filename = f"{sanitized_audio_base}_({sanitized_stem_name})_{sanitized_model_name}.{self.output_format.lower()}"
        return os.path.join(filename)
    
    def _detect_roformer_model(self):
        """
        Detect if the current model is a Roformer model.
        
        Returns:
            bool: True if this is a Roformer model, False otherwise
        """
        if not self.model_data:
            return False
            
        # Check for explicit Roformer flag
        if self.model_data.get("is_roformer", False):
            return True

        # Prefer architecture-defining configuration over renameable paths.
        from .roformer.configuration_normalizer import ConfigurationNormalizer

        if ConfigurationNormalizer().detect_model_type(self.model_data) is not None:
            return True
            
        # Keep filename-based compatibility without letting parent directories
        # influence architecture routing.
        if self.model_path and "roformer" in os.path.basename(self.model_path).lower():
            return True
            
        # Check model name for Roformer indicators
        if self.model_name and "roformer" in self.model_name.lower():
            return True
            
        return False
    
    def _initialize_roformer_loader(self):
        """
        Initialize the Roformer loader for this model.
        """
        try:
            from .roformer.roformer_loader import RoformerLoader
            self.roformer_loader = RoformerLoader()
            self.logger.debug("Initialized Roformer loader for CommonSeparator")
        except ImportError as e:
            self.logger.warning(f"Could not import RoformerLoader: {e}")
            self.roformer_loader = None
    
    def get_roformer_loading_stats(self):
        """
        Get Roformer loading statistics if available.
        
        Returns:
            dict: Loading statistics or empty dict if not available
        """
        if self.roformer_loader:
            return self.roformer_loader.get_loading_stats()
        return {}
    
    def validate_roformer_config(self, config, model_type):
        """
        Validate Roformer configuration if loader is available.
        
        Args:
            config: Configuration dictionary to validate
            model_type: Type of model to validate for
            
        Returns:
            bool: True if valid or validation not available, False if invalid
        """
        if self.roformer_loader:
            return self.roformer_loader.validate_configuration(config, model_type)
        return True  # Assume valid if no loader available
