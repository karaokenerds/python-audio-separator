"""Module for separating audio sources using MDX architecture models."""

import os
import platform
import torch
import onnx
import onnxruntime as ort
import numpy as np
import onnx2torch
from tqdm import tqdm
from audio_separator.separator.uvr_lib_v5 import spec_utils
from audio_separator.separator.uvr_lib_v5.stft import STFT
from audio_separator.separator.common_separator import CommonSeparator


class MDXSeparator(CommonSeparator):
    """
    MDXSeparator is responsible for separating audio sources using MDX models.
    It initializes with configuration parameters and prepares the model for separation tasks.
    """

    def __init__(self, common_config, arch_config):
        # Any configuration values which can be shared between architectures should be set already in CommonSeparator,
        # e.g. user-specified functionality choices (self.output_single_stem) or common model parameters (self.primary_stem_name)
        super().__init__(config=common_config)

        # Initializing user-configurable parameters, passed through with an mdx_from the CLI or Separator instance

        # Pick a segment size to balance speed, resource use, and quality:
        # - Smaller sizes consume less resources.
        # - Bigger sizes consume more resources, but may provide better results.
        # - Default size is 256. Quality can change based on your pick.
        self.segment_size = arch_config.get("segment_size")

        # This option controls the amount of overlap between prediction windows.
        #  - Higher values can provide better results, but will lead to longer processing times.
        #  - For Non-MDX23C models: You can choose between 0.001-0.999
        self.overlap = arch_config.get("overlap")

        # Number of batches to be processed at a time.
        # - Higher values mean more RAM usage but slightly faster processing times.
        # - Lower values mean less RAM usage but slightly longer processing times.
        # - Batch size value has no effect on output quality.
        # BATCH_SIZE = ('1', ''2', '3', '4', '5', '6', '7', '8', '9', '10')
        self.batch_size = arch_config.get("batch_size", 1)

        # hop_length is equivalent to the more commonly used term "stride" in convolutional neural networks
        # In machine learning, particularly in the context of convolutional neural networks (CNNs),
        # the term "stride" refers to the number of pixels by which we move the filter across the input image.
        # Strides are a crucial component in the convolution operation, a fundamental building block of CNNs used primarily in the field of computer vision.
        # Stride is a parameter that dictates the movement of the kernel, or filter, across the input data, such as an image.
        # When performing a convolution operation, the stride determines how many units the filter shifts at each step.
        # The choice of stride affects the model in several ways:
        # Output Size: A larger stride will result in a smaller output spatial dimension.
        # Computational Efficiency: Increasing the stride can decrease the computational load.
        # Field of View: A higher stride means that each step of the filter takes into account a wider area of the input image.
        #   This can be beneficial when the model needs to capture more global features rather than focusing on finer details.
        self.hop_length = arch_config.get("hop_length")

        # If enabled, model will be run twice to reduce noise in output audio.
        self.enable_denoise = arch_config.get("enable_denoise")

        self.logger.debug(f"MDX arch params: batch_size={self.batch_size}, segment_size={self.segment_size}")
        self.logger.debug(f"MDX arch params: overlap={self.overlap}, hop_length={self.hop_length}, enable_denoise={self.enable_denoise}")

        # Initializing model-specific parameters from model_data JSON
        self.compensate = self.model_data["compensate"]
        self.dim_f = self.model_data["mdx_dim_f_set"]
        self.dim_t = 2 ** self.model_data["mdx_dim_t_set"]
        self.n_fft = self.model_data["mdx_n_fft_scale_set"]
        self.config_yaml = self.model_data.get("config_yaml", None)

        self.logger.debug(f"MDX arch params: compensate={self.compensate}, dim_f={self.dim_f}, dim_t={self.dim_t}, n_fft={self.n_fft}")
        self.logger.debug(f"MDX arch params: config_yaml={self.config_yaml}")

        # In UVR, these variables are set but either aren't useful or are better handled in audio-separator.
        # Leaving these comments explaining to help myself or future developers understand why these aren't in audio-separator.

        # "chunks" is not actually used for anything in UVR...
        # self.chunks = 0

        # "adjust" is hard-coded to 1 in UVR, and only used as a multiplier in run_model, so it does nothing.
        # self.adjust = 1

        # "hop" is hard-coded to 1024 in UVR. We have a "hop_length" parameter instead
        # self.hop = 1024

        # "margin" maps to sample rate and is set from the GUI in UVR (default: 44100). We have a "sample_rate" parameter instead.
        # self.margin = 44100

        # "dim_c" is hard-coded to 4 in UVR, seems to be a parameter for the number of channels, and is only used for checkpoint models.
        # We haven't implemented support for the checkpoint models here, so we're not using it.
        # self.dim_c = 4

        self.load_model()

        self.n_bins = 0
        self.trim = 0
        self.chunk_size = 0
        self.gen_size = 0
        self.stft = None

        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None

    def load_model(self):
        """
        Load the model into memory from file on disk, initialize it with config from the model data,
        and prepare for inferencing using hardware accelerated Torch device.
        """
        self.logger.debug("Loading ONNX model for inference...")

        if self.segment_size == self.dim_t:
            self.uses_pytorch_inference = False
            ort_session_options = ort.SessionOptions()
            if self.log_level > 10:
                ort_session_options.log_severity_level = 3
            else:
                ort_session_options.log_severity_level = 0

            ort_inference_session = ort.InferenceSession(self.model_path, providers=self.onnx_execution_provider, sess_options=ort_session_options)
            session_providers = ort_inference_session.get_providers()

            requested_provider = self.onnx_execution_provider[0] if self.onnx_execution_provider else None
            if isinstance(requested_provider, tuple):
                requested_provider = requested_provider[0]
            if requested_provider and requested_provider not in session_providers:
                self.logger.warning(
                    f"ONNX Runtime could not activate requested provider {requested_provider}; "
                    f"session is using {session_providers}. This usually means required CUDA/cuDNN "
                    f"runtime libraries are not visible to the dynamic loader."
                )
            else:
                self.logger.debug(f"ONNX Runtime session providers: {session_providers}")

            self.model_run = lambda spek: ort_inference_session.run(None, {"input": spek.cpu().numpy()})[0]
            self.logger.debug("Model loaded successfully using ONNXruntime inferencing session.")
        else:
            if platform.system() == 'Windows':
                onnx_model = onnx.load(self.model_path)
                self.model_run = onnx2torch.convert(onnx_model)
            else:
                self.model_run = onnx2torch.convert(self.model_path)
   
            self.model_run.to(self.torch_device).eval()
            self.logger.warning("Model converted from onnx to pytorch due to segment size not matching dim_t, processing may be slower.")

    def separate(self, audio_file_path, custom_output_names=None):
        """
        Separates the audio file into primary and secondary sources based on the model's configuration.
        It processes the mix, demixes it into sources, normalizes the sources, and saves the output files.

        Args:
            audio_file_path (str): The path to the audio file to be processed.
            custom_output_names (dict, optional): Custom names for the output files. Defaults to None.

        Returns:
            list: A list of paths to the output files generated by the separation process.
        """
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        # Prepare the mix for processing
        self.logger.debug(f"Preparing mix for input audio file {self.audio_file_path}...")
        mix = self.prepare_mix(self.audio_file_path)

        self.logger.debug("Normalizing mix before demixing...")
        peak = np.abs(mix).max()
        mix = spec_utils.normalize(wave=mix, max_peak=self.normalization_threshold, min_peak=self.amplification_threshold)

        # Start the demixing process
        source = self.demix(mix) * peak
        self.logger.debug("Demixing completed.")


        if not isinstance(self.primary_source, np.ndarray):
            self.primary_source = source.T

        # In UVR, the source is cached here if it's a vocal split model, but we're not supporting that yet

        # Initialize the list for output files
        output_files = []
        self.logger.debug("Processing output files...")

        # Process the secondary source if not already an array
        if not isinstance(self.secondary_source, np.ndarray):
            self.logger.debug("Producing secondary source: demixing in match_mix mode")
            raw_mix = self.demix(mix, is_match_mix=True)

            if self.invert_using_spec:
                self.logger.debug("Inverting secondary stem using spectogram as invert_using_spec is set to True")
                self.secondary_source = spec_utils.invert_stem(raw_mix, self.primary_source * self.compensate)
            else:
                self.logger.debug("Inverting secondary stem by subtracting of transposed demixed stem from transposed original mix")
                self.secondary_source = (-self.primary_source * self.compensate) + mix.T

        # Save and process the secondary stem if needed
        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            self.secondary_stem_output_path = self.get_stem_output_path(self.secondary_stem_name, custom_output_names)

            self.logger.info(f"Saving {self.secondary_stem_name} stem to {self.secondary_stem_output_path}...")
            self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
            output_files.append(self.secondary_stem_output_path)

        # Save and process the primary stem if needed
        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            self.primary_stem_output_path = self.get_stem_output_path(self.primary_stem_name, custom_output_names)
            self.logger.info(f"Saving {self.primary_stem_name} stem to {self.primary_stem_output_path}...")
            self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)
            output_files.append(self.primary_stem_output_path)

        # Not yet implemented from UVR features:
        # self.process_vocal_split_chain(secondary_sources)
        # self.logger.debug("Vocal split chain processed.")

        return output_files

    def initialize_model_settings(self):
        """
        This function sets up the necessary parameters for the model, like the number of frequency bins (n_bins), the trimming size (trim),
        the size of each audio chunk (chunk_size), and the window function for spectral transformations (window).
        It ensures that the model is configured with the correct settings for processing the audio data.
        """
        self.logger.debug("Initializing model settings...")

        # n_bins is half the FFT size plus one (self.n_fft // 2 + 1).
        self.n_bins = self.n_fft // 2 + 1

        # trim is half the FFT size (self.n_fft // 2).
        self.trim = self.n_fft // 2

        # chunk_size is the hop_length size times the segment size minus one
        self.chunk_size = self.hop_length * (self.segment_size - 1)

        # gen_size is the chunk size minus twice the trim size
        self.gen_size = self.chunk_size - 2 * self.trim

        self.stft = STFT(self.logger, self.n_fft, self.hop_length, self.dim_f, self.torch_device)

        self.logger.debug(f"Model input params: n_fft={self.n_fft} hop_length={self.hop_length} dim_f={self.dim_f}")
        self.logger.debug(f"Model settings: n_bins={self.n_bins}, trim={self.trim}, chunk_size={self.chunk_size}, gen_size={self.gen_size}")

    def initialize_mix(self, mix, is_ckpt=False):
        """
        After prepare_mix segments the audio, initialize_mix further processes each segment.
        It ensures each audio segment is in the correct format for the model, applies necessary padding,
        and converts the segments into tensors for processing with the model.
        This step is essential for preparing the audio data in a format that the neural network can process.
        """
        # Log the initialization of the mix and whether checkpoint mode is used
        self.logger.debug(f"Initializing mix with is_ckpt={is_ckpt}. Initial mix shape: {mix.shape}")

        # Ensure the mix is a 2-channel (stereo) audio signal
        if mix.shape[0] != 2:
            error_message = f"Expected a 2-channel audio signal, but got {mix.shape[0]} channels"
            self.logger.error(error_message)
            raise ValueError(error_message)

        # If in checkpoint mode, process the mix differently
        if is_ckpt:
            self.logger.debug("Processing in checkpoint mode...")
            # Calculate padding based on the generation size and trim
            pad = self.gen_size + self.trim - (mix.shape[-1] % self.gen_size)
            self.logger.debug(f"Padding calculated: {pad}")
            # Add padding at the beginning and the end of the mix
            mixture = np.concatenate(
                (
                    np.zeros((2, self.trim), dtype="float32"),  # Pad at the start
                    mix,
                    np.zeros((2, pad), dtype="float32"),        # Pad in the middle (to match chunk size)
                    np.zeros((2, self.trim), dtype="float32"),  # Pad at the end
                ),
                1
            )
            # Determine the number of chunks based on the mixture's length
            num_chunks = mixture.shape[-1] // self.gen_size
            self.logger.debug(f"Mixture shape after padding: {mixture.shape}, Number of chunks: {num_chunks}")
            # Split the mixture into chunks
            mix_waves = [mixture[:, i * self.gen_size : i * self.gen_size + self.chunk_size] for i in range(num_chunks)]
        else:
            # If not in checkpoint mode, process normally
            self.logger.debug("Processing in non-checkpoint mode...")
            mix_waves = []
            n_sample = mix.shape[1]
            # Calculate necessary padding to make the total length divisible by the generation size
            pad = self.gen_size - n_sample % self.gen_size
            self.logger.debug(f"Number of samples: {n_sample}, Padding calculated: {pad}")
            # Apply padding to the mix
            mix_p = np.concatenate((np.zeros((2, self.trim)), mix, np.zeros((2, pad)), np.zeros((2, self.trim))), 1)
            self.logger.debug(f"Shape of mix after padding: {mix_p.shape}")

            # Process the mix in chunks
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + self.chunk_size])
                mix_waves.append(waves)
                self.logger.debug(f"Processed chunk {len(mix_waves)}: Start {i}, End {i + self.chunk_size}")
                i += self.gen_size

        # Convert the list of wave chunks into a tensor for processing on the specified device
        mix_waves_tensor = torch.tensor(mix_waves, dtype=torch.float32).to(self.torch_device)
        self.logger.debug(f"Converted mix_waves to tensor. Tensor shape: {mix_waves_tensor.shape}")

        return mix_waves_tensor, pad

    def demix(self, mix, is_match_mix=False):
        """
        Demixes the input mix into its constituent sources. If is_match_mix is True, the function adjusts the processing
        to better match the mix, affecting chunk sizes and overlaps. The demixing process involves padding the mix,
        processing it in chunks, applying windowing for overlaps, and accumulating the results to separate the sources.
        """
        self.logger.debug(f"Starting demixing process with is_match_mix: {is_match_mix}...")
        self.initialize_model_settings()

        # Preserves the original mix for later use.
        # In UVR, this is used for the pitch fix and VR denoise processes, which aren't yet implemented here.
        org_mix = mix
        self.logger.debug(f"Original mix stored. Shape: {org_mix.shape}")

        # Initializes a list to store the separated waveforms.
        tar_waves_ = []

        # Handling different chunk sizes and overlaps based on the matching requirement.
        if is_match_mix:
            # Sets a smaller chunk size specifically for matching the mix.
            chunk_size = self.hop_length * (self.segment_size - 1)
            # Sets a small overlap for the chunks.
            overlap = 0.02
            self.logger.debug(f"Chunk size for matching mix: {chunk_size}, Overlap: {overlap}")
        else:
            # Uses the regular chunk size defined in model settings.
            chunk_size = self.chunk_size
            # Uses the overlap specified in the model settings.
            overlap = self.overlap
            self.logger.debug(f"Standard chunk size: {chunk_size}, Overlap: {overlap}")

        # Calculates the generated size after subtracting the trim from both ends of the chunk.
        gen_size = chunk_size - 2 * self.trim
        self.logger.debug(f"Generated size calculated: {gen_size}")

        # Calculates padding to make the mix length a multiple of the generated size.
        pad = gen_size + self.trim - ((mix.shape[-1]) % gen_size)
        # Prepares the mixture with padding at the beginning and the end.
        mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")), 1)
        self.logger.debug(f"Mixture prepared with padding. Mixture shape: {mixture.shape}")

        # Calculates the step size for processing chunks based on the overlap.
        step = int((1 - overlap) * chunk_size)
        self.logger.debug(f"Step size for processing chunks: {step} as overlap is set to {overlap}.")

        # Initializes arrays to store the results and to account for overlap.
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)

        # Initializes counters for processing chunks.
        total = 0
        total_chunks = (mixture.shape[-1] + step - 1) // step
        self.logger.debug(f"Total chunks to process: {total_chunks}")

        # Processes each chunk of the mixture.
        for i in tqdm(range(0, mixture.shape[-1], step)):
            total += 1
            start = i
            end = min(i + chunk_size, mixture.shape[-1])
            self.logger.debug(f"Processing chunk {total}/{total_chunks}: Start {start}, End {end}")

            # Handles windowing for overlapping chunks.
            chunk_size_actual = end - start
            window = None
            if overlap != 0:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))
                self.logger.debug("Window applied to the chunk.")

            # Zero-pad the chunk to prepare it for processing.
            mix_part_ = mixture[:, start:end]
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype="float32")), axis=-1)

            # Converts the chunk to a tensor for processing.
            mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(self.torch_device)
            # Splits the chunk into smaller batches if necessary.
            mix_waves = mix_part.split(self.batch_size)
            total_batches = len(mix_waves)
            self.logger.debug(f"Mix part split into batches. Number of batches: {total_batches}")

            with torch.no_grad():
                # Processes each batch in the chunk.
                batches_processed = 0
                for mix_wave in mix_waves:
                    batches_processed += 1
                    self.logger.debug(f"Processing mix_wave batch {batches_processed}/{total_batches}")

                    # Runs the model to separate the sources.
                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)

                    # Applies windowing if needed and accumulates the results.
                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window
                        divider[..., start:end] += window
                    else:
                        divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., : end - start]

        # Normalizes the results by the divider to account for overlap.
        self.logger.debug("Normalizing result by dividing result by divider.")
        tar_waves = result / divider
        tar_waves_.append(tar_waves)

        # Reshapes the results to match the original dimensions.
        tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim : -self.trim]
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, : mix.shape[-1]]

        # Extracts the source from the results.
        source = tar_waves[:, 0:None]
        self.logger.debug(f"Concatenated tar_waves. Shape: {tar_waves.shape}")

        # TODO: In UVR, pitch changing happens here. Consider implementing this as a feature.

        # TODO: In UVR, VR denoise model gets applied here. Consider implementing this as a feature.

        self.logger.debug("Demixing process completed.")
        return source

    def run_model(self, mix, is_match_mix=False):
        """
        Processes the input mix through the model to separate the sources.
        Applies STFT, handles spectrum modifications, and runs the model for source separation.
        """
        # Applying the STFT to the mix. The mix is moved to the specified device (e.g., GPU) before processing.
        # self.logger.debug(f"Running STFT on the mix. Mix shape before STFT: {mix.shape}")
        spek = self.stft(mix.to(self.torch_device))
        self.logger.debug(f"STFT applied on mix. Spectrum shape: {spek.shape}")

        # Zeroing out the first 3 bins of the spectrum. This is often done to reduce low-frequency noise.
        spek[:, :, :3, :] *= 0
        # self.logger.debug("First 3 bins of the spectrum zeroed out.")

        # Handling the case where the mix needs to be matched (is_match_mix = True)
        if is_match_mix:
            # self.logger.debug("Match mix mode is enabled. Converting spectrum to NumPy array.")
            spec_pred = spek.cpu().numpy()
            self.logger.debug("is_match_mix: spectrum prediction obtained directly from STFT output.")
        else:
            # If denoising is enabled, the model is run on both the negative and positive spectrums.
            if self.enable_denoise:
                # Assuming spek is a tensor and self.model_run can process it directly
                spec_pred_neg = self.model_run(-spek)  # Ensure this line correctly negates spek and runs the model
                spec_pred_pos = self.model_run(spek)
                # Ensure both spec_pred_neg and spec_pred_pos are tensors before applying operations
                spec_pred = (spec_pred_neg * -0.5) + (spec_pred_pos * 0.5)  # [invalid-unary-operand-type]
                self.logger.debug("Model run on both negative and positive spectrums for denoising.")
            else:
                spec_pred = self.model_run(spek)
                self.logger.debug("Model run on the spectrum without denoising.")

        # Applying the inverse STFT to convert the spectrum back to the time domain.
        result = self.stft.inverse(torch.tensor(spec_pred).to(self.torch_device)).cpu().detach().numpy()
        self.logger.debug(f"Inverse STFT applied. Returning result with shape: {result.shape}")

        return result
