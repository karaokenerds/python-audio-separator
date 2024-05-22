import os
import sys

import torch
import numpy as np
from tqdm import tqdm
from ml_collections import ConfigDict
from scipy import signal

from audio_separator.separator.common_separator import CommonSeparator
from audio_separator.separator.uvr_lib_v5 import spec_utils
from audio_separator.separator.uvr_lib_v5.tfc_tdf_v3 import TFC_TDF_net
from audio_separator.separator.uvr_lib_v5.mel_band_roformer import MelBandRoformer
from audio_separator.separator.uvr_lib_v5.bs_roformer import BSRoformer


class MDXCSeparator(CommonSeparator):
    """
    MDXCSeparator is responsible for separating audio sources using MDXC models.
    It initializes with configuration parameters and prepares the model for separation tasks.
    """

    def __init__(self, common_config, arch_config):
        # Any configuration values which can be shared between architectures should be set already in CommonSeparator,
        # e.g. user-specified functionality choices (self.output_single_stem) or common model parameters (self.primary_stem_name)
        super().__init__(config=common_config)

        # Model data is basic overview metadata about the model, e.g. which stem is primary and whether it's a karaoke model
        # It's loaded in from model_data_new.json in Separator.load_model and there are JSON examples in that method
        # The instance variable self.model_data is passed through from Separator and set in CommonSeparator
        self.logger.debug(f"Model data: {self.model_data}")

        # Arch Config is the MDXC architecture specific user configuration options, which should all be configurable by the user
        # either by their Separator class instantiation or by passing in a CLI parameter.
        # While there are similarities between architectures for some of these (e.g. batch_size), they are deliberately configured
        # this way as they have architecture-specific default values.
        self.segment_size = arch_config.get("segment_size", 256)

        # Whether or not to use the segment size from model config, or the default
        # The segment size is set based on the value provided in a chosen model's associated config file (yaml).
        self.override_model_segment_size = arch_config.get("override_model_segment_size", False)

        self.overlap = arch_config.get("overlap", 8)
        self.batch_size = arch_config.get("batch_size", 1)

        # Amount of pitch shift to apply during processing (this does NOT affect the pitch of the output audio):
        # • Whole numbers indicate semitones.
        # • Using higher pitches may cut the upper bandwidth, even in high-quality models.
        # • Upping the pitch can be better for tracks with deeper vocals.
        # • Dropping the pitch may take more processing time but works well for tracks with high-pitched vocals.
        self.pitch_shift = arch_config.get("pitch_shift", 0)

        self.logger.debug(f"MDXC arch params: batch_size={self.batch_size}, segment_size={self.segment_size}, overlap={self.overlap}")
        self.logger.debug(f"MDXC arch params: override_model_segment_size={self.override_model_segment_size}, pitch_shift={self.pitch_shift}")

        self.is_roformer = "is_roformer" in self.model_data

        self.load_model()

        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None

        self.is_vocal_main_target = True if self.model_data_cfgdict.training.target_instrument == "Vocals" else False
        self.logger.debug(f"is_vocal_main_target: {self.is_vocal_main_target}")

        self.logger.info("MDXC Separator initialisation complete")

    def load_model(self):
        """
        Load the model into memory from file on disk, initialize it with config from the model data,
        and prepare for inferencing using hardware accelerated Torch device.
        """
        self.logger.debug("Loading checkpoint model for inference...")

        self.model_data_cfgdict = ConfigDict(self.model_data)

        try:
            if self.is_roformer:
                self.logger.debug("Loading Roformer model...")

                # Determine the model type based on the configuration and instantiate it
                if "num_bands" in self.model_data_cfgdict.model:
                    self.logger.debug("Loading MelBandRoformer model...")
                    model = MelBandRoformer(**self.model_data_cfgdict.model)
                elif "freqs_per_bands" in self.model_data_cfgdict.model:
                    self.logger.debug("Loading BSRoformer model...")
                    model = BSRoformer(**self.model_data_cfgdict.model)
                else:
                    raise ValueError("Unknown Roformer model type in the configuration.")

                # Load model checkpoint
                checkpoint = torch.load(self.model_path, map_location="cpu")
                self.model_run = model if not isinstance(model, torch.nn.DataParallel) else model.module
                self.model_run.load_state_dict(checkpoint)
                self.model_run.to(self.torch_device_cpu).eval()

            else:
                self.logger.debug("Loading TFC_TDF_net model...")
                self.model_run = TFC_TDF_net(self.model_data_cfgdict, device=self.torch_device)
                self.model_run.load_state_dict(torch.load(self.model_path, map_location=self.torch_device))
                self.model_run.to(self.torch_device).eval()

        except RuntimeError as e:
            self.logger.error(f"Error: {e}")
            self.logger.error("An error occurred while loading the model file. This often occurs when the model file is corrupt or incomplete.")
            self.logger.error(f"Please try deleting the model file from {self.model_path} and run audio-separator again to re-download it.")
            sys.exit(1)

    def separate(self, audio_file_path):
        """
        Separates the audio file into primary and secondary sources based on the model's configuration.
        It processes the mix, demixes it into sources, normalizes the sources, and saves the output files.

        Args:
            audio_file_path (str): The path to the audio file to be processed.

        Returns:
            list: A list of paths to the output files generated by the separation process.
        """
        self.primary_source = None
        self.secondary_source = None

        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.logger.debug(f"Preparing mix for input audio file {self.audio_file_path}...")
        mix = self.prepare_mix(self.audio_file_path)

        self.logger.debug("Normalizing mix before demixing...")
        mix = spec_utils.normalize(wave=mix, max_peak=self.normalization_threshold)

        source = self.demix(mix=mix)
        self.logger.debug("Demixing completed.")

        output_files = []
        self.logger.debug("Processing output files...")

        if not isinstance(self.primary_source, np.ndarray):
            self.logger.debug(f"Normalizing primary source for primary stem {self.primary_stem_name}...")
            self.primary_source = spec_utils.normalize(wave=source[self.primary_stem_name], max_peak=self.normalization_threshold).T

        if not isinstance(self.secondary_source, np.ndarray):
            self.logger.debug(f"Normalizing secondary source for secondary stem {self.secondary_stem_name}...")
            self.secondary_source = spec_utils.normalize(wave=source[self.secondary_stem_name], max_peak=self.normalization_threshold).T

        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            self.secondary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.secondary_stem_name})_{self.model_name}.{self.output_format.lower()}")

            self.logger.info(f"Saving {self.secondary_stem_name} stem to {self.secondary_stem_output_path}...")
            self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
            output_files.append(self.secondary_stem_output_path)

        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            self.primary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.primary_stem_name})_{self.model_name}.{self.output_format.lower()}")

            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = source.T

            self.logger.info(f"Saving {self.primary_stem_name} stem to {self.primary_stem_output_path}...")
            self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)
            output_files.append(self.primary_stem_output_path)
        return output_files

    def pitch_fix(self, source, sr_pitched, orig_mix):
        """
        Change the pitch of the source audio by a number of semitones.

        Args:
            source (np.ndarray): The source audio to be pitch-shifted.
            sr_pitched (int): The sample rate of the pitch-shifted audio.
            orig_mix (np.ndarray): The original mix, used to match the shape of the pitch-shifted audio.

        Returns:
            np.ndarray: The pitch-shifted source audio.
        """
        source = spec_utils.change_pitch_semitones(source, sr_pitched, semitone_shift=self.pitch_shift)[0]
        source = spec_utils.match_array_shapes(source, orig_mix)
        return source

    def overlap_add(self, result, x, weights, start, length):
        x = x.to(result.device)
        weights = weights.to(result.device)
        result[..., start : start + length] += x[..., :length] * weights[:length]
        return result

    def demix(self, mix: np.ndarray) -> dict:
        """
        Demixes the input mix into primary and secondary sources using the model and model data.

        Args:
            mix (np.ndarray): The mix to be demixed.
        Returns:
            dict: A dictionary containing the demixed sources.
        """
        orig_mix = mix

        if self.pitch_shift != 0:
            self.logger.debug(f"Shifting pitch by -{self.pitch_shift} semitones...")
            mix, sample_rate = spec_utils.change_pitch_semitones(mix, self.sample_rate, semitone_shift=-self.pitch_shift)

        if self.is_roformer:
            mix = torch.tensor(mix, dtype=torch.float32)

            if self.override_model_segment_size:
                mdx_segment_size = self.segment_size
                self.logger.debug(f"Using configured segment size: {mdx_segment_size}")
            else:
                mdx_segment_size = self.model_data_cfgdict.inference.dim_t
                self.logger.debug(f"Using model default segment size: {mdx_segment_size}")

            # num_stems aka "S" in UVR
            num_stems = 1 if self.model_data_cfgdict.training.target_instrument else len(self.model_data_cfgdict.training.instruments)
            self.logger.debug(f"Number of stems: {num_stems}")

            # chunk_size aka "C" in UVR
            chunk_size = self.model_data_cfgdict.audio.hop_length * (mdx_segment_size - 1)
            self.logger.debug(f"Chunk size: {chunk_size}")

            step = int(self.overlap * self.model_data_cfgdict.audio.sample_rate)
            self.logger.debug(f"Step: {step}")

            # Create a weighting table and convert it to a PyTorch tensor
            window = torch.tensor(signal.windows.hamming(chunk_size), dtype=torch.float32)

            device = next(self.model_run.parameters()).device

            # Transfer to the weighting plate for the same device as the other tensors
            window = window.to(device)

            # with torch.cuda.amp.autocast():
            with torch.no_grad():
                req_shape = (len(self.model_data_cfgdict.training.instruments),) + tuple(mix.shape)
                result = torch.zeros(req_shape, dtype=torch.float32).to(device)
                counter = torch.zeros(req_shape, dtype=torch.float32).to(device)

                for i in tqdm(range(0, mix.shape[1], step)):
                    part = mix[:, i : i + chunk_size]
                    length = part.shape[-1]
                    if i + chunk_size > mix.shape[1]:
                        part = mix[:, -chunk_size:]
                        length = chunk_size
                    part = part.to(device)
                    x = self.model_run(part.unsqueeze(0))[0]
                    if i + chunk_size > mix.shape[1]:
                        # Corrigido para adicionar corretamente ao final do tensor
                        result = self.overlap_add(result, x, window, result.shape[-1] - chunk_size, length)
                        counter[..., result.shape[-1] - chunk_size :] += window[:length]
                    else:
                        result = self.overlap_add(result, x, window, i, length)
                        counter[..., i : i + length] += window[:length]

            inferenced_outputs = result / counter.clamp(min=1e-10)

        else:
            mix = torch.tensor(mix, dtype=torch.float32)

            try:
                num_stems = self.model_run.num_target_instruments
            except AttributeError:
                num_stems = self.model_run.module.num_target_instruments
            self.logger.debug(f"Number of stems: {num_stems}")

            if self.override_model_segment_size:
                mdx_segment_size = self.segment_size
                self.logger.debug(f"Using configured segment size: {mdx_segment_size}")
            else:
                mdx_segment_size = self.model_data_cfgdict.inference.dim_t
                self.logger.debug(f"Using model default segment size: {mdx_segment_size}")

            chunk_size = self.model_data_cfgdict.audio.hop_length * (mdx_segment_size - 1)
            self.logger.debug(f"Chunk size: {chunk_size}")

            hop_size = chunk_size // self.overlap
            self.logger.debug(f"Hop size: {hop_size}")

            mix_shape = mix.shape[1]
            pad_size = hop_size - (mix_shape - chunk_size) % hop_size
            self.logger.debug(f"Pad size: {pad_size}")

            mix = torch.cat([torch.zeros(2, chunk_size - hop_size), mix, torch.zeros(2, pad_size + chunk_size - hop_size)], 1)
            self.logger.debug(f"Mix shape: {mix.shape}")

            chunks = mix.unfold(1, chunk_size, hop_size).transpose(0, 1)
            self.logger.debug(f"Chunks length: {len(chunks)} and shape: {chunks.shape}")

            batches = [chunks[i : i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
            self.logger.debug(f"Batch size: {self.batch_size}, number of batches: {len(batches)}")

            # accumulated_outputs is used to accumulate the output from processing each batch of chunks through the model.
            # It starts as a tensor of zeros and is updated in-place as the model processes each batch.
            # The variable holds the combined result of all processed batches, which, after post-processing, represents the separated audio sources.
            accumulated_outputs = torch.zeros(num_stems, *mix.shape) if num_stems > 1 else torch.zeros_like(mix)
            accumulated_outputs = accumulated_outputs.to(self.torch_device)

            with torch.no_grad():
                count = 0
                for batch in tqdm(batches):
                    # Since the model processes the audio data in batches, single_batch_result temporarily holds the model's output
                    # for each batch before it is accumulated into accumulated_outputs.
                    single_batch_result = self.model_run(batch.to(self.torch_device))

                    # Each individual output tensor from the current batch's processing result.
                    # Since single_batch_result can contain multiple output tensors (one for each piece of audio in the batch),
                    # individual_output is used to iterate through these tensors and accumulate them into accumulated_outputs.
                    for individual_output in single_batch_result:
                        accumulated_outputs[..., count * hop_size : count * hop_size + chunk_size] += individual_output
                        count += 1

            self.logger.debug("Calculating inferenced outputs based on accumulated outputs and overlap")
            inferenced_outputs = accumulated_outputs[..., chunk_size - hop_size : -(pad_size + chunk_size - hop_size)] / self.overlap
            self.logger.debug("Deleting accumulated outputs to free up memory")
            del accumulated_outputs

        if num_stems > 1 or self.is_vocal_main_target:
            self.logger.debug("Number of stems is greater than 1 or vocals are main target, detaching individual sources and correcting pitch if necessary...")

            sources = {}

            # Iterates over each instrument specified in the model's configuration and its corresponding separated audio source.
            # self.model_data_cfgdict.training.instruments provides the list of stems.
            # estimated_sources.cpu().detach().numpy() converts the separated sources tensor to a NumPy array for processing.
            # Each iteration provides an instrument name ('key') and its separated audio ('value') for further processing.
            for key, value in zip(self.model_data_cfgdict.training.instruments, inferenced_outputs.cpu().detach().numpy()):
                self.logger.debug(f"Processing instrument: {key}")
                if self.pitch_shift != 0:
                    self.logger.debug(f"Applying pitch correction for {key}")
                    sources[key] = self.pitch_fix(value, sample_rate, orig_mix)
                else:
                    sources[key] = value

            if self.is_vocal_main_target:
                self.logger.debug("Vocals are main target, detaching vocals and matching array shapes if necessary...")
                if sources["Vocals"].shape[1] != orig_mix.shape[1]:
                    sources["Vocals"] = spec_utils.match_array_shapes(sources["Vocals"], orig_mix)
                sources["Instrumental"] = orig_mix - sources["Vocals"]

            self.logger.debug("Deleting inferenced outputs to free up memory")
            del inferenced_outputs

            self.logger.debug("Returning separated sources")
            return sources
        else:
            self.logger.debug("Processing single source...")

            if self.is_roformer:
                sources = {k: v.cpu().detach().numpy() for k, v in zip([self.model_data_cfgdict.training.target_instrument], inferenced_outputs)}
                inferenced_output = sources[self.model_data_cfgdict.training.target_instrument]
            else:
                inferenced_output = inferenced_outputs.cpu().detach().numpy()

            self.logger.debug("Demix process completed for single source.")

            self.logger.debug("Deleting inferenced outputs to free up memory")
            del inferenced_outputs

            if self.pitch_shift != 0:
                self.logger.debug("Applying pitch correction for single instrument")
                return self.pitch_fix(inferenced_output, sample_rate, orig_mix)
            else:
                self.logger.debug("Returning inferenced output for single instrument")
                return inferenced_output
