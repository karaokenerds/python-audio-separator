#!/usr/bin/env python
import argparse
import logging
import json
import sys
import os
from importlib import metadata


def main():
    """Main entry point for the CLI."""
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    parser = argparse.ArgumentParser(description="Separate audio file into different stems.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))

    parser.add_argument("audio_files", nargs="*", help="The audio file paths or directory to separate, in any common format.", default=argparse.SUPPRESS)

    package_version = metadata.distribution("audio-separator").version

    version_help = "Show the program's version number and exit."
    debug_help = "Enable debug logging, equivalent to --log_level=debug."
    env_info_help = "Print environment information and exit."
    list_models_help = "List all supported models and exit. Use --list_filter to filter/sort the list and --list_limit to show only top N results."
    log_level_help = "Log level, e.g. info, debug, warning (default: %(default)s)."

    info_params = parser.add_argument_group("Info and Debugging")
    info_params.add_argument("-v", "--version", action="version", version=f"%(prog)s {package_version}", help=version_help)
    info_params.add_argument("-d", "--debug", action="store_true", help=debug_help)
    info_params.add_argument("-e", "--env_info", action="store_true", help=env_info_help)
    info_params.add_argument("-l", "--list_models", action="store_true", help=list_models_help)
    info_params.add_argument("--log_level", default="info", help=log_level_help)
    info_params.add_argument("--list_filter", help="Filter and sort the model list by 'name', 'filename', or any stem e.g. vocals, instrumental, drums")
    info_params.add_argument("--list_limit", type=int, help="Limit the number of models shown")
    info_params.add_argument("--list_format", choices=["pretty", "json"], default="pretty", help="Format for listing models: 'pretty' for formatted output, 'json' for raw JSON dump")

    model_filename_help = "Model to use for separation (default: %(default)s). Example: -m model1.ckpt"
    extra_models_help = "Additional models for ensembling. Requires -m for the primary model. Example: --extra_models model2.onnx model3.ckpt"
    output_format_help = "Output format for separated files, any common format (default: %(default)s). Example: --output_format=MP3"
    output_bitrate_help = "Output bitrate for separated files, any ffmpeg-compatible bitrate (default: %(default)s). Example: --output_bitrate=320k"
    output_dir_help = "Directory to write output files (default: <current dir>). Example: --output_dir=/app/separated"
    model_file_dir_help = "Model files directory (default: %(default)s or AUDIO_SEPARATOR_MODEL_DIR env var if set). Example: --model_file_dir=/app/models"
    download_model_only_help = "Download a single model file only, without performing separation."

    io_params = parser.add_argument_group("Separation I/O Params")
    io_params.add_argument("-m", "--model_filename", default="model_bs_roformer_ep_317_sdr_12.9755.ckpt", help=model_filename_help)
    io_params.add_argument("--extra_models", nargs="+", default=None, help=extra_models_help)
    io_params.add_argument("--output_format", default="FLAC", help=output_format_help)
    io_params.add_argument("--output_bitrate", default=None, help=output_bitrate_help)
    io_params.add_argument("--output_dir", default=None, help=output_dir_help)
    io_params.add_argument("--model_file_dir", default="/tmp/audio-separator-models/", help=model_file_dir_help)
    io_params.add_argument("--download_model_only", action="store_true", help=download_model_only_help)

    invert_spect_help = "Invert secondary stem using spectrogram (default: %(default)s). Example: --invert_spect"
    normalization_help = "Max peak amplitude to normalize input and output audio to (default: %(default)s). Example: --normalization=0.7"
    amplification_help = "Min peak amplitude to amplify input and output audio to (default: %(default)s). Example: --amplification=0.4"
    single_stem_help = "Output only single stem, e.g. Instrumental, Vocals, Drums, Bass, Guitar, Piano, Other. Example: --single_stem=Instrumental"
    sample_rate_help = "Modify the sample rate of the output audio (default: %(default)s). Example: --sample_rate=44100"
    use_soundfile_help = "Use soundfile to write audio output (default: %(default)s). Example: --use_soundfile"
    use_autocast_help = (
        "Use PyTorch autocast when supported (default: %(default)s). Example: --use_autocast"
    )
    use_native_fp16_help = (
        "Use native float16 for verified model/device combinations (default: %(default)s). "
        "Mutually exclusive with --use_autocast. Example: --use_native_fp16"
    )
    use_torch_compile_help = (
        "Compile verified repeated model blocks when supported (default: %(default)s). "
        "Best for long inputs or repeated same-shape runs; a fresh compiler cache can make "
        "the first run slower. Example: --use_torch_compile"
    )
    use_directml_help = "Use DirectML for hardware-accelerated inference on Windows AMD/Intel GPUs (experimental; requires the 'dml' extra). Example: --use_directml"
    chunk_duration_help = "Split audio into chunks of this duration in seconds (default: %(default)s = no chunking). Useful for processing very long audio files on systems with limited memory. Recommended: 600 (10 minutes) for files >1 hour. Chunks are concatenated without overlap/crossfade. Example: --chunk_duration=600"
    ensemble_algorithm_help = "Algorithm to use for ensembling multiple models (default: avg_wave). Choices: avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft, uvr_max_spec, uvr_min_spec, ensemble_wav. Example: --ensemble_algorithm=uvr_max_spec"
    ensemble_weights_help = "Weights for ensembling multiple models (default: equal). Number of weights must match number of models. Example: --ensemble_weights 1.0 0.5"
    ensemble_preset_help = "Use a named ensemble preset (e.g. vocal_balanced, karaoke). Presets define models + algorithm. Use --list_presets to see all. Example: --ensemble_preset=vocal_balanced"
    list_presets_help = "List all available ensemble presets and exit."
    custom_output_names_help = 'Custom names for all output files in JSON format (default: %(default)s). Example: --custom_output_names=\'{"Vocals": "vocals_output", "Drums": "drums_output"}\''

    common_params = parser.add_argument_group("Common Separation Parameters")
    common_params.add_argument("--invert_spect", action="store_true", help=invert_spect_help)
    common_params.add_argument("--normalization", type=float, default=0.9, help=normalization_help)
    common_params.add_argument("--amplification", type=float, default=0.0, help=amplification_help)
    common_params.add_argument("--single_stem", default=None, help=single_stem_help)
    common_params.add_argument("--sample_rate", type=int, default=44100, help=sample_rate_help)
    common_params.add_argument("--use_soundfile", action="store_true", help=use_soundfile_help)
    precision_params = common_params.add_mutually_exclusive_group()
    precision_params.add_argument("--use_autocast", action="store_true", help=use_autocast_help)
    precision_params.add_argument("--use_native_fp16", action="store_true", help=use_native_fp16_help)
    common_params.add_argument("--use_torch_compile", action="store_true", help=use_torch_compile_help)
    common_params.add_argument("--cuda_device_index", type=int, default=None, help="CUDA device index for both PyTorch and ONNX Runtime (relative to CUDA_VISIBLE_DEVICES).")
    common_params.add_argument("--use_directml", action="store_true", help=use_directml_help)
    common_params.add_argument("--chunk_duration", type=float, default=None, help=chunk_duration_help)
    common_params.add_argument(
        "--ensemble_algorithm",
        default=None,
        choices=["avg_wave", "median_wave", "min_wave", "max_wave", "avg_fft", "median_fft", "min_fft", "max_fft", "uvr_max_spec", "uvr_min_spec", "ensemble_wav"],
        help=ensemble_algorithm_help,
    )
    common_params.add_argument("--ensemble_weights", nargs="+", type=float, default=None, help=ensemble_weights_help)
    common_params.add_argument("--ensemble_preset", default=None, help=ensemble_preset_help)
    common_params.add_argument("--list_presets", action="store_true", help=list_presets_help)
    common_params.add_argument("--custom_output_names", type=json.loads, default=None, help=custom_output_names_help)

    mdx_segment_size_help = "Larger consumes more resources, but may give better results (default: %(default)s). Example: --mdx_segment_size=256"
    mdx_overlap_help = "Amount of overlap between prediction windows, 0.001-0.999. Higher is better but slower (default: %(default)s). Example: --mdx_overlap=0.25"
    mdx_batch_size_help = "Larger consumes more RAM but may process slightly faster (default: %(default)s). Example: --mdx_batch_size=4"
    mdx_hop_length_help = "Usually called stride in neural networks, only change if you know what you're doing (default: %(default)s). Example: --mdx_hop_length=1024"
    mdx_enable_denoise_help = "Enable denoising during separation (default: %(default)s). Example: --mdx_enable_denoise"

    mdx_params = parser.add_argument_group("MDX Architecture Parameters")
    mdx_params.add_argument("--mdx_segment_size", type=int, default=256, help=mdx_segment_size_help)
    mdx_params.add_argument("--mdx_overlap", type=float, default=0.25, help=mdx_overlap_help)
    mdx_params.add_argument("--mdx_batch_size", type=int, default=1, help=mdx_batch_size_help)
    mdx_params.add_argument("--mdx_hop_length", type=int, default=1024, help=mdx_hop_length_help)
    mdx_params.add_argument("--mdx_enable_denoise", action="store_true", help=mdx_enable_denoise_help)

    vr_batch_size_help = "Number of batches to process at a time. Higher = more RAM, slightly faster processing (default: %(default)s). Example: --vr_batch_size=16"
    vr_window_size_help = "Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: %(default)s). Example: --vr_window_size=320"
    vr_aggression_help = "Intensity of primary stem extraction, -100 - 100. Typically, 5 for vocals & instrumentals (default: %(default)s). Example: --vr_aggression=2"
    vr_enable_tta_help = "Enable Test-Time-Augmentation; slow but improves quality (default: %(default)s). Example: --vr_enable_tta"
    vr_high_end_process_help = "Mirror the missing frequency range of the output (default: %(default)s). Example: --vr_high_end_process"
    vr_enable_post_process_help = "Identify leftover artifacts within vocal output; may improve separation for some songs (default: %(default)s). Example: --vr_enable_post_process"
    vr_post_process_threshold_help = "Threshold for post_process feature: 0.1-0.3 (default: %(default)s). Example: --vr_post_process_threshold=0.1"

    vr_params = parser.add_argument_group("VR Architecture Parameters")
    vr_params.add_argument("--vr_batch_size", type=int, default=1, help=vr_batch_size_help)
    vr_params.add_argument("--vr_window_size", type=int, default=512, help=vr_window_size_help)
    vr_params.add_argument("--vr_aggression", type=int, default=5, help=vr_aggression_help)
    vr_params.add_argument("--vr_enable_tta", action="store_true", help=vr_enable_tta_help)
    vr_params.add_argument("--vr_high_end_process", action="store_true", help=vr_high_end_process_help)
    vr_params.add_argument("--vr_enable_post_process", action="store_true", help=vr_enable_post_process_help)
    vr_params.add_argument("--vr_post_process_threshold", type=float, default=0.2, help=vr_post_process_threshold_help)

    demucs_segment_size_help = "Size of segments into which the audio is split, 1-100. Higher = slower but better quality (default: %(default)s). Example: --demucs_segment_size=256"
    demucs_shifts_help = "Number of predictions with random shifts, higher = slower but better quality (default: %(default)s). Example: --demucs_shifts=4"
    demucs_overlap_help = "Overlap between prediction windows, 0.001-0.999. Higher = slower but better quality (default: %(default)s). Example: --demucs_overlap=0.25"
    demucs_segments_enabled_help = "Enable segment-wise processing (default: %(default)s). Example: --demucs_segments_enabled=False"

    demucs_params = parser.add_argument_group("Demucs Architecture Parameters")
    demucs_params.add_argument("--demucs_segment_size", type=str, default="Default", help=demucs_segment_size_help)
    demucs_params.add_argument("--demucs_shifts", type=int, default=2, help=demucs_shifts_help)
    demucs_params.add_argument("--demucs_overlap", type=float, default=0.25, help=demucs_overlap_help)
    demucs_params.add_argument("--demucs_segments_enabled", type=bool, default=True, help=demucs_segments_enabled_help)

    mdxc_segment_size_help = "Larger consumes more resources, but may give better results (default: %(default)s). Example: --mdxc_segment_size=256"
    mdxc_override_model_segment_size_help = "Override model default segment size instead of using the model default value. Example: --mdxc_override_model_segment_size"
    mdxc_overlap_help = "Number of overlapping prediction windows, 2-50. Higher is better but slower (default: model config, falling back to 8). Example: --mdxc_overlap=8"
    mdxc_batch_size_help = "Larger consumes more RAM but may process slightly faster (default: model config, falling back to 1). Example: --mdxc_batch_size=4"
    mdxc_pitch_shift_help = "Shift audio pitch by a number of semitones while processing. May improve output for deep/high vocals. (default: %(default)s). Example: --mdxc_pitch_shift=2"

    mdxc_params = parser.add_argument_group("MDXC Architecture Parameters")
    mdxc_params.add_argument("--mdxc_segment_size", type=int, default=256, help=mdxc_segment_size_help)
    mdxc_params.add_argument("--mdxc_override_model_segment_size", action="store_true", help=mdxc_override_model_segment_size_help)
    mdxc_params.add_argument("--mdxc_overlap", type=int, default=None, help=mdxc_overlap_help)
    mdxc_params.add_argument("--mdxc_batch_size", type=int, default=None, help=mdxc_batch_size_help)
    mdxc_params.add_argument("--mdxc_pitch_shift", type=int, default=0, help=mdxc_pitch_shift_help)

    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

    if args.env_info:
        from audio_separator.separator import Separator
        separator = Separator()
        sys.exit(0)

    if args.list_models:
        from audio_separator.separator import Separator
        separator = Separator(info_only=True)

        if args.list_format == "json":
            model_list = separator.list_supported_model_files()
            print(json.dumps(model_list, indent=2))
        else:
            models = separator.get_simplified_model_list(filter_sort_by=args.list_filter)

            # Apply limit if specified
            if args.list_limit and args.list_limit > 0:
                models = dict(list(models.items())[: args.list_limit])

            # Calculate maximum widths for each column
            filename_width = max(len("Model Filename"), max(len(filename) for filename in models.keys()))
            arch_width = max(len("Arch"), max(len(info["Type"]) for info in models.values()))
            stems_width = max(len("Output Stems (SDR)"), max(len(", ".join(info["Stems"])) for info in models.values()))
            name_width = max(len("Friendly Name"), max(len(info["Name"]) for info in models.values()))

            # Calculate total width for separator line
            total_width = filename_width + arch_width + stems_width + name_width + 15  # 15 accounts for spacing between columns

            # Format the output with dynamic widths and extra spacing
            print("-" * total_width)
            print(f"{'Model Filename':<{filename_width}}  {'Arch':<{arch_width}}  {'Output Stems (SDR)':<{stems_width}}  {'Friendly Name'}")
            print("-" * total_width)

            for filename, info in models.items():
                stems = ", ".join(info["Stems"])
                print(f"{filename:<{filename_width}}  {info['Type']:<{arch_width}}  {stems:<{stems_width}}  {info['Name']}")

        sys.exit(0)

    if args.list_presets:
        from audio_separator.separator import Separator
        separator = Separator(info_only=True)
        presets = separator.list_ensemble_presets()

        if not presets:
            print("No ensemble presets available.")
            sys.exit(0)

        # Calculate column widths
        id_width = max(len("Preset ID"), max(len(k) for k in presets.keys()))
        desc_width = max(len("Description"), max(len(p.get("description", "")[:60]) for p in presets.values()))
        models_width = len("Models")
        algo_width = max(len("Algorithm"), max(len(p.get("algorithm", "")) for p in presets.values()))
        total_width = id_width + desc_width + models_width + algo_width + 12

        print("-" * total_width)
        print(f"{'Preset ID':<{id_width}}  {'Description':<{desc_width}}  {'Models':<{models_width}}  {'Algorithm'}")
        print("-" * total_width)

        for preset_id, preset in presets.items():
            desc = preset.get("description", "")[:60]
            num_models = len(preset.get("models", []))
            algo = preset.get("algorithm", "")
            print(f"{preset_id:<{id_width}}  {desc:<{desc_width}}  {num_models:<{models_width}}  {algo}")

        sys.exit(0)

    if args.download_model_only:
        from audio_separator.separator import Separator
        models_to_download = [args.model_filename] + (args.extra_models or [])
        separator = Separator(log_formatter=log_formatter, log_level=log_level, model_file_dir=args.model_file_dir)
        for model in models_to_download:
            logger.info(f"Separator version {package_version} downloading model {model} to directory {args.model_file_dir}")
            separator.download_model_and_data(model)

        logger.info(f"Model {', '.join(models_to_download)} downloaded successfully.")
        sys.exit(0)

    audio_files = list(getattr(args, "audio_files", []))
    if not audio_files:
        parser.print_help()
        sys.exit(1)

    logger.info(f"Separator version {package_version} beginning with input path(s): {', '.join(audio_files)}")

    from audio_separator.separator import Separator

    separator = Separator(
        log_formatter=log_formatter,
        log_level=log_level,
        model_file_dir=args.model_file_dir,
        output_dir=args.output_dir,
        output_format=args.output_format,
        output_bitrate=args.output_bitrate,
        normalization_threshold=args.normalization,
        amplification_threshold=args.amplification,
        output_single_stem=args.single_stem,
        invert_using_spec=args.invert_spect,
        sample_rate=args.sample_rate,
        use_soundfile=args.use_soundfile,
        use_autocast=args.use_autocast,
        use_native_fp16=args.use_native_fp16,
        use_torch_compile=args.use_torch_compile,
        use_directml=args.use_directml,
        cuda_device_index=args.cuda_device_index,
        chunk_duration=args.chunk_duration,
        ensemble_algorithm=args.ensemble_algorithm,
        ensemble_weights=args.ensemble_weights,
        ensemble_preset=args.ensemble_preset,
        mdx_params={
            "hop_length": args.mdx_hop_length,
            "segment_size": args.mdx_segment_size,
            "overlap": args.mdx_overlap,
            "batch_size": args.mdx_batch_size,
            "enable_denoise": args.mdx_enable_denoise,
        },
        vr_params={
            "batch_size": args.vr_batch_size,
            "window_size": args.vr_window_size,
            "aggression": args.vr_aggression,
            "enable_tta": args.vr_enable_tta,
            "enable_post_process": args.vr_enable_post_process,
            "post_process_threshold": args.vr_post_process_threshold,
            "high_end_process": args.vr_high_end_process,
        },
        demucs_params={
            "segment_size": args.demucs_segment_size,
            "shifts": args.demucs_shifts,
            "overlap": args.demucs_overlap,
            "segments_enabled": args.demucs_segments_enabled,
        },
        mdxc_params={
            "segment_size": args.mdxc_segment_size,
            "batch_size": args.mdxc_batch_size,
            "overlap": args.mdxc_overlap,
            "override_model_segment_size": args.mdxc_override_model_segment_size,
            "pitch_shift": args.mdxc_pitch_shift,
        },
    )

    # Combine primary model with any extra models for ensembling
    # If a preset is active and no explicit models were provided, use preset models via default
    if args.ensemble_preset and args.model_filename == "model_bs_roformer_ep_317_sdr_12.9755.ckpt" and not args.extra_models:
        separator.load_model()
    else:
        model_filenames = [args.model_filename] + (args.extra_models or [])
        if len(model_filenames) == 1:
            model_filenames = model_filenames[0]
        separator.load_model(model_filename=model_filenames)

    try:
        output_files = separator.separate(audio_files, custom_output_names=args.custom_output_names)
    except Exception as e:
        logger.error(f"Separation failed: {e}", exc_info=True)
        sys.exit(1)

    if not output_files:
        logger.error("Separation produced no output files — see errors above.")
        sys.exit(1)

    logger.info(f"Separation complete! Output file(s): {' '.join(output_files)}")
