"""Shared atomic file publication for encoded audio outputs."""

from contextlib import contextmanager
import os
import secrets
import stat
import tempfile

import numpy as np

from audio_separator.separator.exceptions import AudioExportError, InvalidAudioDataError

OUTPUT_SUBTYPES = ("AUTO", "PCM_16", "PCM_24", "PCM_32", "FLOAT")


def normalize_output_subtype(output_subtype, output_format):
    """Validate and normalize an explicit lossless output subtype."""
    subtype = str(output_subtype or "AUTO").upper()
    if subtype not in OUTPUT_SUBTYPES:
        choices = ", ".join(OUTPUT_SUBTYPES)
        raise ValueError(f"output_subtype must be one of: {choices}")

    file_format = str(output_format or "WAV").lower()
    if subtype != "AUTO" and file_format not in ("wav", "flac"):
        raise ValueError("output_subtype can only be set for WAV or FLAC output")
    if file_format == "flac" and subtype in ("PCM_32", "FLOAT"):
        raise ValueError(f"FLAC output does not support {subtype}; use PCM_24 or WAV")
    return subtype


def resolve_output_subtype(requested_subtype, input_subtype, input_bit_depth, output_format):
    """Resolve AUTO to an input-compatible subtype supported by the container."""
    subtype = normalize_output_subtype(requested_subtype, output_format)
    if subtype != "AUTO":
        return subtype

    file_format = str(output_format or "WAV").lower()
    if input_subtype:
        subtype = str(input_subtype).upper()
    else:
        subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(input_bit_depth, "PCM_16")

    if file_format == "flac" and subtype in ("PCM_32", "FLOAT", "DOUBLE"):
        return "PCM_24"
    return subtype


def validate_audio_source(stem_source):
    """Return audio as an array after validating mono/stereo frame layout."""
    stem_source = np.asarray(stem_source)
    if stem_source.ndim not in (1, 2) or (stem_source.ndim == 2 and stem_source.shape[1] not in (1, 2)):
        raise InvalidAudioDataError(f"Audio data has invalid shape {stem_source.shape}; expected mono or stereo frames")
    return stem_source


def _published_file_mode(target_path, target_dir):
    """Preserve an existing mode or derive a new-file mode through the current umask."""
    if os.name == "nt":
        return None
    try:
        return stat.S_IMODE(os.stat(target_path).st_mode)
    except FileNotFoundError:
        pass

    for _ in range(10):
        probe_path = os.path.join(target_dir, f".audio-output-mode-{secrets.token_hex(8)}")
        try:
            probe_fd = os.open(probe_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        try:
            return stat.S_IMODE(os.fstat(probe_fd).st_mode)
        finally:
            os.close(probe_fd)
            os.unlink(probe_path)

    raise FileExistsError("Could not create a unique mode probe for atomic audio output")


@contextmanager
def atomic_output_path(target_path, backend):
    """Yield a same-directory temporary path and atomically publish it on success."""
    temp_fd = None
    temp_path = None
    error = None

    try:
        target_dir = os.path.dirname(target_path) or "."
        suffix = os.path.splitext(target_path)[1]
        temp_fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(target_path)}.", suffix=suffix, dir=target_dir)
        published_mode = _published_file_mode(target_path, target_dir)
        if published_mode is not None:
            os.fchmod(temp_fd, published_mode)
        os.close(temp_fd)
        temp_fd = None

        yield temp_path

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise OSError("Audio backend produced an empty output file")
        os.replace(temp_path, target_path)
    except Exception as exc:
        error = exc
    finally:
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                if error is None:
                    error = cleanup_error
                elif hasattr(error, "add_note"):
                    error.add_note(f"Failed to remove temporary audio output {temp_path}: {cleanup_error}")

    if error is not None:
        if isinstance(error, AudioExportError):
            raise error
        raise AudioExportError(
            f"Failed to publish audio output {target_path} with {backend}: {error}", path=target_path, backend=backend
        ) from error
