"""Keep explicit CUDA selection consistent across inference backends."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from audio_separator.separator import Separator


@pytest.mark.parametrize("index", [0, 1, 3])
def test_selected_device_reaches_torch_and_onnx(index):
    """Keep PyTorch and ONNX Runtime on the same explicitly selected CUDA device."""
    separator = Separator(info_only=True)
    with patch("torch.cuda.device_count", return_value=4):
        separator.configure_cuda(["CUDAExecutionProvider"], device_index=index)
    assert str(separator.torch_device) == f"cuda:{index}"
    assert separator.onnx_execution_provider == [("CUDAExecutionProvider", {"device_id": index})]


@pytest.mark.parametrize("index", [-1, True, 1.5, "1"])
def test_invalid_index_rejected_before_initialization(index):
    """Reject negative, boolean, and non-integer device indices before initializing the separator."""
    with pytest.raises(ValueError, match="cuda_device_index"):
        Separator(info_only=True, cuda_device_index=index)


def test_constructor_selection_is_used_by_device_setup():
    """Apply the constructor CUDA selection when discovering available inference devices."""
    separator = Separator(info_only=True, cuda_device_index=2)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=3),
        patch("audio_separator.separator.separator.ort.get_available_providers", return_value=["CUDAExecutionProvider"]),
    ):
        separator.setup_torch_device(SimpleNamespace(processor="test"))
    assert str(separator.torch_device) == "cuda:2"
    assert separator.onnx_execution_provider == [("CUDAExecutionProvider", {"device_id": 2})]


def test_explicit_index_cannot_silently_fall_back_to_cpu():
    """Fail when an explicit CUDA device is requested but CUDA is unavailable."""
    separator = Separator(info_only=True, cuda_device_index=0)
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(ValueError, match="CUDA"):
        separator.setup_torch_device(SimpleNamespace(processor="test"))


def test_out_of_range_selection_does_not_mutate_devices():
    """Preserve existing device state when the requested CUDA index exceeds the device count."""
    separator = Separator(info_only=True)
    separator.torch_device = "unchanged"
    with patch("torch.cuda.device_count", return_value=2), pytest.raises(ValueError, match="cuda_device_index"):
        separator.configure_cuda(["CUDAExecutionProvider"], device_index=2)
    assert separator.torch_device == "unchanged"


def test_default_selection_keeps_existing_provider_configuration():
    """Retain automatic CUDA placement and provider names when no index is requested."""
    separator = Separator(info_only=True)
    separator.configure_cuda(["CUDAExecutionProvider"])
    assert str(separator.torch_device) == "cuda"
    assert separator.onnx_execution_provider == ["CUDAExecutionProvider"]


@pytest.mark.parametrize("active_providers, warns", [(["CUDAExecutionProvider"], False), (["CPUExecutionProvider"], True)])
def test_mdx_checks_provider_name_and_preserves_device_options(active_providers, warns):
    """Preserve ONNX device options and warn only when the active session lacks the CUDA provider."""
    from unittest.mock import Mock
    from audio_separator.separator.architectures.mdx_separator import MDXSeparator

    separator = MDXSeparator.__new__(MDXSeparator)
    separator.logger = Mock()
    separator.segment_size = separator.dim_t = 256
    separator.log_level = 20
    separator.model_path = "model.onnx"
    separator.onnx_execution_provider = [("CUDAExecutionProvider", {"device_id": 2})]
    with patch("audio_separator.separator.architectures.mdx_separator.ort.InferenceSession") as session:
        session.return_value.get_providers.return_value = active_providers
        separator.load_model()
    assert session.call_args.kwargs["providers"] == [("CUDAExecutionProvider", {"device_id": 2})]
    assert separator.logger.warning.called is warns


@pytest.mark.parametrize("device_name", ["cuda", "cuda:0", "cuda:2"])
def test_cache_cleanup_uses_selected_cuda_context(device_name):
    """Release cached memory within the configured CUDA device context."""
    from unittest.mock import Mock
    import torch
    from audio_separator.separator.common_separator import CommonSeparator

    separator = CommonSeparator.__new__(CommonSeparator)
    separator.logger = Mock()
    separator.torch_device = torch.device(device_name)
    with patch("torch.cuda.device") as context, patch("torch.cuda.empty_cache") as empty_cache:
        separator.clear_gpu_cache()
    context.assert_called_once_with(separator.torch_device)
    context.return_value.__enter__.assert_called_once()
    empty_cache.assert_called_once()
    context.return_value.__exit__.assert_called_once()
