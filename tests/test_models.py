import pytest
import torch
from src.models import ARCHITECTURES


@pytest.mark.parametrize("arch_name", list(ARCHITECTURES.keys()))
def test_model_instantiation(arch_name):
    """Test that models can be instantiated using the factory."""
    model_cls = ARCHITECTURES.get(arch_name)
    assert model_cls is not None, f"Architecture {arch_name} not found in ARCHITECTURES"

    model = model_cls(base_ch=32, num_classes=8)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize("arch_name", list(ARCHITECTURES.keys()))
def test_model_forward_pass(arch_name):
    """Test valid forward pass with dummy input."""
    model_cls = ARCHITECTURES[arch_name]
    model = model_cls(base_ch=32, num_classes=8)

    batch_size = 2
    x = torch.randn(batch_size, 1, 48, 48)
    output = model(x)

    assert output.shape == (batch_size, 8)
    assert not torch.isnan(output).any()
