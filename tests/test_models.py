import pytest
import torch
from src.models import ARCHITECTURES, ResNeXt, ConvNeXt


@pytest.mark.parametrize("arch_name", ["plain", "resnet", "resnext", "convnext"])
def test_model_instantiation(arch_name):
    """Test that models can be instantiated using the factory."""
    model_cls = ARCHITECTURES.get(arch_name)
    assert model_cls is not None, f"Architecture {arch_name} not found in ARCHITECTURES"

    model = model_cls(base_ch=32, num_classes=8)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize("arch_name", ["plain", "resnet", "resnext", "convnext"])
def test_model_forward_pass(arch_name):
    """Test valid forward pass with dummy input."""
    model_cls = ARCHITECTURES[arch_name]
    model = model_cls(base_ch=32, num_classes=8)

    batch_size = 2
    x = torch.randn(batch_size, 1, 48, 48)
    output = model(x)

    assert output.shape == (batch_size, 8)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize("groups,width", [(8, 4), (16, 2), (32, 4), (1, 64)])
def test_resnext_custom_groups(groups, width):
    """Test ResNeXt with various groups and bottleneck configuration."""
    model = ResNeXt(
        base_ch=32, num_classes=8, groups=groups, base_width_per_group=width
    )
    x = torch.randn(2, 1, 48, 48)
    output = model(x)
    assert output.shape == (2, 8)


def test_convnext_scaling():
    """Test ConvNeXt scaling factor and internal dimensions."""
    base_ch = 16
    model = ConvNeXt(base_ch=base_ch, num_classes=5)
    expected_dims = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

    for i, stage in enumerate(model.stages):
        block = stage[0]
        assert block.dwconv.in_channels == expected_dims[i]
        assert block.dwconv.out_channels == expected_dims[i]

    x = torch.randn(1, 1, 48, 48)
    output = model(x)
    assert output.shape == (1, 5)
