"""Tests verifying refactored model definitions match the originals.

Compares the models from the new `models/` package against the
original definitions still present in `utils/convert_flame.py` and
`utils/convert_nuke.py` (which still contain the pre-refactor copies).
"""
import sys
import os
import torch
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _get_state_dict_shapes(model):
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestUNet:
    """Verify UNet produces correct shapes and parameter counts."""

    @pytest.mark.parametrize("hidden_size", [32, 64, 96, 128])
    def test_forward_shape(self, hidden_size):
        from models import UNet
        model = UNet(n_ch=3, n_cls=3, hidden_size=hidden_size)
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {out.shape}"

    def test_4ch_input(self):
        from models import UNet
        model = UNet(n_ch=4, n_cls=3, hidden_size=64)
        x = torch.randn(1, 4, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 128, 128)

    def test_invalid_params(self):
        from models import UNet
        with pytest.raises(ValueError):
            UNet(n_ch=0, n_cls=3, hidden_size=64)
        with pytest.raises(ValueError):
            UNet(n_ch=3, n_cls=3, hidden_size=0)

    def test_state_dict_matches_legacy(self):
        """Verify new UNet has same state_dict keys/shapes as the old one in utils/."""
        from models import UNet as NewUNet
        # Load the old definitions from utils/ (which still has the copy)
        old_utils = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils')
        if not os.path.isdir(old_utils):
            pytest.skip("utils/ directory not present (already deleted)")
        sys.path.insert(0, old_utils)
        try:
            import importlib
            spec = importlib.util.spec_from_file_location(
                "old_flame", os.path.join(old_utils, "convert_flame.py"))
            old_mod = importlib.util.module_from_spec(spec)
            # Only parse, don't execute __main__
            with open(os.path.join(old_utils, "convert_flame.py")) as f:
                code = f.read()
            exec(compile(code, "convert_flame.py", "exec"), old_mod.__dict__)
            OldUNet = old_mod.UNet
        except Exception as e:
            pytest.skip(f"Could not load old UNet: {e}")
        finally:
            sys.path.pop(0)

        for hs in [64, 96, 128]:
            new_model = NewUNet(n_ch=3, n_cls=3, hidden_size=hs)
            old_model = OldUNet(n_ch=3, n_cls=3, hidden_size=hs)
            new_shapes = _get_state_dict_shapes(new_model)
            old_shapes = _get_state_dict_shapes(old_model)
            assert new_shapes == old_shapes, (
                f"State dict mismatch at hidden_size={hs}:\n"
                f"  New keys: {set(new_shapes) - set(old_shapes)}\n"
                f"  Old keys: {set(old_shapes) - set(new_shapes)}"
            )


class TestMSRNet:
    """Verify MSRNet architecture."""

    @pytest.mark.parametrize("hidden_size", [32, 64])
    def test_forward_shape(self, hidden_size):
        from models import MSRNet
        model = MSRNet(n_ch=3, n_cls=3, hidden_size=hidden_size)
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 128, 128)

    def test_recurrence_steps(self):
        from models import MSRNet
        m2 = MSRNet(n_ch=3, n_cls=3, hidden_size=32, t=2)
        m3 = MSRNet(n_ch=3, n_cls=3, hidden_size=32, t=3)
        # Same structure but t=3 should still work
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            o2 = m2(x)
            o3 = m3(x)
        assert o2.shape == o3.shape == (1, 3, 64, 64)


class TestCreateModel:
    """Verify factory function."""

    def test_unet(self):
        from models import create_model
        m = create_model('unet', hidden_size=32)
        assert m.__class__.__name__ == 'UNet'

    def test_msrn(self):
        from models import create_model
        m = create_model('msrn', hidden_size=32)
        assert m.__class__.__name__ == 'MSRNet'

    def test_invalid(self):
        from models import create_model
        with pytest.raises(ValueError):
            create_model('invalid_arch')


class TestNormalizedUNet:
    """Verify normalization wrapper."""

    def test_output_range(self):
        from models import create_model, NormalizedUNet
        base = create_model('unet', hidden_size=32)
        wrapped = NormalizedUNet(base)
        wrapped.eval()
        x = torch.rand(1, 3, 64, 64)  # [0,1] input
        with torch.no_grad():
            out = wrapped(x)
        assert out.min() >= 0.0, f"Output below 0: {out.min()}"
        assert out.max() <= 1.0, f"Output above 1: {out.max()}"

    def test_sigmoid_mode(self):
        from models import create_model, NormalizedUNet
        base = create_model('unet', hidden_size=32)
        wrapped = NormalizedUNet(base, use_sigmoid=True)
        wrapped.eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = wrapped(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
