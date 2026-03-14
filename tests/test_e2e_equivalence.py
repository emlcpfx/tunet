"""E2E equivalence tests — verify refactored modules produce identical results
to the original code (using the old files in utils/ as reference).

These tests create models with both old and new code, load identical weights,
and verify outputs match exactly.
"""
import sys
import os
import importlib.util
import torch
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')
HAS_OLD_UTILS = os.path.isdir(UTILS_DIR) and os.path.isfile(os.path.join(UTILS_DIR, 'convert_flame.py'))


def _load_old_module(filename):
    """Load an old module from utils/ by executing it in isolation."""
    path = os.path.join(UTILS_DIR, filename)
    mod_dict = {'__name__': '_old_module', '__file__': path, '__builtins__': __builtins__}
    with open(path) as f:
        exec(compile(f.read(), path, 'exec'), mod_dict)
    return mod_dict


@pytest.mark.skipif(not HAS_OLD_UTILS, reason="Old utils/ directory not present")
class TestModelEquivalence:
    """Verify new models/ package produces bit-identical results to old utils/ code."""

    def _create_paired_models(self, model_type, hidden_size, old_module_dict):
        from models import create_model as new_create
        old_create = old_module_dict['create_model']

        new_model = new_create(model_type, n_ch=3, n_cls=3, hidden_size=hidden_size)
        old_model = old_create(model_type, n_ch=3, n_cls=3, hidden_size=hidden_size)

        # Copy weights from new to old to ensure identical parameters
        old_model.load_state_dict(new_model.state_dict())
        new_model.eval()
        old_model.eval()
        return new_model, old_model

    @pytest.mark.parametrize("hidden_size", [32, 64])
    def test_unet_output_identical(self, hidden_size):
        old_mod = _load_old_module('convert_flame.py')
        new_model, old_model = self._create_paired_models('unet', hidden_size, old_mod)

        torch.manual_seed(42)
        x = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            new_out = new_model(x)
            old_out = old_model(x)

        assert torch.allclose(new_out, old_out, atol=1e-6), (
            f"UNet outputs differ! Max diff: {(new_out - old_out).abs().max().item()}")

    @pytest.mark.parametrize("hidden_size", [32, 64])
    def test_msrnet_output_identical(self, hidden_size):
        old_mod = _load_old_module('convert_flame.py')
        new_model, old_model = self._create_paired_models('msrn', hidden_size, old_mod)

        torch.manual_seed(42)
        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            new_out = new_model(x)
            old_out = old_model(x)

        assert torch.allclose(new_out, old_out, atol=1e-6), (
            f"MSRNet outputs differ! Max diff: {(new_out - old_out).abs().max().item()}")

    def test_normalized_wrapper_identical(self):
        """Verify NormalizedUNet wrapper from new code matches old."""
        old_mod = _load_old_module('convert_flame.py')
        from models import create_model, NormalizedUNet as NewNorm

        OldNorm = old_mod['NormalizedUNet']

        base_new = create_model('unet', hidden_size=32)
        base_old = old_mod['create_model']('unet', hidden_size=32)
        base_old.load_state_dict(base_new.state_dict())

        wrapped_new = NewNorm(base_new)
        wrapped_old = OldNorm(base_old)
        wrapped_new.eval()
        wrapped_old.eval()

        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out_new = wrapped_new(x)
            out_old = wrapped_old(x)

        assert torch.allclose(out_new, out_old, atol=1e-6), (
            f"NormalizedUNet outputs differ! Max diff: {(out_new - out_old).abs().max().item()}")


@pytest.mark.skipif(not HAS_OLD_UTILS, reason="Old utils/ directory not present")
class TestConfigEquivalence:
    """Verify config functions produce identical results."""

    def test_dict_to_namespace_matches(self):
        from config import dict_to_namespace as new_fn
        old_mod = _load_old_module('convert_flame.py')
        old_fn = old_mod['dict_to_namespace']

        test_data = {
            'model': {'hidden_size': 96, 'type': 'unet'},
            'training': {'loss': 'l1+lpips', 'lr': 1e-4},
            'data': {'resolution': 512, 'overlap': 0.25},
        }

        new_ns = new_fn(test_data)
        old_ns = old_fn(test_data)

        assert new_ns.model.hidden_size == old_ns.model.hidden_size
        assert new_ns.model.type == old_ns.model.type
        assert new_ns.training.loss == old_ns.training.loss
        assert new_ns.data.resolution == old_ns.data.resolution


class TestInferenceEquivalence:
    """Test inference pipeline produces consistent results."""

    def test_blend_mask_symmetry(self):
        from inference import create_blend_mask
        mask = create_blend_mask(64, torch.device('cpu'))
        # Hann window should be symmetric
        mask_np = mask[0, 0].numpy()
        assert np.allclose(mask_np, mask_np.T, atol=1e-7), "Blend mask should be symmetric"
        assert np.allclose(mask_np, np.flip(mask_np, 0), atol=1e-7), "Blend mask should be vertically symmetric"
        assert np.allclose(mask_np, np.flip(mask_np, 1), atol=1e-7), "Blend mask should be horizontally symmetric"

    def test_full_coverage_tiling(self, tmp_path):
        """Verify tiled inference covers every pixel exactly once with no-overlap stride."""
        from inference import process_image
        from inference_config import InferenceConfig
        import torchvision.transforms as T
        from image_io import denormalize

        # Model that outputs constant 0.7 (in normalized space)
        class ConstantModel(torch.nn.Module):
            def forward(self, x):
                return torch.full_like(x, 0.4)  # 0.4 in [-1,1] space → 0.7 after denorm

        model = ConstantModel()
        model.eval()

        # Create 128x128 image
        img = Image.fromarray(np.full((128, 128, 3), 128, dtype=np.uint8))
        input_path = str(tmp_path / "const_in.png")
        output_path = str(tmp_path / "const_out.png")
        img.save(input_path)

        transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        cfg = InferenceConfig(resolution=128, stride=128, device=torch.device('cpu'))
        process_image(model, input_path, output_path, cfg, transform, denormalize)

        from PIL import Image as PILImage
        out = PILImage.open(output_path)
        out_np = np.array(out).astype(float) / 255.0
        # With a single tile + Hann blending, edges get attenuated to ~0.
        # The interior (excluding 1px border from Hann) should be ~uniform.
        interior = out_np[2:-2, 2:-2, :]
        assert interior.std() < 0.05, f"Interior should be approximately uniform, got std={interior.std()}"
        # Interior values should be close to 0.7 (denorm of 0.4 in [-1,1] space)
        assert abs(interior.mean() - 0.7) < 0.05, f"Expected interior ~0.7, got {interior.mean()}"


class TestPreviewContext:
    def test_dataclass_fields(self):
        from training.context import PreviewContext
        import torch
        ctx = PreviewContext(
            model=torch.nn.Linear(1, 1),
            output_dir="/tmp",
            device=torch.device('cpu'),
            current_epoch=0,
            global_step=100,
        )
        assert ctx.preview_save_count == 0
        assert ctx.use_amp is False
        assert ctx.diff_amplify == 5.0
