"""Tests for inference module — tiled processing and blending."""
import sys
import os
import tempfile
import numpy as np
from PIL import Image
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCreateBlendMask:
    def test_shape(self):
        from inference import create_blend_mask
        device = torch.device('cpu')
        mask = create_blend_mask(256, device)
        assert mask.shape == (1, 1, 256, 256)

    def test_hann_properties(self):
        from inference import create_blend_mask
        mask = create_blend_mask(64, torch.device('cpu'))
        # Hann window: center should be max, edges should be ~0
        center_val = mask[0, 0, 32, 32].item()
        corner_val = mask[0, 0, 0, 0].item()
        assert center_val > corner_val
        assert corner_val < 0.01
        assert center_val > 0.9


class TestProcessImage:
    def test_identity_model(self, tmp_path):
        """Test tiled inference with a simple identity-like model."""
        from inference import process_image
        from inference_config import InferenceConfig
        import torchvision.transforms as T
        from image_io import denormalize, NORM_MEAN, NORM_STD

        # Create a simple model that passes input through
        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x

        model = IdentityModel()
        model.eval()

        # Create a test image
        img_np = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_np)
        input_path = str(tmp_path / "input.png")
        output_path = str(tmp_path / "output.png")
        img.save(input_path)

        transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        cfg = InferenceConfig(
            resolution=64, stride=32, device=torch.device('cpu'),
            batch_size=4, use_amp=False)
        process_image(model, input_path, output_path, cfg, transform, denormalize)

        assert os.path.exists(output_path), "Output file should exist"
        out_img = Image.open(output_path)
        assert out_img.size == (128, 128), f"Output size should match input, got {out_img.size}"

    def test_too_small_image(self, tmp_path):
        """Image smaller than resolution should be skipped."""
        from inference import process_image
        from inference_config import InferenceConfig
        import torchvision.transforms as T
        from image_io import denormalize

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return x

        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        input_path = str(tmp_path / "tiny.png")
        output_path = str(tmp_path / "tiny_out.png")
        img.save(input_path)

        transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        cfg = InferenceConfig(resolution=64, stride=32, device=torch.device('cpu'))
        process_image(DummyModel(), input_path, output_path, cfg, transform, denormalize)

        # Should not produce output for too-small image
        assert not os.path.exists(output_path)


class TestInferenceConfig:
    def test_defaults(self):
        from inference_config import InferenceConfig
        cfg = InferenceConfig(resolution=512, stride=256, device=torch.device('cpu'))
        assert cfg.batch_size == 1
        assert cfg.use_amp is False
        assert cfg.half_res is False
        assert cfg.loss_mode == 'l1'

    def test_custom(self):
        from inference_config import InferenceConfig
        cfg = InferenceConfig(
            resolution=256, stride=128, device=torch.device('cpu'),
            batch_size=8, use_amp=True, half_res=True, loss_mode='bce+dice')
        assert cfg.batch_size == 8
        assert cfg.use_amp is True
        assert cfg.half_res is True
        assert cfg.loss_mode == 'bce+dice'
