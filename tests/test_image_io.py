"""Tests for image I/O module."""
import sys
import os
import tempfile
import numpy as np
from PIL import Image
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLoadImageAnyFormat:
    def test_png(self, tmp_path):
        from image_io import load_image_any_format
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        path = str(tmp_path / "test.png")
        img.save(path)
        loaded = load_image_any_format(path)
        assert loaded.mode == 'RGB'
        assert loaded.size == (64, 64)

    def test_jpg(self, tmp_path):
        from image_io import load_image_any_format
        img = Image.fromarray(np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8))
        path = str(tmp_path / "test.jpg")
        img.save(path)
        loaded = load_image_any_format(path)
        assert loaded.mode == 'RGB'
        assert loaded.size == (48, 32)

    def test_missing_file(self):
        from image_io import load_image_any_format
        with pytest.raises(Exception):
            load_image_any_format("/nonexistent/path.png")


class TestLoadMaskImage:
    def test_grayscale_png(self, tmp_path):
        from image_io import load_mask_image
        mask = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        img = Image.fromarray(mask, mode='L')
        path = str(tmp_path / "mask.png")
        img.save(path)
        loaded = load_mask_image(path)
        assert loaded.shape == (64, 64)
        assert loaded.dtype == np.float32
        assert 0.0 <= loaded.min() <= loaded.max() <= 1.0


class TestDenormalize:
    def test_basic(self):
        from image_io import denormalize
        # Normalized tensor: value of 0.0 should map to 0.5 in [0,1]
        t = torch.zeros(1, 3, 4, 4)
        result = denormalize(t)
        assert result.shape == (1, 3, 4, 4)
        assert torch.allclose(result, torch.full_like(result, 0.5), atol=1e-6)

    def test_range(self):
        from image_io import denormalize
        # Input in [-1, 1] should map to [0, 1]
        t = torch.tensor([[[[-1.0]], [[1.0]], [[0.0]]]])  # (1, 3, 1, 1)
        result = denormalize(t)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_none_input(self):
        from image_io import denormalize
        assert denormalize(None) is None

    def test_clamping(self):
        from image_io import denormalize
        t = torch.tensor([[[[5.0]], [[-5.0]], [[0.0]]]])
        result = denormalize(t)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
