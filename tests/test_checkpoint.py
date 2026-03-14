"""Tests for checkpoint management."""
import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPruneCheckpoints:
    def test_prune_keeps_latest(self, tmp_path):
        from training.checkpoint import prune_checkpoints
        # Create fake checkpoint files
        for i in range(5):
            path = tmp_path / f"test_tunet_epoch_{i:09d}.pth"
            path.write_text(f"epoch {i}")
        prune_checkpoints(str(tmp_path), keep_last=2, ckpt_prefix='test')
        remaining = sorted(tmp_path.glob("*_epoch_*.pth"))
        assert len(remaining) == 2
        # Should keep the two highest epochs
        names = [f.name for f in remaining]
        assert 'test_tunet_epoch_000000004.pth' in names
        assert 'test_tunet_epoch_000000003.pth' in names

    def test_prune_keep_zero(self, tmp_path):
        from training.checkpoint import prune_checkpoints
        for i in range(3):
            (tmp_path / f"test_tunet_epoch_{i:09d}.pth").write_text("x")
        prune_checkpoints(str(tmp_path), keep_last=0, ckpt_prefix='test')
        remaining = list(tmp_path.glob("*_epoch_*.pth"))
        assert len(remaining) == 0

    def test_prune_disabled(self, tmp_path):
        from training.checkpoint import prune_checkpoints
        for i in range(3):
            (tmp_path / f"test_tunet_epoch_{i:09d}.pth").write_text("x")
        prune_checkpoints(str(tmp_path), keep_last=-1, ckpt_prefix='test')
        remaining = list(tmp_path.glob("*_epoch_*.pth"))
        assert len(remaining) == 3

    def test_fewer_than_keep(self, tmp_path):
        from training.checkpoint import prune_checkpoints
        (tmp_path / "test_tunet_epoch_000000000.pth").write_text("x")
        prune_checkpoints(str(tmp_path), keep_last=5, ckpt_prefix='test')
        remaining = list(tmp_path.glob("*_epoch_*.pth"))
        assert len(remaining) == 1


class TestCheckpointRoundTrip:
    """Verify model checkpoint save/load preserves weights exactly."""

    def test_unet_save_load(self, tmp_path):
        from models import create_model
        from config import config_to_dict
        from types import SimpleNamespace

        model = create_model('unet', hidden_size=32)
        config = SimpleNamespace(
            model=SimpleNamespace(model_size_dims=32, model_type='unet'),
            training=SimpleNamespace(loss='l1'),
            data=SimpleNamespace(resolution=256),
        )
        ckpt_path = str(tmp_path / "test_ckpt.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config_to_dict(config),
            'effective_model_size': 32,
            'model_type': 'unet',
            'n_input_channels': 3,
            'global_step': 100,
            'epoch': 5,
        }, ckpt_path)

        # Load it back
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model2 = create_model('unet', hidden_size=32)
        model2.load_state_dict(ckpt['model_state_dict'])

        # Verify weights are identical
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert k1 == k2
            assert torch.equal(v1, v2), f"Weight mismatch at {k1}"

    def test_msrnet_save_load(self, tmp_path):
        from models import create_model

        model = create_model('msrn', hidden_size=32, t=2)
        ckpt_path = str(tmp_path / "msrn_ckpt.pth")
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model2 = create_model('msrn', hidden_size=32, t=2)
        model2.load_state_dict(ckpt['model_state_dict'])

        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert torch.equal(v1, v2), f"Weight mismatch at {k1}"
