"""Tests for training utilities — loss functions, auto-mask, helpers."""
import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDiceLoss:
    def test_perfect_match(self):
        from training.loss import dice_loss
        pred = torch.ones(1, 1, 8, 8)
        target = torch.ones(1, 1, 8, 8)
        loss = dice_loss(pred, target)
        assert loss.item() < 0.01, f"Expected ~0 for perfect match, got {loss.item()}"

    def test_no_match(self):
        from training.loss import dice_loss
        pred = torch.ones(1, 1, 8, 8)
        target = torch.zeros(1, 1, 8, 8)
        loss = dice_loss(pred, target)
        assert loss.item() > 0.9, f"Expected ~1 for no match, got {loss.item()}"

    def test_symmetry(self):
        from training.loss import dice_loss
        a = torch.rand(1, 1, 8, 8)
        b = torch.rand(1, 1, 8, 8)
        assert torch.allclose(dice_loss(a, b), dice_loss(b, a), atol=1e-6)


class TestDiffHeatmap:
    def test_output_shape(self):
        from training.loss import diff_heatmap
        a = torch.rand(3, 32, 32)
        b = torch.rand(3, 32, 32)
        hm = diff_heatmap(a, b)
        assert hm.shape == (3, 32, 32)

    def test_identical_inputs(self):
        from training.loss import diff_heatmap
        a = torch.rand(3, 16, 16)
        hm = diff_heatmap(a, a)
        assert hm.max() < 0.01, "Heatmap should be ~0 for identical inputs"


class TestRefineAutoMask:
    def test_output_shape(self):
        from training.loss import refine_auto_mask
        raw = torch.rand(2, 1, 64, 64)
        mask = refine_auto_mask(raw)
        assert mask.shape == (2, 1, 64, 64)

    def test_output_range(self):
        from training.loss import refine_auto_mask
        raw = torch.rand(1, 1, 32, 32)
        mask = refine_auto_mask(raw)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_zero_input(self):
        from training.loss import refine_auto_mask
        raw = torch.zeros(1, 1, 32, 32)
        mask = refine_auto_mask(raw)
        # All-zero input means no significant difference → mask should be ~0
        assert mask.max() < 0.1

    def test_gamma_effect(self):
        from training.loss import refine_auto_mask
        raw = torch.rand(1, 1, 32, 32) * 0.5 + 0.3
        mask_low = refine_auto_mask(raw, gamma=0.5)   # expand white
        mask_high = refine_auto_mask(raw, gamma=2.0)   # contract white
        # Lower gamma should produce higher average mask values
        assert mask_low.mean() >= mask_high.mean() - 0.1


class TestComputeAutoMask:
    def test_identical(self):
        from training.loss import compute_auto_mask
        t = torch.rand(1, 3, 32, 32)
        mask = compute_auto_mask(t, t)
        assert mask.shape == (1, 1, 32, 32)
        assert mask.max() < 0.1, "Auto-mask should be ~0 for identical src/dst"


class TestCycle:
    def test_cycles(self):
        from training.dataloader_utils import cycle
        data = [1, 2, 3]
        c = cycle(data)
        results = [next(c) for _ in range(9)]
        assert results == [1, 2, 3, 1, 2, 3, 1, 2, 3]


class TestCollateSkipNone:
    def test_filters_none(self):
        from training.dataloader_utils import collate_skip_none
        batch = [(torch.tensor([1.0]),), None, (torch.tensor([3.0]),)]
        result = collate_skip_none(batch)
        assert result is not None
        assert len(result[0]) == 2  # Two valid items

    def test_all_none(self):
        from training.dataloader_utils import collate_skip_none
        result = collate_skip_none([None, None])
        assert result is None


class TestAutoDetectNumWorkers:
    def test_returns_tuple(self):
        from training.dataloader_utils import auto_detect_num_workers
        workers, reason = auto_detect_num_workers(512, 'Windows')
        assert isinstance(workers, int)
        assert isinstance(reason, str)
        assert workers >= 0
