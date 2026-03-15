import os
import logging

import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from distributed import is_main_process
from image_io.image_loader import denormalize
from .context import PreviewContext
from .loss import diff_heatmap, refine_auto_mask, compute_auto_mask
from .dataloader_utils import collate_skip_none


def save_previews(ctx: PreviewContext, fixed_src_batch, fixed_dst_batch,
                  fixed_mask_batch=None, fixed_auto_mask_batch=None):
    if not is_main_process() or fixed_src_batch is None or fixed_dst_batch is None or fixed_src_batch.nelement() == 0:
        return
    has_mask = fixed_mask_batch is not None
    has_auto_mask = ctx.use_auto_mask and fixed_auto_mask_batch is not None
    num_samples = min(fixed_src_batch.size(0), 3)
    if num_samples == 0:
        return
    src_select = fixed_src_batch[:num_samples].cpu()
    dst_select = fixed_dst_batch[:num_samples].cpu()
    mask_select = fixed_mask_batch[:num_samples].cpu() if has_mask else None
    ctx.model.eval()
    device_type = ctx.device.type
    use_amp_inf = ctx.use_amp and device_type == 'cuda'
    try:
        with torch.no_grad(), autocast(device_type=device_type, enabled=use_amp_inf):
            src_dev = src_select.to(ctx.device)
            model_module = ctx.model.module if isinstance(ctx.model, DDP) else ctx.model
            if ctx.use_mask_input and mask_select is not None:
                mask_dev = mask_select.to(ctx.device)
                model_input = torch.cat([src_dev, mask_dev], dim=1)
            else:
                model_input = src_dev
            predicted_batch = model_module(model_input)
            if ctx.use_bce_dice:
                predicted_batch = torch.sigmoid(predicted_batch)
        pred_select = predicted_batch.cpu().float()
    except Exception as e:
        logging.error(f"Preview inference error (Step {ctx.global_step}): {e}")
        ctx.model.train()
        return
    ctx.model.train()
    src_denorm = denormalize(src_select)
    if ctx.use_bce_dice:
        pred_denorm = torch.clamp(pred_select, 0, 1)
        dst_denorm = torch.clamp(denormalize(dst_select), 0, 1)
    else:
        pred_denorm = denormalize(pred_select)
        dst_denorm = denormalize(dst_select)
    if src_denorm is None or pred_denorm is None or dst_denorm is None:
        return

    auto_mask_vis = None
    if has_auto_mask:
        auto_mask_refined = refine_auto_mask(fixed_auto_mask_batch[:num_samples], gamma=ctx.auto_mask_gamma)
        auto_mask_vis = auto_mask_refined.repeat(1, 3, 1, 1)

    num_grid_cols = 3 + (1 if has_mask else 0) + 1 + (1 if has_auto_mask else 0)
    if has_mask:
        mask_3ch = mask_select.repeat(1, 3, 1, 1)
        row = lambda i: [src_denorm[i], mask_3ch[i], dst_denorm[i], pred_denorm[i],
                         diff_heatmap(src_denorm[i], dst_denorm[i], amplify=ctx.diff_amplify)] + (
            [auto_mask_vis[i]] if has_auto_mask else [])
    else:
        row = lambda i: [src_denorm[i], dst_denorm[i], pred_denorm[i],
                         diff_heatmap(src_denorm[i], dst_denorm[i], amplify=ctx.diff_amplify)] + (
            [auto_mask_vis[i]] if has_auto_mask else [])

    combined = [item for i in range(num_samples) for item in row(i)]
    if not combined:
        return
    try:
        grid = make_grid(torch.stack(combined), nrow=num_grid_cols, padding=2, normalize=False)
    except Exception as e:
        logging.error(f"Preview grid error (Step {ctx.global_step}): {e}")
        return
    try:
        img_pil = T.functional.to_pil_image(grid)
    except Exception as e:
        logging.error(f"Preview PIL convert error (Step {ctx.global_step}): {e}")
        return
    preview_filename = os.path.join(ctx.output_dir, "training_preview.jpg")
    try:
        os.makedirs(os.path.dirname(preview_filename), exist_ok=True)
        tmp_filename = preview_filename + ".tmp"
        img_pil.save(tmp_filename, "JPEG", quality=90)
        os.replace(tmp_filename, preview_filename)
        refreshed = ctx.preview_refresh_rate > 0 and ctx.preview_save_count > 0 and (ctx.preview_save_count % ctx.preview_refresh_rate == 0)
        log_level = logging.INFO if (refreshed or ctx.preview_save_count == 0) else logging.DEBUG
        logging.log(log_level, f"Saved preview ({num_samples} samples) (E{ctx.current_epoch + 1}, S{ctx.global_step}, #{ctx.preview_save_count})"
                    + (" - Refreshed" if refreshed else ""))
    except Exception as e:
        logging.error(f"Failed to save preview image: {e}")


def capture_preview_batch(config, src_transforms, dst_transforms, shared_transforms,
                          standard_transform, use_masks=False, use_auto_mask=False,
                          skip_empty=False, skip_empty_threshold=1.0,
                          AugmentedDatasetClass=None):
    """Capture a fixed batch of preview samples from the training dataset."""
    if not is_main_process():
        return None, None, None, None
    # Import here to avoid circular imports
    if AugmentedDatasetClass is None:
        from train import AugmentedImagePairSlicingDataset as AugmentedDatasetClass
    num_preview_samples = 3
    try:
        preview_dataset = AugmentedDatasetClass(
            config.data.src_dir, config.data.dst_dir, config.data.resolution,
            config.data.overlap_factor, src_transforms, dst_transforms,
            shared_transforms, standard_transform,
            mask_dir=config.data.mask_dir if use_masks else None,
            use_auto_mask=use_auto_mask, skip_empty_patches=skip_empty,
            skip_empty_threshold=skip_empty_threshold)
        if len(preview_dataset) == 0:
            return None, None, None, None
        num_to_load = min(num_preview_samples, len(preview_dataset))
        preview_loader = DataLoader(preview_dataset, batch_size=num_to_load, shuffle=True,
                                    num_workers=0, collate_fn=collate_skip_none)
        batch_data = next(iter(preview_loader))
        if batch_data is None:
            return None, None, None, None
        has_third = use_masks or use_auto_mask
        if has_third:
            fixed_src, fixed_dst, third = batch_data
            if fixed_src is not None and fixed_dst is not None and fixed_src.size(0) > 0:
                fixed_mask = third.cpu() if use_masks else None
                fixed_auto = third.cpu() if use_auto_mask else None
                return fixed_src.cpu(), fixed_dst.cpu(), fixed_mask, fixed_auto
        else:
            fixed_src, fixed_dst = batch_data
            if fixed_src is not None and fixed_dst is not None and fixed_src.size(0) > 0:
                return fixed_src.cpu(), fixed_dst.cpu(), None, None
        return None, None, None, None
    except Exception as e:
        logging.error(f"Error capturing preview batch: {e}")
        return None, None, None, None


def save_val_previews(ctx: PreviewContext, fixed_src_batch, fixed_dst_batch):
    """Save validation preview grid to val_preview.jpg."""
    if not is_main_process() or fixed_src_batch is None or fixed_src_batch.nelement() == 0:
        return
    has_dst = fixed_dst_batch is not None
    num_samples = min(fixed_src_batch.size(0), 3)
    if num_samples == 0:
        return
    src_select = fixed_src_batch[:num_samples].cpu()
    dst_select = fixed_dst_batch[:num_samples].cpu() if has_dst else None
    ctx.model.eval()
    device_type = ctx.device.type
    use_amp_inf = ctx.use_amp and device_type == 'cuda'
    try:
        with torch.no_grad(), autocast(device_type=device_type, enabled=use_amp_inf):
            model_module = ctx.model.module if isinstance(ctx.model, DDP) else ctx.model
            model_input = src_select.to(ctx.device)
            if ctx.use_mask_input:
                dummy_mask = torch.zeros(num_samples, 1, src_select.shape[2], src_select.shape[3], device=ctx.device)
                model_input = torch.cat([model_input, dummy_mask], dim=1)
            predicted_batch = model_module(model_input)
            if ctx.use_bce_dice:
                predicted_batch = torch.sigmoid(predicted_batch)
        pred_select = predicted_batch.cpu().float()
    except Exception as e:
        logging.error(f"Val preview inference error (Step {ctx.global_step}): {e}")
        ctx.model.train()
        return
    ctx.model.train()
    src_denorm = denormalize(src_select)
    if ctx.use_bce_dice:
        pred_denorm = torch.clamp(pred_select, 0, 1)
        dst_denorm = torch.clamp(denormalize(dst_select), 0, 1) if has_dst else None
    else:
        pred_denorm = denormalize(pred_select)
        dst_denorm = denormalize(dst_select) if has_dst else None
    if src_denorm is None or pred_denorm is None:
        return
    auto_mask_vis = None
    if ctx.use_auto_mask and has_dst:
        auto_mask_batch = compute_auto_mask(src_select, dst_select)
        auto_mask_vis = auto_mask_batch.repeat(1, 3, 1, 1)
    if has_dst:
        row = lambda i: [src_denorm[i], dst_denorm[i], pred_denorm[i],
                         diff_heatmap(src_denorm[i], dst_denorm[i], amplify=ctx.diff_amplify)] + (
            [auto_mask_vis[i]] if auto_mask_vis is not None else [])
        nrow = 4 + (1 if auto_mask_vis is not None else 0)
    else:
        row = lambda i: [src_denorm[i], pred_denorm[i],
                         diff_heatmap(src_denorm[i], pred_denorm[i], amplify=ctx.diff_amplify)]
        nrow = 3
    combined = [item for i in range(num_samples) for item in row(i)]
    if not combined:
        return
    try:
        grid = make_grid(torch.stack(combined), nrow=nrow, padding=2, normalize=False)
    except Exception as e:
        logging.error(f"Val preview grid error: {e}")
        return
    try:
        img_pil = T.functional.to_pil_image(grid)
    except Exception as e:
        logging.error(f"Val preview PIL error: {e}")
        return
    preview_filename = os.path.join(ctx.output_dir, "val_preview.jpg")
    try:
        tmp_filename = preview_filename + ".tmp"
        img_pil.save(tmp_filename, "JPEG", quality=90)
        os.replace(tmp_filename, preview_filename)
    except Exception as e:
        logging.error(f"Failed to save val preview: {e}")


def capture_val_preview_batch(config, standard_transform, SourceOnlyDatasetClass=None,
                              AugmentedDatasetClass=None, use_auto_mask=False,
                              skip_empty=False, skip_empty_threshold=1.0):
    """Capture a fixed batch from the validation dataset for previews."""
    if not is_main_process():
        return None, None
    if not config.data.val_src_dir:
        return None, None
    # Import here to avoid circular imports
    if SourceOnlyDatasetClass is None:
        from train import SourceOnlySlicingDataset as SourceOnlyDatasetClass
    if AugmentedDatasetClass is None:
        from train import AugmentedImagePairSlicingDataset as AugmentedDatasetClass
    num_preview_samples = 3
    try:
        if config.data.val_dst_dir:
            try:
                val_dataset = AugmentedDatasetClass(
                    config.data.val_src_dir, config.data.val_dst_dir,
                    config.data.resolution, config.data.overlap_factor,
                    None, None, None, standard_transform,
                    use_auto_mask=use_auto_mask, skip_empty_patches=skip_empty,
                    skip_empty_threshold=skip_empty_threshold)
                if len(val_dataset) > 0:
                    num_to_load = min(num_preview_samples, len(val_dataset))
                    loader = DataLoader(val_dataset, batch_size=num_to_load, shuffle=True,
                                        num_workers=0, collate_fn=collate_skip_none)
                    batch_data = next(iter(loader))
                    if batch_data is not None:
                        if use_auto_mask and len(batch_data) == 3:
                            fixed_src, fixed_dst, _ = batch_data
                        else:
                            fixed_src, fixed_dst = batch_data
                        if fixed_src is not None and fixed_dst is not None and fixed_src.size(0) > 0:
                            return fixed_src.cpu(), fixed_dst.cpu()
            except Exception as e:
                logging.info(f"Val paired dataset failed ({e}), falling back to src-only preview.")
        # Fallback: src-only
        val_ds = SourceOnlyDatasetClass(config.data.val_src_dir, config.data.resolution,
                                        config.data.overlap_factor, standard_transform)
        if len(val_ds) == 0:
            return None, None
        num_to_load = min(num_preview_samples, len(val_ds))
        loader = DataLoader(val_ds, batch_size=num_to_load, shuffle=True,
                            num_workers=0, collate_fn=collate_skip_none)
        fixed_src = next(iter(loader))
        if fixed_src is not None and fixed_src.size(0) > 0:
            return fixed_src.cpu(), None
        return None, None
    except Exception as e:
        logging.error(f"Val preview capture error: {e}")
        return None, None
