import logging
import math

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed import is_main_process
from .loss import dice_loss


def _psnr(pred, target):
    """Compute PSNR between two tensors in [-1, 1] range.
    Converts to [0, 1] first. Returns scalar float."""
    pred_01 = pred * 0.5 + 0.5
    target_01 = target * 0.5 + 0.5
    mse = (pred_01 - target_01).pow(2).mean()
    if mse < 1e-10:
        return 100.0
    return (10.0 * math.log10(1.0 / mse.item()))


def _ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """Compute mean SSIM between two tensors in [-1, 1] range.
    Uses a simple uniform-window approach (no Gaussian) for speed.
    Returns scalar float."""
    pred_01 = pred * 0.5 + 0.5
    target_01 = target * 0.5 + 0.5

    # Use average pooling as a fast uniform window
    pad = window_size // 2
    # Reshape to (B*C, 1, H, W) for depthwise pooling
    B, C, H, W = pred_01.shape
    p = pred_01.reshape(B * C, 1, H, W)
    t = target_01.reshape(B * C, 1, H, W)

    mu_p = torch.nn.functional.avg_pool2d(p, window_size, stride=1, padding=pad)
    mu_t = torch.nn.functional.avg_pool2d(t, window_size, stride=1, padding=pad)

    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p_sq = torch.nn.functional.avg_pool2d(p * p, window_size, stride=1, padding=pad) - mu_p_sq
    sigma_t_sq = torch.nn.functional.avg_pool2d(t * t, window_size, stride=1, padding=pad) - mu_t_sq
    sigma_pt = torch.nn.functional.avg_pool2d(p * t, window_size, stride=1, padding=pad) - mu_pt

    numerator = (2 * mu_pt + C1) * (2 * sigma_pt + C2)
    denominator = (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean().item()


def run_validation(model, val_dataloader, device, device_type, use_amp, use_lpips,
                   loss_fn_lpips, use_bce_dice, criterion_l1, current_ep_idx,
                   global_step, lambda_lpips=1.0, use_mask_input=False,
                   val_dataset=None):
    """Run validation over entire val dataset, return avg losses.

    Also computes PSNR and SSIM metrics, and logs the worst-performing batch
    with copy-pastable file paths.
    """
    if not is_main_process() or val_dataloader is None:
        return None, None
    model.eval()
    model_module = model.module if isinstance(model, DDP) else model
    total_l1, total_lp, count = 0.0, 0.0, 0
    total_psnr, total_ssim = 0.0, 0.0
    use_amp_inf = use_amp and device_type == 'cuda'

    # Track worst batch for per-image analysis
    worst_l1 = -1.0
    worst_batch_idx = -1

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch is None:
                    continue
                src, dst = batch[0].to(device), batch[1].to(device)
                with autocast(device_type=device_type, enabled=use_amp_inf):
                    model_in = src
                    if use_mask_input:
                        dummy_mask = torch.zeros(src.shape[0], 1, src.shape[2], src.shape[3], device=device)
                        model_in = torch.cat([model_in, dummy_mask], dim=1)
                    out = model_module(model_in)
                    if use_bce_dice:
                        dst_01 = dst * 0.5 + 0.5
                        n_ch = out.shape[1]
                        bce_total = torch.tensor(0.0, device=device)
                        dc_total = torch.tensor(0.0, device=device)
                        criterion_bce = nn.BCEWithLogitsLoss()
                        for ch in range(n_ch):
                            out_ch = out[:, ch:ch + 1, :, :]
                            dst_ch = dst_01[:, ch:ch + 1, :, :]
                            bce_total = bce_total + criterion_bce(out_ch, dst_ch)
                            dc_total = dc_total + dice_loss(torch.sigmoid(out_ch), dst_ch)
                        l1 = (bce_total + dc_total) / n_ch
                        lp = torch.tensor(0.0, device=device)
                        # For PSNR/SSIM, use sigmoid output for BCE+Dice
                        out_for_metrics = torch.sigmoid(out) * 2.0 - 1.0  # back to [-1, 1]
                    else:
                        l1 = criterion_l1(out, dst)
                        lp = torch.tensor(0.0, device=device)
                        if use_lpips and loss_fn_lpips:
                            try:
                                lp = loss_fn_lpips(out, dst).mean()
                            except Exception:
                                lp = torch.tensor(0.0, device=device)
                        out_for_metrics = out

                l1_val = l1.item()
                total_l1 += l1_val
                total_lp += lp.item()

                # Compute PSNR and SSIM
                batch_psnr = _psnr(out_for_metrics.float(), dst.float())
                batch_ssim = _ssim(out_for_metrics.float(), dst.float())
                total_psnr += batch_psnr
                total_ssim += batch_ssim

                # Track worst batch
                if l1_val > worst_l1:
                    worst_l1 = l1_val
                    worst_batch_idx = batch_idx

                count += 1
    except Exception as e:
        logging.error(f"Validation error: {e}")
        model.train()
        return None, None
    model.train()
    if count == 0:
        return None, None
    avg_l1, avg_lp = total_l1 / count, total_lp / count
    avg_psnr, avg_ssim = total_psnr / count, total_ssim / count
    loss_label = 'BCE+Dice' if use_bce_dice else 'L1'
    log_msg = f'Val Epoch[{current_ep_idx + 1}] Step[{global_step}], Val_{loss_label}:{avg_l1:.4f}'
    if use_lpips:
        log_msg += f', Val_LPIPS:{avg_lp:.4f}'
    log_msg += f', PSNR:{avg_psnr:.2f}dB, SSIM:{avg_ssim:.4f}'
    logging.info(log_msg)

    # Log worst batch info with copy-pastable file paths
    if count > 1 and worst_batch_idx >= 0:
        batch_size = val_dataloader.batch_size or 1
        start_idx = worst_batch_idx * batch_size
        worst_paths = []
        if val_dataset is not None and hasattr(val_dataset, 'slice_info'):
            end_idx = min(start_idx + batch_size, len(val_dataset.slice_info))
            seen = set()
            for i in range(start_idx, end_idx):
                p = val_dataset.slice_info[i].get('src_path', '')
                if p and p not in seen:
                    worst_paths.append(p)
                    seen.add(p)
        logging.info(f'  Worst val batch: #{worst_batch_idx} ({loss_label}={worst_l1:.4f}, '
                     f'{worst_l1 / avg_l1:.1f}x avg)')
        if worst_paths:
            logging.info(f'  Worst batch files ({len(worst_paths)}):')
            for p in worst_paths:
                logging.info(f'    {p}')

    return avg_l1, avg_lp
